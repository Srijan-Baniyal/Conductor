"""HTTP API for the harness. Standard library only. Run it with `python -m harness --serve`.

Endpoints (JSON unless noted):
  GET    /api/health
  GET    /api/agents                       agent configs with their tools
  GET    /api/tools                        tool schemas
  GET    /api/models                       model presets (never includes API keys)
  GET    /api/sessions/<id>                saved messages of a session
  DELETE /api/sessions/<id>
  GET    /api/notes?session=<id>&agent=    notes visible to an agent
  POST   /api/chat                         body {agent, message, session?, model?}; response is a
                                           text/event-stream of harness events
  GET    /files/<session>/<name>           files a tool wrote (plots)

Every event on /api/chat is one `data: {...}` line with a "type" field: session, llm_request, token,
llm_response, tool_start, tool_end, error, final, done. API keys stay in the server's environment.
"""
from __future__ import annotations
import json, mimetypes, re, threading, time, uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .agent import Agent
from .config import discover, load_agent_config
from .hooks import Hooks
from .llm import MockLLM, make_client
from .memory import ConversationMemory, NotesStore
from .tools.registry import ToolRegistry

SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_BODY = 1_000_000


@dataclass
class ModelPreset:
    id: str
    label: str
    provider: str = "openai"        # openai | anthropic | mock
    model: str = ""
    base_url: str | None = None
    extra_body: dict = field(default_factory=dict)

    def public(self) -> dict:
        return {"id": self.id, "label": self.label, "provider": self.provider, "model": self.model,
                "base_url": self.base_url, "extra_body": self.extra_body}

    def client(self):
        if self.provider == "mock":
            return MockLLM()
        return make_client(self.provider, self.model, self.base_url, self.extra_body or None)


class App:
    def __init__(self, root: Path, agents_dirs: list[Path], registry: ToolRegistry,
                 presets: list[ModelPreset], cors_origin: str = "http://localhost:3000", auto_approve: bool = False):
        self.root, self.registry, self.presets = Path(root), registry, {p.id: p for p in presets}
        self.default_preset = presets[0].id
        self.agents = discover(*agents_dirs)
        self.cors_origin, self.auto_approve = cors_origin, auto_approve

    # ---- lookups ----
    def agent_info(self, name: str) -> dict:
        cfg = load_agent_config(self.agents[name])
        tools = []
        for t in cfg.tools:
            tool = self.registry.get(t)
            tools.append({"name": t, "description": tool.description if tool else "", "isolated": bool(tool and tool.isolated)})
        return {"name": cfg.name, "description": cfg.description, "system_prompt": cfg.system_prompt,
                "tools": tools, "max_steps": cfg.max_steps, "tool_mode": cfg.tool_mode,
                "notes_scope": cfg.notes_scope, "temperature": cfg.temperature}

    def list_agents(self) -> list[dict]:
        return [self.agent_info(n) for n in sorted(self.agents)]

    def list_tools(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "parameters": t.parameters, "timeout": t.timeout,
                 "isolated": t.isolated, "requires_approval": t.requires_approval}
                for t in (self.registry.get(n) for n in self.registry.names())]

    def session_path(self, session: str) -> Path:
        return self.root / "sessions" / f"{session}.jsonl"

    def notes_for(self, session: str, agent: str) -> list[dict]:
        scope = load_agent_config(self.agents[agent]).notes_scope if agent in self.agents else "session"
        key = "global" if scope == "global" else session
        return NotesStore(self.root / "notes" / f"{key}.json").all()

    def file_url(self, session: str, text: str) -> list[str]:
        base = (self.root / "workspace" / session).resolve()
        urls = []
        for m in re.finditer(r"(/[^\s'\"`]+\.(?:png|jpg|jpeg|svg|gif))", text):
            p = Path(m.group(1))
            try:
                if p.resolve().is_relative_to(base):
                    urls.append(f"/files/{session}/{p.name}")
            except OSError:
                continue
        return urls

    # ---- chat ----
    def run_chat(self, body: dict, emit) -> None:
        agent_name = body.get("agent") or "general"
        message = str(body.get("message", "")).strip()
        session = body.get("session") or uuid.uuid4().hex[:8]
        if agent_name not in self.agents:
            raise ValueError(f"unknown agent '{agent_name}'")
        if not message:
            raise ValueError("message is empty")
        if not SESSION_RE.match(session):
            raise ValueError("bad session id")
        preset = self.presets.get(body.get("model") or self.default_preset)
        if preset is None:
            raise ValueError("unknown model preset")

        t0 = time.perf_counter()
        hooks = Hooks()

        def forward(event: str, data: dict) -> None:
            payload = {"type": event, "t": round(time.perf_counter() - t0, 3), **data}
            if event == "tool_end":
                payload["files"] = self.file_url(session, data.get("output", ""))
            emit(payload)

        hooks.on("*", forward)
        emit({"type": "session", "session": session, "agent": agent_name, "model": preset.id})
        cfg = load_agent_config(self.agents[agent_name])
        agent = Agent(cfg, preset.client(), self.registry, root=self.root, session_id=session,
                      memory=ConversationMemory(self.session_path(session)), hooks=hooks, stream=True,
                      approve=lambda tool, args: self.auto_approve)
        result = agent.run(message)
        emit({"type": "done", "stopped": result.stopped, "steps": result.steps, "tool_calls": result.tool_calls,
              "usage": result.usage, "final": result.final, "ms": round((time.perf_counter() - t0) * 1000)})


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "UniversalAgentHarness"

        def log_message(self, *args) -> None:
            pass

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", app.cors_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Vary", "Origin")

        def _json(self, obj, status: int = 200) -> None:
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self._cors()
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, msg: str) -> None:
            self._json({"error": msg}, status)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self) -> None:
            url = urlparse(self.path)
            parts = [unquote(p) for p in url.path.strip("/").split("/")]
            q = parse_qs(url.query)
            try:
                if parts == ["api", "health"]:
                    return self._json({"ok": True})
                if parts == ["api", "agents"]:
                    return self._json(app.list_agents())
                if parts == ["api", "tools"]:
                    return self._json(app.list_tools())
                if parts == ["api", "models"]:
                    return self._json({"default": app.default_preset, "models": [p.public() for p in app.presets.values()]})
                if parts[:2] == ["api", "sessions"] and len(parts) == 3:
                    if not SESSION_RE.match(parts[2]):
                        return self._error(400, "bad session id")
                    mem = ConversationMemory(app.session_path(parts[2]))
                    return self._json({"session": parts[2], "messages": [m.to_dict() for m in mem.messages]})
                if parts == ["api", "notes"]:
                    session = (q.get("session") or [""])[0]
                    if session and not SESSION_RE.match(session):
                        return self._error(400, "bad session id")
                    return self._json(app.notes_for(session or "none", (q.get("agent") or ["general"])[0]))
                if parts[0] == "files" and len(parts) == 3:
                    return self._file(parts[1], parts[2])
                self._error(404, "not found")
            except KeyError as e:
                self._error(404, f"not found: {e}")

        def _file(self, session: str, name: str) -> None:
            if not SESSION_RE.match(session) or "/" in name or name.startswith("."):
                return self._error(400, "bad path")
            path = app.root / "workspace" / session / name
            if not path.is_file():
                return self._error(404, "file not found")
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self._cors()
            self.end_headers()
            self.wfile.write(data)

        def do_DELETE(self) -> None:
            parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/")]
            if parts[:2] == ["api", "sessions"] and len(parts) == 3 and SESSION_RE.match(parts[2]):
                ConversationMemory(app.session_path(parts[2])).reset()
                return self._json({"ok": True})
            self._error(404, "not found")

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/chat":
                return self._error(404, "not found")
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    return self._error(413, "body too large")
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._error(400, "invalid JSON")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self._cors()
            self.end_headers()
            lock, alive = threading.Lock(), [True]

            def emit(payload: dict) -> None:
                if not alive[0]:
                    return
                line = f"data: {json.dumps(payload, default=str)}\n\n".encode()
                with lock:
                    try:
                        self.wfile.write(line)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionError):
                        alive[0] = False

            try:
                app.run_chat(body, emit)
            except ValueError as e:
                emit({"type": "error", "where": "request", "error": str(e)})
            except Exception as e:  # keep the server alive whatever a model or tool does
                emit({"type": "error", "where": "server", "error": f"{type(e).__name__}: {e}"})
            self.close_connection = True

    return Handler


def serve(app: App, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Create the server. Call .serve_forever() on the result."""
    srv = ThreadingHTTPServer((host, port), make_handler(app))
    srv.daemon_threads = True
    return srv


__all__ = ["App", "ModelPreset", "serve"]
