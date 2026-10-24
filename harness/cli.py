from __future__ import annotations
import argparse, os, sys
from pathlib import Path
from .agent import Agent, llm_summarizer
from .config import discover, load_agent_config
from .hooks import Hooks, JsonlTracer
from .llm import AnthropicClient, MockLLM, OpenAICompatClient
from .memory import ConversationMemory
from .tools.registry import ToolRegistry

BUNDLED = Path(__file__).parent / "agents"
HELP = "/agents  /agent NAME  /tools  /notes  /usage  /reset  /quit"


def _approve(tool, args) -> bool:
    return input(f"  allow {tool.name}({args})? [y/N] ").strip().lower() == "y"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="harness", description="Universal Agent Harness")
    p.add_argument("--agent", "--mode", default="general", help="agent name or path to a .toml file")
    p.add_argument("--agents-dir", action="append", default=[], help="extra folder of agent .toml files")
    p.add_argument("--tools-dir", action="append", default=[], help="folder of plugin .py files")
    p.add_argument("--model", default=os.getenv("HARNESS_MODEL"))
    p.add_argument("--provider", choices=["openai", "anthropic"], default=os.getenv("HARNESS_PROVIDER", "openai"),
                   help="openai (any OpenAI-compatible endpoint) or anthropic (native Messages API)")
    p.add_argument("--base-url", default=None, help="endpoint override, e.g. http://localhost:11434/v1")
    p.add_argument("--stream", action="store_true", help="print tokens as they arrive (if the client supports it)")
    p.add_argument("--mock", action="store_true", help="offline demo model, no API key")
    p.add_argument("--session", default=None, help="session id; reuse to resume")
    p.add_argument("--root", default=".harness")
    p.add_argument("--trace", default=None, help="write a JSONL trace of every event")
    p.add_argument("--summarize", action="store_true", help="LLM-summarize history that leaves the window")
    p.add_argument("-y", "--yes", action="store_true", help="auto-approve tools that require approval")
    p.add_argument("-p", "--prompt", help="run one prompt and exit")
    a = p.parse_args(argv)

    registry = ToolRegistry()
    registry.load_builtin()
    registry.load_entry_points()
    for d in a.tools_dir:
        registry.load_dir(d)
    agents = discover(BUNDLED, *a.agents_dir)

    if a.mock:
        llm = MockLLM()
    elif a.model:
        llm = AnthropicClient(a.model, a.base_url) if a.provider == "anthropic" else OpenAICompatClient(a.model, a.base_url)
    else:
        p.error("set --model (or HARNESS_MODEL), or use --mock")

    hooks = Hooks()
    if a.trace:
        hooks.on("*", JsonlTracer(a.trace))
    streamed = [False]

    def on_token(_event: str, data: dict) -> None:
        streamed[0] = True
        print(data["text"], end="", flush=True)

    if a.stream:
        hooks.on("token", on_token)

    def show(final: str) -> None:
        print() if streamed[0] else print(final)
        streamed[0] = False

    session = a.session or os.urandom(4).hex()
    memory = ConversationMemory(Path(a.root) / "sessions" / f"{session}.jsonl")

    def build(name: str) -> Agent:
        path = Path(name) if name.endswith(".toml") else agents.get(name)
        if not path or not path.exists():
            raise SystemExit(f"unknown agent '{name}'. available: {', '.join(sorted(agents))}")
        return Agent(load_agent_config(path), llm, registry, root=a.root, session_id=session, memory=memory,
                     hooks=hooks, approve=(lambda t, x: True) if a.yes else _approve, stream=a.stream,
                     summarizer=llm_summarizer(llm) if a.summarize else None)

    agent = build(a.agent)
    if a.prompt:
        r = agent.run(a.prompt)
        show(r.final)
        return 0 if r.stopped != "error" else 1

    print(f"agent={agent.cfg.name} session={session}  ({HELP})")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return 0
        if line == "/agents":
            for n, f in sorted(agents.items()):
                print(f"  {n}: {load_agent_config(f).description}")
        elif line.startswith("/agent "):
            agent = build(line.split(None, 1)[1])
            print(f"switched to {agent.cfg.name} (history kept)")
        elif line == "/tools":
            print("  " + ", ".join(agent.registry.names()) or "none")
        elif line == "/notes":
            print("\n".join(f"  #{n['id']} [{n['tag']}] {n['text']}" for n in agent.ctx.notes.all()) or "  none")
        elif line == "/reset":
            memory.reset()
            print("history cleared")
        elif line == "/usage":
            print(f"  messages={len(memory.messages)} summary={'yes' if memory.summary else 'no'}")
        else:
            r = agent.run(line)
            show(r.final)
            print(f"  [{r.stopped}, {r.steps} steps, {r.tool_calls} tool calls, tokens={r.usage or 'n/a'}]")
