from __future__ import annotations
import concurrent.futures as cf
import inspect, json, os, signal, subprocess, sys, time, typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass
class ToolContext:
    workdir: Path
    session_id: str
    notes: Any = None  # NotesStore


@dataclass
class ToolResult:
    ok: bool
    output: str
    ms: float = 0.0


_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def _coerce(v: Any, t: str | None) -> Any:
    if t == "string":
        if isinstance(v, (dict, list)):
            raise TypeError
        return str(v)
    if t in ("integer", "number"):
        if isinstance(v, bool):
            raise TypeError
        f = float(v)
        if t == "number":
            return f
        if not f.is_integer():
            raise ValueError
        return int(f)
    if t == "boolean":
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.lower() in ("true", "false"):
            return v.lower() == "true"
        raise TypeError
    if t == "array":
        if isinstance(v, str):
            v = json.loads(v)
        if not isinstance(v, list):
            raise TypeError
        return v
    if t == "object":
        if isinstance(v, str):
            v = json.loads(v)
        if not isinstance(v, dict):
            raise TypeError
        return v
    return v


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable
    timeout: float = 20.0
    requires_approval: bool = False
    wants_ctx: bool = False
    isolated: bool = False

    def schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                  "parameters": self.parameters}}

    def prepare(self, args: dict) -> tuple[dict, list[str]]:
        if "__raw__" in args:
            return args, [f"arguments were not valid JSON: {str(args['__raw__'])[:200]}"]
        props, errs, out = self.parameters.get("properties", {}), [], {}
        for k in self.parameters.get("required", []):
            if k not in args:
                errs.append(f"missing required argument '{k}'")
        for k, v in args.items():
            if k not in props:
                errs.append(f"unknown argument '{k}' (allowed: {', '.join(props)})")
                continue
            spec = props[k]
            try:
                v = _coerce(v, spec.get("type"))
            except (TypeError, ValueError):
                errs.append(f"argument '{k}' must be of type {spec.get('type')}")
                continue
            if "enum" in spec and v not in spec["enum"]:
                errs.append(f"argument '{k}' must be one of {spec['enum']}")
                continue
            out[k] = v
        return out, errs

    def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        start = time.perf_counter()
        args, errs = self.prepare(args)
        if errs:
            return ToolResult(False, "Invalid arguments: " + "; ".join(errs))
        if self.isolated:
            ok, out = self._run_isolated(args, ctx)
            return ToolResult(ok, out, (time.perf_counter() - start) * 1000)
        pool = cf.ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(lambda: self.fn(**args, ctx=ctx) if self.wants_ctx else self.fn(**args))
        try:
            out, ok = str(fut.result(timeout=self.timeout)), True
        except cf.TimeoutError:
            out, ok = f"Timed out after {self.timeout}s", False
        except Exception as e:  # tool bugs and bad inputs must reach the model, not crash the loop
            out, ok = f"{type(e).__name__}: {e}", False
        finally:
            pool.shutdown(wait=False)
        return ToolResult(ok, out, (time.perf_counter() - start) * 1000)

    def _run_isolated(self, args: dict, ctx: ToolContext | None) -> tuple[bool, str]:
        """Run in a child interpreter and kill its process group on timeout. The child gets the
        workdir and session id but not the notes store, and args and output must be JSON."""
        job = {"module": self.fn.__module__, "file": inspect.getsourcefile(self.fn), "name": self.name,
               "args": args, "sys_path": sys.path, "workdir": str(ctx.workdir) if ctx else "workspace",
               "session_id": ctx.session_id if ctx else ""}
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
        proc = subprocess.Popen([sys.executable, "-m", "harness.tools._isolated_runner"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True,
                                start_new_session=True)
        try:
            stdout, stderr = proc.communicate(json.dumps(job, default=str), timeout=self.timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, AttributeError):
                proc.kill()
            proc.communicate()
            return False, f"Timed out after {self.timeout}s"
        try:
            res = json.loads(stdout)
            return bool(res["ok"]), str(res["output"])
        except (ValueError, KeyError, TypeError):
            return False, f"Tool process failed (exit {proc.returncode}): {stderr.strip()[-300:]}"


def tool(name: str | None = None, description: str | None = None, *, doc: dict[str, str] | None = None,
         enums: dict[str, list] | None = None, timeout: float = 20.0, requires_approval: bool = False, isolated: bool = False):
    """Turn a typed function into a Tool. The JSON schema is inferred from type hints.
    A parameter named `ctx` receives the ToolContext and is hidden from the model."""

    def wrap(fn: Callable) -> Tool:
        hints = typing.get_type_hints(fn)
        props, required, wants_ctx = {}, [], False
        for p in inspect.signature(fn).parameters.values():
            if p.name == "ctx":
                wants_ctx = True
                continue
            spec: dict = {"type": _JSON_TYPES.get(hints.get(p.name, str), "string")}
            if doc and p.name in doc:
                spec["description"] = doc[p.name]
            if enums and p.name in enums:
                spec["enum"] = enums[p.name]
            props[p.name] = spec
            if p.default is inspect.Parameter.empty:
                required.append(p.name)
        desc = description or (inspect.getdoc(fn) or fn.__name__).split("\n\n")[0].replace("\n", " ")
        return Tool(name or fn.__name__, desc,
                    {"type": "object", "properties": props, "required": required, "additionalProperties": False},
                    fn, timeout, requires_approval, wants_ctx, isolated)

    return wrap
