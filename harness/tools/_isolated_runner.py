"""Child process entry point for isolated tools. Reads a JSON job on stdin, writes a JSON result on stdout."""
from __future__ import annotations
import importlib, importlib.util, json, sys
from pathlib import Path


def _find_tool(module: str, file: str | None, name: str):
    from harness.tools.base import Tool
    try:
        mod = importlib.import_module(module)
    except ImportError:
        if not file:
            raise
        spec = importlib.util.spec_from_file_location(module, file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    for obj in [*vars(mod).values(), *getattr(mod, "TOOLS", [])]:
        if isinstance(obj, Tool) and obj.name == name:
            return obj
    raise LookupError(f"tool '{name}' not found in module '{module}'")


def main() -> int:
    job = json.load(sys.stdin)
    real_stdout, sys.stdout = sys.stdout, sys.stderr  # tool prints must not corrupt the result
    sys.path[:0] = [p for p in job["sys_path"] if p not in sys.path]
    try:
        from harness.tools.base import ToolContext
        t = _find_tool(job["module"], job["file"], job["name"])
        kwargs = dict(job["args"])
        if t.wants_ctx:
            kwargs["ctx"] = ToolContext(Path(job["workdir"]), job["session_id"], None)
        result = {"ok": True, "output": str(t.fn(**kwargs))}
    except BaseException as e:
        result = {"ok": False, "output": f"{type(e).__name__}: {e}"}
    real_stdout.write(json.dumps(result))
    real_stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
