from __future__ import annotations
import json, logging, time
from collections import defaultdict
from pathlib import Path
from typing import Callable

log = logging.getLogger("harness")


class Hooks:
    """Observer bus. Events: llm_request, token, llm_response, tool_start, tool_end, error, final.
    Subscribe to "*" for everything. Hook failures never break the agent."""

    def __init__(self) -> None:
        self._h: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event: str, fn: Callable[[str, dict], None]):
        self._h[event].append(fn)
        return fn

    def emit(self, event: str, **data) -> None:
        for fn in (*self._h.get(event, ()), *self._h.get("*", ())):
            try:
                fn(event, data)
            except Exception:
                log.exception("hook failed for %s", event)


class JsonlTracer:
    """Append every event to a JSONL file. Replayable, diffable, greppable."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: str, data: dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"t": round(time.time(), 3), "event": event, **data}, default=str) + "\n")
