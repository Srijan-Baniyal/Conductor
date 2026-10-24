from __future__ import annotations
import importlib, importlib.util, inspect, logging, pkgutil
from importlib.metadata import entry_points
from pathlib import Path
from .base import Tool

log = logging.getLogger("harness")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, t: Tool, replace: bool = False) -> None:
        if t.name in self._tools and not replace:
            raise ValueError(f"tool '{t.name}' already registered")
        self._tools[t.name] = t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def subset(self, names: list[str]) -> "ToolRegistry":
        missing = [n for n in names if n not in self._tools]
        if missing:
            raise KeyError(f"unknown tools {missing}; available: {self.names()}")
        r = ToolRegistry()
        for n in names:
            r._tools[n] = self._tools[n]
        return r

    def _collect(self, mod) -> None:
        found = [o for _, o in inspect.getmembers(mod, lambda o: isinstance(o, Tool))]
        found += list(getattr(mod, "TOOLS", []))
        for t in found:
            self.register(t, replace=True)

    def load_builtin(self) -> None:
        from . import builtin
        for m in pkgutil.iter_modules(builtin.__path__):
            self._collect(importlib.import_module(f"{builtin.__name__}.{m.name}"))

    def load_dir(self, path: str | Path) -> None:
        """Drop a .py file in a folder; every Tool instance in it is registered."""
        for f in sorted(Path(path).glob("*.py")):
            if f.name.startswith("_"):
                continue
            spec = importlib.util.spec_from_file_location(f"harness_plugin_{f.stem}", f)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception:
                log.exception("failed to load plugin %s", f)
                continue
            self._collect(mod)

    def load_entry_points(self) -> None:
        """pip-installable plugins: declare an entry point in group 'harness.tools'
        that resolves to a Tool or a list of Tools."""
        for ep in entry_points(group="harness.tools"):
            try:
                obj = ep.load()
            except Exception:
                log.exception("failed to load entry point %s", ep.name)
                continue
            for t in (obj if isinstance(obj, (list, tuple)) else [obj]):
                self.register(t, replace=True)
