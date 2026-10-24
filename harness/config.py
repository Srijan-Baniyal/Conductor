from __future__ import annotations
import tomllib
from dataclasses import fields
from pathlib import Path
from .agent import AgentConfig

_TOP = {f.name for f in fields(AgentConfig)} - {"max_context_tokens", "notes_scope"}
_MEM = {"max_context_tokens", "notes_scope"}


def load_agent_config(path: str | Path) -> AgentConfig:
    path = Path(path)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    mem = data.pop("memory", {})
    bad = (set(data) - _TOP) | {f"memory.{k}" for k in set(mem) - _MEM}
    if bad:
        raise ValueError(f"{path.name}: unknown keys {sorted(bad)}")
    data.setdefault("name", path.stem)
    if "system_prompt" not in data:
        raise ValueError(f"{path.name}: system_prompt is required")
    return AgentConfig(**data, **mem)


def discover(*dirs: str | Path) -> dict[str, Path]:
    """Later dirs override earlier ones, so user agents can shadow bundled ones."""
    found: dict[str, Path] = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*.toml")):
            found[f.stem] = f
    return found
