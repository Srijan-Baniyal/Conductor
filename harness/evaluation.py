"""Run scripted scenarios against any model and check the outcome. Used by smoke, evaluate and data collection.

A scenario is a dict: {"agent": "math", "turns": ["..."], "expect_tools": ["sympy_math"],
"expect_any_tool": ["search_notes", "list_notes"], "expect_text": ["-3", "-2"]}.
All keys except agent and turns are optional. Extra checks: "forbid_tools" (names that must not be called),
"expect_no_tools" (true means no tool at all), "expect_args" ([{"tool": "add_note", "arg": "tag",
"contains": "a7"}] or "regex": "EXP-A7" for a case-sensitive full match), "expect_regex" (patterns the raw final answer
must match, multiline), and "seed_notes" (notes saved before the first turn, to test untrusted note text)."""
from __future__ import annotations
import concurrent.futures as cf
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .agent import PREAMBLE, Agent
from .config import load_agent_config
from .hooks import Hooks
from .memory import ConversationMemory
from .tools.registry import ToolRegistry
from .types import Message

BUNDLED = Path(__file__).parent / "agents"


@dataclass
class RunResult:
    called: list[str]
    final: str
    stopped: str
    messages: list[Message]
    system: str
    tools: list[dict]
    error: str = ""
    tool_errors: int = 0
    steps: int = 0
    notes: list[str] = field(default_factory=list)


_SUPERSCRIPTS = str.maketrans("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079", "0123456789")


def norm(text: str) -> str:
    """Models often write the Unicode minus sign or thin spaces. Treat them as plain ASCII."""
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("\u00a0", " ").replace("\u202f", " ")
    return text.translate(_SUPERSCRIPTS)


def system_for(agent: str, agents_dir: Path = BUNDLED) -> str:
    """The system prompt a stock agent would send. Used to swap in the student prompt when distilling."""
    return "\n\n".join([load_agent_config(agents_dir / f"{agent}.toml").system_prompt.strip(), PREAMBLE])


def run_scenario(entry: dict, llm, registry: ToolRegistry, temperature: float | None = None,
                 agents_dir: Path = BUNDLED, suffix: str = "") -> RunResult:
    """Run one scenario. `suffix` picks a variant agent file, e.g. "_pro" loads math_pro.toml from agents_dir."""
    cfg = load_agent_config(agents_dir / f"{entry['agent']}{suffix}.toml")
    if temperature is not None:
        cfg.temperature = temperature
    hooks, called = Hooks(), []
    hooks.on("tool_start", lambda e, d: called.append(d["name"]))
    with tempfile.TemporaryDirectory() as tmp:
        agent = Agent(cfg, llm, registry, root=tmp, memory=ConversationMemory(), hooks=hooks,
                      approve=lambda t, a: True)
        for note in entry.get("seed_notes", []):
            agent.ctx.notes.add(note, "")
        final, stopped, steps = "", "done", 0
        for turn in entry["turns"]:
            r = agent.run(turn)
            final, stopped, steps = r.final, r.stopped, steps + r.steps
            if stopped == "error":
                break
        msgs = list(agent.memory.messages)
        errs = sum(1 for m in msgs if m.role == "tool" and m.content.startswith("ERROR:"))
        return RunResult(called, final, stopped, msgs, agent._system(), agent.registry.schemas(),
                         error=final if stopped == "error" else "", tool_errors=errs, steps=steps)


def check(entry: dict, res: RunResult) -> tuple[bool, str]:
    """Return (passed, reason). The reason is empty on success."""
    if res.stopped == "error":
        return False, res.error[:120]
    for t in entry.get("expect_tools", []):
        if t not in res.called:
            return False, f"did not call {t}"
    any_of = entry.get("expect_any_tool")
    if any_of and not set(any_of) & set(res.called):
        return False, f"did not call any of {any_of}"
    if entry.get("expect_no_tools") and res.called:
        return False, f"called tools {res.called} but none were needed"
    for t in entry.get("forbid_tools", []):
        if t in res.called:
            return False, f"called forbidden tool {t}"
    for want in entry.get("expect_args", []):
        def ok(value: str) -> bool:
            if "regex" in want:
                return re.fullmatch(want["regex"], value) is not None
            return want["contains"].lower() in value.lower()

        if not any(c.name == want["tool"] and ok(str(c.arguments.get(want["arg"], "")))
                   for m in res.messages for c in m.tool_calls):
            return False, f"no {want['tool']} call with {want['arg']} matching {want.get('regex') or want['contains']!r}"
    for pat in entry.get("expect_regex", []):
        if not re.search(pat, res.final, re.MULTILINE | re.DOTALL):
            return False, f"answer does not match {pat!r}"
    text = "".join(norm(res.final).lower().replace(",", "").split())  # ignore commas and whitespace
    for s in entry.get("expect_text", []):
        if "".join(norm(str(s)).lower().split()) not in text:
            return False, f"answer is missing {s!r}"
    return True, ""


def is_clean(res: RunResult, entry: dict | None = None) -> bool:
    """A run worth training on: finished normally, used a tool (unless none was expected), no tool errors,
    ended with an answer."""
    last = res.messages[-1] if res.messages else None
    needs_tool = not (entry or {}).get("expect_no_tools")
    return bool(res.stopped == "done" and (res.called or not needs_tool) and not res.tool_errors and last
                and last.role == "assistant" and last.content.strip() and not last.tool_calls)


def run_many(entries: list[dict], llm, registry: ToolRegistry, workers: int = 4, repeat: int = 1,
             temperature: float | None = None, progress=None, agents_dir: Path = BUNDLED, suffix: str = "") -> list[tuple[dict, RunResult | None, str]]:
    """Run every entry `repeat` times in parallel. Returns (entry, result, exception text) in input order."""
    jobs = [e for e in entries for _ in range(repeat)]

    def one(e: dict):
        try:
            return e, run_scenario(e, llm, registry, temperature, agents_dir, suffix), ""
        except Exception as ex:  # one bad run must not stop a long batch
            return e, None, f"{type(ex).__name__}: {ex}"

    out = []
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, item in enumerate(pool.map(one, jobs), 1):
            out.append(item)
            if progress:
                progress(i, len(jobs), item)
    return out
