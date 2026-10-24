from __future__ import annotations
import re, uuid
from typing import Callable
from ..types import Message, LLMResponse, ToolCall


def _call(name: str, **args) -> LLMResponse:
    return LLMResponse("", [ToolCall(f"call_{uuid.uuid4().hex[:8]}", name, args)])


class ScriptedLLM:
    """Deterministic model for tests. Each item is an LLMResponse or a callable(messages, tools)."""
    native_tools = True

    def __init__(self, script: list[LLMResponse | Callable]) -> None:
        self.script, self.calls = list(script), []

    def chat(self, messages, tools=None, temperature=0.0) -> LLMResponse:
        self.calls.append((list(messages), tools))
        item = self.script.pop(0)
        return item(messages, tools) if callable(item) else item


class MockLLM:
    """Offline demo model: pattern-matches a few requests so the whole loop runs with no API key."""
    native_tools = True

    def chat(self, messages, tools=None, temperature=0.0) -> LLMResponse:
        last = messages[-1]
        names = {t["function"]["name"] for t in tools or []}
        if last.role == "tool":
            return LLMResponse(f"Result: {last.content}")
        text = last.content.strip()
        if m := re.match(r"(?i)plot\s+(?:y\s*=\s*)?(.+?)(?:\s+from\s+(-?[\d.]+)\s+to\s+(-?[\d.]+))?$", text):
            if "plot" in names:
                a = {"kind": "function", "expression": m[1]}
                if m[2]:
                    a.update(x_min=float(m[2]), x_max=float(m[3]))
                return _call("plot", **a)
        if m := re.match(r"(?i)solve\s+(.+)", text):
            if "sympy_math" in names:
                return _call("sympy_math", operation="solve", expression=m[1])
        if m := re.match(r"(?i)remember:?\s+(.+)", text):
            if "add_note" in names:
                return _call("add_note", text=m[1])
        if re.match(r"(?i)(list|summari[sz]e).*notes?|summari[sz]e", text) and "list_notes" in names:
            return _call("list_notes")
        return LLMResponse("(mock model) Try: 'solve x^2+5*x+6=0', 'plot sin(x) from -6 to 6', "
                           "'remember: Experiment A uses 5g NaCl in 100ml water', 'summarize notes'.")
