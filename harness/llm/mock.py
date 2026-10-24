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
    """Useful offline model for exercising the full harness without credentials."""
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
        lower = text.lower()
        if re.search(r"\b(hello|hi|hey)\b", lower):
            answer = (
                "## Hello\n\n"
                "I’m the offline harness model. Ask a question, request a plan, "
                "or give me a tool task and I’ll stream the work into the console."
            )
        elif "what can you do" in lower or "help" in lower:
            answer = (
                "## What I can do\n\n"
                "- Answer general questions with a concise explanation\n"
                "- Break projects into practical implementation steps\n"
                "- Solve expressions and generate plots through tools\n"
                "- Save and recall notes across the current session\n"
                "- Show streamed model output, tool calls, artifacts, and errors\n\n"
                "Try: **Explain how an HTTP request works**, or **plan a settings page**."
            )
        elif re.search(r"\b(plan|steps|roadmap|implement)\b", lower):
            answer = (
                f"## Plan: {text}\n\n"
                "1. **Clarify the outcome** — define the user-visible result and constraints.\n"
                "2. **Model the state** — identify data, loading, success, and failure states.\n"
                "3. **Build the smallest slice** — wire the core path end to end first.\n"
                "4. **Add feedback** — expose progress, empty states, and actionable errors.\n"
                "5. **Verify** — test the happy path, edge cases, and responsive behavior."
            )
        elif re.search(r"\b(explain|what is|how does|why)\b", lower):
            answer = (
                f"## Explanation\n\n"
                f"**Question:** {text}\n\n"
                "The useful way to approach this is to separate the system into three "
                "layers: the input and its constraints, the work performed by the system, "
                "and the observable result. Trace those layers in order, then validate the "
                "result against the original requirement.\n\n"
                "> In the full harness, a configured provider can replace this offline "
                "answer while keeping the same streaming and tool lifecycle."
            )
        elif re.search(r"\b(code|function|component|api|debug|error)\b", lower):
            answer = (
                "## Suggested approach\n\n"
                "```text\n"
                "input -> validate -> execute -> stream progress -> render result\n"
                "```\n\n"
                f"For **{text}**, start by reproducing the smallest failing case, "
                "log the boundary where the value changes, and add a focused test before "
                "changing the implementation."
            )
        else:
            answer = (
                f"## Answer\n\n"
                f"You asked: **{text}**\n\n"
                "A practical answer is to make the goal explicit, identify the important "
                "constraints, and work from the smallest verifiable step. The harness is "
                "connected and can stream this response, call tools when the request matches "
                "one, and preserve the session for follow-up questions.\n\n"
                "For a richer provider-backed answer, configure a model preset; the chat "
                "surface and event timeline stay the same."
            )
        return LLMResponse(answer)

    def stream(self, messages, tools=None, temperature=0.0):
        """Yield text in small chunks so the offline harness exercises live rendering."""
        response = self.chat(messages, tools, temperature)
        if response.tool_calls:
            return response
        for chunk in re.findall(r"\S+\s*", response.content):
            yield chunk
        return response
