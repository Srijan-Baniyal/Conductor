"""Tool calling for models without native function calling (small local models, base chat
models). Tools are described in the system prompt and calls are parsed from
<tool_call>{json}</tool_call> tags. Malformed calls become an error the model can correct."""
from __future__ import annotations
import json, re, uuid
from ..types import Message, LLMResponse, ToolCall

TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

INSTRUCTIONS = """You can use tools. To call one, reply with exactly one or more blocks like:
<tool_call>{{"name": "tool_name", "arguments": {{"arg": "value"}}}}</tool_call>
Arguments must be valid JSON matching the schema. After a call, stop and wait: the result arrives as
<tool_result name="tool_name">...</tool_result>. When you need no tool, answer normally.

Tools:
{tools}"""


class TextToolAdapter:
    native_tools = False

    def __init__(self, inner) -> None:
        self.inner = inner

    @staticmethod
    def _describe(tools: list[dict]) -> str:
        return "\n".join(f"- {t['function']['name']}: {t['function']['description']}\n  schema: "
                         f"{json.dumps(t['function']['parameters'])}" for t in tools)

    def _flatten(self, messages: list[Message], tools: list[dict] | None) -> list[Message]:
        out: list[Message] = []
        for m in messages:
            if m.role == "assistant" and m.tool_calls:
                blocks = "".join(f"<tool_call>{json.dumps({'name': c.name, 'arguments': c.arguments})}</tool_call>"
                                 for c in m.tool_calls)
                out.append(Message("assistant", (m.content + "\n" if m.content else "") + blocks))
            elif m.role == "tool":
                out.append(Message("user", f'<tool_result name="{m.name}">{m.content}</tool_result>'))
            else:
                out.append(m)
        if tools and out and out[0].role == "system":
            out[0] = Message("system", out[0].content + "\n\n" + INSTRUCTIONS.format(tools=self._describe(tools)))
        return out

    def chat(self, messages, tools=None, temperature=0.0) -> LLMResponse:
        resp = self.inner.chat(self._flatten(messages, tools), None, temperature)
        calls = []
        for raw in TAG.findall(resp.content):
            try:
                d = json.loads(raw)
                calls.append(ToolCall(f"call_{uuid.uuid4().hex[:8]}", d["name"], d.get("arguments") or {}))
            except (ValueError, KeyError, TypeError, AttributeError):
                calls.append(ToolCall(f"call_{uuid.uuid4().hex[:8]}", "invalid_tool_call", {"__raw__": raw}))
        text = TAG.sub("", resp.content).strip()
        return LLMResponse(text, calls if tools else [], resp.usage)
