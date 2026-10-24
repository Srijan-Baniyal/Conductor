"""Native Anthropic Messages API client. Stdlib only."""
from __future__ import annotations
import json, os, urllib.request
from ..types import Message, LLMResponse, ToolCall
from ._http import open_with_retry
from .base import LLMError


class AnthropicClient:
    native_tools = True

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 max_tokens: int = 4096, timeout: float = 60, max_retries: int = 3,
                 version: str = "2023-06-01", extra_body: dict | None = None) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com").rstrip("/")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("HARNESS_API_KEY") or ""
        self.max_tokens, self.timeout, self.max_retries = max_tokens, timeout, max_retries
        self.version, self.extra = version, extra_body or {}

    @staticmethod
    def _wire(messages: list[Message]) -> tuple[str, list[dict]]:
        """Split out system text and build alternating-role content blocks."""
        system, out = [], []

        def push(role: str, blocks: list[dict]) -> None:
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)  # tool results from one turn share one user message
            else:
                out.append({"role": role, "content": blocks})

        for m in messages:
            if m.role == "system":
                system.append(m.content)
            elif m.role == "tool":
                push("user", [{"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}])
            elif m.role == "assistant":
                blocks = [{"type": "text", "text": m.content}] if m.content else []
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                           for c in m.tool_calls]
                if blocks:
                    push("assistant", blocks)
            else:
                if m.content:
                    push("user", [{"type": "text", "text": m.content}])
        return "\n\n".join(s for s in system if s), out

    @staticmethod
    def _tools(tools: list[dict]) -> list[dict]:
        return [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
                 "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}}
                for t in tools]

    def chat(self, messages, tools=None, temperature=0.0) -> LLMResponse:
        system, wire = self._wire(messages)
        body: dict = {"model": self.model, "max_tokens": self.max_tokens, "messages": wire,
                      "temperature": temperature, **self.extra}
        if system:
            body["system"] = system
        if tools:
            body["tools"] = self._tools(tools)
        req = urllib.request.Request(f"{self.base_url}/v1/messages", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "x-api-key": self.api_key,
                                              "anthropic-version": self.version})
        with open_with_retry(req, self.timeout, self.max_retries) as r:
            try:
                data = json.load(r)
            except ValueError:
                raise LLMError("response was not valid JSON")
        if not isinstance(data.get("content"), list):
            raise LLMError(f"unexpected response shape: {str(data)[:300]}")
        text, calls = [], []
        for b in data["content"]:
            if b.get("type") == "text":
                text.append(b.get("text", ""))
            elif b.get("type") == "tool_use":
                args = b.get("input")
                calls.append(ToolCall(b.get("id") or f"call_{len(calls)}", b["name"],
                                      args if isinstance(args, dict) else {"__raw__": json.dumps(args)}))
        usage = {k: v for k, v in (data.get("usage") or {}).items() if isinstance(v, int)}
        return LLMResponse("".join(text), calls, usage)
