"""One client for OpenAI, Azure-compatible gateways, Ollama, vLLM, LM Studio, llama.cpp server,
OpenRouter, Groq, Together, Gemini's OpenAI endpoint. Stdlib only."""
from __future__ import annotations
import json, os, urllib.request
from typing import Iterator
from ..types import Message, LLMResponse, ToolCall
from ._http import RETRY_CODES, open_with_retry  # noqa: F401 (RETRY_CODES kept for compatibility)
from .base import LLMError

USER_AGENT = "universal-agent-harness/0.1"


def _make_call(i: int, call_id: str | None, name: str, raw: str | None) -> ToolCall:
    raw = raw or "{}"
    try:
        args = json.loads(raw)
        if not isinstance(args, dict):
            raise ValueError
    except ValueError:
        args = {"__raw__": raw}
    return ToolCall(call_id or f"call_{i}", name, args)


def _usage(d: dict | None) -> dict:
    return {k: v for k, v in (d or {}).items() if isinstance(v, int)}


class OpenAICompatClient:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 native_tools: bool = True, timeout: float = 60, max_retries: int = 3,
                 extra_body: dict | None = None) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("HARNESS_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.getenv("HARNESS_API_KEY") or os.getenv("OPENAI_API_KEY") or "none"
        self.native_tools = native_tools
        self.timeout, self.max_retries, self.extra = timeout, max_retries, extra_body or {}

    @staticmethod
    def _wire(m: Message) -> dict:
        if m.role == "tool":
            return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
        d: dict = {"role": m.role, "content": m.content}
        if m.tool_calls:
            d["content"] = m.content or None
            d["tool_calls"] = [{"id": c.id, "type": "function",
                                "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                               for c in m.tool_calls]
        return d

    def _request(self, messages, tools, temperature, stream: bool) -> urllib.request.Request:
        body = {"model": self.model, "messages": [self._wire(m) for m in messages],
                "temperature": temperature, **self.extra}
        if tools:
            body["tools"] = tools
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                      headers={"Content-Type": "application/json", "User-Agent": USER_AGENT,
                                               "Authorization": "Bearer " + self.api_key})

    def chat(self, messages, tools=None, temperature=0.0) -> LLMResponse:
        req = self._request(messages, tools, temperature, False)
        with open_with_retry(req, self.timeout, self.max_retries) as r:
            try:
                data = json.load(r)
            except ValueError:
                raise LLMError("response was not valid JSON")
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected response shape: {str(data)[:300]}")
        calls = [_make_call(i, c.get("id"), c["function"]["name"], c["function"].get("arguments"))
                 for i, c in enumerate(msg.get("tool_calls") or [])]
        return LLMResponse(msg.get("content") or "", calls, _usage(data.get("usage")))

    def stream(self, messages, tools=None, temperature=0.0) -> Iterator[str]:
        """Yield text deltas over SSE. The generator's return value is the final LLMResponse,
        including any tool calls and usage."""
        req = self._request(messages, tools, temperature, True)
        text: list[str] = []
        partial: dict[int, dict] = {}
        usage: dict = {}
        with open_with_retry(req, self.timeout, self.max_retries) as r:
            for line in r:
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                if chunk.get("usage"):
                    usage = _usage(chunk["usage"])
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        text.append(delta["content"])
                        yield delta["content"]
                    for tc in delta.get("tool_calls") or []:
                        p = partial.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                        p["id"] = tc.get("id") or p["id"]
                        fn = tc.get("function") or {}
                        p["name"] += fn.get("name") or ""
                        p["args"] += fn.get("arguments") or ""
        calls = [_make_call(i, p["id"], p["name"], p["args"]) for i, (_, p) in enumerate(sorted(partial.items()))]
        return LLMResponse("".join(text), calls, usage)
