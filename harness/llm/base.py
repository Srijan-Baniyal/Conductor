from __future__ import annotations
from typing import Iterator, Protocol
from ..types import Message, LLMResponse


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    """Implement this to support any model. One method, one flag."""
    native_tools: bool  # False => harness uses the text tool protocol

    def chat(self, messages: list[Message], tools: list[dict] | None = None,
             temperature: float = 0.0) -> LLMResponse: ...

    def stream(self, messages: list[Message], tools: list[dict] | None = None,
               temperature: float = 0.0) -> Iterator[str]:
        """Optional. Yield text deltas. The generator may return the final LLMResponse (with tool
        calls and usage); if it returns None the harness builds one from the joined text. Clients
        without stream() are called through chat()."""
        ...
