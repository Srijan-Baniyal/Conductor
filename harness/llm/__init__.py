from .base import LLMClient, LLMError
from .openai_compat import OpenAICompatClient
from .anthropic_client import AnthropicClient
from .text_adapter import TextToolAdapter
from .mock import MockLLM, ScriptedLLM

def make_client(provider: str, model: str, base_url: str | None = None, extra_body: dict | None = None):
    """Build a client from CLI-style options. extra_body is sent with every request (e.g. reasoning_effort)."""
    if provider == "anthropic":
        return AnthropicClient(model, base_url, extra_body=extra_body)
    return OpenAICompatClient(model, base_url, extra_body=extra_body)


__all__ = ["make_client", "LLMClient", "LLMError", "OpenAICompatClient", "AnthropicClient", "TextToolAdapter", "MockLLM", "ScriptedLLM"]
