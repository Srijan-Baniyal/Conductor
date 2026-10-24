# Contributing

1. Fork, then run `uv sync --all-extras`.
2. Add a tool: one file in `harness/tools/builtin/` using `@tool`. Add a test in `tests/` using `ScriptedLLM`. Use `isolated=True` for tools that can hang or run heavy code.
3. Add a provider: implement `LLMClient` in `harness/llm/`, with an optional `stream()`. Keep the core dependency-free. Test it against a local fake HTTP server, never a real API.
4. Run `uv run pytest -q` before opening a PR. CI runs it on Python 3.11 and 3.12, plus a mock CLI run.
5. Add dev tools with `uv add --optional dev <pkg>`. Run everything through `uv run`.

Rules: standard library only in `harness/`, optional dependencies as extras in `pyproject.toml`, no secrets in commits (read keys from environment variables), no em dashes in docs or comments.

To try a real model, run `uv run scripts/smoke.py --model ... --base-url ...`. The fine-tuning walkthrough is in `docs/GUIDE.md`.

Good first issues: SQLite notes backend, streaming for the Anthropic client, HTTP server mode, richer sandboxing (resource limits, no network).

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
