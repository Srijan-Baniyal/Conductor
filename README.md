# Universal Agent Harness

A small, model-agnostic harness that turns any chat LLM into an agent. It owns the parts around the model: prompts, tools, memory, context budget and the action loop. Agents are TOML files, tools are typed Python functions, and the core has zero runtime dependencies.

```
User -> Agent loop -> LLM -> tool calls (validated, sandboxed, timed) -> LLM -> ... -> answer
                 |-> memory (windowed history, durable notes, rolling summary)
                 |-> hooks (trace, metrics, UI)
```

## Try it in 30 seconds (no API key)

Needs [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone <YOUR_REPO_URL> && cd universal-agent-harness
```

```bash
uv sync --all-extras
uv run python -m harness --mock --agent math -p "solve x^2+5*x+6=0"
uv run python -m harness --mock --agent math -p "plot sin(x) from -6 to 6"
uv run python -m harness --mock --agent lab  -p "remember: Experiment A uses 5g NaCl in 100ml water"
uv run pytest -q
```

## Use a real model

Two providers ship in the box, both standard library only.

`--provider openai` (default) works with any OpenAI-compatible endpoint: OpenAI, Ollama, vLLM, LM Studio, llama.cpp, OpenRouter, Groq, Together.

```bash
# hosted
export OPENAI_API_KEY=...
uv run python -m harness --agent math --model gpt-4o-mini

# local, no key
uv run python -m harness --agent lab --model llama3.1 --base-url http://localhost:11434/v1
```

`--provider anthropic` uses the native Messages API (`x-api-key`, tool_use and tool_result blocks, top-level system prompt):

```bash
export ANTHROPIC_API_KEY=...
uv run python -m harness --provider anthropic --model <model-id> --agent math
```

`HARNESS_MODEL`, `HARNESS_BASE_URL`, `HARNESS_API_KEY` and `HARNESS_PROVIDER` can replace the flags. Keys are read from environment variables only.

Other providers: implement `chat()` (see `harness/llm/base.py`).

Models without native function calling (small local models) are handled automatically by a text protocol adapter. Force it with `tool_mode = "text"` in the agent file.

### Streaming

```bash
uv run python -m harness --stream --agent lab --model llama3.1 --base-url http://localhost:11434/v1
```

`--stream` prints tokens as they arrive and emits a `token` hook event for each one. Streaming is implemented for the OpenAI-compatible client over SSE, including streamed tool calls. Any client without a `stream()` method, including the Anthropic client, the mock models and text-protocol mode, falls back to `chat()`, and the full answer prints at the end.

To add streaming to your own client, implement `stream(messages, tools, temperature)` as a generator that yields text deltas and returns the final `LLMResponse`.

## Test any model

`scripts/smoke.py` checks that a real model can drive the harness. It runs the math and lab agents and asserts that tools were called.

```bash
uv run scripts/smoke.py --model qwen3:8b --base-url http://localhost:11434/v1 --runs 5
uv run scripts/smoke.py --provider anthropic --model <model-id>
```

It prints PASS or FAIL per scenario and a pass rate with `--runs`. Without a model it prints SKIP.

## Fine-tune your own model

The full path is in [docs/GUIDE.md](docs/GUIDE.md): run a local model with Ollama, collect tool-calling data from a teacher model, fine-tune a LoRA on [Tinker](https://tinker-docs.thinkingmachines.ai/), merge it, import it into Ollama, and compare pass rates.

```bash
uv run scripts/collect_data.py --model <teacher> --base-url <url> --out data/train.jsonl
uv run --with tinker --with tinker-cookbook scripts/finetune_tinker.py --data data/train.jsonl
uv run scripts/ollama_import.py --model-dir build/tuned/merged --name harness-qwen3
```

The data collection and conversion are tested offline. The Tinker, merge and Ollama import steps have not been run end to end yet, and Tinker needs an account and `TINKER_API_KEY`.

## Demo (60 seconds)

No API key needed for the first two steps.

```bash
uv sync --all-extras

# 1. Math plot. The PNG path is printed. Math and plot run in a child process.
uv run python -m harness --mock --agent math -p "plot sin(x) from -6 to 6"

# 2. Lab notes recall: save a note, then recall it in a second run of the same session.
uv run python -m harness --mock --agent lab --session demo -p "remember: Experiment A uses 5g NaCl in 100ml water"
uv run python -m harness --mock --agent lab --session demo -p "summarize notes"

# 3. Custom tool via --tools-dir. The mock model cannot call it, so this lists it:
printf '/tools\n/quit\n' | uv run python -m harness --mock --agents-dir examples/agents --tools-dir examples/custom_tools --agent research
#    With a real model it gets used:
uv run python -m harness --agents-dir examples/agents --tools-dir examples/custom_tools --agent research \
  --model llama3.1 --base-url http://localhost:11434/v1 -p "How many feet is 100 meters?"
```

To check a real model end to end: `HARNESS_MODEL=... uv run scripts/smoke.py`. It runs the math and lab agents, asserts that tools were called, and prints PASS or FAIL per scenario. Without a model it prints a skip message.

## Define an agent (no code)

`agents/research.toml`:

```toml
description = "Looks things up and keeps notes."
system_prompt = "You are a careful research assistant."
tools = ["search_notes", "add_note", "unit_convert"]
max_steps = 8
temperature = 0.2

[memory]
max_context_tokens = 6000
notes_scope = "global"     # or "session"
```

```bash
uv run python -m harness --agents-dir agents --tools-dir examples/custom_tools --agent research --model gpt-4o-mini
```

Inside the REPL: `/agents`, `/agent NAME` (switch agent, keep history), `/tools`, `/notes`, `/usage`, `/reset`.

## Add a tool (5 lines)

```python
from harness import tool

@tool("unit_convert", enums={"to_unit": ["m", "ft"]})
def unit_convert(value: float, to_unit: str) -> str:
    """Convert meters to the target unit."""
    return f"{value * (3.28084 if to_unit == 'ft' else 1):.4g} {to_unit}"
```

The JSON schema comes from the type hints. Drop the file in a folder and pass `--tools-dir`, or publish a package with an entry point in the `harness.tools` group. Options: `timeout`, `requires_approval=True` (asks the user before running), `isolated=True` (see below), and a `ctx` parameter that receives the working directory, session id and notes store.

### Subprocess isolation

`@tool(..., isolated=True)` runs the tool in a child Python process. On timeout the child and its process group are killed with SIGKILL, so a hung tool really stops and returns `Timed out after Ns`. `sympy_math` and `plot` use it. Rules for isolated tools:

- Define the tool at module level in an importable file (the child re-imports it by module name or file path).
- Arguments and the return value must be JSON-friendly. The child gets `ctx.workdir` and `ctx.session_id`, but `ctx.notes` is `None`.
- Each call pays interpreter startup, and imports like SymPy count against the timeout.
- This is process isolation, not a security sandbox. The child has your user's file and network access.

## What the harness guarantees

| Concern | Behavior |
|---|---|
| Tool protocol | Native function calling when the model supports it, `<tool_call>` JSON text protocol otherwise |
| Argument safety | Validated against the schema, numeric strings coerced, bad calls return a readable error the model can fix |
| Tool failures | Exceptions, timeouts and bad JSON become tool results, never crash the loop |
| Hung tools | `isolated=True` tools run in a child process that is hard-killed on timeout |
| Loops | `max_steps`, identical-call guard (3 repeats), forced no-tools final answer at the limit |
| Parallelism | Multiple tool calls in one turn run concurrently (disabled when approval is needed) |
| Context | Token-budgeted window on whole turns, tool calls never split from results, optional rolling LLM summary of dropped turns |
| Memory | Append-only JSONL per session (resume with `--session ID`), notes with atomic writes and keyword search |
| Math safety | SymPy input is AST-whitelisted: no attribute access, only listed functions, bounded constants and exponents |
| Prompt injection | Tool output is truncated and the system prompt marks it untrusted. This reduces risk, it does not remove it |
| Retries | Exponential backoff on 429 and 5xx, honors Retry-After |
| Observability | `--trace file.jsonl` records every LLM call, tool call, latency and token usage; subscribe with `Hooks.on` (events: `llm_request`, `token`, `llm_response`, `tool_start`, `tool_end`, `error`, `final`) |

## Use as a library

```python
from harness import Agent, AgentConfig, ToolRegistry
from harness.llm import OpenAICompatClient

reg = ToolRegistry(); reg.load_builtin()
agent = Agent(AgentConfig("math", "You are a math assistant.", tools=["sympy_math"]),
              OpenAICompatClient("gpt-4o-mini"), reg)
print(agent.run("Integrate x^2 from 0 to 3").final)
```

## Layout

```
harness/
  agent.py        loop, config, context assembly, loop guards
  memory.py       ConversationMemory (windowing), NotesStore
  tools/          Tool, @tool, registry (builtin, folder, entry-point loading)
    builtin/      sympy_math, plot, add_note, search_notes, list_notes, safe_math
  llm/            OpenAI-compatible and Anthropic clients, text-protocol adapter, mock models
  agents/         bundled general, math, lab
  hooks.py        event bus and JSONL tracer
  cli.py          REPL and one-shot mode
scripts/           smoke test, training data collection, Tinker fine-tune, Ollama import
data/prompts.jsonl  starter scenarios for data collection
docs/GUIDE.md     model, test and fine-tune walkthrough
tests/            43 tests, all offline (scripted model and local fake HTTP servers)
```

## Known limits

- Only `isolated=True` tools can be killed on timeout. Other tools run in a thread, so a timeout stops waiting but the thread keeps running. Isolation is process level, with no filesystem or network restrictions. Use a container for untrusted code.
- Isolated tools cannot use the notes store, and each call pays interpreter startup.
- Streaming exists only for the OpenAI-compatible client. The Anthropic client and text-protocol mode use `chat()`.
- Token counts are estimated (chars / 4), not tokenizer-exact.
- Notes search is keyword based. Swap `NotesStore` for embeddings if you need semantic recall.
- Tested end to end with scripted models, mock models and local fake servers. The OpenAI-compatible and Anthropic clients have not been run against the real services in CI. Use `scripts/smoke.py` for that. The fine-tuning scripts are untested against Tinker and Ollama. Real-model behavior depends on the model's tool-calling quality.

## License

MIT. See `LICENSE`. Contributions welcome, see `CONTRIBUTING.md`.
