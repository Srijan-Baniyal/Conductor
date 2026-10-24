# Guide: run any model, then fine-tune your own

This guide takes you from a clean checkout to a fine-tuned local model that runs inside the harness. Every command uses `uv`.

Status of each step, so you know what to trust:

| Step | Status |
|---|---|
| 1 to 3 (install, run a model, smoke test any model) | Offline tests pass. Run against a real local model: not yet verified by the author |
| 4 (collect data) | Covered by offline tests with a fake server. Not yet run with a real teacher model |
| 5 to 7 (Tinker, export, Ollama import) | The data conversion is tested offline. The Tinker, merge and Ollama calls were written from the docs and have **not been run**. Expect to adjust them |

## 1. Install

```bash
git clone <YOUR_REPO_URL> && cd universal-agent-harness
uv sync --all-extras
uv run pytest -q
uv run python -m harness --mock --agent math -p "solve x^2+5*x+6=0"   # prints: Result: [-3, -2]
```

## 2. Run a local model with Ollama

```bash
ollama serve                      # skip if the Ollama app is already running
ollama pull qwen3:8b              # about 5 GB. Any model with tool support works
uv run python -m harness --agent math --model qwen3:8b --base-url http://localhost:11434/v1
```

Notes:

- Ollama exposes an OpenAI-compatible API at `http://localhost:11434/v1`, so the default `--provider openai` is right.
- Models without native tool calling are handled by the text protocol. Force it with `tool_mode = "text"` in the agent file.
- Qwen3 "thinks" by default. That is slower but harmless for the harness.
- `:cloud` models in Ollama run on ollama.com and may need an account with usage credits. They are not local.

## 3. Test any model with the smoke test

`scripts/smoke.py` runs the math and lab agents against a real model and asserts that the right tools were called. It works with any model the harness can reach: Ollama, vLLM, LM Studio, llama.cpp, OpenRouter, OpenAI, or Anthropic.

```bash
# Ollama
uv run scripts/smoke.py --model qwen3:8b --base-url http://localhost:11434/v1

# any OpenAI-compatible server, key from the environment
export HARNESS_API_KEY=...
uv run scripts/smoke.py --model <model-id> --base-url https://openrouter.ai/api/v1

# Anthropic native API
export ANTHROPIC_API_KEY=...
uv run scripts/smoke.py --provider anthropic --model <model-id>

# estimate a pass rate over 5 runs per scenario
uv run scripts/smoke.py --model qwen3:8b --base-url http://localhost:11434/v1 --runs 5
```

It prints `PASS` or `FAIL` per scenario and exits 1 if any fail. With no model configured it prints `SKIP` and exits 0. Environment variables `HARNESS_MODEL`, `HARNESS_BASE_URL`, `HARNESS_PROVIDER` work in place of flags.

Record the pass rate of the base model. It is your baseline for step 8.

## 4. Collect training data from a teacher model

The idea: run a strong model through the harness, keep only the runs where it used tools correctly, and train a small model to imitate those runs.

```bash
uv run scripts/collect_data.py \
  --model <teacher-model> --base-url <teacher-url> \
  --prompts data/prompts.jsonl --out data/train.jsonl --repeat 3
```

- `data/prompts.jsonl` has 36 starter scenarios (math, plots, lab notes with recall). Each line is `{"agent": "math", "turns": ["..."]}`. Add your own, especially for tools you wrote.
- A run is kept only if it called at least one tool, no tool returned an error, and it ended normally.
- `--repeat 3` samples each prompt three times for variety. Expect to want a few hundred examples, not 36. Write more prompts.
- Use a teacher that is better than your student. Using the student as its own teacher only teaches it to repeat what it already does.
- Review `data/train.jsonl` by eye before training. Remove wrong answers.

Output lines look like `{"messages": [...], "tools": [...]}` in OpenAI chat format.

## 5. Fine-tune on Tinker

Tinker is a hosted service that trains LoRA adapters on its GPUs. You need an account and an API key. Training costs money, so check pricing first.

```bash
export TINKER_API_KEY=...     # never commit this

# check the data conversion with no network access
uv run scripts/finetune_tinker.py --data data/train.jsonl --dry-run

# train, download the adapter, and merge it into a full HuggingFace model
uv run --with tinker --with tinker-cookbook scripts/finetune_tinker.py \
  --data data/train.jsonl --base-model Qwen/Qwen3-8B --out build/tuned
```

What the script does:

1. Rewrites each conversation into Qwen's native tool format: the tool list in the system prompt, calls as `<tool_call>{...}</tool_call>`, and results as `<tool_response>` user turns. This matches what Ollama's Qwen template sends at inference time.
2. Trains a LoRA adapter (rank 32, learning rate 2e-4, 3 epochs, batch 8 by default) with cross-entropy loss on assistant tokens only.
3. Saves a sampler checkpoint, downloads it to `build/tuned/adapter`, and merges it into `build/tuned/merged`.

If training succeeded but export failed, resume with `--checkpoint tinker://<run_id>/sampler_weights/final`.

The base model must match the Ollama model you plan to replace. `Qwen/Qwen3-8B` on Tinker pairs with `qwen3:8b` in Ollama. The renderer name `qwen3_disable_thinking` trains the model to answer without a thinking block. If your Tinker version names it differently, pass `--renderer`.

Tinker is not a project dependency, which is why the command uses `uv run --with`. Nothing is added to `pyproject.toml`.

## 6. Import the tuned model into Ollama

```bash
uv run scripts/ollama_import.py --model-dir build/tuned/merged --name harness-qwen3 --base qwen3:8b
```

This copies the chat template and parameters from `qwen3:8b`, points the new model at your merged weights, and quantizes to `q4_K_M` so it fits in memory. The merged 8B model is about 16 GB in bf16, so you need that much free disk during import.

Ollama can only import some architectures from safetensors. Qwen3 is expected to work. If the import fails, convert to GGUF with llama.cpp's `convert_hf_to_gguf.py` and use `FROM ./model.gguf` in a Modelfile.

## 7. Use it

```bash
uv run python -m harness --agent math --model harness-qwen3 --base-url http://localhost:11434/v1
```

## 8. Check that it actually got better

```bash
uv run scripts/smoke.py --model qwen3:8b       --base-url http://localhost:11434/v1 --runs 10
uv run scripts/smoke.py --model harness-qwen3  --base-url http://localhost:11434/v1 --runs 10
```

Compare the `pass rate:` lines. Two scenarios is a small test. For a real claim, add your own scenarios to `scripts/smoke.py`, and hold out prompts that were not in the training data. Fine-tuning on 36 prompts and testing on the same 36 proves nothing.

## Troubleshooting

- `The model call failed: HTTP 402`: the provider wants credits. Common with Ollama `:cloud` models.
- `ollama pull` ends with `EOF`: the download was cut off. Run it again, it resumes. On a bad connection, try a smaller model.
- The model prints tool JSON as text instead of calling the tool: set `tool_mode = "text"` in the agent file, or pick a model with tool support.
- Every call fails argument validation: read the error in the trace (`--trace trace.jsonl`). The harness returns readable errors to the model, and small models often need a clearer tool description.
