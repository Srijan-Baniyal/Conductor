"""LoRA supervised fine-tuning on Tinker, then download and merge the result.

Needs a Tinker account and TINKER_API_KEY in the environment. Tinker is not a core dependency, so run:

    uv run --with tinker --with tinker-cookbook scripts/finetune_tinker.py \
        --data data/train.jsonl --base-model Qwen/Qwen3-8B --out build/tuned

Steps: train, save a sampler checkpoint, download the adapter, merge it into a HuggingFace model
directory at <out>/merged. Then import it into Ollama with scripts/ollama_import.py.

Tool calls are written into the text in Qwen's native <tool_call> and <tool_response> format, so the
tuned model works with Ollama's Qwen chat template. Use --dry-run to check the data conversion
without any network access.
"""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path

TOOLS_HEADER = ("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>"
                "\n\nFor each function call, return a json object with function name and arguments within "
                "<tool_call></tool_call> XML tags:\n<tool_call>\n"
                '{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>')


def to_qwen_messages(example: dict) -> list[dict]:
    """Flatten OpenAI-style tool messages into plain chat turns that follow Qwen's tool format."""
    out: list[dict] = []
    for m in example["messages"]:
        role, content = m["role"], m.get("content") or ""
        if role == "system":
            tools = "\n".join(json.dumps(t) for t in example.get("tools", []))
            out.append({"role": "system", "content": content + (TOOLS_HEADER.format(tools=tools) if tools else "")})
        elif role == "assistant":
            calls = []
            for c in m.get("tool_calls") or []:
                fn = c["function"]
                args = fn["arguments"] if isinstance(fn["arguments"], dict) else json.loads(fn["arguments"] or "{}")
                calls.append("<tool_call>\n" + json.dumps({"name": fn["name"], "arguments": args}) + "\n</tool_call>")
            out.append({"role": "assistant", "content": "\n".join(p for p in [content, *calls] if p)})
        elif role == "tool":
            block = f"<tool_response>\n{content}\n</tool_response>"
            if out and out[-1]["role"] == "user" and out[-1]["content"].startswith("<tool_response>"):
                out[-1]["content"] += "\n" + block
            else:
                out.append({"role": "user", "content": block})
        else:
            out.append({"role": role, "content": content})
    return out


def load_examples(path: str | Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"{path} has no examples. Run scripts/collect_data.py first.")
    return rows


def train(a: argparse.Namespace, examples: list[dict]) -> str:
    import tinker
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.supervised.data import conversation_to_datum

    service = tinker.ServiceClient()
    client = service.create_lora_training_client(base_model=a.base_model, rank=a.rank)
    renderer = get_renderer(a.renderer, client.get_tokenizer())
    data = [conversation_to_datum(to_qwen_messages(e), renderer, a.max_length, TrainOnWhat.ALL_ASSISTANT_MESSAGES)
            for e in examples]
    rng = random.Random(0)
    for epoch in range(a.epochs):
        rng.shuffle(data)
        for i in range(0, len(data), a.batch_size):
            batch = data[i:i + a.batch_size]
            client.forward_backward(batch, loss_fn="cross_entropy").result()
            client.optim_step(tinker.AdamParams(learning_rate=a.lr)).result()
            print(f"epoch {epoch + 1}/{a.epochs} step {i // a.batch_size + 1}", flush=True)
    path = client.save_weights_for_sampler("final").result().path
    print(f"checkpoint: {path}")
    return path


def export(a: argparse.Namespace, tinker_path: str) -> None:
    from tinker_cookbook import weights
    out = Path(a.out)
    adapter = weights.download(tinker_path=tinker_path, output_dir=str(out / "adapter"))
    weights.build_hf_model(base_model=a.base_model, adapter_path=adapter, output_path=str(out / "merged"))
    print(f"merged model: {out / 'merged'}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/train.jsonl")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--renderer", default="qwen3_disable_thinking")
    p.add_argument("--out", default="build/tuned")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--checkpoint", help="skip training and export this tinker:// path")
    p.add_argument("--dry-run", action="store_true", help="convert and print the first example, then exit")
    a = p.parse_args(argv)

    if a.dry_run:
        examples = load_examples(a.data)
        print(f"{len(examples)} examples")
        for m in to_qwen_messages(examples[0]):
            print(f"--- {m['role']}\n{m['content']}")
        return 0
    if not os.getenv("TINKER_API_KEY"):
        print("SKIP: set TINKER_API_KEY to train on Tinker (use --dry-run to check the data offline).")
        return 0
    path = a.checkpoint or train(a, load_examples(a.data))
    export(a, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
