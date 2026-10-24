"""LoRA supervised fine-tuning on Tinker, then download and merge the result.

Needs a Tinker account and TINKER_API_KEY in the environment. Tinker is not a core dependency, so run:

    uv run --with tinker --with tinker-cookbook scripts/finetune_tinker.py \
        --data data/train.jsonl --base-model Qwen/Qwen3.8-27B --out build/tuned

Steps: train, save a sampler checkpoint, download the adapter, merge it into a HuggingFace model
directory at <out>/merged. Then import it into Ollama with scripts/ollama_import.py.

Tool calls are written into the text in the base model's native format, so the tuned model works with
Ollama's chat template. Qwen3.5 uses <tool_call><function=name><parameter=key>value</parameter> XML,
Qwen3 uses JSON inside <tool_call>. The format is picked from the base model name, or set --tool-format. Use --dry-run to check the data conversion
without any network access.
"""
from __future__ import annotations
import argparse, json, os, random, sys
from pathlib import Path

JSON_TOOLS_HEADER = ("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                     "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tools}\n</tools>"
                     "\n\nFor each function call, return a json object with function name and arguments within "
                     "<tool_call></tool_call> XML tags:\n<tool_call>\n"
                     '{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>')

# Copied from the Qwen3.5-4B chat_template.jinja on HuggingFace.
XML_TOOLS_HEADER = ("# Tools\n\nYou have access to the following functions:\n\n<tools>{tools}\n</tools>"
                    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
                    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
                    "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
                    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
                    "Reminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> "
                    "block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be "
                    "specified\n- You may provide optional reasoning for your function call in natural language BEFORE "
                    "the function call, but NOT after\n- If there is no function call available, answer the question "
                    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>")


def default_tool_format(base_model: str) -> str:
    name = base_model.lower()
    if "qwen3.8" in name:
        return "qwen38_xml"
    return "qwen35_xml" if ("qwen3.5" in name or "qwen3.6" in name) else "qwen3_json"


def _xml_call(name: str, args: dict, json_scalars: bool = False) -> str:
    """Qwen3.5 writes scalars with str(). Qwen3.8 writes non-strings as JSON (true, not True)."""
    out = f"<tool_call>\n<function={name}>\n"
    for k, v in args.items():
        if json_scalars:
            text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        else:
            text = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
        out += f"<parameter={k}>\n{text}\n</parameter>\n"
    return out + "</function>\n</tool_call>"


def to_qwen_messages(example: dict, fmt: str = "qwen35_xml") -> list[dict]:
    """Flatten OpenAI-style tool messages into plain chat turns that follow the model's tool format."""
    xml = fmt in ("qwen35_xml", "qwen38_xml")
    out: list[dict] = []
    for m in example["messages"]:
        role, content = m["role"], m.get("content") or ""
        if role == "system":
            tools = [json.dumps(t, ensure_ascii=False) for t in example.get("tools", [])]
            if not tools:
                out.append({"role": "system", "content": content})
            elif xml:
                head = XML_TOOLS_HEADER.format(tools="".join("\n" + t for t in tools))
                out.append({"role": "system", "content": head + ("\n\n" + content.strip() if content.strip() else "")})
            else:
                out.append({"role": "system", "content": content + JSON_TOOLS_HEADER.format(tools="\n".join(tools))})
        elif role == "assistant":
            calls = []
            for c in m.get("tool_calls") or []:
                fn = c["function"]
                args = fn["arguments"] if isinstance(fn["arguments"], dict) else json.loads(fn["arguments"] or "{}")
                calls.append(_xml_call(fn["name"], args, fmt == "qwen38_xml") if xml else
                             "<tool_call>\n" + json.dumps({"name": fn["name"], "arguments": args}) + "\n</tool_call>")
            if xml:
                text = content.strip() + ("\n\n" if content.strip() and calls else "") + "\n".join(calls)
            else:
                text = "\n".join(p for p in [content, *calls] if p)
            out.append({"role": "assistant", "content": text})
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
    data = [conversation_to_datum(to_qwen_messages(e, a.tool_format), renderer, a.max_length, TrainOnWhat.ALL_ASSISTANT_MESSAGES)
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
    p.add_argument("--base-model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--renderer", default="qwen3_8_disable_thinking")
    p.add_argument("--tool-format", choices=["qwen38_xml", "qwen35_xml", "qwen3_json"], help="default: chosen from the base model name")
    p.add_argument("--out", default="build/tuned")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--checkpoint", help="skip training and export this tinker:// path")
    p.add_argument("--dry-run", action="store_true", help="convert and print the first example, then exit")
    a = p.parse_args(argv)
    a.tool_format = a.tool_format or default_tool_format(a.base_model)

    if a.dry_run:
        examples = load_examples(a.data)
        print(f"{len(examples)} examples")
        for m in to_qwen_messages(examples[0], a.tool_format):
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
