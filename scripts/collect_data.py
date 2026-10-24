"""Collect tool-calling training data by running a teacher model through the harness.

    uv run --env-file .env scripts/collect_data.py --model thinkingmachines/Inkling-Small \
        --base-url https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1 \
        --prompts data/train_prompts.jsonl --out data/train.jsonl --workers 8

Prompt distillation: pass --agents-dir examples/agents --suffix _pro to run the teacher with the long house-format
prompt (math_pro.toml, lab_pro.toml). The saved examples swap in the stock agent's short system prompt, so the
student learns the format without being told. A run is kept only if it is clean (finished normally, used a tool, no tool errors) and, when the scenario
has checks, passes them. Output lines are {"messages": [...], "tools": [...]} in OpenAI chat format, ready for
scripts/finetune_tinker.py. Make prompts with scripts/make_prompts.py.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

from harness import Message, ToolRegistry
from harness.evaluation import BUNDLED, RunResult, check, is_clean, run_many, system_for


def to_openai(system: str, messages: list[Message]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "name": m.name, "content": m.content})
        elif m.role == "assistant" and m.tool_calls:
            out.append({"role": "assistant", "content": m.content, "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in m.tool_calls]})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def to_example(res: RunResult, system: str | None = None) -> dict:
    return {"messages": to_openai(system or res.system, res.messages), "tools": res.tools}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.getenv("HARNESS_MODEL"))
    p.add_argument("--base-url", default=os.getenv("HARNESS_BASE_URL"))
    p.add_argument("--provider", choices=["openai", "anthropic"], default=os.getenv("HARNESS_PROVIDER", "openai"))
    p.add_argument("--prompts", default="data/train_prompts.jsonl")
    p.add_argument("--out", default="data/train.jsonl")
    p.add_argument("--repeat", type=int, default=1, help="samples per prompt")
    p.add_argument("--temperature", type=float, default=0.3, help="above 0 so repeats differ")
    p.add_argument("--extra-body", default=None, help='JSON merged into every request, e.g. \'{"reasoning_effort": "none"}\'')
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--agents-dir", default=str(BUNDLED), help="folder with the teacher agent files")
    p.add_argument("--suffix", default="", help='teacher agent file suffix, e.g. "_pro"')
    a = p.parse_args(argv)
    if not a.model:
        print("SKIP: set --model or HARNESS_MODEL to the teacher model.")
        return 0

    from harness.llm import make_client
    llm = make_client(a.provider, a.model, a.base_url, json.loads(a.extra_body) if a.extra_body else None)
    registry = ToolRegistry()
    registry.load_builtin()
    entries = [json.loads(line) for line in Path(a.prompts).read_text().splitlines() if line.strip()]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    kept, dropped = 0, {"error": 0, "unclean": 0, "failed check": 0}

    def progress(i, n, item):
        if i % 10 == 0 or i == n:
            print(f"  {i}/{n} runs", flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        for entry, res, err in run_many(entries, llm, registry, a.workers, a.repeat, a.temperature, progress,
                                      Path(a.agents_dir), a.suffix):
            if res is None:
                dropped["error"] += 1
                print(f"  error on {entry['turns'][0]!r}: {err}", file=sys.stderr)
            elif not is_clean(res, entry):
                dropped["unclean"] += 1
            elif not check(entry, res)[0]:
                dropped["failed check"] += 1
            else:
                student = system_for(entry["agent"]) if a.suffix else None
                f.write(json.dumps(to_example(res, student)) + "\n")
                f.flush()
                kept += 1
    print(f"kept {kept} runs -> {a.out}; dropped {dropped}")
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
