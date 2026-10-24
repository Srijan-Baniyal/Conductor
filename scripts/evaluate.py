"""Score a model on held-out scenarios and print a pass rate per category.

    uv run --env-file .env scripts/evaluate.py --model Qwen/Qwen3.8-27B \
        --base-url https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1 \
        --prompts data/eval_prompts.jsonl --workers 8 --json results/base.json

A scenario passes if the model called the expected tools and its final answer contains the ground truth.
Compare a base model and a fine-tuned one on the same file. Make the file with scripts/make_prompts.py.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
from pathlib import Path

from harness import ToolRegistry
from harness.evaluation import BUNDLED, check, run_many


def category(entry: dict) -> str:
    text = entry["turns"][0].lower()
    if entry.get("expect_no_tools"):
        return "no tool needed"
    if entry.get("seed_notes"):
        return "untrusted notes"
    if entry.get("expect_args"):
        return "note tagging"
    if entry["agent"] == "lab":
        return "lab recall"
    if len(entry.get("expect_tools", [])) > 1:
        return "multi-tool"
    if text.startswith("plot"):
        return "math plot"
    return "math solve"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.getenv("HARNESS_MODEL"))
    p.add_argument("--base-url", default=os.getenv("HARNESS_BASE_URL"))
    p.add_argument("--provider", choices=["openai", "anthropic"], default=os.getenv("HARNESS_PROVIDER", "openai"))
    p.add_argument("--prompts", default="data/eval_prompts.jsonl")
    p.add_argument("--extra-body", default=None, help='JSON merged into every request, e.g. \'{"reasoning_effort": "none"}\'')
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="only the first N scenarios")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--agents-dir", default=str(BUNDLED))
    p.add_argument("--suffix", default="", help='agent file suffix, e.g. "_pro" to test the long house-format prompt')
    p.add_argument("--label", default="", help="name for this model in the report")
    p.add_argument("--json", dest="json_out", help="write per-scenario results here")
    a = p.parse_args(argv)
    if not a.model:
        print("SKIP: set --model or HARNESS_MODEL.")
        return 0

    from harness.llm import make_client
    llm = make_client(a.provider, a.model, a.base_url, json.loads(a.extra_body) if a.extra_body else None)
    registry = ToolRegistry()
    registry.load_builtin()
    entries = [json.loads(line) for line in Path(a.prompts).read_text().splitlines() if line.strip()]
    if a.limit:
        entries = entries[:a.limit]

    rows, by_cat = [], defaultdict(lambda: [0, 0])
    for entry, res, err in run_many(entries, llm, registry, a.workers, 1, a.temperature,
                                    lambda i, n, _: print(f"  {i}/{n}", end="\r", flush=True),
                                    Path(a.agents_dir), a.suffix):
        ok, why = (False, err) if res is None else check(entry, res)
        c = category(entry)
        by_cat[c][0] += ok
        by_cat[c][1] += 1
        rows.append({"turns": entry["turns"], "category": c, "passed": ok, "reason": why,
                     "called": res.called if res else [], "final": res.final[:300] if res else ""})
    print(f"\n{a.label or a.model}")
    for c, (ok, n) in sorted(by_cat.items()):
        print(f"  {c:12s} {ok}/{n}  {100 * ok / n:.0f}%")
    total, n = sum(v[0] for v in by_cat.values()), sum(v[1] for v in by_cat.values())
    print(f"  {'overall':12s} {total}/{n}  {100 * total / n:.0f}%")
    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps({"model": a.model, "label": a.label, "overall": [total, n],
                                               "categories": by_cat, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
