"""Collect tool-calling training data by running a teacher model through the harness.

    uv run scripts/collect_data.py --model minimax-m3:cloud --base-url http://localhost:11434/v1 \
        --prompts data/prompts.jsonl --out data/train.jsonl

Each line of the prompts file is {"agent": "math", "turns": ["user message", ...]}.
A run is kept only if it called at least one tool, no tool returned an error, and it ended normally.
Output lines are {"messages": [...], "tools": [...]} in OpenAI chat format, ready for
scripts/finetune_tinker.py. Use --repeat to sample each prompt several times.
"""
from __future__ import annotations
import argparse, json, os, sys, tempfile
from pathlib import Path

from harness import Agent, ConversationMemory, Message, ToolRegistry
from harness.cli import BUNDLED
from harness.config import load_agent_config


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


def is_good(messages: list[Message], stopped: str) -> bool:
    tools = [m for m in messages if m.role == "tool"]
    last = messages[-1] if messages else None
    return bool(stopped == "done" and tools and not any(m.content.startswith("ERROR:") for m in tools)
                and last and last.role == "assistant" and last.content.strip() and not last.tool_calls)


def collect_one(entry: dict, llm, registry: ToolRegistry) -> dict | None:
    cfg = load_agent_config(BUNDLED / f"{entry['agent']}.toml")
    with tempfile.TemporaryDirectory() as tmp:
        agent = Agent(cfg, llm, registry, root=tmp, memory=ConversationMemory(), approve=lambda t, a: True)
        stopped = "done"
        for turn in entry["turns"]:
            stopped = agent.run(turn).stopped
            if stopped != "done":
                break
        msgs = agent.memory.messages
        if not is_good(msgs, stopped):
            return None
        return {"messages": to_openai(agent._system(), msgs), "tools": agent.registry.schemas()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.getenv("HARNESS_MODEL"))
    p.add_argument("--base-url", default=os.getenv("HARNESS_BASE_URL"))
    p.add_argument("--provider", choices=["openai", "anthropic"], default=os.getenv("HARNESS_PROVIDER", "openai"))
    p.add_argument("--prompts", default="data/prompts.jsonl")
    p.add_argument("--out", default="data/train.jsonl")
    p.add_argument("--repeat", type=int, default=1, help="samples per prompt")
    a = p.parse_args(argv)
    if not a.model:
        print("SKIP: set --model or HARNESS_MODEL to the teacher model.")
        return 0

    from harness.llm import AnthropicClient, OpenAICompatClient
    llm = AnthropicClient(a.model, a.base_url) if a.provider == "anthropic" else OpenAICompatClient(a.model, a.base_url)
    registry = ToolRegistry()
    registry.load_builtin()
    entries = [json.loads(line) for line in Path(a.prompts).read_text().splitlines() if line.strip()]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    kept = total = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for entry in entries:
            for _ in range(a.repeat):
                total += 1
                try:
                    ex = collect_one(entry, llm, registry)
                except Exception as e:
                    print(f"  error on {entry['turns'][0]!r}: {e}", file=sys.stderr)
                    ex = None
                if ex:
                    f.write(json.dumps(ex) + "\n")
                    kept += 1
    print(f"kept {kept} of {total} runs -> {a.out}")
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())
