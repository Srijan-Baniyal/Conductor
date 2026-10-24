"""Real-model smoke test. Runs the math and lab agents and checks that tools were actually called.

    HARNESS_MODEL=llama3.1 HARNESS_BASE_URL=http://localhost:11434/v1 uv run scripts/smoke.py
    uv run scripts/smoke.py --model qwen3:8b --base-url http://localhost:11434/v1 --runs 5

Works with any model the harness can talk to. Flags override the environment variables.
Optional: HARNESS_PROVIDER=anthropic, HARNESS_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY.
Exit codes: 0 all passed or skipped, 1 a scenario failed.
"""
from __future__ import annotations
import argparse, os, sys, tempfile
from pathlib import Path

from harness import Agent, Hooks, ToolRegistry
from harness.cli import BUNDLED
from harness.config import load_agent_config
from harness.llm import AnthropicClient, OpenAICompatClient


def build_llm(model: str, base: str | None, provider: str):
    if provider == "anthropic":
        return AnthropicClient(model, base)
    return OpenAICompatClient(model, base)


def run_agent(name: str, llm, registry: ToolRegistry, root: str, prompts: list[str]) -> tuple[list[str], str]:
    hooks, called = Hooks(), []
    hooks.on("tool_start", lambda e, d: called.append(d["name"]))
    agent = Agent(load_agent_config(BUNDLED / f"{name}.toml"), llm, registry, root=root, hooks=hooks,
                  approve=lambda t, a: True)
    final = ""
    for p in prompts:
        final = agent.run(p).final
    return called, final


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Real-model smoke test for the harness")
    p.add_argument("--model", default=os.getenv("HARNESS_MODEL"))
    p.add_argument("--base-url", default=os.getenv("HARNESS_BASE_URL"))
    p.add_argument("--provider", choices=["openai", "anthropic"], default=os.getenv("HARNESS_PROVIDER", "openai"))
    p.add_argument("--runs", type=int, default=1, help="repeat each scenario to estimate a pass rate")
    a = p.parse_args(argv)
    if not a.model:
        print("SKIP: no model configured. Set HARNESS_MODEL (and HARNESS_BASE_URL for local or "
              "non-OpenAI endpoints) or pass --model to run the smoke test.")
        return 0
    llm = build_llm(a.model, a.base_url, a.provider)
    registry = ToolRegistry()
    registry.load_builtin()
    scenarios = {
        "math": ("math", ["Solve x^2+5*x+6=0 using your tools and state the roots."],
                 lambda c, f: "sympy_math" in c and "-3" in f and "-2" in f),
        "lab": ("lab", ["Remember this: Experiment A uses 5g NaCl in 100ml water.",
                        "What does Experiment A use? Check your notes."],
                lambda c, f: "add_note" in c and bool({"search_notes", "list_notes"} & set(c)) and "NaCl" in f),
    }
    passed = {label: 0 for label in scenarios}
    with tempfile.TemporaryDirectory() as tmp:
        for label, (agent, prompts, check) in scenarios.items():
            for i in range(a.runs):
                tag = f"{label} #{i + 1}" if a.runs > 1 else label
                try:
                    called, final = run_agent(agent, llm, registry, str(Path(tmp) / f"{label}{i}"), prompts)
                    ok = bool(check(called, final))
                    print(f"{'PASS' if ok else 'FAIL'} {tag}: tools={called} answer={final[:120]!r}")
                except Exception as e:
                    ok = False
                    print(f"FAIL {tag}: {type(e).__name__}: {e}")
                passed[label] += ok
    if a.runs > 1:
        print("pass rate: " + ", ".join(f"{k} {v}/{a.runs}" for k, v in passed.items()))
    return 0 if all(v == a.runs for v in passed.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
