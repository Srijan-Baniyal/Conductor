"""Import a merged HuggingFace model directory into Ollama, reusing the base model's template.

    uv run scripts/ollama_import.py --model-dir build/tuned/merged --name harness-qwen35 --base qwen3.5:4b

Then test it with: HARNESS_MODEL=harness-qwen35 HARNESS_BASE_URL=http://localhost:11434/v1 uv run scripts/smoke.py
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path


def build_modelfile(base_modelfile: str, model_dir: Path) -> str:
    """Copy TEMPLATE, PARAMETER, SYSTEM and friends from the base model, but load weights from model_dir."""
    kept, seen_from = [], False
    for line in base_modelfile.splitlines():
        if not seen_from:  # header comments and the FROM line come first; later '#' lines may be template text
            seen_from = line.startswith("FROM ")
            continue
        kept.append(line)
    return f"FROM {model_dir.resolve()}\n" + "\n".join(kept).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--name", required=True, help="name for the new Ollama model")
    p.add_argument("--base", default="qwen3.5:4b", help="Ollama model whose template to reuse")
    p.add_argument("--quantize", default="q4_K_M", help="quantization passed to ollama create, '' to skip")
    a = p.parse_args(argv)
    model_dir = Path(a.model_dir)
    if not (model_dir / "config.json").exists():
        print(f"{model_dir} does not look like a HuggingFace model directory (no config.json).")
        return 1
    shown = subprocess.run(["ollama", "show", "--modelfile", a.base], capture_output=True, text=True)
    if shown.returncode:
        print(f"Could not read the base model '{a.base}'. Pull it first: ollama pull {a.base}\n{shown.stderr}")
        return 1
    mf = model_dir.parent / "Modelfile"
    mf.write_text(build_modelfile(shown.stdout, model_dir), encoding="utf-8")
    cmd = ["ollama", "create", a.name, "-f", str(mf)] + (["--quantize", a.quantize] if a.quantize else [])
    print("running:", " ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
