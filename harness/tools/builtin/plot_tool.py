from __future__ import annotations
import threading, uuid
from ..base import tool, ToolContext

_LOCK = threading.Lock()  # matplotlib's pyplot state is not thread-safe
KINDS = ["function", "line", "scatter", "bar"]


def _nums(v, name):
    if not v:
        raise ValueError(f"'{name}' is required for this kind")
    if len(v) > 10000:
        raise ValueError(f"'{name}' has more than 10000 points")
    try:
        return [float(x) for x in v]
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a list of numbers")


@tool("plot", enums={"kind": KINDS}, timeout=30, isolated=True,
      doc={"kind": "function plots an expression in x; line/scatter/bar plot y_values.",
           "expression": "For kind=function: expression in x, e.g. 'sin(x)' or 'y = x^2'.",
           "x_min": "Left bound for function plots.", "x_max": "Right bound for function plots.",
           "y_values": "Numbers to plot for line/scatter/bar.",
           "x_values": "Optional x positions for line/scatter; defaults to 0..n-1.",
           "title": "Plot title."})
def plot(kind: str, expression: str = "", x_min: float = -10.0, x_max: float = 10.0, y_values: list = None,
         x_values: list = None, title: str = "", ctx: ToolContext = None) -> str:
    """Draw a plot and save it as a PNG. Returns the file path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from .safe_math import parse

    if kind == "function":
        s = expression.strip()
        if "=" in s:
            lhs, rhs = s.split("=", 1)
            if lhs.strip() != "y":
                raise ValueError("equations must have the form 'y = f(x)'")
            s = rhs
        e = parse(s)
        syms = sorted(e.free_symbols, key=str)
        if len(syms) > 1:
            raise ValueError("function plots support one variable")
        if not x_min < x_max:
            raise ValueError("x_min must be less than x_max")
        xs = np.linspace(x_min, x_max, 600)
        f = sp.lambdify(syms[0] if syms else sp.Symbol("x"), e, "numpy")
        with np.errstate(all="ignore"):
            c = np.asarray(f(xs), dtype=complex) * np.ones_like(xs)
        ys = np.where(np.abs(c.imag) < 1e-12, c.real, np.nan)
    else:
        ys = np.array(_nums(y_values, "y_values"))
        xs = np.array(_nums(x_values, "x_values")) if x_values else np.arange(len(ys), dtype=float)
        if len(xs) != len(ys):
            raise ValueError("x_values and y_values must have the same length")

    out_dir = ctx.workdir if ctx else __import__("pathlib").Path("workspace")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"plot_{uuid.uuid4().hex[:8]}.png"
    with _LOCK:
        fig, ax = plt.subplots(figsize=(6, 4))
        if kind == "bar":
            ax.bar(xs, ys)
        elif kind == "scatter":
            ax.scatter(xs, ys)
        else:
            ax.plot(xs, ys, marker="o" if kind == "line" else None)
        ax.set_title(title or (expression if kind == "function" else kind))
        ax.grid(True, alpha=0.3)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return f"Saved plot to {path}"
