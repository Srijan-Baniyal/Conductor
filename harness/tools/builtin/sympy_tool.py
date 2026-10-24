from __future__ import annotations
from ..base import tool

OPS = ["solve", "simplify", "expand", "factor", "diff", "integrate", "evaluate"]


@tool("sympy_math", enums={"operation": OPS}, timeout=10, isolated=True,
      doc={"operation": "What to do with the expression.",
           "expression": "Math in Python syntax with explicit *, e.g. 'x^2 + 5*x + 6 = 0'. For solve, '=' is allowed.",
           "variable": "Variable for solve, diff, integrate.",
           "lower": "Lower bound for a definite integral (optional).",
           "upper": "Upper bound for a definite integral (optional)."})
def sympy_math(operation: str, expression: str, variable: str = "x", lower: str = "", upper: str = "") -> str:
    """Exact symbolic math with SymPy: solve, simplify, expand, factor, differentiate, integrate, evaluate numerically."""
    import sympy as sp
    from .safe_math import parse, parse_equation
    var = sp.Symbol(variable)
    if operation == "solve":
        return str(sp.solve(parse_equation(expression), var))
    e = parse(expression)
    if operation == "simplify":
        return str(sp.simplify(e))
    if operation == "expand":
        return str(sp.expand(e))
    if operation == "factor":
        return str(sp.factor(e))
    if operation == "diff":
        return str(sp.diff(e, var))
    if operation == "integrate":
        if lower and upper:
            return str(sp.integrate(e, (var, parse(lower), parse(upper))))
        return str(sp.integrate(e, var)) + " + C"
    if e.free_symbols:
        raise ValueError(f"cannot evaluate numerically, free symbols: {sorted(map(str, e.free_symbols))}")
    return str(sp.N(e, 15))
