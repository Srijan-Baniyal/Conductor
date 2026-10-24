"""Safe expression parsing. sympify/parse_expr call eval(), so untrusted model output must be
checked first. We whitelist AST nodes, function names and magnitudes, and hand parse_expr an
empty builtins namespace."""
from __future__ import annotations
import ast, re

CONSTS = ("pi", "E", "I", "oo")
FUNCS = ("sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "log", "sqrt",
         "Abs", "floor", "ceiling", "factorial", "gamma")
_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Call, ast.Load,
          ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd)
_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9]{0,5}")


def check(src: str) -> str:
    src = src.strip().replace("^", "**")
    if not src or len(src) > 300:
        raise ValueError("expression empty or longer than 300 characters")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError:
        raise ValueError("syntax error; use explicit * for multiplication, e.g. 5*x")
    pows = 0
    for n in ast.walk(tree):
        if not isinstance(n, _NODES):
            raise ValueError(f"disallowed syntax: {type(n).__name__}")
        if isinstance(n, ast.Constant) and (isinstance(n.value, bool) or not isinstance(n.value, (int, float))
                                            or abs(n.value) > 10**6):
            raise ValueError("only numeric constants up to 1e6 are allowed")
        if isinstance(n, ast.Name) and n.id not in CONSTS + FUNCS and not _SYMBOL.fullmatch(n.id):
            raise ValueError(f"disallowed name '{n.id}'")
        if isinstance(n, ast.Call) and (not isinstance(n.func, ast.Name) or n.func.id not in FUNCS or n.keywords):
            raise ValueError("only whitelisted functions may be called: " + ", ".join(FUNCS))
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow):
            pows += 1
            for sub in ast.walk(n.right):  # blocks 9**9**9 and 2**100000 style CPU/memory bombs
                if (isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Pow)) or (isinstance(sub, ast.Constant) and abs(sub.value) > 1000):
                    raise ValueError("exponents must be constants up to 1000 with no nested powers")
    if pows > 3:
        raise ValueError("too many exponent operators")
    return src


def parse(src: str):
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations
    src = check(src)
    local = {n: getattr(sp, n) for n in CONSTS + FUNCS}
    glob = {"__builtins__": {}, "Symbol": sp.Symbol, "Integer": sp.Integer, "Float": sp.Float,
            "Rational": sp.Rational, "Function": sp.Function}
    return parse_expr(src, local_dict=local, global_dict=glob, transformations=standard_transformations)


def parse_equation(src: str):
    """'lhs = rhs' becomes Eq(lhs, rhs); anything else is an expression."""
    import sympy as sp
    if "=" in src and not re.search(r"[<>!=]=", src):
        lhs, rhs = src.split("=", 1)
        return sp.Eq(parse(lhs), parse(rhs))
    return parse(src)
