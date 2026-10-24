"""Drop-in plugin: harness --tools-dir examples/custom_tools
Any Tool instance defined at module level is registered automatically."""
from harness import tool

_TO_M = {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001, "mi": 1609.344, "ft": 0.3048, "in": 0.0254}


@tool("unit_convert", enums={"from_unit": list(_TO_M), "to_unit": list(_TO_M)})
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a length between m, km, cm, mm, mi, ft, in."""
    return f"{value * _TO_M[from_unit] / _TO_M[to_unit]:.6g} {to_unit}"
