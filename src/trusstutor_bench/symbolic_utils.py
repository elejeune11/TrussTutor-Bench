import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

# Single-letter names SymPy uses for its own objects. Parsing them as plain
# symbols avoids clashes with problem symbols such as "Q".
SYMPY_RESERVED_CLASH_NAMES = ("C", "O", "S", "I", "N", "E", "Q")


def safe_sympify(val):
    """
    Parses a value into a SymPy expression, treating C, O, S, I, N, E and Q as plain symbols.

    Args:
        val: String or other value to parse.

    Returns:
        SymPy expression.
    """
    if isinstance(val, str):
        global_dict = {}
        exec("from sympy import *", global_dict)
        for name in SYMPY_RESERVED_CLASH_NAMES:
            global_dict.pop(name, None)
        return parse_expr(val, global_dict=global_dict, transformations=standard_transformations)
    return sp.sympify(val)
