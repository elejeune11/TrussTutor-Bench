import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

# sympy.abc documents these single-letter names as clashing with real sympy
# objects (C=Catalan, O=Order, S=Singleton registry, I=imaginary unit,
# N=numeric evaluator, E=Euler's number, Q=assumptions namespace). Trace
# values and LLM answers are logged/compared as plain strings (e.g.
# "5*P/2 - Q/3") and re-parsed here; a problem symbol named e.g. "Q" would
# otherwise silently resolve to sympy's assumptions object instead of a
# fresh Symbol('Q').
SYMPY_RESERVED_CLASH_NAMES = ("C", "O", "S", "I", "N", "E", "Q")


def safe_sympify(val):
    """Parses a value, immune to single-letter sympy name collisions."""
    if isinstance(val, str):
        global_dict = {}
        exec("from sympy import *", global_dict)
        for name in SYMPY_RESERVED_CLASH_NAMES:
            global_dict.pop(name, None)
        return parse_expr(val, global_dict=global_dict, transformations=standard_transformations)
    return sp.sympify(val)
