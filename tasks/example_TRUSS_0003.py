# Standard library imports
import numpy as np
import os
import sympy as sp

# trusstutor_bench tools
from trusstutor_bench.truss_state import TrussState
from trusstutor_bench.schemas import FORCE_PROPERTY_SCHEMA
from trusstutor_bench.statics_tools import (
    autonomous_moj_macro,
    identify_zfms_by_inspection_tool,
    solve_method_of_sections_tool,
    solve_single_joint_tool,
    solve_global_reactions_tool,
    programmatic_solution,
    find_controlling_members
)
from trusstutor_bench.draw_truss import draw_truss_adapter

#############################################################
# PROBLEM DEFINITION TEMPLATE
# For instructions on how to fill out this file, please see
# the README.md file.
#############################################################

#############################################################
# METADATA
#############################################################

METADATA = {
    "benchmark_id": "TRUSS_0003",
    "topic": ["find_unknown_load_or_dimension"],
    "DEFAULT_RANDOM_SEED": 11,
    "task_creator": "eapostolakis99",
    "source_type": "original",
    "source_name": "eapostolakis99 handwritten",
    "source_problem_pointer": [""],
}

#############################################################
# SYMBOLIC DEFINITIONS
#############################################################

P = sp.Symbol("P", positive=True)

SYMBOLIC_DEFINITIONS = [P]

#############################################################
# NODES
#############################################################

# Maintain pure SymPy exactness to prevent floating point drift

NODES = {
    "A": (0, 0),
    "B": (5, sp.Rational(5, 2)),
    "C": (5, 0),
    "D": (10, 6),
    "E": (10, sp.Rational(5, 2)),
    "F": (10, 0),
    "G": (15, 6),
    "H": (15, 0),
    "I": (20, 6),
    "J": (20, sp.Rational(5, 2)),
    "K": (20, 0),
    "L": (25, sp.Rational(5, 2)),
    "M": (25, 0),
    "N": (30, 0)
}

#############################################################
# ELEMENTS
#############################################################

DEFAULT_PROPS = {'E': 1.0, 'A': 1.0}

connectivity = [
    ('AB', 'A', 'B'),
    ('AC', 'A', 'C'),
    ('BD', 'B', 'D'),
    ('BC', 'B', 'C'),
    ('BE', 'B', 'E'),
    ('CF', 'C', 'F'),
    ('DG', 'D', 'G'),
    ('DE', 'D', 'E'),
    ('EG', 'E', 'G'),
    ('EH', 'E', 'H'),
    ('EF', 'E', 'F'),
    ('FH', 'F', 'H'),
    ('GH', 'G', 'H'),
    ('GJ', 'G', 'J'),
    ('GI', 'G', 'I'),
    ('HJ', 'H', 'J'),
    ('HK', 'H', 'K'),
    ('IJ', 'I', 'J'),
    ('IL', 'I', 'L'),
    ('JK', 'J', 'K'),
    ('JL', 'J', 'L'),
    ('KM', 'K', 'M'),
    ('LM', 'L', 'M'),
    ('LN', 'L', 'N'),
    ('MN', 'M', 'N')
]

ELEMENTS = [{'id': eid, 'nodes': (n1, n2), **DEFAULT_PROPS} for eid, n1, n2 in connectivity]

#############################################################
# LOADS
#############################################################

LOADS = {
    "C": (0, -P),
    "F": (0, -P),
    "H": (0, -P)
}

#############################################################
# SUPPORTS
#############################################################

SUPPORTS = {
    "A": ['x', 'y'],
    "N": ['y']  
}

#############################################################
# PROBLEM STATEMENT
#############################################################

PROBLEM_SCHEMA = {
    "template": (
        "Determine the maximum force P which may be applied to the truss without "
        "exceeding {max_tension} kN in tension or {max_compression} kN in compression in any member."
        "Identify the maximum allowable force, the critical member(s) that govern this limit "
        "(list every member that reaches the limit at that load, in case of a tie), "
        "and whether they are in tension or compression."
    ),
    "units": {
        "force_unit": "kN"
    },
    "variables": {
        "max_tension": {"type": "float", "default": 300.0, "min": 200.0, "max": 600.0, "step": 5.0, "format": ".1f"},
        "max_compression": {"type": "float", "default": 500.0, "min": 300.0, "max": 900.0, "step": 5.0, "format": ".1f"}
    },
    "expected_llm_output": {
        "type": "object",
        "properties": {
            "max_allowable_force": {
                "type": "number",
                "description": "The maximum force in kN."
            },
            "controlling_member": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "The ID(s) of the member(s) that reach their limit first "
                    "(e.g., ['AB']). List every member tied for the governing "
                    "load, not just one, if more than one reaches the limit "
                    "at the exact same load."
                )
            },
            "controlling_state": {
                "type": "string",
                "enum": ["tension", "compression"],
                "description": "The failure mode of the controlling member."
            },
            "units": {
                "type": "string",
                "description": "Units of the force."
            }
        },
        "required": ["max_allowable_force", "controlling_member", "controlling_state", "units"],
        "additionalProperties": False
    }
}

#############################################################
# PROGRAMMATIC PROBLEM SOLUTION
#############################################################

def initialize_truss_state(sampled_vars, nodes, elements_list, loads, supports):
    """
    BESPOKE SETUP: Leaves 'P' as a pure symbol for the statics engine.
    Attaches sampled limits to the state object for the extractor.
    """
    evaluated_nodes = {}
    for nid, (x, y) in nodes.items():
        evaluated_nodes[nid] = (sp.sympify(x), sp.sympify(y))

    evaluated_loads = {}
    for nid, (fx, fy) in loads.items():
        evaluated_loads[nid] = (sp.sympify(fx), sp.sympify(fy))

    state = TrussState(evaluated_nodes, elements_list, supports, evaluated_loads)

    # Attach the randomly sampled limits to the state so the extractor can evaluate failure
    state.sampled_vars = sampled_vars
    return state


def extract_solution(state, problem_schema):
    """
    BESPOKE EXTRACTION: Inverse Optimization Problem.
    Solves for the critical load 'P' that causes the first member to fail.
    """
    # 1. Retrieve limits
    limit_T_N = state.sampled_vars['max_tension']
    limit_C_N = state.sampled_vars['max_compression']

    P_sym = sp.Symbol('P', positive=True)

    # 2. Extract numerical coefficients for P
    coeffs = {}
    for eid, force_expr in state.known_forces.items():
        if hasattr(force_expr, 'simplify'):
            # Substitute P=1 to find the multiplier for the load
            val = float(force_expr.subs(P_sym, 1).evalf())
        else:
            val = float(force_expr)

        if abs(val) > 1e-9:
            coeffs[eid] = val

    if not coeffs:
        raise ValueError("No internal forces found.")

    # 3. Locate the governing failure mode and every member tied for it
    controlling_P, controlling_state, controlling_members = find_controlling_members(
        coeffs, limit_T_N, limit_C_N
    )

    return {
        "max_allowable_force": round(controlling_P, 2),
        "controlling_member": controlling_members,
        "controlling_state": controlling_state,
        "units": problem_schema["units"]["force_unit"]
    }


SOLUTION_RECIPE = {
    1: {
        "function": initialize_truss_state,
        "inputs": {
            "nodes": NODES,
            "elements_list": ELEMENTS,
            "loads": LOADS,
            "supports": SUPPORTS
        }
    },
    2: {
        "function": solve_global_reactions_tool,
        "inputs": {}
    },
    3: {
        "function": autonomous_moj_macro,
        "inputs": {}
    },
    4: {
        "function": extract_solution,
        "inputs": {
            "problem_schema": PROBLEM_SCHEMA
        }
    }
}

#############################################################
# DRAWING VARIABLES
#############################################################

# We provide a dummy 'P' strictly for the drawing engine
# so it can compute float coordinate lengths for the load vectors.
DRAWING_VARIABLES = {
    "P": 35.0
}

#############################################################
# DEVELOPMENT EXECUTION BLOCK
#############################################################

if __name__ == "__main__":
    # 1. Extract limit values dynamically for the solver
    solver_vars = {
        "max_tension": PROBLEM_SCHEMA["variables"]["max_tension"].get("default"),
        "max_compression": PROBLEM_SCHEMA["variables"]["max_compression"].get("default")
    }

    print(f"--- Testing Programmatic Solution ---")
    try:
        truth_data = programmatic_solution(
            solution_recipe=SOLUTION_RECIPE,
            sampled_vars=solver_vars
        )
        import pprint
        print("Success! Output:")
        pprint.pprint(truth_data)
        print("\n")
    except Exception as e:
        print(f"Solver Failed: {e}\n")

    # 2. Generate schematic using the top-level DRAWING_VARIABLES
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_save_path = os.path.join(current_dir, "draw_truss")

    print(f"--- Testing Schematic Generation ---")
    draw_truss_adapter(
        nodes=NODES,
        elements_list=ELEMENTS,
        loads=LOADS,
        supports=SUPPORTS,
        metadata=METADATA,
        subs_dict=DRAWING_VARIABLES,
        output_dir=local_save_path
    )
