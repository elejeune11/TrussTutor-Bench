# Standard library imports
import numpy as np
import os
import sympy as sp

# trusstutor_bench tools
from trusstutor_bench.truss_state import TrussState
from trusstutor_bench.statics_tools import (
    autonomous_moj_macro,
    identify_zfms_by_inspection_tool,
    solve_method_of_sections_tool,
    solve_single_joint_tool,
    solve_global_reactions_tool,
    programmatic_solution
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
    "benchmark_id": "TRUSS_0001",
    "topic": ["method_of_joints"],
    "DEFAULT_RANDOM_SEED": 11,
    "task_creator": "elejeune11",
    "source_type": "original",
    "source_name": "github test example",
    "source_problem_pointer": [""],
}

#############################################################
# SYMBOLIC DEFINITIONS
#############################################################

# Leave as an empty list if there are no symbolic variables.
L = sp.Symbol('L', positive=True)
P = sp.Symbol('P')

SYMBOLIC_DEFINITIONS = [L, P]

#############################################################
# NODES
#############################################################

# Note: Use sp.Rational for fractions to prevent premature float conversion.
NODES = {
    "A": (0, 0),
    "B": (L, 0),
    "C": (sp.Rational(1, 2) * L, sp.Rational(1, 2) * sp.sqrt(3) * L)
}

#############################################################
# ELEMENTS
#############################################################

# Default properties required for direct stiffness method compatibility.
DEFAULT_PROPS = {'E': 1.0, 'A': 1.0}

connectivity = [
    ("AB", "A", "B"),
    ("BC", "B", "C"),
    ("CA", "C", "A")
]

ELEMENTS = [{'id': eid, 'nodes': (n1, n2), **DEFAULT_PROPS} for eid, n1, n2 in connectivity]

#############################################################
# LOADS
#############################################################

# Note: Ensure vector signs match the problem text and intended schematic.
LOADS = {
    "C": (0, -P)
}

#############################################################
# SUPPORTS
#############################################################

# 2D determinate trusses typically require exactly 3 constraints.
SUPPORTS = {
    "A": ['x', 'y'],  # Pin
    "B": ['y']        # Roller
}

#############################################################
# PROBLEM STATEMENT
#############################################################

PROBLEM_SCHEMA = {
    "template": (
        "A statically determinate equilateral truss is supported by a pin at A "
        "and a roller at B. A point load of P {P} {force_unit} "
        "is applied at the apex (Node C). If the bottom chord length is L, "
        "determine the maximum compression force in the structure."
    ),
    "units": {
        "force_unit": "kN",
        "length_unit": "m"
    },
    "variables": {
        "P": {"type": "float", "default": 50.0, "min": 10.0, "max": 100.0, "step": 5.0, "format": ".1f"},
        "L_val": {"type": "symbolic"}
    },
    "expected_llm_output": {
        "type": "object",
        "properties": {
            "max_compression_magnitude": {
                "type": "number",
                "description": "The numerical value of the maximum compressive force."
            },
            "units": {
                "type": "string",
                "description": "The units of the force."
            }
        },
        "required": ["max_compression_magnitude", "units"],
        "additionalProperties": False
    }
}

#############################################################
# PROGRAMMATIC PROBLEM SOLUTION
#############################################################

def initialize_truss_state(sampled_vars, nodes, elements_list, loads, supports):
    """
    BESPOKE SETUP: Inject dynamic numerical values into symbolic loads/geometry.
    Deliberately leaves L as a purely symbolic variable for the statics engine.
    """
    P_val = sampled_vars['P']
    P_sym = sp.Symbol('P')

    evaluated_nodes = {}
    for nid, (x, y) in nodes.items():
        evaluated_nodes[nid] = (sp.sympify(x), sp.sympify(y))

    evaluated_loads = {}
    for nid, (fx, fy) in loads.items():
        eval_fx = sp.sympify(fx).subs(P_sym, P_val)
        eval_fy = sp.sympify(fy).subs(P_sym, P_val)
        evaluated_loads[nid] = (eval_fx, eval_fy)

    state = TrussState(evaluated_nodes, elements_list, supports, evaluated_loads)
    return state


def extract_solution(state, problem_schema):
    """
    BESPOKE EXTRACTION: Target exactly what the problem schema requests.
    """
    # Extract all forces into a simple list of floats
    forces = [float(expr.simplify().evalf()) for expr in state.known_forces.values()]
        
    # Find the most negative force, defaulting to 0.0 if the list is empty
    max_comp_val = min(forces, default=0.0)
    
    # Isolate the magnitude (ignoring near-zero floating point noise)
    max_comp_magnitude = abs(max_comp_val) if max_comp_val < -1e-9 else 0.0

    return {
        "max_compression_magnitude": max_comp_magnitude,
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

# Specific numerical values used exclusively to render the static schematic.
# Keys must exactly match the variable names in SYMBOLIC_DEFINITIONS.
DRAWING_VARIABLES = {
    "P": PROBLEM_SCHEMA["variables"]["P"].get("default"),
    "L": 10.0
}

#############################################################
# DEVELOPMENT EXECUTION BLOCK
#############################################################

if __name__ == "__main__":
    # 1. Extract values dynamically for the solver (L remains purely symbolic)
    default_p = PROBLEM_SCHEMA["variables"]["P"].get("default")
    solver_vars = {"P": default_p}
    
    print(f"--- Testing Programmatic Solution ---")
    try:
        truth_data = programmatic_solution(
            solution_recipe=SOLUTION_RECIPE, 
            sampled_vars=solver_vars
        )
        print(f"Success! Output: {truth_data}\n")
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
