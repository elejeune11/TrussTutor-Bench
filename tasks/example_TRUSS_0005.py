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
from trusstutor_bench.schemas import FORCE_PROPERTY_SCHEMA


#############################################################
# PROBLEM DEFINITION TEMPLATE
# For instructions on how to fill out this file, please see
# the README.md file.
#############################################################

#############################################################
# METADATA
#############################################################

METADATA = {
    "benchmark_id": "TRUSS_0005",
    "topic": ["method_of_joints"],
    "DEFAULT_RANDOM_SEED": 11,
    "task_creator": "eapostolakis99",
    "source_type": "original",
    "source_name": "eapostolakis99 handwritten",
    "source_problem_pointer": [""],
}


#############################################################
# SYMBOLIC DEFINITIONS
#############################################################

L = sp.Symbol('L', positive=True)
P = sp.Symbol('P')

SYMBOLIC_DEFINITIONS = [L, P]

#############################################################
# NODES
#############################################################

NODES = {
    "A": (0, L),
    "B": (4 * L, L),
    "C": (L, 0),
    "D": (2 * L, 0),
    "E": (3 * L, 0),
    "F": (L, L),
    "G": (2 * L, L),
    "H": (3 * L, L),
    "I": (L, 2 * L),
    "K": (2 * L, 3 * L),
    "L": (3 * L, 2 * L)
}

#############################################################
# ELEMENTS
#############################################################

DEFAULT_PROPS = {'E': 1.0, 'A': 1.0}

connectivity = [
    ("AC", "A", "C"),
    ('AF', "A", "F"),
    ('AI', "A", "I"),
    ('BE', "B", "E"),
    ('BH', "B", "H"),
    ('BL', "B", "L"),
    ('CD', "C", "D"),
    ('CF', "C", "F"),
    ('CG', "C", "G"),
    ('DG', "D", "G"),
    ('DE', "D", "E"),
    ('EG', "E", "G"),
    ('EH', "E", "H"),
    ('FG', "F", "G"),
    ('FI', "F", "I"),
    ('GH', "G", "H"),
    ('HL', "H", "L"),
    ('IK', "I", "K"),
    ('KL', "K", "L"),
]

ELEMENTS = [{'id': eid, 'nodes': (n1, n2), **DEFAULT_PROPS} for eid, n1, n2 in connectivity]

#############################################################
# LOADS
#############################################################

LOADS = {
    "F": (0, -P),
    "G": (0, -P),
    "H": (0, -P)
}


#############################################################
# SUPPORTS
#############################################################

SUPPORTS = {
    "A": ['x', 'y'],
    "B": ['y']
}

#############################################################
# PROBLEM STATEMENT
#############################################################


PROBLEM_SCHEMA = {
    "template": (
        "A statically determinate truss is supported by a pin at A and a roller at B. "
        "Downward loads of {P} {force_unit} are applied at joints F, G, and H. "
        "Determine the internal force in each member of the truss, specifying its magnitude "
        "and whether it is in tension or compression. Also report the global support reactions "
        "at A and B."
    ),
    "units": {
        "force_unit": "kN",
        "length_unit": "m"
    },
    "variables": {
        "P": {"type": "float", "default": 500.0, "min": 100.0, "max": 5000.0, "step": 100.0, "format": ".1f"},
        "L": {"type": "symbolic"}
    },

    # 3. Construct the explicit LLM schema dynamically
    "expected_llm_output": {
        "type": "object",
        "properties": {
            "global_reactions": {
                "type": "object",
                "description": "Global support reactions.",
                "properties": {
                    "R_A_x": {"type": "number"},
                    "R_A_y": {"type": "number"},
                    "R_B_y": {"type": "number"}
                },
                "required": ["R_A_x", "R_A_y", "R_B_y"],
                "additionalProperties": False
            },
            "member_forces": {
                "type": "object",
                "description": "Exact forces for every member in the truss.",
                "properties": {
                    member: FORCE_PROPERTY_SCHEMA for member in [eid for eid, n1, n2 in connectivity]
                },
                "required": [eid for eid, n1, n2 in connectivity],
                "additionalProperties": False
            },
            "units": {
                "type": "string",
                "description": "The units of the forces."
            }
        },
        "required": ["global_reactions", "member_forces", "units"],
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
        eval_fx = sp.sympify(fx).subs({P_sym: P_val})
        eval_fy = sp.sympify(fy).subs({P_sym: P_val})
        evaluated_loads[nid] = (eval_fx, eval_fy)

    state = TrussState(evaluated_nodes, elements_list, supports, evaluated_loads)
    return state


def extract_solution(state, problem_schema):
    """
    BESPOKE EXTRACTION: Target exactly what the problem schema requests.
    Builds a dictionary of both global reactions and member forces.
    """
    member_forces = {}

    for eid, expr in state.known_forces.items():
        val = float(expr.simplify().evalf()) if hasattr(expr, 'simplify') else float(expr)

        # Strip floating point noise
        if abs(val) < 1e-9:
            magnitude = 0.0
            status = "zero"
        elif val > 0:
            magnitude = val
            status = "tension"
        else:
            magnitude = abs(val)
            status = "compression"

        member_forces[eid] = {
            "magnitude": magnitude,
            "state": status
        }

    reactions = {
        "R_A_x": float(state.global_reactions.get("R_A_x", 0.0)),
        "R_A_y": float(state.global_reactions.get("R_A_y", 0.0)),
        "R_B_y": float(state.global_reactions.get("R_B_y", 0.0)),
    }

    return {
        "global_reactions": reactions,
        "member_forces": member_forces,
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
DRAWING_VARIABLES = {
    "P": PROBLEM_SCHEMA["variables"]["P"].get("default"),
    "L": 10.0
}

#############################################################
# DEVELOPMENT EXECUTION BLOCK
#############################################################

if __name__ == "__main__":
    # 1. Extract values dynamically for the solver
    solver_vars = {
        "P": PROBLEM_SCHEMA["variables"]["P"].get("default")
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
