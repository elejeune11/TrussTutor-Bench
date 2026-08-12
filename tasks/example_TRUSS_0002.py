# Standard library imports
import numpy as np
import os
import sympy as sp

# trusstutor_support_code tools
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
    "benchmark_id": "TRUSS_0002",
    "topic": ["method_of_joints"],
    "DEFAULT_RANDOM_SEED": 11,
    "task_creator": "elejeune11",
    "source_type": "textbook",
    "source_name": "Engineering Mechanics: Statics, Osgood et al.",
    "source_problem_pointer": ["https://pressbooks.library.upei.ca/statics/chapter/method-of-joints/"],
}

#############################################################
# SYMBOLIC DEFINITIONS
#############################################################

L = sp.Symbol('L', positive=True)
P1 = sp.Symbol('P1')
P2 = sp.Symbol('P2')

SYMBOLIC_DEFINITIONS = [L, P1, P2]

#############################################################
# NODES
#############################################################

NODES = {
    "A": (0, 0),
    "B": (sp.Rational(1, 2) * L, sp.Rational(1, 2) * sp.sqrt(3) * L),
    "C": (sp.Rational(3, 2) * L, sp.Rational(1, 2) * sp.sqrt(3) * L),
    "D": (sp.Rational(5, 2) * L, sp.Rational(1, 2) * sp.sqrt(3) * L),
    "E": (3 * L, 0),
    "F": (2 * L, 0),
    "G": (L, 0),
}

#############################################################
# ELEMENTS
#############################################################

DEFAULT_PROPS = {'E': 1.0, 'A': 1.0}

connectivity = [
    ("AB", "A", "B"),
    ("BC", "B", "C"),
    ("CD", "C", "D"),
    ("DE", "D", "E"),
    ("EF", "E", "F"),
    ("FG", "F", "G"),
    ("AG", "A", "G"),
    ("BG", "B", "G"),
    ("CG", "C", "G"),
    ("CF", "C", "F"),
    ("DF", "D", "F")
]

ELEMENTS = [{'id': eid, 'nodes': (n1, n2), **DEFAULT_PROPS} for eid, n1, n2 in connectivity]

#############################################################
# LOADS
#############################################################

LOADS = {
    "G": (0, -P1),
    "F": (0, -P2)
}

#############################################################
# SUPPORTS
#############################################################

SUPPORTS = {
    "A": ['x', 'y'],
    "E": ['y']
}

#############################################################
# PROBLEM STATEMENT
#############################################################


PROBLEM_SCHEMA = {
    "template": (
        "A statically determinate truss is supported by a pin at a "
        "and a roller at e. Vertical downward loads of {P1} {force_unit} "
        "and {P2} {force_unit} are applied at joints g and f, respectively. "
        "If the bay length is L, determine the internal force in each member "
        "of the structure, specifying its magnitude and whether it is in tension or compression."
    ),
    "units": {
        "force_unit": "kN",
        "length_unit": "m"
    },
    "variables": {
        "P1": {"type": "float", "default": 100.0, "min": 50.0, "max": 500.0, "step": 10.0, "format": ".1f"},
        "P2": {"type": "float", "default": 50.0, "min": 25.0, "max": 250.0, "step": 5.0, "format": ".1f"},
        "L_val": {"type": "symbolic"}
    },
    
    # 3. Construct the explicit LLM schema dynamically
    "expected_llm_output": {
        "type": "object",
        "properties": {
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
        "required": ["member_forces", "units"],
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
    P1_val = sampled_vars['P1']
    P2_val = sampled_vars['P2']
    P1_sym = sp.Symbol('P1')
    P2_sym = sp.Symbol('P2')

    evaluated_nodes = {}
    for nid, (x, y) in nodes.items():
        evaluated_nodes[nid] = (sp.sympify(x), sp.sympify(y))

    evaluated_loads = {}
    for nid, (fx, fy) in loads.items():
        eval_fx = sp.sympify(fx).subs({P1_sym: P1_val, P2_sym: P2_val})
        eval_fy = sp.sympify(fy).subs({P1_sym: P1_val, P2_sym: P2_val})
        evaluated_loads[nid] = (eval_fx, eval_fy)

    state = TrussState(evaluated_nodes, elements_list, supports, evaluated_loads)
    return state


def extract_solution(state, problem_schema):
    """
    BESPOKE EXTRACTION: Target exactly what the problem schema requests.
    Builds a dictionary of member forces with explicit magnitude and state.
    """
    member_forces = {}
    
    for eid, expr in state.known_forces.items():
        val = float(expr.simplify().evalf())
        
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

    return {
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
    "P1": PROBLEM_SCHEMA["variables"]["P1"].get("default"),
    "P2": PROBLEM_SCHEMA["variables"]["P2"].get("default"),
    "L": 10.0
}

#############################################################
# DEVELOPMENT EXECUTION BLOCK
#############################################################

if __name__ == "__main__":
    # 1. Extract values dynamically for the solver
    solver_vars = {
        "P1": PROBLEM_SCHEMA["variables"]["P1"].get("default"),
        "P2": PROBLEM_SCHEMA["variables"]["P2"].get("default")
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
