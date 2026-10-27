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
    "benchmark_id": "TRUSS_0002",
    "topic": ["method_of_sections"],
    "DEFAULT_RANDOM_SEED": 11,
    "task_creator": "elejeune11",
    "source_type": "original",
    "source_name": "elejeune11 handwritten",
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
    "A": (0, 9 * L / 2),
    "B": (3 * L / 2, 9 * L / 2),
    "C": (0, 7 * L / 2),
    "D": (3 * L / 2, 7 * L / 2),
    "E": (0, 5 * L / 2),
    "F": (3 * L / 2, 5 * L / 2),
    "G": (0, 3 * L / 2),
    "H": (3 * L / 2, 3 * L / 2),
    "I": (0, 0),
    "J": (3 * L / 2, 0),
}

#############################################################
# ELEMENTS
#############################################################

DEFAULT_PROPS = {'E': 1.0, 'A': 1.0}

connectivity = [
    ('AB', 'A', 'B'),
    ('AC', 'A', 'C'),
    ('BC', 'B', 'C'),
    ('BD', 'B', 'D'),
    ('CD', 'C', 'D'),
    ('CE', 'C', 'E'),
    ('CF', 'C', 'F'),
    ('DF', 'D', 'F'),
    ('EF', 'E', 'F'),
    ('EG', 'E', 'G'),
    ('FG', 'F', 'G'),
    ('FH', 'F', 'H'),
    ('GH', 'G', 'H'),
    ('GI', 'G', 'I'),
    ('GJ', 'G', 'J'),
    ('HJ', 'H', 'J'),
    ('IJ', 'I', 'J')
]

ELEMENTS = [{'id': eid, 'nodes': (n1, n2), **DEFAULT_PROPS} for eid, n1, n2 in connectivity]

#############################################################
# LOADS
#############################################################

LOADS = {
    "B": (-P, 0)
}

#############################################################
# SUPPORTS
#############################################################

SUPPORTS = {
    "I": ['x', 'y'],
    "J": ['y']
}

#############################################################
# PROBLEM STATEMENT
#############################################################

PROBLEM_SCHEMA = {
    "template": (
        "If P = {P} {force_unit}, what is the force in members CE, CF, and DF?"
    ),
    "units": {
        "force_unit": "kN",
        "length_unit": "m"
    },
    "variables": {
        "P": {"type": "float", "default": 50.0, "min": 20.0, "max": 100.0, "step": 10.0, "format": ".1f"},
        "L_val": {"type": "symbolic"}
    },
    
    # 3. Construct the explicit LLM schema dynamically
    "expected_llm_output": {
        "type": "object",
        "properties": {
            "member_forces": {
                "type": "object",
                "description": "Exact forces for the requested members.",
                "properties": {
                    member: FORCE_PROPERTY_SCHEMA for member in ["CE", "CF", "DF"]
                },
                "required": ["CE", "CF", "DF"],
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
    BESPOKE SETUP: Inject dynamic numerical values into symbolic loads.
    """
    subs_dict = {
        'P': sampled_vars['P'],
    }

    evaluated_nodes = {}
    for nid, (x, y) in nodes.items():
        evaluated_nodes[nid] = (sp.sympify(x), sp.sympify(y))

    evaluated_loads = {}
    for nid, (fx, fy) in loads.items():
        evaluated_loads[nid] = (
            sp.sympify(fx).subs(subs_dict),
            sp.sympify(fy).subs(subs_dict)
        )

    state = TrussState(evaluated_nodes, elements_list, supports, evaluated_loads)
    return state


def extract_solution(state, problem_schema, target_elements):
    """
    BESPOKE EXTRACTION: Target exactly what the problem schema requests.
    Only pulls data for the requested Method of Sections target elements.
    """
    member_forces = {}
    
    for eid in target_elements:
        if eid not in state.known_forces:
            raise ValueError(f"Force for member '{eid}' was not resolved by the solver.")
            
        val = float(state.known_forces[eid].simplify().evalf())
        
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
        "function": solve_method_of_sections_tool,
        "inputs": {
            "cut_element_ids": ["CE", "CF", "DF"]
        }
    },
    4: {
        "function": extract_solution,
        "inputs": {
            "problem_schema": PROBLEM_SCHEMA,
            "target_elements": ["CE", "CF", "DF"]
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
        "P": PROBLEM_SCHEMA["variables"]["P"].get("default"),
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
