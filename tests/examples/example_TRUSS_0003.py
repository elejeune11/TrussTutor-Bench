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
    "benchmark_id": "TRUSS_0003",
    "topic": ["method_of_sections"],
    "DEFAULT_RANDOM_SEED": 11,
    "task_creator": "elejeune11",
    "source_type": "textbook",
    "source_name": "Engineering Mechanics: Statics, Osgood et al.",
    "source_problem_pointer": ["https://pressbooks.library.upei.ca/statics/chapter/method-of-sections/"],
}

#############################################################
# SYMBOLIC DEFINITIONS
#############################################################

L = sp.Symbol('L', positive=True)
theta = sp.Symbol('theta', real=True)
P1 = sp.Symbol('P1')
P2 = sp.Symbol('P2')

SYMBOLIC_DEFINITIONS = [L, theta, P1, P2]

#############################################################
# NODES
#############################################################

NODES = {
    "0": (0, 0),
    "1": (L, 0),
    "2": (2 * L, 0),
    "3": (3 * L, 0),
    # Use float value here so the coordinates are resolved correctly
    "4": (2 * L, -L * np.tan(20 * np.pi / 180)),
    "5": (L, -L * np.tan(20 * np.pi / 180))
}

#############################################################
# ELEMENTS
#############################################################

DEFAULT_PROPS = {'E': 1.0, 'A': 1.0}

connectivity = [
    ('A', '0', '1'),
    ('B', '0', '5'),
    ('C', '1', '5'),
    ('D', '1', '2'),
    ('E', '1', '4'),
    ('F', '4', '5'),
    ('G', '2', '4'),
    ('H', '2', '3'),
    ('I', '3', '4')
]

ELEMENTS = [{'id': eid, 'nodes': (n1, n2), **DEFAULT_PROPS} for eid, n1, n2 in connectivity]

#############################################################
# LOADS
#############################################################

LOADS = {
    "1": (0, -P1),
    "2": (0, -P2)
}

#############################################################
# SUPPORTS
#############################################################

SUPPORTS = {
    "0": ['x', 'y'],
    "3": ['y']
}

#############################################################
# PROBLEM STATEMENT
#############################################################

PROBLEM_SCHEMA = {
    "template": (
        "A statically determinate truss is illustrated in the attached schematic. "
        "Vertical downward loads of {P1} {force_unit} and {P2} {force_unit} are applied at joints. "
        "The top chord has bay lengths of L, and the bottom chord slopes downward "
        "at an angle of {theta} degrees relative to the horizontal. "
        "Determine the internal forces in members D, E, and F. "
        "Specify the magnitude and whether each member is in tension or compression."
    ),
    "units": {
        "force_unit": "kN",
        "length_unit": "m"
    },
    "variables": {
        "P1": {"type": "float", "default": 60.0, "min": 20.0, "max": 100.0, "step": 10.0, "format": ".1f"},
        "P2": {"type": "float", "default": 80.0, "min": 20.0, "max": 100.0, "step": 10.0, "format": ".1f"},
        "theta": {"type": "float", "default": 20.0, "min": 10.0, "max": 45.0, "step": 5.0, "format": ".1f"},
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
                    member: FORCE_PROPERTY_SCHEMA for member in ["D", "E", "F"]
                },
                "required": ["D", "E", "F"],
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
    Geometry is already resolved using THETA_VAL.
    """
    subs_dict = {
        'P1': sampled_vars['P1'],
        'P2': sampled_vars['P2']
    }

    evaluated_nodes = {}
    for nid, (x, y) in nodes.items():
        # Nodes are already floats from THETA_VAL, sympify handles them safely
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
            "cut_element_ids": ["D", "E", "F"]
        }
    },
    4: {
        "function": extract_solution,
        "inputs": {
            "problem_schema": PROBLEM_SCHEMA,
            "target_elements": ["D", "E", "F"]
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
    "theta": PROBLEM_SCHEMA["variables"]["theta"].get("default") * np.pi / 180,
    "L": 10.0
}

#############################################################
# DEVELOPMENT EXECUTION BLOCK
#############################################################

if __name__ == "__main__":
    # 1. Extract values dynamically for the solver
    solver_vars = {
        "P1": PROBLEM_SCHEMA["variables"]["P1"].get("default"),
        "P2": PROBLEM_SCHEMA["variables"]["P2"].get("default"),
        "theta": PROBLEM_SCHEMA["variables"]["theta"].get("default")
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
