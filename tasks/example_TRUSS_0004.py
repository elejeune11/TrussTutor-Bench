# Standard library imports
import numpy as np
import os
import sympy as sp

# trusstutor_bench tools
from trusstutor_bench.truss_state import TrussState
from trusstutor_bench.statics_tools import (
    identify_zfms_by_inspection_tool,
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
    "benchmark_id": "TRUSS_0004",
    "topic": ["identify_zfm"],
    "DEFAULT_RANDOM_SEED": 11,
    "task_creator": "elejeune11",
    "source_type": "original",
    "source_name": "github test example",
    "source_problem_pointer": [""],
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

# Base intervals abstracted to L
NODES = {
    "A": (0, 0),
    "B": (L, 0),
    "C": (2 * L, 0),
    "D": (3 * L, 0),
    "E": (4 * L, 0),
    "F": (3 * L, L),
    "G": (2 * L, 2 * L),
    "H": (L, L),
}

#############################################################
# ELEMENTS
#############################################################

DEFAULT_PROPS = {'E': 1.0, 'A': 1.0}

connectivity = [
    ('AB', 'A', 'B'),
    ('BC', 'B', 'C'),
    ('CD', 'C', 'D'),
    ('DE', 'D', 'E'),
    ('EF', 'E', 'F'),
    ('FG', 'F', 'G'),
    ('GH', 'G', 'H'),
    ('HA', 'H', 'A'),
    ('BH', 'B', 'H'),
    ('CH', 'C', 'H'),
    ('CG', 'C', 'G'),
    ('CF', 'C', 'F'),
    ('DF', 'D', 'F')
]

ELEMENTS = [{'id': eid, 'nodes': (n1, n2), **DEFAULT_PROPS} for eid, n1, n2 in connectivity]

#############################################################
# LOADS
#############################################################

LOADS = {
    "A": (0.0, -P1),
    "H": (0.0, -P2),
    "E": (0.0, -P1)
}

#############################################################
# SUPPORTS
#############################################################

SUPPORTS = {
    "A": ["y"],
    "E": ["x", "y"]
}

#############################################################
# PROBLEM STATEMENT
#############################################################

# Extract specific target members dynamically
expected_members = [eid for eid, n1, n2 in connectivity]

PROBLEM_SCHEMA = {
    "template": (
        "A statically determinate truss is supported by a roller at A and a pin at E. "
        "Vertical downward loads of {P1} {force_unit} are applied at joints A and E, "
        "and a load of {P2} {force_unit} is applied at joint H. "
        "Identify all Zero Force Members (ZFMs) in the truss by inspection."
    ),
    "units": {
        "force_unit": "kN",
        "length_unit": "m"
    },
    "variables": {
        "P1": {"type": "float", "default": 300.0, "min": 100.0, "max": 500.0, "step": 50.0, "format": ".1f"},
        "P2": {"type": "float", "default": 600.0, "min": 200.0, "max": 1000.0, "step": 100.0, "format": ".1f"},
        "L_val": {"type": "symbolic"}
    },
    "expected_llm_output": {
        "type": "object",
        "properties": {
            "zero_force_members": {
                "type": "array",
                "description": "A list of member IDs that are identified as zero force members. Return an empty list if there are none.",
                "items": {
                    "type": "string",
                    "enum": expected_members
                },
                "uniqueItems": True
            }
        },
        "required": ["zero_force_members"],
        "additionalProperties": False
    }
}

#############################################################
# PROGRAMMATIC PROBLEM SOLUTION
#############################################################

def initialize_truss_state(sampled_vars, nodes, elements_list, loads, supports):
    """
    BESPOKE SETUP: Uses string substitution dictionary to prevent SymPy assumption clashes.
    """
    subs_dict = {
        'P1': sampled_vars['P1'],
        'P2': sampled_vars['P2']
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


def extract_solution(state, problem_schema):
    """
    BESPOKE EXTRACTION: Target exactly what the problem schema requests.
    Safely handles both native Python floats (from ZFM tool) and SymPy expressions.
    """
    zfms = []
    
    for eid, force in state.known_forces.items():
        # Safely extract the float value regardless of the object type
        if hasattr(force, 'simplify'):
            val = float(force.simplify().evalf())
        else:
            val = float(force)
            
        # Any resolved force with a magnitude near zero is a ZFM
        if abs(val) < 1e-9:
            zfms.append(eid)
            
    return {
        "zero_force_members": sorted(zfms)
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
    # Note: Global reactions intentionally skipped; ZFM rules run pure topology inspection
    2: {
        "function": identify_zfms_by_inspection_tool,
        "inputs": {}
    },
    3: {
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
    "L": 2.5
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
