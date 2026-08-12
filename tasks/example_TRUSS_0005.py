# Standard library imports
import numpy as np
import os
import sympy as sp

# trusstutor_support_code tools
from trusstutor_bench.truss_state import TrussState
from trusstutor_bench.schemas import FORCE_PROPERTY_SCHEMA
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
    "benchmark_id": "TRUSS_0005",
    "topic": ["find_unknown_load_or_dimension"],
    "DEFAULT_RANDOM_SEED": 11,
    "task_creator": "elejeune11",
    "source_type": "textbook",
    "source_name": "Engineering Statics Open and Interactive, Baker and Haynes",
    "source_problem_pointer": ["https://engineeringstatics.org/Chapter_06-10.html"],
}

#############################################################
# SYMBOLIC DEFINITIONS
#############################################################

m = sp.Symbol("m", positive=True)

SYMBOLIC_DEFINITIONS = [m]

#############################################################
# NODES
#############################################################

# Maintain pure SymPy exactness to prevent floating point drift
theta = sp.atan(sp.Rational(4, 10))
phi = sp.pi / 4 - theta
L_geom = 4 * sp.sqrt(2) * sp.cos(phi)

NODES = {
    "A": (0, 6),
    "B": (0, 0),
    "C": (10, 10),
    "D": (10 - sp.cos(theta) * L_geom, 10 - sp.sin(theta) * L_geom),
    "E": (3, 3),
    "F": (6, 6)
}

#############################################################
# ELEMENTS
#############################################################

DEFAULT_PROPS = {'E': 1.0, 'A': 1.0}

connectivity = [
    ('AB', 'A', 'B'),
    ('BE', 'B', 'E'),
    ('AE', 'A', 'E'),
    ('AF', 'A', 'F'),
    ('EF', 'E', 'F'),
    ('AD', 'A', 'D'),
    ('DF', 'D', 'F'),
    ('CF', 'C', 'F'),
    ('CD', 'C', 'D')
]

ELEMENTS = [{'id': eid, 'nodes': (n1, n2), **DEFAULT_PROPS} for eid, n1, n2 in connectivity]

#############################################################
# LOADS
#############################################################

LOADS = {
    "C": (-1 * m * 9.81, -1 * m * 9.81)
}

#############################################################
# SUPPORTS
#############################################################

SUPPORTS = {
    "A": ['x', 'y'],
    "B": ['x']  # Horizontally constrained roller
}

#############################################################
# PROBLEM STATEMENT
#############################################################

PROBLEM_SCHEMA = {
    "template": (
        "Determine the maximum mass m which may be applied to the truss without "
        "exceeding {max_tension} kN in tension or {max_compression} kN in compression in any member."
        "Identify the maximum allowable mass, the critical member that governs this limit, "
        "and whether that member is in tension or compression."
    ),
    "units": {
        "mass_unit": "kg"
    },
    "variables": {
        "max_tension": {"type": "float", "default": 30.0, "min": 20.0, "max": 60.0, "step": 5.0, "format": ".1f"},
        "max_compression": {"type": "float", "default": 50.0, "min": 30.0, "max": 90.0, "step": 5.0, "format": ".1f"}
    },
    "expected_llm_output": {
        "type": "object",
        "properties": {
            "max_allowable_mass": {
                "type": "number",
                "description": "The maximum mass in kg."
            },
            "controlling_member": {
                "type": "string",
                "description": "The ID of the member that reaches its limit first (e.g., 'AB')."
            },
            "controlling_state": {
                "type": "string",
                "enum": ["tension", "compression"],
                "description": "The failure mode of the controlling member."
            },
            "units": {
                "type": "string",
                "description": "Units of the mass."
            }
        },
        "required": ["max_allowable_mass", "controlling_member", "controlling_state", "units"],
        "additionalProperties": False
    }
}

#############################################################
# PROGRAMMATIC PROBLEM SOLUTION
#############################################################

def initialize_truss_state(sampled_vars, nodes, elements_list, loads, supports):
    """
    BESPOKE SETUP: Leaves 'm' as a pure symbol for the statics engine.
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
    Solves for the critical mass 'm' that causes the first member to fail.
    """
    # 1. Retrieve limits and convert from kN to N (since gravity is 9.81 m/s^2)
    limit_T_N = state.sampled_vars['max_tension'] * 1000
    limit_C_N = state.sampled_vars['max_compression'] * 1000

    m_sym = sp.Symbol('m', positive=True)

    # 2. Extract numerical coefficients for m
    coeffs = {}
    for eid, force_expr in state.known_forces.items():
        if hasattr(force_expr, 'simplify'):
            # Substitute m=1 to find the multiplier for the mass
            val = float(force_expr.subs(m_sym, 1).evalf())
        else:
            val = float(force_expr)
            
        if abs(val) > 1e-9:
            coeffs[eid] = val

    if not coeffs:
        raise ValueError("No internal forces found.")

    # 3. Locate the members carrying the most extreme forces
    t_eid = max(coeffs, key=coeffs.get)
    c_eid = min(coeffs, key=coeffs.get)

    max_t_coeff = coeffs[t_eid] if coeffs[t_eid] > 0 else 0.0
    max_c_coeff = abs(coeffs[c_eid]) if coeffs[c_eid] < 0 else 0.0

    # 4. Calculate allowable mass for both failure modes
    m_tension = (limit_T_N / max_t_coeff) if max_t_coeff > 0 else float('inf')
    m_comp = (limit_C_N / max_c_coeff) if max_c_coeff > 0 else float('inf')

    # 5. The lowest mass is the bottleneck (controlling limit)
    controlling_m = min(m_tension, m_comp)
    controlling_state = "tension" if m_tension < m_comp else "compression"
    controlling_member = t_eid if m_tension < m_comp else c_eid

    return {
        "max_allowable_mass": round(controlling_m, 2),
        "controlling_member": controlling_member,
        "controlling_state": controlling_state,
        "units": problem_schema["units"]["mass_unit"]
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

# We provide a dummy 'm' strictly for the drawing engine 
# so it can compute float coordinate lengths for the load vectors.
DRAWING_VARIABLES = {
    "m": 100.0 
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
