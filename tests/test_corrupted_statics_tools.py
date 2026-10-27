import sympy as sp
import pytest
from unittest.mock import patch

from trusstutor_bench.corrupted_statics_tools import (
    corrupted_solve_global_reactions_tool,
    corrupted_autonomous_moj_macro
)

import importlib.util
import copy
from pathlib import Path

from trusstutor_bench.task_base import BenchmarkTask
from trusstutor_bench.task_to_prompt import sample_variables
from trusstutor_bench.statics_tools import solve_global_reactions_tool, autonomous_moj_macro

from trusstutor_bench.statics_tools import solve_single_joint_tool
from trusstutor_bench.corrupted_statics_tools import corrupted_solve_single_joint_tool
from trusstutor_bench.statics_tools import solve_method_of_sections_tool
from trusstutor_bench.corrupted_statics_tools import corrupted_solve_method_of_sections_tool


from trusstutor_bench.truss_state import TrussState


class MockTrussState:
    """A lightweight mock state designed to expose the moment sign error."""
    def __init__(self):
        # A simple right triangle truss
        self.nodes = {
            "A": (0, 0),    # Moment center
            "B": (10, 0),   # Roller
            "C": (5, 5)     # Point of load application
        }
        self.supports = {
            "A": ["x", "y"], 
            "B": ["y"]
        }
        # A horizontal load applied at C.
        # Correct Moment about A: -(5 * 100) = -500
        # Corrupted Moment about A: +(5 * 100) = +500
        self.loads = {
            "C": (100, 0)  
        }
        self.global_reactions = {}
        self.elements = [
            {"id": "AB", "nodes": ("A", "B")},
            {"id": "BC", "nodes": ("B", "C")},
            {"id": "CA", "nodes": ("C", "A")}
        ]
        self.active_element_ids = ["AB", "BC", "CA"]
        self.known_forces = {}
        self.trace = []

    def log_step(self, step_type, target, results, explanation):
        self.trace.append({
            "step_type": step_type,
            "target": target,
            "results": results,
            "explanation": explanation
        })


def test_corrupted_solve_global_reactions_tool_sign_error():
    """The corrupted reactions tool uses the wrong moment sign but still looks like a flawless solve."""
    state = MockTrussState()
    
    # Execute the corrupted solver
    corrupted_solve_global_reactions_tool(state)
    
    assert "R_B_y" in state.global_reactions
    assert float(state.global_reactions["R_B_y"]) == -50.0
    
    assert len(state.trace) == 1
    assert state.trace[0]["step_type"] == "Global Statics"
    assert "Entire Truss" in state.trace[0]["target"]
    assert "Solved rigid-body equilibrium" in state.trace[0]["explanation"]


@patch("trusstutor_bench.corrupted_statics_tools.corrupted_solve_global_reactions_tool")
@patch("trusstutor_bench.corrupted_statics_tools.identify_zfms_by_inspection_tool")
def test_corrupted_autonomous_moj_macro(mock_zfm, mock_corrupted_reactions):
    """The corrupted macro runs through the corrupted global reactions tool."""
    state = MockTrussState()
    
    # Clear active elements to prevent the un-mocked joint solvers from looping
    state.active_element_ids = []
    
    corrupted_autonomous_moj_macro(state)
    
    # Assert that the corrupted global reactions tool was invoked by the macro
    mock_corrupted_reactions.assert_called_once_with(state)


def load_real_task(benchmark_id: str = "example_TRUSS_0001"):
    """
    Loads a task from the tasks folder, or skips the test if the file is missing.

    Args:
        benchmark_id (str): Task file name without .py.

    Returns:
        BenchmarkTask: The loaded task.
    """
    current_dir = Path(__file__).parent
    
    examples_dir = current_dir.parent / "tasks"
        
    example_path = examples_dir / f"{benchmark_id}.py"
    
    if not example_path.exists():
        pytest.skip(f"Could not find {example_path} for real example testing.")
        
    spec = importlib.util.spec_from_file_location(benchmark_id, example_path)
    truss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(truss_module)
    
    return BenchmarkTask.from_module(truss_module)


def test_corrupted_global_reactions_real_example():
    """On TRUSS_0001, the clean and corrupted reactions tools give different results."""
    task = load_real_task("example_TRUSS_0001")
    
    sampled_vars = sample_variables(task.problem_schema["variables"])
    init_step = task.solution_recipe[1]
    base_state = init_step["function"](sampled_vars=sampled_vars, **init_step["inputs"])
    
    clean_state = copy.deepcopy(base_state)
    corrupted_state = copy.deepcopy(base_state)
    
    solve_global_reactions_tool(clean_state)
    corrupted_solve_global_reactions_tool(corrupted_state)
    
    assert clean_state.global_reactions != corrupted_state.global_reactions
    
    assert len(corrupted_state.trace) > 0
    assert corrupted_state.trace[-1]["step_type"] == "Global Statics"


def test_corrupted_autonomous_moj_macro_real_example():
    """On TRUSS_0001, the corrupted macro still solves every member, carrying the flawed reactions forward."""
    task = load_real_task("example_TRUSS_0001")
    
    sampled_vars = sample_variables(task.problem_schema["variables"])
    init_step = task.solution_recipe[1]
    base_state = init_step["function"](sampled_vars=sampled_vars, **init_step["inputs"])
    
    clean_state = copy.deepcopy(base_state)
    corrupted_state = copy.deepcopy(base_state)
    
    autonomous_moj_macro(clean_state)
    corrupted_autonomous_moj_macro(corrupted_state)
    
    assert len(clean_state.active_element_ids) == 0
    assert len(corrupted_state.active_element_ids) == 0
    
    assert clean_state.known_forces != corrupted_state.known_forces


class OverdeterminedMockState:
    """Mock state whose equations contradict each other, for the overdetermined joint check."""
    def __init__(self):
        self.nodes = {
            "A": (0, 0), 
            "B": (10, 0)
        }
        self.elements = [
            {"id": "AB", "nodes": ("A", "B")}
        ]
        self.active_element_ids = ["AB"]
        self.supports = {
            "A": ["x", "y"]
        }
        
        # INTENTIONAL CONTRADICTION: 
        # R_A_y = 5 is applied, but the only member connected (AB) is entirely horizontal.
        # This means Sum Fy = 5 != 0. The joint is physically impossible to balance.
        self.global_reactions = {
            "R_A_x": 10, 
            "R_A_y": 5
        }
        
        self.loads = {}
        self.known_forces = {}
        self.trace = []

    def log_step(self, step_type, target, results, explanation):
        self.trace.append({
            "step_type": step_type,
            "target": target,
            "results": results,
            "explanation": explanation
        })


def test_corrupted_solve_single_joint_tool_forces_solution():
    """The corrupted joint tool solves an overdetermined joint from one equation and ignores the contradiction."""
    base_state = OverdeterminedMockState()
    
    clean_state = copy.deepcopy(base_state)
    corrupted_state = copy.deepcopy(base_state)
    
    clean_result = solve_single_joint_tool(clean_state, "A")
    
    assert clean_result is False
    assert "AB" in clean_state.active_element_ids
    assert "AB" not in clean_state.known_forces
    
    corrupted_result = corrupted_solve_single_joint_tool(corrupted_state, "A")
    
    assert corrupted_result is True
    assert "AB" not in corrupted_state.active_element_ids
    assert corrupted_state.known_forces["AB"] == -10
    
    # Verify the trace was logged to fool the LLM
    assert len(corrupted_state.trace) == 1
    assert corrupted_state.trace[0]["step_type"] == "Method of Joints"
    assert corrupted_state.trace[0]["target"] == "A"


def test_corrupted_single_joint_tool_real_example():
    """On TRUSS_0001 with flawed reactions, the corrupted joint tool forces a solution at the final check joint, where the clean tool detects the contradiction."""
    task = load_real_task("example_TRUSS_0001")
    
    sampled_vars = sample_variables(task.problem_schema["variables"])
    init_step = task.solution_recipe[1]
    base_state = init_step["function"](sampled_vars=sampled_vars, **init_step["inputs"])
    
    corrupted_solve_global_reactions_tool(base_state)
    
    made_progress = True
    while made_progress and base_state.active_element_ids:
        made_progress = False
        for node_id in sorted(base_state.nodes.keys(), key=str):
            if solve_single_joint_tool(base_state, node_id):
                made_progress = True
                break
                
    # Verify the clean solver actually got stuck due to the contradiction
    assert len(base_state.active_element_ids) > 0
    
    stuck_node = None
    for node_id in base_state.nodes:
        unknowns = [
            el for el in base_state.elements 
            if node_id in el['nodes'] and el['id'] in base_state.active_element_ids
        ]
        if 0 < len(unknowns) <= 2:
            stuck_node = node_id
            break
            
    assert stuck_node is not None, "Could not find an overdetermined node to test."
    
    clean_state = copy.deepcopy(base_state)
    corrupted_state = copy.deepcopy(base_state)
    
    assert solve_single_joint_tool(clean_state, stuck_node) is False
    assert len(clean_state.active_element_ids) == len(base_state.active_element_ids)
    
    assert corrupted_solve_single_joint_tool(corrupted_state, stuck_node) is True
    assert len(corrupted_state.active_element_ids) < len(clean_state.active_element_ids)


def create_mock_truss_state():
    """
    Builds a two-panel truss state with known reactions, so a method-of-sections cut can run immediately.

    Returns:
        TrussState: The state.
    """
    nodes = {
        "0": (0, 0),
        "1": (1, 0),
        "2": (2, 0),
        "3": (0.5, 1),
        "4": (1.5, 1)
    }
    
    elements = [
        {'id': 'bot1', 'nodes': ('0', '1')},
        {'id': 'bot2', 'nodes': ('1', '2')},
        {'id': 'top', 'nodes': ('3', '4')},
        {'id': 'diag1', 'nodes': ('0', '3')},
        {'id': 'diag2', 'nodes': ('1', '3')},
        {'id': 'diag3', 'nodes': ('1', '4')},
        {'id': 'diag4', 'nodes': ('2', '4')}
    ]
    
    supports = {"0": ['x', 'y'], "2": ['y']}
    loads = {"1": (0, -10)}

    state = TrussState(
        nodes={k: (sp.sympify(v[0]), sp.sympify(v[1])) for k, v in nodes.items()},
        elements=elements,
        supports=supports,
        loads={k: (sp.sympify(v[0]), sp.sympify(v[1])) for k, v in loads.items()}
    )
    
    # Inject the known global reactions (Sum Fy = 0, Sum M = 0)
    state.global_reactions = {
        "R_0_x": sp.sympify(0),
        "R_0_y": sp.sympify(5),
        "R_2_y": sp.sympify(5)
    }
    
    return state


def test_corrupted_solve_method_of_sections_tc_flip():
    """The corrupted sections tool does the matrix math correctly but flips the sign of the member forces."""
    # A standard cut through the middle of the 2-panel truss
    cut_elements = ['top', 'diag3', 'bot2']
    
    clean_state = create_mock_truss_state()
    success_clean = solve_method_of_sections_tool(clean_state, cut_elements)
    
    assert success_clean is True, "Authentic tool failed to solve the cut."
    
    corrupt_state = create_mock_truss_state()
    success_corrupt = corrupted_solve_method_of_sections_tool(corrupt_state, cut_elements)
    
    assert success_corrupt is True, "Corrupted tool failed to solve the cut."
    
    for eid in cut_elements:
        clean_force = clean_state.known_forces[eid]
        corrupt_force = corrupt_state.known_forces[eid]
        
        # Ensure the member isn't a zero-force member (which would trivially pass 0 == -0)
        assert clean_force != 0, f"Member {eid} had 0 force, cannot test sign flip."
        
        # The core check: The corrupted force must be the exact mathematical negative
        assert corrupt_force == -clean_force, (
            f"Failed on member {eid}: Expected { -clean_force }, got { corrupt_force }"
        )
        
    assert len(corrupt_state.trace) == 1
    assert corrupt_state.trace[0]["step_type"] == "Method of Sections"


def test_corrupted_mos_with_truss_0002():
    """On TRUSS_0002, the corrupted sections tool returns exactly the negative of the clean forces."""
    current_dir = Path(__file__).parent
    examples_dir = current_dir.parent / "tasks"
    example_path = examples_dir / "example_TRUSS_0002.py"

    if not example_path.exists():
        pytest.skip("example_TRUSS_0002.py not found in examples directory.")

    spec = importlib.util.spec_from_file_location("example_TRUSS_0002", example_path)
    truss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(truss_module)

    recipe = truss_module.SOLUTION_RECIPE
    schema = truss_module.PROBLEM_SCHEMA
    
    # Grab the default numeric values (P=50.0)
    sampled_vars = {k: v.get("default") for k, v in schema["variables"].items() if "default" in v}

    # Helper function to generate a state prepared up to Step 2
    def get_prepared_state():
        # Step 1: Initialize TrussState
        init_func = recipe[1]["function"]
        state = init_func(sampled_vars=sampled_vars, **recipe[1]["inputs"])
        
        # Step 2: Global Reactions (Required before MoS can run)
        global_func = recipe[2]["function"]
        global_func(state, **recipe[2]["inputs"])
        
        return state

    clean_state = get_prepared_state()
    corrupt_state = get_prepared_state()

    # The target cut elements for TRUSS_0002: ["CE", "CF", "DF"]
    cut_elements = recipe[3]["inputs"]["cut_element_ids"]

    success_clean = solve_method_of_sections_tool(clean_state, cut_elements)
    assert success_clean is True, "Authentic TRUSS_0002 MoS tool failed."

    success_corrupt = corrupted_solve_method_of_sections_tool(corrupt_state, cut_elements)
    assert success_corrupt is True, "Corrupted TRUSS_0002 MoS tool failed."

    for eid in cut_elements:
        clean_force = clean_state.known_forces[eid]
        corrupt_force = corrupt_state.known_forces[eid]
        
        assert clean_force != 0, f"Member {eid} had 0 force, cannot test sign flip."
        
        # Verify the corrupted force is the exact mathematical negative
        assert corrupt_force == -clean_force, (
            f"Failed on member {eid}: Expected { -clean_force }, got { corrupt_force }"
        )

    assert corrupt_state.trace[-1]["step_type"] == "Method of Sections"
