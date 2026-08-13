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
    """
    Verifies that the corrupted global reactions tool correctly injects a 
    Right-Hand Rule sign error into the moment equation while maintaining 
    the appearance of a flawless solve.
    """
    state = MockTrussState()
    
    # Execute the corrupted solver
    corrupted_solve_global_reactions_tool(state)
    
    # 1. Verify the math was corrupted.
    # The correct physical reaction at B should be +50.0 to counteract the clockwise moment.
    # Because of the corrupted cross product, it will calculate as -50.0.
    assert "R_B_y" in state.global_reactions
    assert float(state.global_reactions["R_B_y"]) == -50.0
    
    # 2. Verify the trace still looks completely normal to fool the LLM.
    assert len(state.trace) == 1
    assert state.trace[0]["step_type"] == "Global Statics"
    assert "Entire Truss" in state.trace[0]["target"]
    assert "Solved rigid-body equilibrium" in state.trace[0]["explanation"]


@patch("trusstutor_bench.corrupted_statics_tools.corrupted_solve_global_reactions_tool")
@patch("trusstutor_bench.corrupted_statics_tools.identify_zfms_by_inspection_tool")
def test_corrupted_autonomous_moj_macro(mock_zfm, mock_corrupted_reactions):
    """
    Verifies that the corrupted macro specifically routes the execution path 
    through the faulty global reactions tool.
    """
    state = MockTrussState()
    
    # Clear active elements to prevent the un-mocked joint solvers from looping
    state.active_element_ids = []
    
    corrupted_autonomous_moj_macro(state)
    
    # Assert that the corrupted global reactions tool was invoked by the macro
    mock_corrupted_reactions.assert_called_once_with(state)


def load_real_task(benchmark_id: str = "example_TRUSS_0001"):
    """Helper to dynamically load a real benchmark task from the examples directory."""
    current_dir = Path(__file__).parent
    
    # Try to find the examples directory (handles different possible pytest execution roots)
    examples_dir = current_dir / "examples"
    if not examples_dir.exists():
        examples_dir = current_dir.parent / "examples"
        
    example_path = examples_dir / f"{benchmark_id}.py"
    
    if not example_path.exists():
        pytest.skip(f"Could not find {example_path} for real example testing.")
        
    spec = importlib.util.spec_from_file_location(benchmark_id, example_path)
    truss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(truss_module)
    
    return BenchmarkTask.from_module(truss_module)


def test_corrupted_global_reactions_real_example():
    """
    Loads TRUSS_0001, applies both the clean and corrupted global reactions 
    tools to identical states, and verifies that the calculated physics diverge.
    """
    task = load_real_task("example_TRUSS_0001")
    
    # 1. Initialize the raw TrussState using the template's creator step
    sampled_vars = sample_variables(task.problem_schema["variables"])
    init_step = task.solution_recipe[1]
    base_state = init_step["function"](sampled_vars=sampled_vars, **init_step["inputs"])
    
    # 2. Create two identical, independent copies of the state
    clean_state = copy.deepcopy(base_state)
    corrupted_state = copy.deepcopy(base_state)
    
    # 3. Execute the respective tools
    solve_global_reactions_tool(clean_state)
    corrupted_solve_global_reactions_tool(corrupted_state)
    
    # 4. Assert that the mathematical outputs are fundamentally different 
    # (proving the Right-Hand Rule error successfully altered the physics)
    assert clean_state.global_reactions != corrupted_state.global_reactions
    
    # 5. Assert that the corrupted trace still appended the "Global Statics" log step
    # so the LLM has something to read.
    assert len(corrupted_state.trace) > 0
    assert corrupted_state.trace[-1]["step_type"] == "Global Statics"


def test_corrupted_autonomous_moj_macro_real_example():
    """
    Loads TRUSS_0001 and verifies that running the corrupted MOJ macro 
    successfully completes the internal force evaluations while carrying 
    forward the flawed global reactions.
    """
    task = load_real_task("example_TRUSS_0001")
    
    sampled_vars = sample_variables(task.problem_schema["variables"])
    init_step = task.solution_recipe[1]
    base_state = init_step["function"](sampled_vars=sampled_vars, **init_step["inputs"])
    
    clean_state = copy.deepcopy(base_state)
    corrupted_state = copy.deepcopy(base_state)
    
    # 1. Execute the full macros
    autonomous_moj_macro(clean_state)
    corrupted_autonomous_moj_macro(corrupted_state)
    
    # 2. Verify that both macros successfully solved all active elements
    assert len(clean_state.active_element_ids) == 0
    assert len(corrupted_state.active_element_ids) == 0
    
    # 3. Verify that the internal forces diverged because the corrupted macro 
    # fed flawed support reactions into the subsequent joint evaluations.
    assert clean_state.known_forces != corrupted_state.known_forces


class OverdeterminedMockState:
    """
    A lightweight mock state specifically designed with a physical contradiction 
    to test the overdetermined safety check bypass.
    """
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
    """
    Verifies that the corrupted joint tool bypasses the mathematical safety 
    check for overdetermined systems and forces a solution using the first 
    available equation, simulating a student ignoring a math contradiction.
    """
    base_state = OverdeterminedMockState()
    
    clean_state = copy.deepcopy(base_state)
    corrupted_state = copy.deepcopy(base_state)
    
    # 1. Execute the Clean Tool
    # The standard solver should detect that the system has 1 unknown (AB) 
    # but 2 active equations that contradict each other. It should safely 
    # abort and return False.
    clean_result = solve_single_joint_tool(clean_state, "A")
    
    assert clean_result is False
    assert "AB" in clean_state.active_element_ids
    assert "AB" not in clean_state.known_forces
    
    # 2. Execute the Corrupted Tool
    # The corrupted solver should ignore the contradiction in the Y-direction, 
    # force a solution using the X-direction equation (10 + F_AB = 0), and 
    # confidently return True.
    corrupted_result = corrupted_solve_single_joint_tool(corrupted_state, "A")
    
    assert corrupted_result is True
    assert "AB" not in corrupted_state.active_element_ids
    assert corrupted_state.known_forces["AB"] == -10
    
    # Verify the trace was logged to fool the LLM
    assert len(corrupted_state.trace) == 1
    assert corrupted_state.trace[0]["step_type"] == "Method of Joints"
    assert corrupted_state.trace[0]["target"] == "A"


def test_corrupted_single_joint_tool_real_example():
    """
    Loads TRUSS_0001, injects flawed support reactions, and verifies that the 
    corrupted single joint tool forces a solution on the final overdetermined 
    'check' joint where the clean tool correctly identifies a contradiction and halts.
    """
    task = load_real_task("example_TRUSS_0001")
    
    # 1. Initialize the raw TrussState
    sampled_vars = sample_variables(task.problem_schema["variables"])
    init_step = task.solution_recipe[1]
    base_state = init_step["function"](sampled_vars=sampled_vars, **init_step["inputs"])
    
    # 2. Inject flawed global reactions
    # This guarantees that the final joint equations will contradict each other.
    corrupted_solve_global_reactions_tool(base_state)
    
    # 3. Run the standard (clean) joint solver loop until it stalls out
    made_progress = True
    while made_progress and base_state.active_element_ids:
        made_progress = False
        for node_id in sorted(base_state.nodes.keys(), key=str):
            if solve_single_joint_tool(base_state, node_id):
                made_progress = True
                break
                
    # Verify the clean solver actually got stuck due to the contradiction
    assert len(base_state.active_element_ids) > 0
    
    # 4. Find the node where the solver stalled (the "check" joint)
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
    
    # 5. Create two identical copies of this stalled state
    clean_state = copy.deepcopy(base_state)
    corrupted_state = copy.deepcopy(base_state)
    
    # 6. Prove the clean tool STILL refuses to solve it
    assert solve_single_joint_tool(clean_state, stuck_node) is False
    assert len(clean_state.active_element_ids) == len(base_state.active_element_ids)
    
    # 7. Prove the corrupted tool successfully forces a solution
    assert corrupted_solve_single_joint_tool(corrupted_state, stuck_node) is True
    assert len(corrupted_state.active_element_ids) < len(clean_state.active_element_ids)


def create_mock_truss_state():
    """
    Helper function to generate a fresh, simple 2-panel truss state 
    with predefined global reactions so Method of Sections can immediately run.
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
    """
    Verifies that the corrupted Method of Sections tool perfectly executes 
    the matrix math but flips the sign of the final internal forces to 
    simulate a student misinterpreting Tension/Compression.
    """
    # A standard cut through the middle of the 2-panel truss
    cut_elements = ['top', 'diag3', 'bot2']
    
    # 1. Run the Authentic Tool (Ground Truth)
    clean_state = create_mock_truss_state()
    success_clean = solve_method_of_sections_tool(clean_state, cut_elements)
    
    assert success_clean is True, "Authentic tool failed to solve the cut."
    
    # 2. Run the Corrupted Tool
    corrupt_state = create_mock_truss_state()
    success_corrupt = corrupted_solve_method_of_sections_tool(corrupt_state, cut_elements)
    
    assert success_corrupt is True, "Corrupted tool failed to solve the cut."
    
    # 3. Assert the Error Injection (The Tension/Compression Flip)
    for eid in cut_elements:
        clean_force = clean_state.known_forces[eid]
        corrupt_force = corrupt_state.known_forces[eid]
        
        # Ensure the member isn't a zero-force member (which would trivially pass 0 == -0)
        assert clean_force != 0, f"Member {eid} had 0 force, cannot test sign flip."
        
        # The core check: The corrupted force must be the exact mathematical negative
        assert corrupt_force == -clean_force, (
            f"Failed on member {eid}: Expected { -clean_force }, got { corrupt_force }"
        )
        
    # 4. Verify the trace logged confidently (student thinks they did it right)
    assert len(corrupt_state.trace) == 1
    assert corrupt_state.trace[0]["step_type"] == "Method of Sections"


def test_corrupted_mos_with_truss_0003():
    """
    Validates the Tension/Compression flip explicitly using the real TRUSS_0003 template.
    """
    # 1. Dynamically load the TRUSS_0003 template
    current_dir = Path(__file__).parent
    examples_dir = current_dir / "examples"
    example_path = examples_dir / "example_TRUSS_0003.py"

    if not example_path.exists():
        pytest.skip("example_TRUSS_0003.py not found in examples directory.")

    spec = importlib.util.spec_from_file_location("example_TRUSS_0003", example_path)
    truss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(truss_module)

    # 2. Extract the recipe and default variables
    recipe = truss_module.SOLUTION_RECIPE
    schema = truss_module.PROBLEM_SCHEMA
    
    # Grab the default numeric values (P1=60.0, P2=80.0, theta=20.0)
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

    # 3. Create two identical, prepared states
    clean_state = get_prepared_state()
    corrupt_state = get_prepared_state()

    # The target cut elements for TRUSS_0003: ["D", "E", "F"]
    cut_elements = recipe[3]["inputs"]["cut_element_ids"]

    # 4. Run Authentic Tool
    success_clean = solve_method_of_sections_tool(clean_state, cut_elements)
    assert success_clean is True, "Authentic TRUSS_0003 MoS tool failed."

    # 5. Run Corrupted Tool
    success_corrupt = corrupted_solve_method_of_sections_tool(corrupt_state, cut_elements)
    assert success_corrupt is True, "Corrupted TRUSS_0003 MoS tool failed."

    # 6. Assert the Error Injection (The Tension/Compression Flip)
    for eid in cut_elements:
        clean_force = clean_state.known_forces[eid]
        corrupt_force = corrupt_state.known_forces[eid]
        
        assert clean_force != 0, f"Member {eid} had 0 force, cannot test sign flip."
        
        # Verify the corrupted force is the exact mathematical negative
        assert corrupt_force == -clean_force, (
            f"Failed on member {eid}: Expected { -clean_force }, got { corrupt_force }"
        )

    # 7. Verify the trace logged confidently at the last step
    assert corrupt_state.trace[-1]["step_type"] == "Method of Sections"