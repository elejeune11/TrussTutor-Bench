from trusstutor_bench.markdown_generator import parse_trace_value
from trusstutor_bench.markdown_generator import render_trace_step
from trusstutor_bench.markdown_generator import generate_markdown_workflow
from trusstutor_bench.statics_tools import programmatic_solution
import pytest
from pathlib import Path
import importlib.util


def test_parse_internal_force_tension():
    result = parse_trace_value("F_AB", 50.0)
    assert result == "$F_{AB}$ = 50.00 (Tension)"


def test_parse_internal_force_compression():
    # Ensures the negative sign is stripped when the "(Compression)" label is added
    result = parse_trace_value("F_BC", -28.867)
    assert result == "$F_{BC}$ = 28.87 (Compression)"


def test_parse_internal_force_zfm_with_noise():
    # Ensures floating point artifacts trigger the ZFM logic
    result = parse_trace_value("F_CD", 1e-12)
    assert result == "$F_{CD}$ = 0.00 (Zero-Force Member)"


def test_parse_support_reaction_positive():
    result = parse_trace_value("R_A_y", 25.0)
    assert result == "$R_{Ay}$ = 25.00"


def test_parse_support_reaction_negative():
    # Reactions should keep their negative signs instead of using tension/compression
    result = parse_trace_value("R_B_x", -10.5)
    assert result == "$R_{Bx}$ = -10.50"


def test_parse_support_reaction_zero_noise():
    # Ensures float noise on reactions snaps to a clean 0.00 without a negative sign
    result = parse_trace_value("R_A_x", -1e-15)
    assert result == "$R_{Ax}$ = 0.00"


def test_parse_with_units():
    result = parse_trace_value("F_EF", 100.0, unit="kN")
    assert result == "$F_{EF}$ = 100.00 kN (Tension)"
    
    result = parse_trace_value("R_A_y", 50.0, unit="N")
    assert result == "$R_{Ay}$ = 50.00 N"


def test_parse_symbolic_fallback():
    # If the user passes an unresolved SymPy string, it should safely fall back
    result = parse_trace_value("F_AG", "P/2", unit="kN")
    assert result == "$F_{AG}$ = P/2 kN (Algebraic)"


def test_parse_trace_value_with_real_truss_0001():
    """
    Tests the value parser using the actual mathematical output generated 
    by the programmatic solver for TRUSS_0001.
    """
    # 1. Resolve the path to the real example file
    current_dir = Path(__file__).parent
    example_path = current_dir / "examples" / "example_TRUSS_0001.py"
    
    if not example_path.exists():
        pytest.skip(f"Could not find {example_path}")

    # 2. Dynamically import the example module
    spec = importlib.util.spec_from_file_location("example_TRUSS_0001", example_path)
    truss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(truss_module)
    
    # 3. Extract the recipe and default variables
    recipe = truss_module.SOLUTION_RECIPE
    default_p = truss_module.PROBLEM_SCHEMA["variables"]["P"]["default"]
    sampled_vars = {"P": default_p}  # P = 50.0
    
    # 4. Execute the Creator (Step 1) to get the TrussState
    init_step = recipe[1]
    state = init_step["function"](sampled_vars=sampled_vars, **init_step["inputs"])
    
    # 5. Execute the Mutators (Step 2 and 3) to solve the physics
    reaction_step = recipe[2]
    reaction_step["function"](state, **reaction_step["inputs"])
    
    moj_step = recipe[3]
    moj_step["function"](state, **moj_step["inputs"])
    
    # 6. Test the Support Reaction Parser
    # For a symmetrical truss with a 50.0 kN downward load at the apex, R_A_y = 25.0
    r_a_y_val = state.global_reactions["R_A_y"]
    parsed_reaction = parse_trace_value("R_A_y", r_a_y_val, unit="kN")
    assert parsed_reaction == "$R_{Ay}$ = 25.00 kN"
    
    r_a_x_val = state.global_reactions["R_A_x"]
    parsed_x_reaction = parse_trace_value("R_A_x", r_a_x_val, unit="kN")
    assert parsed_x_reaction == "$R_{Ax}$ = 0.00 kN"
    
    # 7. Test the Internal Force Parser
    # Member BC will be under compression
    f_bc_val = state.known_forces["BC"]
    parsed_bc_force = parse_trace_value("F_BC", f_bc_val, unit="kN")
    assert parsed_bc_force == "$F_{BC}$ = 28.87 kN (Compression)"
    
    # Member AB will be under tension
    f_ab_val = state.known_forces["AB"]
    parsed_ab_force = parse_trace_value("F_AB", f_ab_val, unit="kN")
    assert parsed_ab_force == "$F_{AB}$ = 14.43 kN (Tension)"


def test_render_trace_step_mock_data():
    mock_trace = {
        "step_type": "Method of Joints",
        "target": "A",
        "results": {"AB": 50.0, "AC": -25.5},
        "explanation": "Evaluated joint A (Coupled System: 2 unknowns)."
    }
    
    result = render_trace_step(step_index=2, trace_entry=mock_trace, unit="kN")
    
    expected_lines = [
        "**Step 2: Method of Joints (A)**",
        "Evaluated joint A (Coupled System: 2 unknowns).",
        "*   $F_{AB}$ = 50.00 kN (Tension)",
        "*   $F_{AC}$ = 25.50 kN (Compression)"
    ]
    
    # Check that all expected lines are present in the exact joined output
    assert result == "\n".join(expected_lines)


def test_render_trace_step_with_real_truss_0001():
    # 1. Resolve the path to the real example file
    current_dir = Path(__file__).parent
    example_path = current_dir / "examples" / "example_TRUSS_0001.py"
    
    if not example_path.exists():
        import pytest
        pytest.skip(f"Could not find {example_path}")

    # 2. Dynamically import the example module
    spec = importlib.util.spec_from_file_location("example_TRUSS_0001", example_path)
    truss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(truss_module)
    
    # 3. Execute the solver up to the global reactions
    recipe = truss_module.SOLUTION_RECIPE
    sampled_vars = {"P": 50.0} 
    
    init_step = recipe[1]
    state = init_step["function"](sampled_vars=sampled_vars, **init_step["inputs"])
    
    reaction_step = recipe[2]
    reaction_step["function"](state, **reaction_step["inputs"])
    
    # 4. Extract the newly generated trace entry
    assert len(state.trace) == 1
    trace_entry = state.trace[0]
    
    # 5. Render the step
    rendered_markdown = render_trace_step(step_index=1, trace_entry=trace_entry, unit="kN")
    
    # 6. Verify the output format and math
    assert "**Step 1: Global Statics (Entire Truss)**" in rendered_markdown
    assert "Solved rigid-body equilibrium" in rendered_markdown
    assert "*   $R_{Ay}$ = 25.00 kN" in rendered_markdown
    assert "*   $R_{By}$ = 25.00 kN" in rendered_markdown


class MockTrussState:
    """A lightweight mock of TrussState for isolated testing."""
    def __init__(self, trace):
        self.trace = trace


def test_generate_markdown_workflow_mock():
    mock_trace = [
        {
            "step_type": "Global Statics",
            "target": "Entire Truss",
            "results": {"R_A_x": 0.0, "R_A_y": 50.0},
            "explanation": "Solved rigid-body equilibrium."
        }
    ]
    mock_state = MockTrussState(mock_trace)
    final_answers = {"max_compression": 100.0, "units": "kN"}
    
    result = generate_markdown_workflow(
        state=mock_state, 
        benchmark_id="MOCK_0001", 
        unit="kN", 
        final_answers=final_answers
    )
    
    # Verify Structural Elements
    assert "### Problem: MOCK_0001" in result
    assert "**Step 1: Global Statics (Entire Truss)**" in result
    assert "**Final Extraction**" in result
    
    # Verify Formatting
    assert "*   $R_{Ax}$ = 0.00 kN" in result
    assert "*   $max_{compression}$ = 100.00 kN" in result


def test_generate_markdown_workflow_real_truss_0001():
    # 1. Resolve the path to the real example file
    current_dir = Path(__file__).parent
    example_path = current_dir / "examples" / "example_TRUSS_0001.py"
    
    if not example_path.exists():
        import pytest
        pytest.skip(f"Could not find {example_path}")

    # 2. Dynamically import the example module
    spec = importlib.util.spec_from_file_location("example_TRUSS_0001", example_path)
    truss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(truss_module)
    
    # 3. Execute the full programmatic solver
    recipe = truss_module.SOLUTION_RECIPE
    sampled_vars = {"P": 50.0} 
    
    # Pass it through the standard pipeline runner
    
    # We have to extract the state manually because programmatic_solution 
    # only returns the final dict, not the state object.
    init_step = recipe[1]
    state = init_step["function"](sampled_vars=sampled_vars, **init_step["inputs"])
    
    recipe[2]["function"](state, **recipe[2]["inputs"])
    recipe[3]["function"](state, **recipe[3]["inputs"])
    
    final_answers = recipe[4]["function"](state, **recipe[4]["inputs"])
    
    # 4. Generate the document
    document = generate_markdown_workflow(
        state=state,
        benchmark_id="TRUSS_0001",
        unit="kN",
        final_answers=final_answers
    )
    
    # 5. Verify the full textbook-style output
    assert "### Problem: TRUSS_0001" in document
    assert "**Step 1: Global Statics (Entire Truss)**" in document
    assert "*   $R_{Ay}$ = 25.00 kN" in document
    
    # These are now correctly back to Step 2 and Step 3
    assert "**Step 2: Method of Joints (A)**" in document
    assert "*   $F_{CA}$ = 28.87 kN (Compression)" in document
    assert "**Step 3: Method of Joints (B)**" in document
    
    assert "**Final Extraction**" in document
    assert "*   $max_{compression_magnitude}$ = 28.87 kN" in document

