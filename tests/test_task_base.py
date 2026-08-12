import pytest
from dataclasses import FrozenInstanceError
from trusstutor_bench.task_base import BenchmarkTask


# --- Helper to instantly create a mock module in memory ---
def create_mock_module(**attributes):
    return type('MockModule', (), attributes)


def test_from_module_happy_path():
    """Test that a fully populated module maps perfectly to the dataclass."""
    mock_module = create_mock_module(
        METADATA={"benchmark_id": "TEST_001", "topic": ["method_of_joints"]},
        SYMBOLIC_DEFINITIONS=["x", "y"],
        NODES={"A": (0, 0)},
        ELEMENTS=[{"id": "AB"}],
        LOADS={"A": (0, -100)},
        SUPPORTS={"A": ["x", "y"]},
        PROBLEM_SCHEMA={"template": "Solve this."},
        SOLUTION_RECIPE={1: "step_1"},
        DRAWING_VARIABLES={"L": 10}
    )
    
    task = BenchmarkTask.from_module(mock_module)
    
    assert task.benchmark_id == "TEST_001"
    assert task.nodes == {"A": (0, 0)}
    assert task.loads == {"A": (0, -100)}


def test_from_module_optional_fallbacks():
    """Test that missing optional attributes safely default to empty structures."""
    # Notice: Missing LOADS, SUPPORTS, SYMBOLIC_DEFINITIONS, DRAWING_VARIABLES
    mock_module = create_mock_module(
        METADATA={"benchmark_id": "TEST_002"},
        NODES={"A": (0, 0)},
        ELEMENTS=[{"id": "AB"}],
        PROBLEM_SCHEMA={"template": "Solve this."},
        SOLUTION_RECIPE={1: "step_1"}
    )
    
    task = BenchmarkTask.from_module(mock_module)
    
    assert task.loads == {}  # The getattr default
    assert task.supports == {}
    assert task.symbolic_definitions == []
    assert task.drawing_variables == {}


def test_from_module_missing_required_raises_error():
    """Test that missing a mandatory attribute (no getattr) correctly crashes."""
    # Notice: Missing mandatory 'NODES'
    mock_module = create_mock_module(
        METADATA={"benchmark_id": "TEST_003"},
        ELEMENTS=[{"id": "AB"}],
        PROBLEM_SCHEMA={},
        SOLUTION_RECIPE={}
    )
    
    with pytest.raises(AttributeError):
        BenchmarkTask.from_module(mock_module)


def test_frozen_immutability():
    """Test that downstream scripts cannot overwrite the locked template data."""
    mock_module = create_mock_module(
        METADATA={"benchmark_id": "TEST_004"},
        NODES={"A": (0, 0)},
        ELEMENTS=[],
        PROBLEM_SCHEMA={},
        SOLUTION_RECIPE={}
    )
    task = BenchmarkTask.from_module(mock_module)
    
    # Attempting to reassign an attribute should throw a FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        task.nodes = {"B": (10, 10)}