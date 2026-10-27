import pytest
from unittest.mock import MagicMock, patch
from trusstutor_bench.verify_physics import validate_task_physics


@pytest.fixture
def mock_task():
    """Provides a fake BenchmarkTask with one load and one support.

    Returns:
        MagicMock: The fake task.
    """
    task = MagicMock()
    task.benchmark_id = "TEST_TRUSS_001"

    task.problem_schema = {
        "variables": {
            "P": {"type": "float", "default": 100.0},
        }
    }

    task.drawing_variables = {}
    task.nodes = {"A": (0, 0)}
    task.elements = [{"id": "AB"}]
    task.supports = {"A": ["x", "y"]}
    task.loads = {"A": (0, -100)}

    return task


@patch("trusstutor_bench.verify_physics.check_stability_and_determinacy")
@patch("trusstutor_bench.verify_physics.solve_truss")
def test_validate_physics_happy_path(mock_solve_truss, mock_check, mock_task):
    """Returns True when the recipe and the DSM give the same signed force."""
    mock_check.return_value = {"is_singular": False}

    mock_state = MagicMock()
    mock_state.known_forces = {"AB": 50.0}

    mock_task.solution_recipe = {
        1: {"function": MagicMock(return_value=mock_state), "inputs": {}},
        2: {"function": MagicMock(), "inputs": {}},
        3: {"function": MagicMock(), "inputs": {}} # Extraction step, not run here
    }

    dsm_internal_forces = {"AB": 50.0}
    mock_solve_truss.return_value = (None, dsm_internal_forces, None)

    result = validate_task_physics(mock_task)
    assert result is True


@patch("trusstutor_bench.verify_physics.check_stability_and_determinacy")
def test_validate_physics_catches_mechanism(mock_check, mock_task):
    """Fails at once for a geometrically unstable truss."""
    mock_check.return_value = {"is_singular": True}

    with pytest.raises(ValueError, match="geometric mechanism"):
        validate_task_physics(mock_task)


@patch("trusstutor_bench.verify_physics.check_stability_and_determinacy")
@patch("trusstutor_bench.verify_physics.solve_truss")
def test_validate_physics_catches_mismatch(mock_solve_truss, mock_check, mock_task):
    """Raises AssertionError when the two solvers disagree on a member force."""
    mock_check.return_value = {"is_singular": False}

    mock_state = MagicMock()
    # The recipe finds 50 in compression (-50).
    mock_state.known_forces = {"AB": -50.0}

    mock_task.solution_recipe = {
        1: {"function": MagicMock(return_value=mock_state), "inputs": {}},
        2: {"function": MagicMock(), "inputs": {}},
        3: {"function": MagicMock(), "inputs": {}}
    }

    # The DSM finds 50 in tension (+50).
    dsm_internal_forces = {
        "AB": 50.0
    }
    mock_solve_truss.return_value = (None, dsm_internal_forces, None)

    with pytest.raises(AssertionError, match="Mismatch in Element AB"):
        validate_task_physics(mock_task)
