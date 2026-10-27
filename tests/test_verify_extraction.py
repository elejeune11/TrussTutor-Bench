import pytest
from unittest.mock import MagicMock, patch
from trusstutor_bench.verify_extraction import validate_task_extraction


@pytest.fixture
def mock_task_extraction():
    """Provides a fake BenchmarkTask with a strict output schema.

    Returns:
        MagicMock: The fake task.
    """
    task = MagicMock()
    task.benchmark_id = "TEST_EXTRACT_001"

    task.problem_schema = {
        "variables": {
            "P": {"type": "float", "default": 100.0}
        },
        "expected_llm_output": {
            "type": "object",
            "properties": {
                "max_compression_magnitude": {"type": "number"},
                "units": {"type": "string"}
            },
            "required": ["max_compression_magnitude", "units"]
        }
    }

    task.solution_recipe = {}

    return task


@patch("trusstutor_bench.verify_extraction.programmatic_solution")
def test_validate_extraction_happy_path(mock_prog_sol, mock_task_extraction):
    """Returns True when the output matches the schema."""
    mock_prog_sol.return_value = {
        "max_compression_magnitude": 50.0,
        "units": "kN"
    }

    result = validate_task_extraction(mock_task_extraction)

    assert result is True
    mock_prog_sol.assert_called_once()


@patch("trusstutor_bench.verify_extraction.programmatic_solution")
def test_validate_extraction_schema_mismatch(mock_prog_sol, mock_task_extraction):
    """Raises ValueError when the output has a wrong type and a missing key."""
    mock_prog_sol.return_value = {
        "max_compression_magnitude": "fifty"
    }

    with pytest.raises(ValueError, match="Schema Validation Failed"):
        validate_task_extraction(mock_task_extraction)


@patch("trusstutor_bench.verify_extraction.programmatic_solution")
def test_validate_extraction_pipeline_crash(mock_prog_sol, mock_task_extraction):
    """Raises a clear RuntimeError when the solution recipe crashes."""
    mock_prog_sol.side_effect = Exception("Matrix is singular")

    with pytest.raises(RuntimeError, match="Pipeline execution failed"):
        validate_task_extraction(mock_task_extraction)


@patch("trusstutor_bench.verify_extraction.programmatic_solution")
def test_validate_extraction_missing_schema(mock_prog_sol):
    """Raises ValueError when the task defines no output schema."""
    task = MagicMock()
    task.benchmark_id = "TEST_EXTRACT_002"
    task.problem_schema = {"variables": {}}
    task.solution_recipe = {}

    mock_prog_sol.return_value = {"some": "data"}

    with pytest.raises(ValueError, match="Missing 'expected_llm_output'"):
        validate_task_extraction(task)


@patch("trusstutor_bench.verify_extraction.programmatic_solution")
def test_validate_extraction_rejects_vacuous_schema(mock_prog_sol):
    """Rejects a schema with the answer keys at the root, which would pass anything."""
    task = MagicMock()
    task.benchmark_id = "TEST_EXTRACT_003"
    task.problem_schema = {
        "variables": {},
        "expected_llm_output": {
            "max_compression_magnitude": {"type": "number"},
            "units": {"type": "string"}
        }
    }
    task.solution_recipe = {}
    mock_prog_sol.return_value = {"completely": "wrong output"}

    with pytest.raises(ValueError, match="must be a valid JSON Schema"):
        validate_task_extraction(task)
