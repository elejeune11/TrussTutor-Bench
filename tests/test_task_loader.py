import pytest
from pathlib import Path
from trusstutor_bench.task_loader import TaskLoader


VALID_TEMPLATE = """
METADATA = {
    "benchmark_id": "TEST_001",
    "topic": ["method_of_joints"],
    "source_type": "original",
    "source_name": "github test example"
}
NODES = {"A": (0, 0)}
ELEMENTS = []
PROBLEM_SCHEMA = {}
SOLUTION_RECIPE = {}
"""

INVALID_METADATA_TEMPLATE = """
METADATA = {
    "benchmark_id": "TEST_002",
    "topic": ["made_up_topic_that_fails_linting"],  # This should trigger a crash
    "source_type": "original",
    "source_name": "github test example"
}
NODES = {"A": (0, 0)}
ELEMENTS = []
PROBLEM_SCHEMA = {}
SOLUTION_RECIPE = {}
"""

DUPLICATE_TEMPLATE = """
METADATA = {
    "benchmark_id": "TEST_001",  # Intentionally duplicating TEST_001
    "topic": ["method_of_joints"],
    "source_type": "original",
    "source_name": "github test example"
}
NODES = {"A": (0, 0)}
ELEMENTS = []
PROBLEM_SCHEMA = {}
SOLUTION_RECIPE = {}
"""

NO_METADATA_TEMPLATE = """
# A utility file or broken template missing the METADATA block
def some_helper_function():
    return True
"""


def test_load_directory_success(tmp_path: Path):
    """Test that a perfectly formatted template loads and registers successfully."""
    file_path = tmp_path / "TRUSS_VALID.py"
    file_path.write_text(VALID_TEMPLATE)

    registry = TaskLoader.load_directory(str(tmp_path))

    assert len(registry) == 1
    assert "TEST_001" in registry
    assert registry["TEST_001"].benchmark_id == "TEST_001"
    assert registry["TEST_001"].metadata["topic"] == ["method_of_joints"]


def test_load_directory_not_found():
    """Test that providing a bad directory path raises a FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        TaskLoader.load_directory("/this/path/definitely/does/not/exist")


def test_load_directory_skips_irrelevant_files(tmp_path: Path):
    """Test that it ignores files without METADATA, hidden files, and __init__.py."""
    (tmp_path / "helper.py").write_text(NO_METADATA_TEMPLATE)
    (tmp_path / "__init__.py").write_text(VALID_TEMPLATE)  # Starts with __
    (tmp_path / ".hidden.py").write_text(VALID_TEMPLATE)   # Starts with .

    registry = TaskLoader.load_directory(str(tmp_path))

    # All three files should be cleanly ignored without crashing
    assert len(registry) == 0


def test_load_directory_invalid_metadata_crashes(tmp_path: Path):
    """Test that unauthorized vocabularies trigger a fail-fast RuntimeError."""
    file_path = tmp_path / "TRUSS_INVALID.py"
    file_path.write_text(INVALID_METADATA_TEMPLATE)

    with pytest.raises(RuntimeError) as exc_info:
        TaskLoader.load_directory(str(tmp_path))

    # Verify the underlying ValueError bubbled up correctly
    assert "Unauthorized topic" in str(exc_info.value)


def test_load_directory_duplicate_id_crashes(tmp_path: Path):
    """Test that two files sharing the same benchmark_id crash the loader."""
    file1 = tmp_path / "TRUSS_001.py"
    file1.write_text(VALID_TEMPLATE)

    file2 = tmp_path / "TRUSS_001_COPY.py"
    file2.write_text(DUPLICATE_TEMPLATE)

    with pytest.raises(RuntimeError) as exc_info:
        TaskLoader.load_directory(str(tmp_path))

    assert "Duplicate benchmark_id found" in str(exc_info.value)


def test_load_real_examples_directory():
    """Test that the loader correctly parses the core benchmark example files."""
    # Resolve the absolute path to the 'examples' folder next to this test file
    current_dir = Path(__file__).parent
    examples_dir = current_dir.parent / "tasks"
    
    # Ensure the directory exists before attempting to load
    assert examples_dir.exists(), f"Expected examples directory at {examples_dir}"
    
    registry = TaskLoader.load_directory(str(examples_dir))
    
    # Check for specific benchmark IDs to ensure core files are loaded.
    # Checking existence rather than length prevents the test from failing 
    # when new examples are added to the directory later.
    expected_trusses = [
        "TRUSS_0001",
        "TRUSS_0002",
        "TRUSS_0003",
        "TRUSS_0004",
        "TRUSS_0005"
    ]
    
    for truss_id in expected_trusses:
        assert truss_id in registry, f"Expected {truss_id} to be loaded into the registry."
