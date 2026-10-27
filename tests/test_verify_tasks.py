import shutil
from pathlib import Path
import pytest
from verify_tasks import run_verification

TASKS_DIR = Path(__file__).parent.parent / "tasks"


def test_example_tasks_pass_verification():
    """The five example tasks pass all verification phases."""
    with pytest.raises(SystemExit) as exit_info:
        run_verification(str(TASKS_DIR))

    assert exit_info.value.code == 0


def test_missing_figure_fails_verification(tmp_path):
    """A task without its figure in figs/ fails the asset phase."""
    shutil.copy(TASKS_DIR / "example_TRUSS_0001.py", tmp_path / "example_TRUSS_0001.py")

    with pytest.raises(SystemExit) as exit_info:
        run_verification(str(tmp_path))

    assert exit_info.value.code == 1


def test_missing_directory_fails_verification(tmp_path):
    """A directory that does not exist exits with status 1."""
    with pytest.raises(SystemExit) as exit_info:
        run_verification(str(tmp_path / "does_not_exist"))

    assert exit_info.value.code == 1
