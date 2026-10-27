import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.append(str(PROJECT_ROOT / "src"))

from trusstutor_bench.task_loader import TaskLoader
from trusstutor_bench.verify_physics import validate_task_physics
from trusstutor_bench.verify_extraction import validate_task_extraction


def run_verification(target_directory: str):
    """
    Runs the four verification phases on every task in a directory, prints a summary and
    exits with status 0 if all tasks pass, or 1 otherwise.

    Args:
        target_directory (str): Folder with the task files and a "figs" folder.
    """
    target_dir = Path(target_directory).resolve()

    if not target_dir.exists() or not target_dir.is_dir():
        print(f"[FATAL] Directory not found: {target_dir}")
        sys.exit(1)

    print("==================================================")
    print(f" STARTING TASK VERIFICATION ")
    print(f" Target: {target_dir}")
    print("==================================================\n")

    print("Executing Phase 0: Structural Ingest (BenchmarkTask Instantiation)...")
    try:
        registry = TaskLoader.load_directory(str(target_dir))
    except Exception as e:
        print(f"\n[FATAL] Loader crashed during Phase 0 ingest. Template violates BenchmarkTask schema.\n{e}")
        sys.exit(1)

    if not registry:
        print("\n[WARNING] No valid task templates found in the target directory.")
        sys.exit(0)

    figs_dir = target_dir / "figs"
    if not figs_dir.exists() or not figs_dir.is_dir():
        print(f"\n[WARNING] 'figs' directory not found at {figs_dir}. Image validation will fail.")

    print(f"\n==================================================")
    print(f" RUNNING VERIFICATION ({len(registry)} TASKS) ")
    print(f" Phase 0: Structural Ingest (Passed automatically during load) ")
    print(f" Phase 1: Physics (SymPy vs NumPy) ")
    print(f" Phase 2: Schema (JSON Output Extraction) ")
    print(f" Phase 3: Assets (Matching .png in figs/ folder) ")
    print(f"==================================================\n")

    success_count = 0
    failures = []

    for benchmark_id, task in registry.items():
        try:
            validate_task_physics(task)

            validate_task_extraction(task)

            expected_image = figs_dir / f"{benchmark_id}.png"
            if not expected_image.exists():
                raise FileNotFoundError(f"Missing image asset: '{expected_image.name}' in the 'figs' folder.")

            print(f"[PASSED] {benchmark_id}")
            success_count += 1

        except Exception as e:
            print(f"[FAILED] {benchmark_id}: {e}")
            failures.append(benchmark_id)

    print("\n==================================================")
    print(" VERIFICATION SUMMARY ")
    print("==================================================")
    print(f"Total Tasks Found: {len(registry)}")
    print(f"Passed Validation: {success_count}")
    print(f"Failed Validation: {len(failures)}")

    if failures:
        print(f"\n[ERROR] Failing Tasks: {', '.join(failures)}")
        print("Please check the error messages above and fix the templates or missing images.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All tasks passed structural ingest, schema linting, physics validation, extraction formatting, and asset verification!")
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verifies the task templates of the TrussTutor benchmark."
    )
    parser.add_argument(
        "directory",
        type=str,
        nargs="?",
        default="tasks",
        help="Path to the directory containing the task templates (defaults to 'tasks')."
    )

    args = parser.parse_args()
    run_verification(args.directory)
