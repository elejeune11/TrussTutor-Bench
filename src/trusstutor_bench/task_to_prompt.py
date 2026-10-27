import base64
from pathlib import Path
from trusstutor_bench.task_base import BenchmarkTask
from trusstutor_bench.task_loader import TaskLoader
from trusstutor_bench.statics_tools import programmatic_solution
import json


def sample_variables(variables_schema: dict) -> dict:
    """
    Returns the default value of each variable in the problem schema.

    Args:
        variables_schema (dict): The 'variables' dictionary from PROBLEM_SCHEMA.

    Returns:
        dict: Variable name -> default value, for variables that define one.
    """
    sampled_vars = {}

    for var_name, config in variables_schema.items():
        if "default" in config:
            sampled_vars[var_name] = config["default"]

    return sampled_vars


def encode_image_to_base64(image_path: Path) -> str:
    """
    Reads an image file and encodes it as a base64 string. Raises FileNotFoundError
    if the file does not exist.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        str: Base64-encoded image.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())

    return encoded_bytes.decode('utf-8')


def render_prompt(problem_schema: dict, sampled_vars: dict) -> str:
    """
    Fills the problem template with the variable values and units.

    Args:
        problem_schema (dict): The PROBLEM_SCHEMA dictionary from a task.
        sampled_vars (dict): Values returned by sample_variables().

    Returns:
        str: The problem statement for the model.
    """
    template_str = problem_schema["template"]
    units = problem_schema.get("units", {})

    format_kwargs = {**sampled_vars, **units}

    return template_str.format(**format_kwargs)


def compile_task_payload(task: BenchmarkTask, figs_dir: Path) -> dict:
    """
    Builds the payload for one task: prompt, image, expected schema and ground truth.

    Args:
        task (BenchmarkTask): The loaded task.
        figs_dir (Path): Folder with the <benchmark_id>.png figures.

    Returns:
        dict: JSON-serializable payload.
    """
    sampled_vars = sample_variables(task.problem_schema["variables"])

    user_prompt = render_prompt(task.problem_schema, sampled_vars)

    image_path = figs_dir / f"{task.benchmark_id}.png"
    image_base64 = encode_image_to_base64(image_path)

    structured_schema = task.problem_schema["expected_llm_output"]

    ground_truth = programmatic_solution(task.solution_recipe, sampled_vars)

    payload = {
        "benchmark_id": task.benchmark_id,
        "user_prompt": user_prompt,
        "image_base64": image_base64,
        "structured_schema": structured_schema,
        "ground_truth_variables": sampled_vars,
        "ground_truth": ground_truth
    }

    return payload


def save_payload(payload: dict, output_dir: Path, benchmark_id: str) -> Path:
    """
    Saves a payload as <benchmark_id>_payload.json.

    Args:
        payload (dict): The compiled task data.
        output_dir (Path): Folder where the file is saved (created if missing).
        benchmark_id (str): ID used to name the file (e.g., TRUSS_0001).

    Returns:
        Path: Path of the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / f"{benchmark_id}_payload.json"

    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=4)

    return file_path


def task_to_prompt_single(task: BenchmarkTask, figs_dir: Path, output_dir: Path) -> Path:
    """
    Builds the payload for one task and saves it.

    Args:
        task (BenchmarkTask): The loaded task.
        figs_dir (Path): Folder with the <benchmark_id>.png figures.
        output_dir (Path): Folder where the payload is saved.

    Returns:
        Path: Path of the saved file.
    """
    payload = compile_task_payload(task, figs_dir)
    saved_path = save_payload(payload, output_dir, task.benchmark_id)
    return saved_path


def task_to_prompt_all(examples_dir: Path, figs_dir: Path, output_dir: Path, task_ids: list[str] | None = None) -> None:
    """
    Builds and saves the payload of every task in a directory.

    Args:
        examples_dir (Path): Folder with the task files.
        figs_dir (Path): Folder with the <benchmark_id>.png figures.
        output_dir (Path): Folder where the payloads are saved.
        task_ids (list[str] | None): If given, only these benchmark IDs are built.
    """
    print(f"Loading tasks from {examples_dir}...")
    registry = TaskLoader.load_directory(str(examples_dir))

    if task_ids is not None:
        missing = [tid for tid in task_ids if tid not in registry]
        if missing:
            print(f"[WARNING] Requested ids not found in registry, skipping: {missing}")
        registry = {tid: registry[tid] for tid in task_ids if tid in registry}

    if not registry:
        print("No tasks found. Exiting.")
        return

    print(f"Found {len(registry)} tasks. Compiling payloads...")
    success_count = 0

    for benchmark_id, task in registry.items():
        try:
            saved_path = task_to_prompt_single(task, figs_dir, output_dir)
            print(f"  [SUCCESS] Saved {benchmark_id} -> {saved_path.name}")
            success_count += 1
        except Exception as e:
            print(f"  [ERROR] Failed to compile {benchmark_id}: {e}")

    print(f"\nAll done! Successfully generated {success_count}/{len(registry)} payloads in: {output_dir}")
