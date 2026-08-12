import base64
from pathlib import Path
from trusstutor_bench.task_base import BenchmarkTask
from trusstutor_bench.task_loader import TaskLoader
from trusstutor_bench.statics_tools import programmatic_solution
import json


def sample_variables(variables_schema: dict) -> dict:
    """
    Extracts the default values from the problem schema's variables configuration.
    
    Args:
        variables_schema (dict): The 'variables' dictionary from PROBLEM_SCHEMA.
        
    Returns:
        dict: A mapping of variable names to their default numerical values.
    """
    sampled_vars = {}
    
    for var_name, config in variables_schema.items():
        # Only extract parameters that actually have a 'default' defined
        if "default" in config:
            sampled_vars[var_name] = config["default"]
            
    return sampled_vars


def encode_image_to_base64(image_path: Path) -> str:
    """
    Reads an image file from disk and encodes it as a base64 string.
    
    Args:
        image_path (Path): The path to the image file (e.g., a .png).
        
    Returns:
        str: The base64 encoded string representation of the image.
        
    Raises:
        FileNotFoundError: If the specified image path does not exist.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at path: {image_path}")
        
    with open(image_path, "rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
        
    # Decode the bytes object to a standard string for JSON serialization
    return encoded_bytes.decode('utf-8')


def render_prompt(problem_schema: dict, sampled_vars: dict) -> str:
    """
    Formats the problem template string using the sampled variables and units.
    
    Args:
        problem_schema (dict): The PROBLEM_SCHEMA dictionary from a task.
        sampled_vars (dict): The numerical values extracted by sample_variables().
        
    Returns:
        str: The fully formatted problem statement ready for the LLM.
    """
    template_str = problem_schema["template"]
    units = problem_schema.get("units", {})
    
    # Merge the sampled variables and the units into a single dictionary
    format_kwargs = {**sampled_vars, **units}
    
    # Inject the values into the {placeholders} in the string
    return template_str.format(**format_kwargs)


def compile_task_payload(task: BenchmarkTask, figs_dir: Path) -> dict:
    """
    Compiles all components of a single task into a self-contained JSON-serializable payload.
    """
    # 1. Sample the ground truth variables
    sampled_vars = sample_variables(task.problem_schema["variables"])
    
    # 2. Render the final text prompt
    user_prompt = render_prompt(task.problem_schema, sampled_vars)
    
    # 3. Locate and encode the image
    image_path = figs_dir / f"{task.benchmark_id}.png"
    image_base64 = encode_image_to_base64(image_path)
    
    # 4. Extract the JSON schema for structural enforcement
    structured_schema = task.problem_schema["expected_llm_output"]
    
    # 5. Compute canonical ground truth at payload creation time
    ground_truth = programmatic_solution(task.solution_recipe, sampled_vars)
    
    # 6. Assemble the final payload
    payload = {
        "benchmark_id": task.benchmark_id,
        "user_prompt": user_prompt,
        "image_base64": image_base64,
        "structured_schema": structured_schema,
        "ground_truth_variables": sampled_vars,
        "ground_truth": ground_truth  # Embedded directly into the artifact
    }
    
    return payload


def save_payload(payload: dict, output_dir: Path, benchmark_id: str) -> Path:
    """
    Saves the compiled payload dictionary to a JSON file.
    
    Args:
        payload (dict): The compiled task data.
        output_dir (Path): The directory where the JSON file should be saved.
        benchmark_id (str): The ID used to name the file (e.g., TRUSS_0001).
        
    Returns:
        Path: The absolute path to the newly created JSON file.
    """
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct the file name
    file_path = output_dir / f"{benchmark_id}_payload.json"
    
    # Write the payload to disk with indent=4 for human readability
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=4)
        
    return file_path


def task_to_prompt_single(task: BenchmarkTask, figs_dir: Path, output_dir: Path) -> Path:
    """
    Compiles the payload for a single benchmark task and saves it to disk.
    
    Args:
        task (BenchmarkTask): The loaded task object.
        figs_dir (Path): Directory containing the generated .png schematics.
        output_dir (Path): Directory where the final JSON file will be saved.
        
    Returns:
        Path: The absolute path to the saved JSON file.
    """
    payload = compile_task_payload(task, figs_dir)
    saved_path = save_payload(payload, output_dir, task.benchmark_id)
    return saved_path


def task_to_prompt_all(examples_dir: Path, figs_dir: Path, output_dir: Path) -> None:
    """
    Iterates through all benchmark tasks in a directory and compiles their payloads.
    
    Args:
        examples_dir (Path): Directory containing the example_TRUSS_xxxx.py files.
        figs_dir (Path): Directory containing the generated .png schematics.
        output_dir (Path): Directory where the final JSON files will be saved.
    """
    print(f"Loading tasks from {examples_dir}...")
    registry = TaskLoader.load_directory(str(examples_dir))
    
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
