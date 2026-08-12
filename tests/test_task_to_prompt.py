import base64
import pytest
import json
from pathlib import Path
from trusstutor_bench.task_loader import TaskLoader
from trusstutor_bench.task_to_prompt import sample_variables
from trusstutor_bench.task_to_prompt import encode_image_to_base64
from trusstutor_bench.task_to_prompt import render_prompt
from trusstutor_bench.task_to_prompt import compile_task_payload
from trusstutor_bench.task_to_prompt import save_payload
from trusstutor_bench.task_to_prompt import task_to_prompt_single, task_to_prompt_all
from trusstutor_bench.task_loader import TaskLoader


def test_sample_variables_with_real_examples():
    """Test that sample_variables correctly extracts defaults from real tasks."""
    # Resolve the absolute path to the 'examples' folder
    examples_dir = Path(__file__).parent / "examples"
    
    # Ensure the directory exists before attempting to load
    assert examples_dir.exists(), f"Expected examples directory at {examples_dir}"
    
    # Load the real benchmark tasks into the registry
    registry = TaskLoader.load_directory(str(examples_dir))
    
    # ---------------------------------------------------------
    # Test TRUSS_0001 Extraction
    # ---------------------------------------------------------
    task_1 = registry["TRUSS_0001"]
    sampled_1 = sample_variables(task_1.problem_schema["variables"])
    
    # It should extract the default 50.0 for P
    assert "P" in sampled_1
    assert sampled_1["P"] == 50.0
    
    # It should ignore L_val because it has no default
    assert "L_val" not in sampled_1

    # ---------------------------------------------------------
    # Test TRUSS_0002 Extraction
    # ---------------------------------------------------------
    task_2 = registry["TRUSS_0002"]
    sampled_2 = sample_variables(task_2.problem_schema["variables"])
    
    # It should extract the defaults for both P1 and P2
    assert sampled_2 == {"P1": 100.0, "P2": 50.0}
    
    # It should ignore L_val
    assert "L_val" not in sampled_2


def test_encode_image_to_base64_success(tmp_path: Path):
    """Test that a file is correctly read and encoded to a base64 string."""
    # Create a temporary dummy image file
    dummy_image_path = tmp_path / "dummy_fig.png"
    
    # Write some known bytes to it (the word 'trusstutor' in bytes)
    known_bytes = b"trusstutor"
    dummy_image_path.write_bytes(known_bytes)
    
    # Run the function
    encoded_str = encode_image_to_base64(dummy_image_path)
    
    # Verify the output matches the expected manual base64 encoding
    expected_str = base64.b64encode(known_bytes).decode('utf-8')
    
    assert isinstance(encoded_str, str)
    assert encoded_str == expected_str


def test_encode_image_to_base64_file_not_found():
    """Test that a missing file correctly raises a FileNotFoundError."""
    bad_path = Path("/this/path/does/not/exist.png")
    
    with pytest.raises(FileNotFoundError):
        encode_image_to_base64(bad_path)


def test_render_prompt_with_real_example():
    """Test that variables and units are correctly injected into the template string."""
    # Resolve the absolute path and load the registry
    examples_dir = Path(__file__).parent / "examples"
    registry = TaskLoader.load_directory(str(examples_dir))
    
    # Use TRUSS_0001 for the test
    task_1 = registry["TRUSS_0001"]
    schema = task_1.problem_schema
    
    # Get the variables (we expect {'P': 50.0})
    sampled_1 = sample_variables(schema["variables"])
    
    # Render the prompt
    final_text = render_prompt(schema, sampled_1)
    
    # Verify the formatting worked
    assert "P 50.0 kN" in final_text
    assert "{P}" not in final_text
    assert "{force_unit}" not in final_text


def test_compile_task_payload_success():
    """Test that the full payload is correctly assembled using real task data and images."""
    current_dir = Path(__file__).parent
    examples_dir = current_dir / "examples"
    figs_dir = examples_dir / "figs"
    
    from trusstutor_bench.task_loader import TaskLoader
    registry = TaskLoader.load_directory(str(examples_dir))
    task_1 = registry["TRUSS_0001"]
    
    payload = compile_task_payload(task_1, figs_dir)
    
    assert payload["benchmark_id"] == "TRUSS_0001"
    assert "P 50.0 kN" in payload["user_prompt"]
    assert payload["ground_truth_variables"] == {"P": 50.0}
    assert "max_compression_magnitude" in payload["structured_schema"]["properties"]
    
    # Verify Pattern A Ground Truth was embedded
    assert "ground_truth" in payload
    assert isinstance(payload["ground_truth"], dict)
    
    assert isinstance(payload["image_base64"], str)


def test_save_payload_success_real_example(tmp_path: Path):
    """Test that a real task payload is correctly serialized and saved to disk."""
    # 1. Resolve paths to the real examples and figs directories
    current_dir = Path(__file__).parent
    examples_dir = current_dir / "examples"
    figs_dir = examples_dir / "figs"
    
    # Ensure directories exist
    assert examples_dir.exists(), f"Expected examples directory at {examples_dir}"
    assert figs_dir.exists(), f"Expected figs directory at {figs_dir}"
    
    # 2. Load the real TRUSS_0001 example
    registry = TaskLoader.load_directory(str(examples_dir))
    task_1 = registry["TRUSS_0001"]
    
    # 3. Compile the real payload
    real_payload = compile_task_payload(task_1, figs_dir)
    
    # 4. Define the output directory (using the temp path so we don't clutter the repo)
    output_folder = tmp_path / "payloads"
    
    # 5. Save the payload
    saved_file_path = save_payload(real_payload, output_folder, task_1.benchmark_id)
    
    # 6. Verify the file was created in the correct location
    assert saved_file_path.exists()
    assert saved_file_path.name == "TRUSS_0001_payload.json"
    
    # 7. Read it back and verify the contents survived serialization
    with open(saved_file_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
        
    # Check that the loaded data matches the payload we saved
    assert loaded_data["benchmark_id"] == "TRUSS_0001"
    assert loaded_data["ground_truth_variables"] == {"P": 50.0}
    assert loaded_data["image_base64"] == real_payload["image_base64"]


def test_task_to_prompt_single_success(tmp_path: Path):
    """Test that compiling a single task produces exactly one JSON payload."""
    current_dir = Path(__file__).parent
    examples_dir = current_dir / "examples"
    figs_dir = examples_dir / "figs"
    output_folder = tmp_path / "payloads"
    
    # Load just TRUSS_0001
    registry = TaskLoader.load_directory(str(examples_dir))
    task_1 = registry["TRUSS_0001"]
    
    # Process single
    saved_path = task_to_prompt_single(task_1, figs_dir, output_folder)
    
    assert saved_path.exists()
    assert saved_path.name == "TRUSS_0001_payload.json"
    
    # Ensure no other files were created
    generated_files = list(output_folder.glob("*_payload.json"))
    assert len(generated_files) == 1


def test_task_to_prompt_all_success(tmp_path: Path):
    """Test the batch processor compiles all files in the directory."""
    current_dir = Path(__file__).parent
    examples_dir = current_dir / "examples"
    figs_dir = examples_dir / "figs"
    output_folder = tmp_path / "payloads"
    
    # Process all
    task_to_prompt_all(examples_dir, figs_dir, output_folder)
    
    assert output_folder.exists()
    
    # Verify exactly 5 payloads were generated
    generated_files = list(output_folder.glob("*_payload.json"))
    assert len(generated_files) == 5
    
    # Verify expected files exist
    expected_ids = ["TRUSS_0001", "TRUSS_0002", "TRUSS_0003", "TRUSS_0004", "TRUSS_0005"]
    for task_id in expected_ids:
        assert (output_folder / f"{task_id}_payload.json").exists()