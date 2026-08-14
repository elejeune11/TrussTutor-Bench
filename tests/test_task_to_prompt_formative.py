from trusstutor_bench.task_to_prompt_formative import get_formative_schema
from trusstutor_bench.task_base import BenchmarkTask
import importlib.util
from pathlib import Path
import pytest
from trusstutor_bench.task_to_prompt_formative import generate_formative_blocks
from trusstutor_bench.task_to_prompt_formative import compile_formative_payload
import json
from trusstutor_bench.task_to_prompt_formative import task_to_prompt_formative_single
from trusstutor_bench.task_to_prompt_formative import task_to_prompt_formative_all


def test_get_formative_schema():
    """
    Validates that the formative schema strictly enforces the required 
    JSON structure for automated LLM grading.
    """
    schema = get_formative_schema()

    # 1. Check outer structure
    assert isinstance(schema, dict)
    assert schema.get("type") == "object"
    assert schema.get("additionalProperties") is False
    assert "properties" in schema
    assert "required" in schema

    # 2. Check properties keys
    properties = schema["properties"]
    expected_keys = {"is_work_correct", "error_step_number", "explanation_to_student"}
    assert set(properties.keys()) == expected_keys

    # 3. Check specific types inside properties
    assert properties["is_work_correct"]["type"] == "boolean"
    
    # Ensure error_step_number uses the Gemini-friendly nullable flag
    assert properties["error_step_number"]["type"] == "integer"
    assert properties["error_step_number"].get("nullable") is True
    
    assert properties["explanation_to_student"]["type"] == "string"

    # 4. Check required list ensures all keys are returned
    assert set(schema["required"]) == expected_keys


def test_get_student_posture():
    """
    Validates the retrieval of student postures and ensures 
    invalid keys raise the appropriate ValueError.
    """
    import pytest
    from trusstutor_bench.task_to_prompt_formative import get_student_posture

    # 1. Test valid posture_1 retrieval
    expected_text_1 = "I know my answer is wrong. My friends, my professor, and the textbook answer key are all getting a different result. Where did I make a mistake in my work? Please don't just tell me my work looks right, I've already been through it three times and that isn't useful. Where's the mistake?"
    assert get_student_posture("posture_1") == expected_text_1

    # 2. Test valid posture_2 retrieval
    expected_text_2 = "Is my work correct?"
    assert get_student_posture("posture_2") == expected_text_2

    # 3. Test valid posture_3 retrieval
    expected_text_3 = "Is my work correct?"
    assert get_student_posture("posture_3") == expected_text_3

    # 4. Test invalid posture exception handling
    with pytest.raises(ValueError) as excinfo:
        get_student_posture("invalid_posture")
        
    # Verify the error message contains the expected string format
    error_msg = str(excinfo.value)
    assert "Unknown posture_type: invalid_posture" in error_msg
    assert "'posture_1'" in error_msg
    assert "'posture_2'" in error_msg
    assert "'posture_3'" in error_msg


def test_generate_formative_blocks_real_example(tmp_path):
    """
    Executes TRUSS_0001, compiles the formative blocks, and verifies 
    the correct interleaving of Markdown text and Base64 images using 
    the real BenchmarkTask object.
    """
    # 1. Resolve paths
    current_dir = Path(__file__).parent
    example_path = current_dir / "examples" / "example_TRUSS_0001.py"
    
    if not example_path.exists():
        pytest.skip(f"Could not find {example_path}")

    # 2. Dynamically import the example
    spec = importlib.util.spec_from_file_location("example_TRUSS_0001", example_path)
    truss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(truss_module)

    # 3. Instantiate the real BenchmarkTask directly from the module
    task = BenchmarkTask.from_module(truss_module)

    # 4. Extract solver recipe and drawing variables
    recipe = truss_module.SOLUTION_RECIPE
    subs_dict = truss_module.DRAWING_VARIABLES

    # 5. Execute the solver trace up to Method of Joints
    init_step = recipe[1]
    state = init_step["function"](sampled_vars=subs_dict, **init_step["inputs"])
    
    reaction_step = recipe[2]
    reaction_step["function"](state, **reaction_step["inputs"])
    
    moj_step = recipe[3]
    moj_step["function"](state, **moj_step["inputs"])

    # 6. Generate the blocks using pytest's temporary directory for image storage
    blocks = generate_formative_blocks(
        task=task, 
        state=state, 
        subs_dict=subs_dict, 
        output_dir=tmp_path
    )

    # 7. Verify the output structure
    assert isinstance(blocks, list)
    assert len(blocks) > 0, "Blocks list should not be empty."

    # Check the first two blocks (Global FBD intro text and image)
    assert blocks[0]["type"] == "text"
    assert "global Free Body Diagram" in blocks[0]["content"]
    
    assert blocks[1]["type"] == "image"
    assert blocks[1]["name"] == f"{task.benchmark_id}_global_FBD"
    assert "content_base64" in blocks[1]
    assert len(blocks[1]["content_base64"]) > 100  # Ensure the string is actually populated

    # Verify that the sequence contains a logical mix of text and images
    types = [b["type"] for b in blocks]
    assert "text" in types
    assert "image" in types

    # Check for Method of Joints injection
    # Since TRUSS_0001 executes MoJ, we expect at least one Node FBD image block
    image_names = [b.get("name", "") for b in blocks if b["type"] == "image"]
    node_fbd_found = any("_node_" in name for name in image_names)
    
    assert node_fbd_found, "Should have generated at least one Node FBD block for Method of Joints."


def test_render_formative_prompt():
    """
    Verifies that the formative prompt correctly injects the base problem 
    statement and the student's query without unnecessary instruction bloat.
    """
    from trusstutor_bench.task_to_prompt_formative import render_formative_prompt
    
    # 1. Mock the inputs
    mock_problem_schema = {
        "template": "A truss with a load of {P} {force_unit} is analyzed.",
        "units": {"force_unit": "kN"}
    }
    mock_sampled_vars = {"P": 50.0}
    mock_student_query = "Where did I make a mistake in my work?"
    
    # 2. Execute the function
    result_prompt = render_formative_prompt(
        problem_schema=mock_problem_schema, 
        sampled_vars=mock_sampled_vars, 
        student_query=mock_student_query
    )
    
    # 3. Verify the injections
    assert "A truss with a load of 50.0 kN is analyzed." in result_prompt
    assert f"\"{mock_student_query}\"" in result_prompt
    
    # 4. Verify minimalism (ensure no legacy instructions remain)
    assert "JSON object" not in result_prompt
    assert "You are an expert" not in result_prompt


def test_compile_formative_payload(tmp_path):
    """
    Executes the full pipeline for TRUSS_0001 to ensure the final 
    payload dictionary contains all required text, schemas, and base64 blocks.
    """
    # 1. Resolve paths and dynamically load TRUSS_0001
    current_dir = Path(__file__).parent
    example_path = current_dir / "examples" / "example_TRUSS_0001.py"
    
    if not example_path.exists():
        pytest.skip(f"Could not find {example_path}")

    spec = importlib.util.spec_from_file_location("example_TRUSS_0001", example_path)
    truss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(truss_module)

    # 2. Instantiate the BenchmarkTask
    task = BenchmarkTask.from_module(truss_module)

    # 3. Create a dummy base schematic image to satisfy the encoder
    figs_dir = tmp_path / "figs"
    figs_dir.mkdir()
    dummy_image_path = figs_dir / f"{task.benchmark_id}.png"
    
    # Write a tiny valid 1x1 pixel PNG byte string
    tiny_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    dummy_image_path.write_bytes(tiny_png)

    # 4. Compile the payload
    output_dir = tmp_path / "output"
    payload = compile_formative_payload(
        task=task,
        figs_dir=figs_dir,
        output_dir=output_dir,
        posture_type="posture_1"
    )

    # 5. Verify top-level structure
    assert isinstance(payload, dict)
    assert payload["benchmark_id"] == "TRUSS_0001"
    assert payload["posture_type"] == "posture_1"
    
    # 6. Verify components
    assert "Problem Statement" in payload["system_prompt"]
    assert "Where did I make a mistake" in payload["system_prompt"]
    
    assert isinstance(payload["base_image_base64"], str)
    assert len(payload["base_image_base64"]) > 0
    
    assert isinstance(payload["structured_schema"], dict)
    assert "is_work_correct" in payload["structured_schema"]["properties"]
    
    # 7. Verify the interleaved block array generated correctly
    student_work = payload["student_work"]
    assert isinstance(student_work, list)
    assert len(student_work) > 0
    
    block_types = [block["type"] for block in student_work]
    assert "text" in block_types
    assert "image" in block_types


def test_compile_formative_payload_all_examples(tmp_path):
    """
    Crawls the examples directory and executes the full formative payload 
    pipeline for every example_TRUSS_*.py file found, ensuring robust 
    execution across diverse geometries and topologies.
    """
    import importlib.util
    from pathlib import Path
    import pytest
    from trusstutor_bench.task_base import BenchmarkTask
    from trusstutor_bench.task_to_prompt_formative import compile_formative_payload

    # 1. Locate the examples directory
    current_dir = Path(__file__).parent
    examples_dir = current_dir / "examples"
    example_files = list(examples_dir.glob("example_TRUSS_*.py"))
    
    if not example_files:
        pytest.skip(f"No example files found in {examples_dir}")

    # 2. Setup mock image directory
    figs_dir = tmp_path / "figs"
    figs_dir.mkdir()
    output_dir = tmp_path / "output"
    
    # Tiny valid 1x1 pixel PNG byte string to satisfy the base64 encoder
    tiny_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'

    # 3. Iterate over all found examples
    success_count = 0
    for example_path in example_files:
        try:
            # Dynamically import the example
            spec = importlib.util.spec_from_file_location(example_path.stem, example_path)
            truss_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(truss_module)

            # Instantiate the task
            task = BenchmarkTask.from_module(truss_module)

            # Provide the dummy image
            dummy_image_path = figs_dir / f"{task.benchmark_id}.png"
            if not dummy_image_path.exists():
                dummy_image_path.write_bytes(tiny_png)

            # Compile the payload
            payload = compile_formative_payload(
                task=task,
                figs_dir=figs_dir,
                output_dir=output_dir,
                posture_type="posture_1"
            )

            # Verify core payload structures
            assert isinstance(payload, dict), f"[{task.benchmark_id}] Payload is not a dict"
            assert payload["benchmark_id"] == task.benchmark_id, f"[{task.benchmark_id}] ID mismatch"
            assert "Problem Statement" in payload["system_prompt"], f"[{task.benchmark_id}] Prompt missing problem statement"
            assert "student_work" in payload, f"[{task.benchmark_id}] Missing student_work block"
            
            # Verify the multimodal blocks
            student_work = payload["student_work"]
            assert isinstance(student_work, list), f"[{task.benchmark_id}] student_work is not a list"
            assert len(student_work) > 0, f"[{task.benchmark_id}] student_work is empty"
            
            block_types = [block["type"] for block in student_work]
            assert "text" in block_types, f"[{task.benchmark_id}] No text blocks found"
            
            # --- Conditional Image Check ---
            topics = task.metadata.get("topic", [])
            is_pure_zfm = len(topics) == 1 and topics[0] == "identify_zfm"
            
            if not is_pure_zfm:
                assert "image" in block_types, f"[{task.benchmark_id}] No image blocks found for non-ZFM problem"
            else:
                assert "image" not in block_types, f"[{task.benchmark_id}] Unwanted image blocks found in pure ZFM problem"
            
            success_count += 1
            
        except Exception as e:
            # Use pytest.fail to immediately flag which specific file broke the pipeline
            pytest.fail(f"Pipeline failed on {example_path.name}: {str(e)}")
            
    # Final sanity check to ensure the loop actually ran
    assert success_count == len(example_files), "Did not successfully process all examples."


def test_task_to_prompt_formative_single(tmp_path):
    """
    Verifies that the single task orchestrator correctly compiles the payload 
    and writes it to a valid JSON file on disk.
    """
    # 1. Resolve paths and load task
    current_dir = Path(__file__).parent
    example_path = current_dir / "examples" / "example_TRUSS_0001.py"
    if not example_path.exists():
        pytest.skip(f"Could not find {example_path}")

    spec = importlib.util.spec_from_file_location("example_TRUSS_0001", example_path)
    truss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(truss_module)

    task = BenchmarkTask.from_module(truss_module)

    # 2. Setup mock directories and images
    figs_dir = tmp_path / "figs"
    figs_dir.mkdir()
    output_dir = tmp_path / "output"
    
    tiny_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    (figs_dir / f"{task.benchmark_id}.png").write_bytes(tiny_png)

    # 3. Execute the single generator
    saved_file = task_to_prompt_formative_single(
        task=task,
        figs_dir=figs_dir,
        output_dir=output_dir,
        posture_type="posture_1"
    )

    # 4. Verify the file exists and is named correctly
    assert saved_file.exists()
    assert saved_file.suffix == ".json"
    assert "formative_posture_1" in saved_file.name

    # 5. Verify the JSON contents
    with open(saved_file, "r") as f:
        data = json.load(f)
        
    assert data["benchmark_id"] == task.benchmark_id
    assert data["posture_type"] == "posture_1"
    assert "student_work" in data


def test_compile_formative_payload_postures(tmp_path):
    """
    Verifies that compile_formative_payload dynamically swaps solver tools 
    and updates the ground truth block based on the selected posture type.
    """
    # 1. Dynamically load a real template (e.g., TRUSS_0001)
    current_dir = Path(__file__).parent
    examples_dir = current_dir / "examples"
    example_path = examples_dir / "example_TRUSS_0001.py"
    
    if not example_path.exists():
        pytest.skip("example_TRUSS_0001.py not found in examples directory.")

    spec = importlib.util.spec_from_file_location("example_TRUSS_0001", example_path)
    truss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(truss_module)
    task = BenchmarkTask.from_module(truss_module)

    # 2. Setup mock directories and a dummy image for the encoder
    figs_dir = tmp_path / "figs"
    figs_dir.mkdir()
    output_dir = tmp_path / "output"
    
    tiny_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    dummy_image_path = figs_dir / f"{task.benchmark_id}.png"
    dummy_image_path.write_bytes(tiny_png)

    # 3. Test Posture 1 (Flawless Execution)
    payload_p1 = compile_formative_payload(
        task=task,
        figs_dir=figs_dir,
        output_dir=output_dir,
        posture_type="posture_1"
    )
    
    assert payload_p1["posture_type"] == "posture_1"
    assert "ground_truth" in payload_p1
    assert payload_p1["ground_truth"]["is_work_correct"] is True
    assert payload_p1["ground_truth"]["error_step_number"] is None

    # 4. Test Posture 2 (Flawed Execution via Corrupted Tools)
    payload_p2 = compile_formative_payload(
        task=task,
        figs_dir=figs_dir,
        output_dir=output_dir,
        posture_type="posture_2"
    )
    
    assert payload_p2["posture_type"] == "posture_2"
    assert "ground_truth" in payload_p2
    assert payload_p2["ground_truth"]["is_work_correct"] is False
    assert payload_p2["ground_truth"]["error_step_number"] == 1
    
    # 5. Verify the prompts mapped correctly
    assert "Where did I make a mistake" in payload_p1["system_prompt"]
    assert "Is my work correct?" in payload_p2["system_prompt"]


def test_compile_formative_payload_posture_2_no_error_injection(tmp_path):
    """
    Verifies the edge case where posture_2 is selected, but the task's recipe 
    does not utilize any tools that have corrupted variants (e.g., TRUSS_0004).
    The compiler should dynamically detect that no error was injected and 
    grade the work as correct.
    """
    # 1. Dynamically load TRUSS_0004 (ZFM inspection only, no global reactions)
    current_dir = Path(__file__).parent
    examples_dir = current_dir / "examples"
    example_path = examples_dir / "example_TRUSS_0004.py"
    
    if not example_path.exists():
        pytest.skip("example_TRUSS_0004.py not found in examples directory.")

    spec = importlib.util.spec_from_file_location("example_TRUSS_0004", example_path)
    truss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(truss_module)
    task = BenchmarkTask.from_module(truss_module)

    # 2. Setup mock directories and a dummy image
    figs_dir = tmp_path / "figs"
    figs_dir.mkdir()
    output_dir = tmp_path / "output"
    
    tiny_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    dummy_image_path = figs_dir / f"{task.benchmark_id}.png"
    dummy_image_path.write_bytes(tiny_png)

    # 3. Test Posture 2
    payload = compile_formative_payload(
        task=task,
        figs_dir=figs_dir,
        output_dir=output_dir,
        posture_type="posture_2"
    )
    
    # 4. Assert that because no corrupted tools were called, the work remains correct
    assert payload["posture_type"] == "posture_2"
    assert payload["ground_truth"]["is_work_correct"] is True
    assert payload["ground_truth"]["error_step_number"] is None


def test_task_to_prompt_formative_all(tmp_path, capsys):
    """
    Verifies that the batch iterator crawls the directory and successfully 
    generates JSON payloads for all discovered examples.
    """
    # 1. Resolve paths
    current_dir = Path(__file__).parent
    examples_dir = current_dir / "examples"
    
    if not list(examples_dir.glob("example_TRUSS_*.py")):
        pytest.skip("No examples found")

    figs_dir = tmp_path / "figs"
    figs_dir.mkdir()
    output_dir = tmp_path / "output"
    
    # 2. Write dummy images for all examples to avoid FileNotFoundError
    tiny_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    for ex in examples_dir.glob("example_TRUSS_*.py"):
        spec = importlib.util.spec_from_file_location(ex.stem, ex)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        bid = mod.METADATA["benchmark_id"]
        (figs_dir / f"{bid}.png").write_bytes(tiny_png)

    # 3. Execute the batch iterator
    task_to_prompt_formative_all(
        examples_dir=examples_dir,
        figs_dir=figs_dir,
        output_dir=output_dir,
        posture_type="posture_1"
    )

    # 4. Verify terminal output
    captured = capsys.readouterr()
    assert "Successfully generated" in captured.out
    
    # 5. Verify the files hit the disk
    json_files = list(output_dir.glob("*_formative_posture_1_payload.json"))
    assert len(json_files) > 0


def test_task_to_prompt_formative_all_posture_2(tmp_path, capsys):
    """
    Verifies that the batch iterator crawls the directory and successfully 
    generates JSON payloads for all discovered examples using posture_2.
    It also verifies that the dynamic error tracking correctly identifies 
    which files have injected errors and which do not.
    """
    # 1. Resolve paths
    current_dir = Path(__file__).parent
    examples_dir = current_dir / "examples"
    
    if not list(examples_dir.glob("example_TRUSS_*.py")):
        pytest.skip("No examples found")

    figs_dir = tmp_path / "figs"
    figs_dir.mkdir()
    output_dir = tmp_path / "output"
    
    # 2. Write dummy images for all examples to avoid FileNotFoundError
    tiny_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    for ex in examples_dir.glob("example_TRUSS_*.py"):
        spec = importlib.util.spec_from_file_location(ex.stem, ex)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        bid = mod.METADATA["benchmark_id"]
        (figs_dir / f"{bid}.png").write_bytes(tiny_png)

    # 3. Execute the batch iterator
    task_to_prompt_formative_all(
        examples_dir=examples_dir,
        figs_dir=figs_dir,
        output_dir=output_dir,
        posture_type="posture_2"
    )

    # 4. Verify terminal output
    captured = capsys.readouterr()
    assert "Successfully generated" in captured.out
    
    # 5. Verify the files hit the disk with the posture_2 suffix
    json_files = list(output_dir.glob("*_formative_posture_2_payload.json"))
    assert len(json_files) > 0
    
    # 6. Peek inside TRUSS_0001 to confirm the error payload generated correctly.
    # This task uses global reactions, so it should have an injected error.
    target_file_1 = output_dir / "example_TRUSS_0001_formative_posture_2_payload.json"
    if target_file_1.exists():
        with open(target_file_1, "r") as f:
            payload_data_1 = json.load(f)
            
        assert payload_data_1["posture_type"] == "posture_2"
        assert payload_data_1["ground_truth"]["is_work_correct"] is False
        assert payload_data_1["ground_truth"]["error_step_number"] == 1
        
    # 7. Peek inside TRUSS_0004 to confirm the edge case worked.
    # This task skips global reactions, so no corrupted tools are called, meaning work is correct.
    target_file_4 = output_dir / "example_TRUSS_0004_formative_posture_2_payload.json"
    if target_file_4.exists():
        with open(target_file_4, "r") as f:
            payload_data_4 = json.load(f)
            
        assert payload_data_4["posture_type"] == "posture_2"
        assert payload_data_4["ground_truth"]["is_work_correct"] is True
        assert payload_data_4["ground_truth"]["error_step_number"] is None


def test_compile_formative_payload_posture_3_with_mos(tmp_path):
    """
    Verifies that posture_3 correctly injects a late-stage Tension/Compression
    error when the task utilizes the Method of Sections tool (e.g., TRUSS_0003).
    """
    import importlib.util
    from pathlib import Path
    from trusstutor_bench.task_base import BenchmarkTask
    from trusstutor_bench.task_to_prompt_formative import compile_formative_payload

    # 1. Dynamically load TRUSS_0003 (Method of Sections)
    current_dir = Path(__file__).parent
    examples_dir = current_dir / "examples"
    example_path = examples_dir / "example_TRUSS_0003.py"
    
    if not example_path.exists():
        pytest.skip("example_TRUSS_0003.py not found in examples directory.")

    spec = importlib.util.spec_from_file_location("example_TRUSS_0003", example_path)
    truss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(truss_module)
    task = BenchmarkTask.from_module(truss_module)

    # 2. Setup mock directories and a dummy image for the encoder
    figs_dir = tmp_path / "figs"
    figs_dir.mkdir()
    output_dir = tmp_path / "output"
    
    tiny_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    dummy_image_path = figs_dir / f"{task.benchmark_id}.png"
    dummy_image_path.write_bytes(tiny_png)

    # 3. Compile the payload targeting Posture 3
    payload = compile_formative_payload(
        task=task,
        figs_dir=figs_dir,
        output_dir=output_dir,
        posture_type="posture_3"
    )
    
    # 4. Verify Posture 3 successfully injected the error
    assert payload["posture_type"] == "posture_3"
    assert payload["ground_truth"]["is_work_correct"] is False
    assert payload["ground_truth"]["error_step_number"] is not None
    assert isinstance(payload["ground_truth"]["error_step_number"], int)
    
    # 5. Verify the prompt matches the Posture 3 student query
# 5. Verify the prompt matches the Posture 3 student query
    assert "Is my work correct?" in payload["system_prompt"]


def test_compile_formative_payload_posture_3_with_moj_edge_case(tmp_path):
    """
    Verifies the edge case where posture_3 is selected, but the task's recipe 
    DOES NOT utilize the Method of Sections tool (e.g., TRUSS_0001). 
    The compiler should dynamically detect that no error was injected and 
    grade the work as correct.
    """
    import importlib.util
    from pathlib import Path
    from trusstutor_bench.task_base import BenchmarkTask
    from trusstutor_bench.task_to_prompt_formative import compile_formative_payload

    # 1. Dynamically load TRUSS_0001 (Method of Joints ONLY)
    current_dir = Path(__file__).parent
    examples_dir = current_dir / "examples"
    example_path = examples_dir / "example_TRUSS_0001.py"
    
    if not example_path.exists():
        pytest.skip("example_TRUSS_0001.py not found in examples directory.")

    spec = importlib.util.spec_from_file_location("example_TRUSS_0001", example_path)
    truss_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(truss_module)
    task = BenchmarkTask.from_module(truss_module)

    # 2. Setup mock directories and a dummy image for the encoder
    figs_dir = tmp_path / "figs"
    figs_dir.mkdir()
    output_dir = tmp_path / "output"
    
    tiny_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    dummy_image_path = figs_dir / f"{task.benchmark_id}.png"
    dummy_image_path.write_bytes(tiny_png)

    # 3. Compile the payload targeting Posture 3
    payload = compile_formative_payload(
        task=task,
        figs_dir=figs_dir,
        output_dir=output_dir,
        posture_type="posture_3"
    )
    
    # 4. Assert that because no corrupted Method of Sections tools were called, the work remains correct
    assert payload["posture_type"] == "posture_3"
    assert payload["ground_truth"]["is_work_correct"] is True
    assert payload["ground_truth"]["error_step_number"] is None