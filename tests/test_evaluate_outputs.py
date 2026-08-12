from trusstutor_bench.evaluate_outputs import compare_field_values  # Adjust import based on where you put it
from trusstutor_bench.evaluate_outputs import evaluate_single_task
from trusstutor_bench.evaluate_outputs import generate_minimal_report
from trusstutor_bench.evaluate_outputs import evaluate_outputs_minimal
from pathlib import Path
import pytest
import json

from trusstutor_bench.evaluate_outputs import (
    evaluate_formative_task,
    evaluate_outputs_formative,
    cross_evaluate_2x2_matrix,
    print_2x2_summary
)


def test_compare_field_values_numbers():
    schema = {"type": "number"}
    
    # Exact match
    assert compare_field_values(50.0, 50.0, schema) is True
    # Within absolute tolerance (default atol=1e-3)
    assert compare_field_values(50.0, 50.0001, schema) is True
    # Outside tolerance
    assert compare_field_values(50.0, 60.0, schema) is False
    # Type mismatch (LLM hallucinates text)
    assert compare_field_values(50.0, "fifty", schema) is False


def test_compare_field_values_arrays():
    schema = {"type": "array"}
    
    # Exact match, same order
    assert compare_field_values(["AB", "BC"], ["AB", "BC"], schema) is True
    # Exact match, different order (valid for sets)
    assert compare_field_values(["AB", "BC"], ["BC", "AB"], schema) is True
    # Missing items
    assert compare_field_values(["AB", "BC"], ["AB"], schema) is False
    # Hallucinated extra items
    assert compare_field_values(["AB"], ["AB", "BC"], schema) is False
    # Type mismatch
    assert compare_field_values(["AB"], "AB", schema) is False


def test_compare_field_values_strings():
    schema = {"type": "string"}
    
    # Exact match
    assert compare_field_values("kN", "kN", schema) is True
    # Case normalization
    assert compare_field_values("kN", "kn", schema) is True
    # Whitespace normalization
    assert compare_field_values("kN", " kN ", schema) is True
    # Completely wrong string
    assert compare_field_values("kN", "N", schema) is False
    # Type mismatch
    assert compare_field_values("kN", 100, schema) is False


def test_evaluate_single_task_perfect_match():
    schema = {
        "properties": {
            "chain_of_thought": {"type": "string"},
            "max_compression": {"type": "number"},
            "units": {"type": "string"}
        }
    }
    ground_truth = {"max_compression": 150.5, "units": "kN"}
    prediction = {
        "chain_of_thought": "Sum of forces in Y is zero...",
        "max_compression": 150.5001,  # Within default tolerance
        "units": "kn"                 # Case-insensitive match
    }
    
    result = evaluate_single_task(ground_truth, prediction, schema)
    
    assert result["passed"] is True
    assert "max_compression" in result["fields"]
    assert "units" in result["fields"]
    assert "chain_of_thought" not in result["fields"]


def test_evaluate_single_task_partial_failure():
    schema = {
        "properties": {
            "max_compression": {"type": "number"},
            "units": {"type": "string"}
        }
    }
    ground_truth = {"max_compression": 150.5, "units": "kN"}
    prediction = {
        "max_compression": 999.9,  # Incorrect math
        "units": "kN"
    }
    
    result = evaluate_single_task(ground_truth, prediction, schema)
    
    assert result["passed"] is False
    assert result["fields"]["max_compression"]["passed"] is False
    assert result["fields"]["units"]["passed"] is True


def test_evaluate_single_task_missing_key():
    schema = {
        "properties": {
            "max_compression": {"type": "number"},
            "units": {"type": "string"}
        }
    }
    ground_truth = {"max_compression": 150.5, "units": "kN"}
    prediction = {
        "max_compression": 150.5
        # "units" key is completely missing
    }
    
    result = evaluate_single_task(ground_truth, prediction, schema)
    
    assert result["passed"] is False
    assert result["fields"]["max_compression"]["passed"] is True
    assert result["fields"]["units"]["passed"] is False
    assert result["fields"]["units"]["reason"] == "Missing key in prediction"


def test_evaluate_single_task_real_example():
    """
    Tests the evaluation logic using the actual payload and LLM response 
    generated for TRUSS_0001.
    """
    # 1. Resolve paths based on the project structure
    # Assuming this test file is located inside the 'tests' directory
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    payload_path = project_root / "prompts" / "TRUSS_0001_payload.json"
    response_path = current_dir / "llm_outputs" / "TRUSS_0001_response.json"
    
    # 2. Skip the test gracefully if the pipeline hasn't been executed yet
    if not payload_path.exists() or not response_path.exists():
        pytest.skip(f"Required artifacts for TRUSS_0001 not found. Please run the pipeline first.")
        
    # 3. Load the artifacts from disk
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
        
    with open(response_path, "r", encoding="utf-8") as f:
        prediction = json.load(f)
        
    schema = payload["structured_schema"]
    ground_truth = payload["ground_truth"]
    
    # 4. Execute the evaluator
    result = evaluate_single_task(ground_truth, prediction, schema)
    
    # 5. Verify the scorecard structure
    assert isinstance(result, dict)
    assert "passed" in result
    assert isinstance(result["passed"], bool)
    assert "fields" in result
    
    # 6. Ensure every expected field (except the scratchpad) was graded
    expected_keys = [k for k in schema.get("properties", {}) if k != "chain_of_thought"]
    
    for key in expected_keys:
        assert key in result["fields"], f"Expected field '{key}' missing from scorecard."
        
        field_score = result["fields"][key]
        assert "passed" in field_score
        assert isinstance(field_score["passed"], bool)
        assert "ground_truth" in field_score
        assert "prediction" in field_score
        
    # 7. Ensure the internal reasoning scratchpad was properly ignored
    assert "chain_of_thought" not in result["fields"]


def test_generate_minimal_report(tmp_path):
    """
    Tests that the minimal report generator correctly calculates aggregates
    and writes the expected JSON structure to disk.
    """
    # 1. Setup mock evaluation results (3 passed, 1 failed)
    mock_results = {
        "TRUSS_0001": {"passed": True, "fields": {}},
        "TRUSS_0002": {"passed": False, "fields": {}},
        "TRUSS_0003": {"passed": True, "fields": {}},
        "TRUSS_0004": {"passed": True, "fields": {}}
    }
    
    # 2. Define a temporary file path
    report_file = tmp_path / "test_evaluation_report.json"
    
    # 3. Execute the function
    generate_minimal_report(mock_results, report_file)
    
    # 4. Assert the file was successfully created
    assert report_file.exists(), "The report file was not created."
    
    # 5. Read the file back and verify the data
    with open(report_file, "r", encoding="utf-8") as f:
        saved_report = json.load(f)
        
    # Verify the top-level structure
    assert "aggregate_metrics" in saved_report
    assert "detailed_results" in saved_report
    
    # Verify the math
    metrics = saved_report["aggregate_metrics"]
    assert metrics["total_tasks"] == 4
    assert metrics["total_passed"] == 3
    assert metrics["accuracy_percentage"] == 75.0
    
    # Verify the original payload passed through
    assert "TRUSS_0001" in saved_report["detailed_results"]
    assert saved_report["detailed_results"]["TRUSS_0002"]["passed"] is False


def test_evaluate_outputs_minimal_integration(tmp_path):
    """
    Tests the full evaluation orchestrator by generating mock directories, 
    payloads, and responses, and verifying the pipeline executes end-to-end.
    """
    # 1. Setup mock directories inside the temporary Pytest environment
    prompts_dir = tmp_path / "prompts"
    outputs_dir = tmp_path / "llm_outputs"
    prompts_dir.mkdir()
    outputs_dir.mkdir()
    report_path = tmp_path / "evaluation_report.json"
    
    # 2. Create a mock payload with embedded ground truth
    mock_payload = {
        "benchmark_id": "TRUSS_9999",
        "structured_schema": {
            "properties": {
                "max_compression": {"type": "number"},
                "units": {"type": "string"}
            }
        },
        "ground_truth": {
            "max_compression": 100.0,
            "units": "kN"
        }
    }
    with open(prompts_dir / "TRUSS_9999_payload.json", "w", encoding="utf-8") as f:
        json.dump(mock_payload, f)
        
    # 3. Create a mock LLM prediction
    mock_prediction = {
        "max_compression": 100.0,
        "units": "kn" # Should pass due to string case-normalization
    }
    with open(outputs_dir / "TRUSS_9999_response.json", "w", encoding="utf-8") as f:
        json.dump(mock_prediction, f)
        
    # 4. Execute the orchestrator
    results = evaluate_outputs_minimal(prompts_dir, outputs_dir, report_path)
    
    # 5. Assert the return object contains the graded task
    assert "TRUSS_9999" in results
    assert results["TRUSS_9999"]["passed"] is True
    
    # 6. Assert the report generator was successfully triggered and wrote the file
    assert report_path.exists()
    
    with open(report_path, "r", encoding="utf-8") as f:
        saved_report = json.load(f)
        
    assert saved_report["aggregate_metrics"]["total_tasks"] == 1
    assert saved_report["aggregate_metrics"]["total_passed"] == 1
    assert saved_report["aggregate_metrics"]["accuracy_percentage"] == 100.0


def test_evaluate_outputs_minimal_real_examples(tmp_path):
    """
    Tests the full evaluation orchestrator using the actual generated payloads 
    and LLM responses located in the project repository.
    """
    # 1. Resolve paths to the real project directories
    # Assuming this test file is located inside the 'tests' directory
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    prompts_dir = project_root / "prompts"
    outputs_dir = current_dir / "llm_outputs"
    
    # 2. Output the report to a temporary directory to avoid overwriting production data
    report_path = tmp_path / "test_real_evaluation_report.json"
    
    # 3. Skip gracefully if the user hasn't generated the real pipeline artifacts yet
    if not prompts_dir.exists() or not outputs_dir.exists():
        pytest.skip("Real prompt or output directories not found. Please run the pipeline first.")
        
    payload_files = list(prompts_dir.glob("*_payload.json"))
    response_files = list(outputs_dir.glob("*_response.json"))
    
    if not payload_files or not response_files:
        pytest.skip("Real payloads or responses are missing. Please run the pipeline first.")
        
    # 4. Execute the orchestrator
    results = evaluate_outputs_minimal(prompts_dir, outputs_dir, report_path)
    
    # 5. Basic assertions on the returned results dictionary
    assert isinstance(results, dict)
    assert len(results) > 0, "No tasks were evaluated."
    
    # Ensure at least TRUSS_0001 was processed (as a known benchmark ID)
    assert "TRUSS_0001" in results
    assert "passed" in results["TRUSS_0001"]
    assert "fields" in results["TRUSS_0001"]
    
    # 6. Assert the JSON report was successfully generated on disk
    assert report_path.exists(), "Evaluation report was not created."
    
    # 7. Validate the structure of the saved report
    with open(report_path, "r", encoding="utf-8") as f:
        saved_report = json.load(f)
        
    assert "aggregate_metrics" in saved_report
    assert "detailed_results" in saved_report
    
    metrics = saved_report["aggregate_metrics"]
    assert metrics["total_tasks"] == len(results)
    assert "total_passed" in metrics
    assert "accuracy_percentage" in metrics


def test_evaluate_formative_task():
    """
    Validates that the formative task evaluator correctly grades the LLM's 
    is_work_correct prediction against the expected reality.
    """
    # 1. Test True Positive (Model correctly identifies flawless work)
    pred_true = {"is_work_correct": True, "error_step_number": None}
    res_true = evaluate_formative_task(pred_true, is_student_work_flawless=True)
    assert res_true["passed"] is True
    
    # 2. Test False Negative (Model hallucinated an error in flawless work)
    pred_false_neg = {"is_work_correct": False, "error_step_number": 2}
    res_false_neg = evaluate_formative_task(pred_false_neg, is_student_work_flawless=True)
    assert res_false_neg["passed"] is False

    # 3. Test True Negative (Model correctly identifies flawed work)
    res_true_neg = evaluate_formative_task(pred_false_neg, is_student_work_flawless=False)
    assert res_true_neg["passed"] is True

    # 4. Test False Positive (Model rubber-stamped flawed work)
    res_false_pos = evaluate_formative_task(pred_true, is_student_work_flawless=False)
    assert res_false_pos["passed"] is False


def test_evaluate_outputs_formative_integration(tmp_path):
    """
    Tests the formative orchestrator using mock directories and responses.
    """
    # 1. Setup mock directories
    prompts_dir = tmp_path / "prompts_formative"
    outputs_dir = tmp_path / "llm_outputs_formative"
    prompts_dir.mkdir()
    outputs_dir.mkdir()
    report_path = tmp_path / "formative_evaluation_report.json"
    
    # 2. Create a mock LLM prediction
    mock_prediction = {
        "is_work_correct": True,
        "error_step_number": None,
        "explanation_to_student": "Great job!"
    }
    with open(outputs_dir / "TRUSS_0001_formative_posture_1_response.json", "w", encoding="utf-8") as f:
        json.dump(mock_prediction, f)
        
    # 3. Execute the orchestrator
    results = evaluate_outputs_formative(prompts_dir, outputs_dir, report_path)
    
    # 4. Assert the return object contains the graded task
    assert "TRUSS_0001_formative_posture_1" in results
    assert results["TRUSS_0001_formative_posture_1"]["passed"] is True
    
    # 5. Assert the report was generated
    assert report_path.exists()
    
    with open(report_path, "r", encoding="utf-8") as f:
        saved_report = json.load(f)
        
    assert saved_report["aggregate_metrics"]["total_tasks"] == 1
    assert saved_report["aggregate_metrics"]["total_passed"] == 1


def test_cross_evaluate_2x2_matrix(tmp_path):
    """
    Verifies that the cross-evaluator correctly matches benchmark IDs across 
    both reports and accurately sorts them into the 4 quadrants of the 2x2 matrix.
    """
    baseline_report_path = tmp_path / "baseline_report.json"
    formative_report_path = tmp_path / "formative_report.json"
    
    # 1. Setup Mock Baseline Report
    baseline_data = {
        "detailed_results": {
            "TRUSS_0001": {"passed": True},   # Will be Baseline Pass / Formative Pass
            "TRUSS_0002": {"passed": True},   # Will be Baseline Pass / Formative Fail
            "TRUSS_0003": {"passed": False},  # Will be Baseline Fail / Formative Pass
            "TRUSS_0004": {"passed": False},  # Will be Baseline Fail / Formative Fail
            "TRUSS_0005": {"passed": True}    # Will be ignored (no formative counterpart)
        }
    }
    with open(baseline_report_path, "w") as f:
        json.dump(baseline_data, f)
        
    # 2. Setup Mock Formative Report
    formative_data = {
        "detailed_results": {
            "TRUSS_0001_formative_posture_1": {"passed": True},
            "TRUSS_0002_formative_posture_1": {"passed": False},
            "TRUSS_0003_formative_posture_1": {"passed": True},
            "TRUSS_0004_formative_posture_1": {"passed": False},
            "TRUSS_0006_formative_posture_1": {"passed": True} # Will be ignored (no baseline counterpart)
        }
    }
    with open(formative_report_path, "w") as f:
        json.dump(formative_data, f)
        
    # 3. Execute Cross-Evaluation
    matrix = cross_evaluate_2x2_matrix(baseline_report_path, formative_report_path)
    
    # 4. Verify Quadrants
    assert "TRUSS_0001" in matrix["baseline_pass_formative_pass"]
    assert len(matrix["baseline_pass_formative_pass"]) == 1
    
    assert "TRUSS_0002" in matrix["baseline_pass_formative_fail"]
    assert len(matrix["baseline_pass_formative_fail"]) == 1
    
    assert "TRUSS_0003" in matrix["baseline_fail_formative_pass"]
    assert len(matrix["baseline_fail_formative_pass"]) == 1
    
    assert "TRUSS_0004" in matrix["baseline_fail_formative_fail"]
    assert len(matrix["baseline_fail_formative_fail"]) == 1
    
    # 5. Verify un-matched tasks are cleanly dropped
    all_matrix_ids = sum(matrix.values(), [])
    assert "TRUSS_0005" not in all_matrix_ids
    assert "TRUSS_0006" not in all_matrix_ids


def test_print_2x2_summary(capsys):
    """
    Verifies that the formatting function successfully prints the 2x2 table 
    to the terminal without crashing.
    """
    mock_matrix = {
        "baseline_pass_formative_pass": ["1", "2"],
        "baseline_pass_formative_fail": ["3"],
        "baseline_fail_formative_pass": ["4", "5", "6"],
        "baseline_fail_formative_fail": []
    }
    
    print_2x2_summary(mock_matrix)
    
    captured = capsys.readouterr()
    
    # Verify key headers and table structure are in the output
    assert "2x2 CROSS-EVALUATION MATRIX" in captured.out
    assert "Baseline CORRECT" in captured.out
    assert "Baseline INCORRECT" in captured.out
    
    # Verify the math was summed correctly in the terminal output
    assert "Total Overlapping Tasks Evaluated: 6" in captured.out