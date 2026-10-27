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
    """Numbers match within tolerance; different numbers and text fail."""
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
    """Arrays match regardless of order and fail when elements differ."""
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
    """Strings match ignoring case and whitespace; different units fail."""
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


def test_compare_field_values_member_name_strips_latex_wrapping():
    """Member names ignore '$...$' wrapping and member direction."""
    schema = {"type": "string"}

    # Model wraps the member name in LaTeX math mode ("$AB$") -- semantically
    # identical to the plain "AB" ground truth.
    assert compare_field_values("AB", "$AB$", schema, field_name="controlling_member") is True
    # Direction still shouldn't matter once the LaTeX wrapping is stripped.
    assert compare_field_values("AB", "$BA$", schema, field_name="controlling_member") is True
    # A genuinely different member is still a real failure.
    assert compare_field_values("AB", "$CD$", schema, field_name="controlling_member") is False


def test_compare_field_values_units_tolerates_trailing_explanation():
    """A units answer with extra commentary is accepted when it contains the right unit."""
    schema = {"type": "string"}

    # Model states the correct unit but pads it with harmless commentary,
    # either after it or as a lead-in before it.
    assert compare_field_values("kN", "kN (positive = tension, negative = compression)", schema, field_name="units") is True
    assert compare_field_values("P", "P (multiples of the applied load P)", schema, field_name="units") is True
    assert compare_field_values("P", "Multiples of the applied load P", schema, field_name="units") is True
    assert compare_field_values("kips", "kips (k)", schema, field_name="units") is True
    # lb/lbs are used interchangeably across this dataset's pound-based tasks.
    assert compare_field_values("lbs", "lb", schema, field_name="units") is True
    assert compare_field_values("lb", "lbs", schema, field_name="units") is True
    assert compare_field_values("lb", "pounds", schema, field_name="units") is True
    assert compare_field_values("lbs", "kg", schema, field_name="units") is False

    # A genuinely wrong unit must still fail, even when it's a substring of
    # the other (a real 1000x order-of-magnitude error, not a formatting
    # difference) -- word-boundary matching keeps this safe.
    assert compare_field_values("N", "kN", schema, field_name="units") is False
    assert compare_field_values("kN", "N", schema, field_name="units") is False
    # Unrelated commentary with no real unit match anywhere.
    assert compare_field_values("N", "Cannot be determined from the drawing.", schema, field_name="units") is False

    # This leniency is scoped to the "units" field only -- an ordinary
    # string field with the same text should not get the same pass.
    assert compare_field_values("kN", "kN (positive = tension, negative = compression)", schema) is False


def test_evaluate_single_task_perfect_match():
    """A fully correct answer passes, and the scratchpad is ignored."""
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
    """One wrong field fails the task, while the correct field still passes."""
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
    """A missing field fails with the reason 'Missing key in prediction'."""
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
    Grades the sample payload for TRUSS_0001 against its fixture response in
    tests/llm_outputs/, which is the ground-truth answer.
    """
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    payload_path = project_root / "prompts" / "TRUSS_0001_payload.json"
    response_path = current_dir / "llm_outputs" / "TRUSS_0001_response.json"
    
    if not payload_path.exists() or not response_path.exists():
        pytest.skip(f"Required artifacts for TRUSS_0001 not found. Please run the pipeline first.")
        
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
        
    with open(response_path, "r", encoding="utf-8") as f:
        prediction = json.load(f)
        
    schema = payload["structured_schema"]
    ground_truth = payload["ground_truth"]
    
    result = evaluate_single_task(ground_truth, prediction, schema)
    
    assert isinstance(result, dict)
    assert "passed" in result
    assert isinstance(result["passed"], bool)
    assert "fields" in result
    
    expected_keys = [k for k in schema.get("properties", {}) if k != "chain_of_thought"]
    
    for key in expected_keys:
        assert key in result["fields"], f"Expected field '{key}' missing from scorecard."
        
        field_score = result["fields"][key]
        assert "passed" in field_score
        assert isinstance(field_score["passed"], bool)
        assert "ground_truth" in field_score
        assert "prediction" in field_score
        
    assert "chain_of_thought" not in result["fields"]


def test_generate_minimal_report(tmp_path):
    """Computes the pass count and accuracy and writes the report JSON."""
    mock_results = {
        "TRUSS_0001": {"passed": True, "fields": {}},
        "TRUSS_0002": {"passed": False, "fields": {}},
        "TRUSS_0003": {"passed": True, "fields": {}},
        "TRUSS_0004": {"passed": True, "fields": {}}
    }
    
    report_file = tmp_path / "test_evaluation_report.json"
    
    generate_minimal_report(mock_results, report_file)
    
    assert report_file.exists(), "The report file was not created."
    
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
    """Grades mock payloads and responses end to end and writes the report."""
    prompts_dir = tmp_path / "prompts"
    outputs_dir = tmp_path / "llm_outputs"
    prompts_dir.mkdir()
    outputs_dir.mkdir()
    report_path = tmp_path / "evaluation_report.json"
    
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
        
    mock_prediction = {
        "max_compression": 100.0,
        "units": "kn" # Should pass due to string case-normalization
    }
    with open(outputs_dir / "TRUSS_9999_response.json", "w", encoding="utf-8") as f:
        json.dump(mock_prediction, f)
        
    results = evaluate_outputs_minimal(prompts_dir, outputs_dir, report_path)
    
    assert "TRUSS_9999" in results
    assert results["TRUSS_9999"]["passed"] is True
    
    assert report_path.exists()
    
    with open(report_path, "r", encoding="utf-8") as f:
        saved_report = json.load(f)
        
    assert saved_report["aggregate_metrics"]["total_tasks"] == 1
    assert saved_report["aggregate_metrics"]["total_passed"] == 1
    assert saved_report["aggregate_metrics"]["accuracy_percentage"] == 100.0


def test_evaluate_outputs_minimal_real_examples(tmp_path):
    """
    Runs the evaluation orchestrator on the sample payloads in prompts/ and the
    fixture responses in tests/llm_outputs/ (ground-truth answers).
    """
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    
    prompts_dir = project_root / "prompts"
    outputs_dir = current_dir / "llm_outputs"
    
    report_path = tmp_path / "test_real_evaluation_report.json"
    
    if not prompts_dir.exists() or not outputs_dir.exists():
        pytest.skip("Real prompt or output directories not found. Please run the pipeline first.")
        
    payload_files = list(prompts_dir.glob("*_payload.json"))
    response_files = list(outputs_dir.glob("*_response.json"))
    
    if not payload_files or not response_files:
        pytest.skip("Real payloads or responses are missing. Please run the pipeline first.")
        
    results = evaluate_outputs_minimal(prompts_dir, outputs_dir, report_path)
    
    assert isinstance(results, dict)
    assert len(results) > 0, "No tasks were evaluated."
    
    # Ensure at least TRUSS_0001 was processed (as a known benchmark ID)
    assert "TRUSS_0001" in results
    assert "passed" in results["TRUSS_0001"]
    assert "fields" in results["TRUSS_0001"]
    
    assert report_path.exists(), "Evaluation report was not created."
    
    with open(report_path, "r", encoding="utf-8") as f:
        saved_report = json.load(f)
        
    assert "aggregate_metrics" in saved_report
    assert "detailed_results" in saved_report
    
    metrics = saved_report["aggregate_metrics"]
    assert metrics["total_tasks"] == len(results)
    assert "total_passed" in metrics
    assert "accuracy_percentage" in metrics


def test_evaluate_formative_task():
    """Grades both the correctness judgement and the error step number."""
    # Define Ground Truth states
    gt_flawless = {"is_work_correct": True, "error_step_number": None}
    gt_flawed = {"is_work_correct": False, "error_step_number": 2}

    pred_flawless = {"is_work_correct": True, "error_step_number": None}
    res_tp = evaluate_formative_task(pred_flawless, gt_flawless)
    assert res_tp["passed"] is True
    
    pred_hallucinated_error = {"is_work_correct": False, "error_step_number": 2}
    res_fn = evaluate_formative_task(pred_hallucinated_error, gt_flawless)
    assert res_fn["passed"] is False

    pred_correct_detection = {"is_work_correct": False, "error_step_number": 2}
    res_tn = evaluate_formative_task(pred_correct_detection, gt_flawed)
    assert res_tn["passed"] is True

    res_fp = evaluate_formative_task(pred_flawless, gt_flawed)
    assert res_fp["passed"] is False
    
    pred_wrong_step = {"is_work_correct": False, "error_step_number": 3}
    res_loc_fail = evaluate_formative_task(pred_wrong_step, gt_flawed)
    assert res_loc_fail["passed"] is False


def test_evaluate_outputs_formative_integration(tmp_path):
    """Grades mock formative payloads and responses for one posture."""
    prompts_dir = tmp_path / "prompts_formative"
    outputs_dir = tmp_path / "llm_outputs_formative"
    prompts_dir.mkdir()
    outputs_dir.mkdir()
    report_path = tmp_path / "formative_evaluation_report.json"
    
    benchmark_id = "TRUSS_0001"
    
    mock_payload = {
        "benchmark_id": benchmark_id,
        "posture_type": "posture_1",
        "ground_truth": {
            "is_work_correct": True,
            "error_step_number": None
        }
    }
    with open(prompts_dir / f"{benchmark_id}_payload.json", "w", encoding="utf-8") as f:
        json.dump(mock_payload, f)
        
    mock_prediction = {
        "is_work_correct": True,
        "error_step_number": None,
        "explanation_to_student": "Great job!"
    }
    with open(outputs_dir / f"{benchmark_id}_response.json", "w", encoding="utf-8") as f:
        json.dump(mock_prediction, f)
        
    results = evaluate_outputs_formative(prompts_dir, outputs_dir, "posture_1", report_path)
    
    assert benchmark_id in results
    assert results[benchmark_id]["passed"] is True
    
    assert report_path.exists()
    
    with open(report_path, "r", encoding="utf-8") as f:
        saved_report = json.load(f)
        
    assert saved_report["aggregate_metrics"]["total_tasks"] == 1
    assert saved_report["aggregate_metrics"]["total_passed"] == 1


def test_cross_evaluate_2x2_matrix(tmp_path):
    """Matches task IDs across both reports and sorts them into the four quadrants."""
    baseline_report_path = tmp_path / "baseline_report.json"
    formative_report_path = tmp_path / "formative_report.json"
    
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
        
    matrix = cross_evaluate_2x2_matrix(baseline_report_path, formative_report_path)
    
    assert "TRUSS_0001" in matrix["baseline_pass_formative_pass"]
    assert len(matrix["baseline_pass_formative_pass"]) == 1
    
    assert "TRUSS_0002" in matrix["baseline_pass_formative_fail"]
    assert len(matrix["baseline_pass_formative_fail"]) == 1
    
    assert "TRUSS_0003" in matrix["baseline_fail_formative_pass"]
    assert len(matrix["baseline_fail_formative_pass"]) == 1
    
    assert "TRUSS_0004" in matrix["baseline_fail_formative_fail"]
    assert len(matrix["baseline_fail_formative_fail"]) == 1
    
    all_matrix_ids = sum(matrix.values(), [])
    assert "TRUSS_0005" not in all_matrix_ids
    assert "TRUSS_0006" not in all_matrix_ids


def test_print_2x2_summary(capsys):
    """Prints the 2x2 table without crashing."""
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
