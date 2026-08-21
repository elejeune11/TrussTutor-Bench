import math
import json
from pathlib import Path


def _normalize_member_id(member_id: str) -> str:
    """
    Canonicalizes a member/edge name so direction doesn't matter (a truss
    member is undirected: "BG" and "GB" name the same physical member).
    Handles both naming conventions used across the dataset: two-letter
    joint pairs ("AB", "BG") and dash-separated numeric joints ("9-8").
    Falls back to a plain lowercase string for anything else, so this is
    safe to apply even when a value isn't actually a two-endpoint name.
    """
    s = member_id.strip()
    if "-" in s:
        parts = s.split("-")
    elif len(s) == 2 and s.isalpha():
        parts = list(s)
    else:
        return s.lower()
    return "-".join(sorted(p.strip().lower() for p in parts))


def _is_member_name_field(field_name: str) -> bool:
    """Field-naming convention in this dataset: member-identifier fields are
    always named "*member*" (e.g. "controlling_member", "zero_force_members")."""
    return field_name is not None and "member" in field_name.lower()


def compare_field_values(truth_val, pred_val, field_schema: dict, rtol=1e-2, atol=1e-3, field_name: str = None) -> bool:
    """
    Compares a single ground-truth value against an LLM prediction based on the expected JSON schema type.
    """
    expected_type = field_schema.get("type")

    # 1. Numerical Comparison
    if expected_type in ["number", "float", "integer"] or isinstance(truth_val, (int, float)):
        try:
            return math.isclose(float(truth_val), float(pred_val), rel_tol=rtol, abs_tol=atol)
        except (ValueError, TypeError):
            return False

    # 2. Array / Set Comparison
    if expected_type == "array" or isinstance(truth_val, list):
        if not isinstance(pred_val, list):
            return False
        if _is_member_name_field(field_name):
            return {_normalize_member_id(v) for v in truth_val} == {_normalize_member_id(v) for v in pred_val}
        return set(truth_val) == set(pred_val)

    # 3. String / Enum Comparison
    if expected_type == "string" or isinstance(truth_val, str):
        if not isinstance(pred_val, str):
            return False
        if _is_member_name_field(field_name):
            return _normalize_member_id(truth_val) == _normalize_member_id(pred_val)
        return str(truth_val).strip().lower() == str(pred_val).strip().lower()

    # 4. Object / Dictionary Comparison
    if expected_type == "object" or isinstance(truth_val, dict):
        if not isinstance(pred_val, dict):
            return False

        properties = field_schema.get("properties", {})

        for key, true_sub_val in truth_val.items():
            pred_sub_val = pred_val.get(key)
            if pred_sub_val is None:
                return False

            # Fetch the nested schema, or default to an empty dict to rely on Python type inference
            sub_schema = properties.get(key, {})

            # Recursively check the nested values
            if not compare_field_values(true_sub_val, pred_sub_val, sub_schema, rtol, atol, field_name=key):
                return False
        return True

    # 5. Fallback for booleans or exact programmatic matches
    return truth_val == pred_val


def evaluate_single_task(ground_truth: dict, prediction: dict, schema: dict) -> dict:
    """
    Grades an LLM output against ground truth using the task's expected schema.
    """
    properties = schema.get("properties", {})
    field_results = {}
    all_passed = True

    for field, field_schema in properties.items():
        # Skip internal reasoning scratchpads since they don't have a direct ground truth equivalent
        if field == "chain_of_thought":
            continue

        truth_val = ground_truth.get(field)
        pred_val = prediction.get(field)

        # Handle cases where the LLM completely forgot to include a required key
        if pred_val is None:
            field_results[field] = {
                "passed": False, 
                "reason": "Missing key in prediction",
                "ground_truth": truth_val,
                "prediction": None
            }
            all_passed = False
            continue

        # Evaluate the specific field using the type-aware comparator
        passed = compare_field_values(truth_val, pred_val, field_schema, field_name=field)
        
        field_results[field] = {
            "passed": passed,
            "ground_truth": truth_val,
            "prediction": pred_val
        }
        
        if not passed:
            all_passed = False

    return {
        "passed": all_passed,
        "fields": field_results
    }


def generate_minimal_report(overall_results: dict, report_path: Path) -> None:
    """
    Calculates basic aggregate metrics, prints a terminal summary, 
    and saves the complete evaluation report to disk.
    """
    total_tasks = len(overall_results)
    total_passed = sum(1 for result in overall_results.values() if result.get("passed", False))
    
    accuracy = (total_passed / total_tasks * 100) if total_tasks > 0 else 0.0
    
    final_report = {
        "aggregate_metrics": {
            "total_tasks": total_tasks,
            "total_passed": total_passed,
            "accuracy_percentage": round(accuracy, 2)
        },
        "detailed_results": overall_results
    }
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)
        
    print(f"Total Tasks:  {total_tasks}")
    print(f"Total Passed: {total_passed}")
    print(f"Accuracy:     {accuracy:.2f}%")
    print(f"Saved to:     {report_path.name}\n")


def evaluate_outputs_minimal(prompts_dir: Path, outputs_dir: Path, report_path: Path) -> dict:
    """
    Orchestrates the evaluation of all LLM outputs against their corresponding payloads.
    """
    print("=== Starting TrussTutor-Bench Evaluation ===")
    
    if not outputs_dir.exists():
        print(f"[ERROR] Outputs directory not found at {outputs_dir}")
        return {}
        
    response_files = list(outputs_dir.glob("*_response.json"))
    if not response_files:
        print("[ERROR] No response files found to evaluate.")
        return {}
        
    overall_results = {}
    
    for response_file in response_files:
        benchmark_id = response_file.stem.replace("_response", "")
        payload_file = prompts_dir / f"{benchmark_id}_payload.json"
        
        if not payload_file.exists():
            print(f"[WARNING] Missing payload for {benchmark_id}. Skipping.")
            continue
            
        with open(payload_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
            
        with open(response_file, "r", encoding="utf-8") as f:
            prediction = json.load(f)
            
        schema = payload["structured_schema"]
        ground_truth = payload["ground_truth"]
        
        overall_results[benchmark_id] = evaluate_single_task(ground_truth, prediction, schema)
        
    generate_minimal_report(overall_results, report_path)
    
    return overall_results


def evaluate_formative_task(prediction: dict, ground_truth: dict) -> dict:
    """
    Grades a single formative tutoring response against the expected ground truth.
    Evaluates both error identification and step localization.
    """
    expected_is_correct = ground_truth.get("is_work_correct")
    expected_error_step = ground_truth.get("error_step_number")
    
    pred_correct = prediction.get("is_work_correct")
    pred_error_step = prediction.get("error_step_number")
    
    passed_correctness = (pred_correct is expected_is_correct)
    passed_step = (pred_error_step == expected_error_step)
    
    passed = passed_correctness and passed_step
    
    return {
        "passed": passed,
        "prediction": prediction,
        "ground_truth": ground_truth
    }


def evaluate_outputs_formative(prompts_dir: Path, outputs_dir: Path, target_posture: str, report_path: Path) -> dict:
    """
    Orchestrates the evaluation of formative LLM outputs for a specific posture.
    For flawed postures (2 and 3), it skips examples where the ground truth is flawless to isolate error detection.
    """
    print(f"\n=== Starting Formative Evaluation for {target_posture} ===")
    
    if not outputs_dir.exists() or not list(outputs_dir.glob("*_response.json")):
        print(f"[ERROR] No formative responses found in {outputs_dir}")
        return {}
        
    overall_results = {}
    
    for response_file in outputs_dir.glob("*_response.json"):
        benchmark_id = response_file.stem.replace("_response", "")
        payload_file = prompts_dir / f"{benchmark_id}_payload.json"
        
        if not payload_file.exists():
            continue
            
        with open(payload_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
            
        # --- Filter by the requested posture ---
        if payload.get("posture_type") != target_posture:
            continue
            
        ground_truth = payload.get("ground_truth", {})
        
        # --- Posture-specific skipping ---
        # If we are evaluating a flawed posture (2 or 3), strictly isolate error detection 
        # by skipping tasks where no error was successfully injected.
        if target_posture in ["posture_2", "posture_3"] and ground_truth.get("is_work_correct") is True:
            continue
            
        with open(response_file, "r", encoding="utf-8") as f:
            prediction = json.load(f)
            
        overall_results[benchmark_id] = evaluate_formative_task(prediction, ground_truth)
        
    generate_minimal_report(overall_results, report_path)
    
    return overall_results


def print_2x2_summary(matrix_data: dict) -> None:
    """
    Prints a clean 2x2 contingency table to the terminal.
    """
    print("\n" + "="*50)
    print(" 2x2 CROSS-EVALUATION MATRIX: Baseline vs Formative")
    print("="*50)
    print(f"{'':<20} | {'Formative CORRECT':<20} | {'Formative INCORRECT':<20}")
    print("-" * 67)
    
    # Baseline Correct Row
    b_pass_f_pass = len(matrix_data["baseline_pass_formative_pass"])
    b_pass_f_fail = len(matrix_data["baseline_pass_formative_fail"])
    print(f"{'Baseline CORRECT':<20} | {b_pass_f_pass:<20} | {b_pass_f_fail:<20}")
    
    # Baseline Incorrect Row
    b_fail_f_pass = len(matrix_data["baseline_fail_formative_pass"])
    b_fail_f_fail = len(matrix_data["baseline_fail_formative_fail"])
    print(f"{'Baseline INCORRECT':<20} | {b_fail_f_pass:<20} | {b_fail_f_fail:<20}")
    print("-" * 67)
    print(f"Total Overlapping Tasks Evaluated: {b_pass_f_pass + b_pass_f_fail + b_fail_f_pass + b_fail_f_fail}\n")


def cross_evaluate_2x2_matrix(baseline_report_path: Path, formative_report_path: Path) -> dict:
    """
    Loads both baseline and formative reports, matches tasks by ID, 
    and categorizes them into a 2x2 matrix.
    """
    if not baseline_report_path.exists() or not formative_report_path.exists():
        print("[ERROR] Both baseline and formative reports must exist to generate the 2x2 matrix.")
        return {}

    with open(baseline_report_path, "r") as f:
        baseline_data = json.load(f).get("detailed_results", {})
        
    with open(formative_report_path, "r") as f:
        formative_data = json.load(f).get("detailed_results", {})

    matrix = {
        "baseline_pass_formative_pass": [],
        "baseline_pass_formative_fail": [],
        "baseline_fail_formative_pass": [],
        "baseline_fail_formative_fail": []
    }

    # Find the overlapping benchmark IDs
    # (Formative IDs have "_formative_posture_1" appended, so we need to strip it to match)
    for form_id, form_result in formative_data.items():
        base_id = form_id.split("_formative_")[0]
        
        if base_id in baseline_data:
            b_pass = baseline_data[base_id].get("passed", False)
            f_pass = form_result.get("passed", False)
            
            if b_pass and f_pass:
                matrix["baseline_pass_formative_pass"].append(base_id)
            elif b_pass and not f_pass:
                matrix["baseline_pass_formative_fail"].append(base_id)
            elif not b_pass and f_pass:
                matrix["baseline_fail_formative_pass"].append(base_id)
            else:
                matrix["baseline_fail_formative_fail"].append(base_id)

    print_2x2_summary(matrix)
    return matrix