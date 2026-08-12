import math
import json
from pathlib import Path


def compare_field_values(truth_val, pred_val, field_schema: dict, rtol=1e-2, atol=1e-3) -> bool:
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
        return set(truth_val) == set(pred_val)

    # 3. String / Enum Comparison
    if expected_type == "string" or isinstance(truth_val, str):
        if not isinstance(pred_val, str):
            return False
        return str(truth_val).strip().lower() == str(pred_val).strip().lower()

    # 4. Object / Dictionary Comparison (The Fix)
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
            if not compare_field_values(true_sub_val, pred_sub_val, sub_schema, rtol, atol):
                return False
        return True

    # 5. Fallback for booleans or exact programmatic matches
    return truth_val == pred_val


def evaluate_single_task(ground_truth: dict, prediction: dict, schema: dict) -> dict:
    """
    Grades an LLM output against ground truth using the task's expected schema.
    
    Args:
        ground_truth (dict): The canonical answers embedded in the payload.
        prediction (dict): The parsed JSON output from the LLM.
        schema (dict): The expected_llm_output schema definition.
        
    Returns:
        dict: A scorecard containing an overall pass/fail status and field-level details.
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
        passed = compare_field_values(truth_val, pred_val, field_schema)
        
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
    
    Args:
        overall_results (dict): The dictionary of evaluated tasks from evaluate_single_task.
        report_path (Path): The file path where the JSON report should be saved.
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
        
    print("\n=== Minimal Evaluation Report ===")
    print(f"Total Tasks:  {total_tasks}")
    print(f"Total Passed: {total_passed}")
    print(f"Accuracy:     {accuracy:.2f}%")
    print(f"Saved to:     {report_path.name}\n")


def evaluate_outputs_minimal(prompts_dir: Path, outputs_dir: Path, report_path: Path) -> dict:
    """
    Orchestrates the evaluation of all LLM outputs against their corresponding payloads.
    
    Args:
        prompts_dir (Path): Directory containing the generated *_payload.json files.
        outputs_dir (Path): Directory containing the LLM *_response.json files.
        report_path (Path): File path where the final JSON report should be saved.
        
    Returns:
        dict: The overall results mapped by benchmark ID.
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
        
        # Evaluate the single task and store the scorecard
        overall_results[benchmark_id] = evaluate_single_task(ground_truth, prediction, schema)
        
    # Generate the final aggregate report
    generate_minimal_report(overall_results, report_path)
    
    return overall_results


def evaluate_formative_task(prediction: dict, is_student_work_flawless: bool = True) -> dict:
    """
    Grades a single formative tutoring response.
    Currently defaults to expecting flawless student work (is_work_correct == True) 
    until the state corruptor is implemented.
    """
    pred_correct = prediction.get("is_work_correct")
    
    # Did the model correctly evaluate the student's work?
    passed = (pred_correct is is_student_work_flawless)
    
    return {
        "passed": passed,
        "prediction": prediction,
        "expected_is_work_correct": is_student_work_flawless
    }


def evaluate_outputs_formative(prompts_dir: Path, outputs_dir: Path, report_path: Path) -> dict:
    """
    Orchestrates the evaluation of all formative LLM outputs.
    """
    print("=== Starting Formative Evaluation ===")
    
    if not outputs_dir.exists() or not list(outputs_dir.glob("*_response.json")):
        print(f"[ERROR] No formative responses found in {outputs_dir}")
        return {}
        
    overall_results = {}
    
    for response_file in outputs_dir.glob("*_response.json"):
        benchmark_id = response_file.stem.replace("_response", "")
        
        with open(response_file, "r", encoding="utf-8") as f:
            prediction = json.load(f)
            
        # Grade the task (assuming perfect student work for now)
        overall_results[benchmark_id] = evaluate_formative_task(prediction)
        
    # Reuse your existing minimal report generator to save the formative scorecard
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