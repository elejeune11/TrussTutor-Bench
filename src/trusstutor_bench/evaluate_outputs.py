import math
import json
import re
from pathlib import Path

from trusstutor_bench.symbolic_utils import safe_sympify


_UNIT_SYNONYMS = [
    {"kip", "kips", "k"},
    {"lb", "lbs", "pound", "pounds"},
]


def _unit_synonym_equal(truth_val: str, pred_val: str) -> bool:
    """
    Checks whether two unit strings are synonyms, such as "kip", "kips" and "k".

    Args:
        truth_val (str): Ground-truth unit.
        pred_val (str): Predicted unit.

    Returns:
        bool: True if both belong to the same synonym group.
    """
    truth_norm = truth_val.strip().lower()
    pred_norm = pred_val.strip().lower()
    return any(truth_norm in group and pred_norm in group for group in _UNIT_SYNONYMS)


def _units_value_present(truth_val: str, pred_val: str) -> bool:
    """
    Checks whether the true unit appears in the prediction as a whole word, so extra
    commentary is accepted but "N" does not match inside "kN".

    Args:
        truth_val (str): Ground-truth unit.
        pred_val (str): Predicted units text.

    Returns:
        bool: True if the unit is present as a standalone word.
    """
    pattern = r"\b" + re.escape(truth_val.strip()) + r"\b"
    return re.search(pattern, pred_val, re.IGNORECASE) is not None


def _looks_algebraic(s: str) -> bool:
    """
    Cheap pre-filter: only strings that could be math expressions get a symbolic parse.

    Args:
        s (str): Answer string.

    Returns:
        bool: True if the string contains a digit or an operator.
    """
    return any(c.isdigit() or c in "+-*/^()" for c in s)


def _symbolic_string_equal(truth_val: str, pred_val: str) -> bool:
    """
    Checks whether two algebraic strings are equal by evaluating both at a fixed test point,
    with the same tolerance as numeric fields. This accepts equivalent forms and decimal
    approximations of exact values.

    Args:
        truth_val (str): Ground-truth expression.
        pred_val (str): Predicted expression.

    Returns:
        bool: True if both expressions agree at the test point.
    """
    if not (_looks_algebraic(truth_val) and _looks_algebraic(pred_val)):
        return False
    try:
        truth_expr = safe_sympify(truth_val)
        pred_expr = safe_sympify(pred_val)

        free_symbols = truth_expr.free_symbols | pred_expr.free_symbols
        test_values = {sym: 1.3 + 0.7 * i for i, sym in enumerate(free_symbols)}

        truth_num = complex(truth_expr.evalf(subs=test_values))
        pred_num = complex(pred_expr.evalf(subs=test_values))
        return math.isclose(truth_num.real, pred_num.real, rel_tol=1e-2, abs_tol=1e-3) and \
            math.isclose(truth_num.imag, pred_num.imag, rel_tol=1e-2, abs_tol=1e-3)
    except Exception:
        return False


def _normalize_member_id(member_id: str) -> str:
    """
    Canonicalizes a member name so direction and "$...$" wrapping do not matter
    ("BG", "GB" and "$BG$" are the same member). Handles "AB" and "9-8" styles.

    Args:
        member_id (str): Member name.

    Returns:
        str: Canonical lowercase name.
    """
    s = member_id.strip().replace("$", "").strip()
    if "-" in s:
        parts = s.split("-")
    elif len(s) == 2 and s.isalpha():
        parts = list(s)
    else:
        return s.lower()
    return "-".join(sorted(p.strip().lower() for p in parts))


def _is_member_name_field(field_name: str) -> bool:
    """
    Member-name fields are the ones whose name contains "member".

    Args:
        field_name (str): Name of a schema field.

    Returns:
        bool: True if the field holds member names.
    """
    return field_name is not None and "member" in field_name.lower()


def compare_field_values(truth_val, pred_val, field_schema: dict, rtol=1e-2, atol=1e-3, field_name: str = None) -> bool:
    """
    Compares one ground-truth value with the prediction, according to the field's schema type.

    Args:
        truth_val: Ground-truth value.
        pred_val: Predicted value.
        field_schema (dict): Schema of the field.
        rtol (float): Relative tolerance for numbers.
        atol (float): Absolute tolerance for numbers.
        field_name (str): Name of the field, used for member names and units.

    Returns:
        bool: True if the prediction matches.
    """
    expected_type = field_schema.get("type")

    if expected_type in ["number", "float", "integer"] or isinstance(truth_val, (int, float)):
        try:
            return math.isclose(float(truth_val), float(pred_val), rel_tol=rtol, abs_tol=atol)
        except (ValueError, TypeError):
            return False

    if expected_type == "array" or isinstance(truth_val, list):
        if not isinstance(pred_val, list):
            return False
        if _is_member_name_field(field_name):
            return {_normalize_member_id(v) for v in truth_val} == {_normalize_member_id(v) for v in pred_val}
        return set(truth_val) == set(pred_val)

    if expected_type == "string" or isinstance(truth_val, str):
        if not isinstance(pred_val, str):
            return False
        if _is_member_name_field(field_name):
            return _normalize_member_id(truth_val) == _normalize_member_id(pred_val)
        if str(truth_val).strip().lower() == str(pred_val).strip().lower():
            return True
        if _unit_synonym_equal(truth_val, pred_val):
            return True
        if field_name == "units" and _units_value_present(truth_val, pred_val):
            return True
        return _symbolic_string_equal(truth_val, pred_val)

    if expected_type == "object" or isinstance(truth_val, dict):
        if not isinstance(pred_val, dict):
            return False

        properties = field_schema.get("properties", {})

        for key, true_sub_val in truth_val.items():
            pred_sub_val = pred_val.get(key)
            if pred_sub_val is None:
                return False

            sub_schema = properties.get(key, {})

            if not compare_field_values(true_sub_val, pred_sub_val, sub_schema, rtol, atol, field_name=key):
                return False
        return True

    # Booleans and exact matches.
    return truth_val == pred_val


def evaluate_single_task(ground_truth: dict, prediction: dict, schema: dict) -> dict:
    """
    Grades a model's answer against the ground truth, field by field.

    Args:
        ground_truth (dict): Ground-truth answer.
        prediction (dict): Model's answer.
        schema (dict): The task's expected output schema.

    Returns:
        dict: "passed" (all fields correct) and per-field "fields" results.
    """
    properties = schema.get("properties", {})
    field_results = {}
    all_passed = True

    for field, field_schema in properties.items():
        # Skip the scratchpad, which has no ground truth.
        if field == "chain_of_thought":
            continue

        truth_val = ground_truth.get(field)
        pred_val = prediction.get(field)

        if pred_val is None:
            field_results[field] = {
                "passed": False,
                "reason": "Missing key in prediction",
                "ground_truth": truth_val,
                "prediction": None
            }
            all_passed = False
            continue

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
    Computes the pass count and accuracy, prints a summary, and saves the full report.

    Args:
        overall_results (dict): Benchmark ID -> result of the grading function.
        report_path (Path): Where the JSON report is saved.
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
    Grades every baseline response against its payload and saves a report.

    Args:
        prompts_dir (Path): Folder with the *_payload.json files.
        outputs_dir (Path): Folder with the *_response.json files.
        report_path (Path): Where the JSON report is saved.

    Returns:
        dict: Benchmark ID -> grading result, or an empty dict if nothing was graded.
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
    Grades one formative response: it must judge the work correctly and name the right error step.

    Args:
        prediction (dict): Model's answer.
        ground_truth (dict): Expected "is_work_correct" and "error_step_number".

    Returns:
        dict: "passed", "prediction" and "ground_truth".
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
    Grades the formative responses of one posture and saves a report. For postures 2 and 3,
    tasks where no error was injected are skipped, so only error detection is measured.

    Args:
        prompts_dir (Path): Folder with the *_payload.json files.
        outputs_dir (Path): Folder with the *_response.json files.
        target_posture (str): "posture_1", "posture_2" or "posture_3".
        report_path (Path): Where the JSON report is saved.

    Returns:
        dict: Payload ID -> grading result, or an empty dict if nothing was graded.
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

        if payload.get("posture_type") != target_posture:
            continue

        ground_truth = payload.get("ground_truth", {})

        if target_posture in ["posture_2", "posture_3"] and ground_truth.get("is_work_correct") is True:
            continue

        with open(response_file, "r", encoding="utf-8") as f:
            prediction = json.load(f)

        overall_results[benchmark_id] = evaluate_formative_task(prediction, ground_truth)

    generate_minimal_report(overall_results, report_path)

    return overall_results


def print_2x2_summary(matrix_data: dict) -> None:
    """
    Prints the 2x2 table of baseline result against formative result.

    Args:
        matrix_data (dict): The four ID lists returned by cross_evaluate_2x2_matrix.
    """
    print("\n" + "="*50)
    print(" 2x2 CROSS-EVALUATION MATRIX: Baseline vs Formative")
    print("="*50)
    print(f"{'':<20} | {'Formative CORRECT':<20} | {'Formative INCORRECT':<20}")
    print("-" * 67)

    b_pass_f_pass = len(matrix_data["baseline_pass_formative_pass"])
    b_pass_f_fail = len(matrix_data["baseline_pass_formative_fail"])
    print(f"{'Baseline CORRECT':<20} | {b_pass_f_pass:<20} | {b_pass_f_fail:<20}")

    b_fail_f_pass = len(matrix_data["baseline_fail_formative_pass"])
    b_fail_f_fail = len(matrix_data["baseline_fail_formative_fail"])
    print(f"{'Baseline INCORRECT':<20} | {b_fail_f_pass:<20} | {b_fail_f_fail:<20}")
    print("-" * 67)
    print(f"Total Overlapping Tasks Evaluated: {b_pass_f_pass + b_pass_f_fail + b_fail_f_pass + b_fail_f_fail}\n")


def cross_evaluate_2x2_matrix(baseline_report_path: Path, formative_report_path: Path) -> dict:
    """
    Matches baseline and formative results by task ID and sorts them into a 2x2 matrix.

    Args:
        baseline_report_path (Path): Baseline report JSON.
        formative_report_path (Path): Formative report JSON.

    Returns:
        dict: Four lists of task IDs, one per baseline/formative pass/fail combination.
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

    # Formative IDs end in "_formative_<posture>", which is stripped to match the baseline ID.
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
