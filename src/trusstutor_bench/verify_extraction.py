import jsonschema
from trusstutor_bench.task_base import BenchmarkTask
from trusstutor_bench.statics_tools import programmatic_solution


def validate_task_extraction(task: BenchmarkTask) -> bool:
    """
    Checks that the solution recipe runs to completion and that its output matches the task's
    expected_llm_output schema. Raises RuntimeError if the recipe fails, and ValueError if the
    schema is missing or not a JSON Schema object, or if the output does not match it.

    Args:
        task (BenchmarkTask): The task to check.

    Returns:
        bool: True if the output matches the schema.
    """
    test_vars = {}
    for var_name, rules in task.problem_schema.get("variables", {}).items():
        if "default" in rules:
            test_vars[var_name] = rules["default"]
        elif "value" in rules:
            test_vars[var_name] = rules["value"]

    try:
        final_output = programmatic_solution(
            solution_recipe=task.solution_recipe,
            sampled_vars=test_vars
        )
    except Exception as e:
        raise RuntimeError(f"[{task.benchmark_id}] Pipeline execution failed: {e}")

    expected_schema = task.problem_schema.get("expected_llm_output", {})

    if not expected_schema:
        raise ValueError(f"[{task.benchmark_id}] Missing 'expected_llm_output' in PROBLEM_SCHEMA.")

    # A schema without 'type' and 'properties' makes jsonschema ignore the root keys and pass anything.
    if expected_schema.get("type") != "object" or "properties" not in expected_schema:
        raise ValueError(
            f"[{task.benchmark_id}] 'expected_llm_output' must be a valid JSON Schema "
            f"object: {{'type': 'object', 'properties': {{...}}, 'required': [...]}}. "
            f"Answer keys at the root are ignored by jsonschema."
        )

    try:
        jsonschema.validate(instance=final_output, schema=expected_schema)
    except jsonschema.exceptions.SchemaError as e:
        raise ValueError(f"[{task.benchmark_id}] 'expected_llm_output' is not a valid JSON Schema: {e.message}")
    except jsonschema.exceptions.ValidationError as e:
        path = " -> ".join(str(p) for p in e.absolute_path) if e.absolute_path else "Root"
        raise ValueError(f"[{task.benchmark_id}] Schema Validation Failed at '{path}': {e.message}")

    return True
