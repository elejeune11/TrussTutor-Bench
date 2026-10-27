import numpy as np
import sympy as sp
from trusstutor_bench.task_base import BenchmarkTask
from trusstutor_bench.direct_stiffness_method import solve_truss, check_stability_and_determinacy


def validate_task_physics(task: BenchmarkTask, tolerance: float = 1e-4) -> bool:
    """
    Checks that the symbolic solution recipe and the numerical direct stiffness method
    give the same member forces. Raises ValueError for a geometric mechanism and
    AssertionError for a force mismatch.

    Args:
        task (BenchmarkTask): The task to check.
        tolerance (float): Absolute tolerance for comparing forces.

    Returns:
        bool: True if every solved member force matches.
    """
    test_vars = {}
    for var_name, rules in task.problem_schema["variables"].items():
        if "default" in rules:
            test_vars[var_name] = rules["default"]
        elif "value" in rules:
            test_vars[var_name] = rules["value"]

    raw_subs = {**test_vars, **task.drawing_variables}

    # Match symbols by name, because Symbol('L', positive=True) != Symbol('L').
    sym_map = {str(sym): sym for sym in task.symbolic_definitions}
    subs_dict = {}

    for key, val in raw_subs.items():
        if key in sym_map:
            subs_dict[sym_map[key]] = val
        else:
            subs_dict[key] = val

    numeric_nodes = {}
    for nid, (x, y) in task.nodes.items():
        numeric_nodes[nid] = (
            float(sp.sympify(x).subs(subs_dict).evalf()),
            float(sp.sympify(y).subs(subs_dict).evalf())
        )

    numeric_loads = {}
    for nid, (fx, fy) in task.loads.items():
        numeric_loads[nid] = (
            float(sp.sympify(fx).subs(subs_dict).evalf()),
            float(sp.sympify(fy).subs(subs_dict).evalf())
        )

    diagnostic = check_stability_and_determinacy(
        numeric_nodes, task.elements, task.supports
    )
    if diagnostic["is_singular"]:
        raise ValueError("Failed: Structure is a geometric mechanism.")

    _, dsm_forces, _ = solve_truss(
        nodes=numeric_nodes,
        elements=task.elements,
        supports=task.supports,
        loads=numeric_loads
    )

    # Run the recipe without its last step, the answer extraction.
    init_step = task.solution_recipe[1]
    state = init_step["function"](sampled_vars=test_vars, **init_step["inputs"])

    step_keys = sorted(list(task.solution_recipe.keys()))
    for step_idx in step_keys[1:-1]:
        step_data = task.solution_recipe[step_idx]
        step_data["function"](state, **step_data.get("inputs", {}))

    for eid, sym_force in state.known_forces.items():
        # Substitute any symbols that remain (like 'm' or 'L') into the force.
        if hasattr(sym_force, 'free_symbols') and sym_force.free_symbols:
            sym_force = sym_force.subs(subs_dict)

        recipe_val = float(sp.sympify(sym_force).evalf())
        dsm_val = float(dsm_forces.get(eid, 0.0))

        if not np.isclose(recipe_val, dsm_val, atol=tolerance):
            raise AssertionError(
                f"Mismatch in Element {eid}: Recipe says {recipe_val:.3f}, DSM says {dsm_val:.3f}"
            )

    return True
