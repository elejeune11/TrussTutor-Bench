import sympy as sp

from trusstutor_bench.symbolic_utils import safe_sympify


def parse_trace_value(var_name: str, val, unit: str = "") -> str:
    """
    Turns a solver variable and value into a Markdown line with LaTeX,
    such as "$F_{AB}$ = 50.00 kN (Compression)".

    Args:
        var_name (str): Solver variable name (e.g., 'F_AB' or 'R_A_x').
        val: Float, integer or SymPy string from the solver trace.
        unit (str, optional): Unit appended to the value (e.g., 'kN').

    Returns:
        str: The formatted line.
    """
    if var_name.startswith("F_"):
        # 'F_AB' -> '$F_{AB}$'
        subscript = var_name.split("_", 1)[1]
        latex_var = f"$F_{{{subscript}}}$"
        is_internal_force = True
    elif var_name.startswith("R_"):
        # 'R_A_x' -> '$R_{Ax}$'
        parts = var_name.split("_")
        if len(parts) >= 3:
            latex_var = f"$R_{{{parts[1]}{parts[2]}}}$"
        else:
            latex_var = f"${var_name}$"
        is_internal_force = False
    else:
        # Any other name, e.g. "max_compression" -> "$max_{compression}$"
        if "_" in var_name:
            parts = var_name.split("_", 1)
            latex_var = f"${parts[0]}_{{{parts[1]}}}$"
        else:
            latex_var = f"${var_name}$"
        is_internal_force = False

    unit_str = f" {unit}" if unit else ""

    try:
        num_val = float(safe_sympify(val).evalf())
        abs_val = abs(num_val)

        # Treat floating point noise as zero.
        is_zero = abs_val < 1e-9

        if is_internal_force:
            if is_zero:
                val_str = f"0.00{unit_str} (Zero-Force Member)"
            elif num_val > 0:
                val_str = f"{num_val:.2f}{unit_str} (Tension)"
            else:
                val_str = f"{abs_val:.2f}{unit_str} (Compression)"
        else:
            # Reactions keep their sign, which depends on the coordinate system.
            if is_zero:
                num_val = 0.0
            val_str = f"{num_val:.2f}{unit_str}"

    except (ValueError, TypeError):
        # Purely symbolic values (e.g., "P/2") cannot be converted to a float.
        clean_symbolic = str(sp.simplify(safe_sympify(val)))
        val_str = f"{clean_symbolic}{unit_str} (Algebraic)"

    return f"{latex_var} = {val_str}"


def render_trace_step(step_index: int, trace_entry: dict, unit: str = "") -> str:
    """
    Formats one solver trace entry as a Markdown block: a bold header, the
    explanation and a bullet for each result.

    Args:
        step_index (int): Step number (1, 2, 3, ...).
        trace_entry (dict): Entry with step_type, target, explanation and results.
        unit (str, optional): Unit appended to the values.

    Returns:
        str: Multi-line Markdown text.
    """
    step_type = trace_entry.get("step_type", "Calculation")
    target = trace_entry.get("target", "Unknown")
    explanation = trace_entry.get("explanation", "")
    results = trace_entry.get("results", {})

    header = f"**Step {step_index}: {step_type} ({target})**"

    formatted_results = []
    for raw_key, raw_val in results.items():
        # Internal forces are logged by member ID only (e.g., "AB"), so "F_" is added.
        var_name = raw_key
        if not var_name.startswith("R_") and not var_name.startswith("F_"):
            var_name = f"F_{raw_key}"

        formatted_item = parse_trace_value(var_name, raw_val, unit)
        formatted_results.append(f"*   {formatted_item}")

    lines = [header]
    if explanation:
        lines.append(explanation)
    if formatted_results:
        lines.extend(formatted_results)

    return "\n".join(lines)
