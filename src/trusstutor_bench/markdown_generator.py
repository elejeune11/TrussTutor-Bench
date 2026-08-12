import sympy as sp


def parse_trace_value(var_name: str, val, unit: str = "") -> str:
    """
    Translates a raw mathematical solver trace variable and value into 
    a textbook-style Markdown string with LaTeX formatting.
    
    Args:
        var_name (str): The raw string ID (e.g., 'F_AB' or 'R_A_x').
        val (any): The float, integer, or SymPy string from the solver trace.
        unit (str, optional): The physical unit to append (e.g., 'kN'). Defaults to "".
        
    Returns:
        str: A formatted string (e.g., '$F_{AB}$ = 50.00 kN (Compression)').
    """
    # 1. Format the variable name into LaTeX
    if var_name.startswith("F_"):
        # Example: 'F_AB' -> '$F_{AB}$'
        subscript = var_name.split("_", 1)[1]
        latex_var = f"$F_{{{subscript}}}$"
        is_internal_force = True
    elif var_name.startswith("R_"):
        # Example: 'R_A_x' -> '$R_{Ax}$'
        parts = var_name.split("_")
        if len(parts) >= 3:
            latex_var = f"$R_{{{parts[1]}{parts[2]}}}$"
        else:
            latex_var = f"${var_name}$"
        is_internal_force = False
    else:
        # Fallback for unexpected variables (e.g., "max_compression" -> "$max_{compression}$")
        if "_" in var_name:
            parts = var_name.split("_", 1)
            latex_var = f"${parts[0]}_{{{parts[1]}}}$"
        else:
            latex_var = f"${var_name}$"
        is_internal_force = False

    unit_str = f" {unit}" if unit else ""

    # 2. Parse the numerical value and determine semantic state
    try:
        # sympify handles fractions, algebraic strings, and native numbers
        num_val = float(sp.sympify(val).evalf())
        abs_val = abs(num_val)
        
        # Filter out extreme floating point noise (e.g., 1e-15 -> 0.0)
        is_zero = abs_val < 1e-9

        if is_internal_force:
            if is_zero:
                val_str = f"0.00{unit_str} (Zero-Force Member)"
            elif num_val > 0:
                val_str = f"{num_val:.2f}{unit_str} (Tension)"
            else:
                val_str = f"{abs_val:.2f}{unit_str} (Compression)"
        else:
            # Reactions maintain their signs depending on the assumed global coordinate system
            if is_zero:
                num_val = 0.0
            val_str = f"{num_val:.2f}{unit_str}"
            
    except (ValueError, TypeError):
        # Fallback for purely symbolic outputs (e.g., "P/2") that cannot be coerced to float
        clean_symbolic = str(sp.simplify(val))
        val_str = f"{clean_symbolic}{unit_str} (Algebraic)"

    return f"{latex_var} = {val_str}"


def render_trace_step(step_index: int, trace_entry: dict, unit: str = "") -> str:
    """
    Takes a single trace dictionary logged by the solver and formats it 
    into a cohesive Markdown block.
    
    Args:
        step_index (int): The chronological step number (e.g., 1, 2, 3).
        trace_entry (dict): The dictionary containing step_type, target, explanation, and results.
        unit (str, optional): The physical unit to append. Defaults to "".
        
    Returns:
        str: A multi-line Markdown string representing the solver step.
    """
    step_type = trace_entry.get("step_type", "Calculation")
    target = trace_entry.get("target", "Unknown")
    explanation = trace_entry.get("explanation", "")
    results = trace_entry.get("results", {})

    # 1. Format the bold header
    header = f"**Step {step_index}: {step_type} ({target})**"

    # 2. Format the bulleted list of parsed mathematical results
    formatted_results = []
    for raw_key, raw_val in results.items():
        # The solver often logs internal forces using just the member ID (e.g., "AB").
        # We prepend "F_" if it's not a reaction so the value parser knows it's a force.
        var_name = raw_key
        if not var_name.startswith("R_") and not var_name.startswith("F_"):
            var_name = f"F_{raw_key}"
            
        formatted_item = parse_trace_value(var_name, raw_val, unit)
        formatted_results.append(f"*   {formatted_item}")

    # 3. Assemble the final block
    lines = [header]
    if explanation:
        lines.append(explanation)
    if formatted_results:
        lines.extend(formatted_results)
        
    return "\n".join(lines)


def generate_markdown_workflow(state, benchmark_id: str = "Truss Problem", unit: str = "", final_answers: dict = None) -> str:
    """
    Assembles the complete step-by-step Markdown solution from a resolved TrussState.
    Filters out redundant steps (like Global Statics running twice) to ensure a clean pedagogical trace.
    """
    lines = [f"### Problem: {benchmark_id}", "---", ""]
    
    # 0. Deduplicate the trace to protect against redundant solver calls
    filtered_trace = []
    seen_steps = set()
    
    for entry in state.trace:
        # Create a unique signature for this specific action
        step_signature = (entry.get("step_type"), entry.get("target"))
        
        if step_signature not in seen_steps:
            filtered_trace.append(entry)
            seen_steps.add(step_signature)
    
    # 1. Iterate through the filtered solver trace
    for i, trace_entry in enumerate(filtered_trace, start=1):
        rendered_step = render_trace_step(i, trace_entry, unit)
        lines.append(rendered_step)
        lines.append("")  # Add a blank line for readability between steps
        
    # 2. Optionally append the final extracted answers
    if final_answers:
        lines.append("---")
        lines.append("**Final Extraction**")
        
        for key, val in final_answers.items():
            if isinstance(val, str) and key == "units":
                continue
                
            if isinstance(val, dict):
                lines.append(f"**{key.replace('_', ' ').title()}:**")
                for sub_key, sub_val in val.items():
                    raw_val = sub_val.get("magnitude", sub_val) if isinstance(sub_val, dict) else sub_val
                    
                    var_name = sub_key
                    if not var_name.startswith("R_") and not var_name.startswith("F_"):
                        var_name = f"F_{sub_key}"
                        
                    parsed_val = parse_trace_value(var_name, raw_val, unit)
                    lines.append(f"*   {parsed_val}")
            else:
                parsed_val = parse_trace_value(key, val, unit)
                lines.append(f"*   {parsed_val}")
                
    return "\n".join(lines).strip()