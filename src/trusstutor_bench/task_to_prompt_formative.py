import json
from pathlib import Path
from trusstutor_bench.task_base import BenchmarkTask
from trusstutor_bench.task_to_prompt import (
    sample_variables,
    encode_image_to_base64,
    save_payload
)

from trusstutor_bench.task_to_prompt import render_prompt

from trusstutor_bench.task_to_prompt import encode_image_to_base64

from trusstutor_bench.draw_FBDs import (
    generate_global_fbd,
    generate_node_fbd,
    generate_section_fbd
)

from trusstutor_bench.markdown_generator import render_trace_step
from trusstutor_bench.task_loader import TaskLoader

from trusstutor_bench.corrupted_statics_tools import (
    corrupted_solve_global_reactions_tool,
    corrupted_autonomous_moj_macro,
    corrupted_solve_method_of_sections_tool
)

def get_formative_schema() -> dict:
    """
    Returns the strict JSON schema required for automated grading of the LLM's tutoring ability.
    """
    return {
        "type": "object",
        "properties": {
            "is_work_correct": {
                "type": "boolean",
                "description": "True if the student's work is mathematically and physically flawless. False if there is any error."
            },
            "error_step_number": {
                "type": "integer",
                "nullable": True,  # <-- Gemini-friendly null handling
                "description": "The step number where the first error occurred. Must be null if is_work_correct is true."
            },
            "explanation_to_student": {
                "type": "string",
                "description": "A brief, empathetic explanation of what you found, speaking directly to the student."
            }
        },
        "required": ["is_work_correct", "error_step_number", "explanation_to_student"],
        "additionalProperties": False
    }


def get_student_posture(posture_type: str) -> str:
    """
    Returns the specific query from the student based on the testing scenario.
    """
    postures = {
        "posture_1": "I am getting the wrong answer according to my professor and the homework answer key. Where did I make a mistake in my work?",
        "posture_2": "Is my work correct?",
        "posture_3": "Is my work correct?"
    }
    
    if posture_type not in postures:
        raise ValueError(f"Unknown posture_type: {posture_type}. Must be one of {list(postures.keys())}")
        
    return postures[posture_type]


def generate_formative_blocks(task: BenchmarkTask, state, subs_dict: dict, output_dir: Path) -> list:
    """
    Translates a fully resolved TrussState into a structured array of interleaved 
    text and image blocks by walking through the solver's chronological trace.
    """
    blocks = []
    task_id = task.benchmark_id
    unit = task.problem_schema.get("units", {}).get("force_unit", "")
    
    # Ensure output directory exists for temp FBD images
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if this is exclusively a ZFM problem
    topics = task.metadata.get("topic", [])
    is_pure_zfm = len(topics) == 1 and topics[0] == "identify_zfm"

    # ==========================================
    # 1. INITIAL SETUP: GLOBAL FBD
    # ==========================================
    # Only draw the Global FBD if the problem involves more than just finding ZFMs
    if not is_pure_zfm:
        global_img_path = output_dir / f"{task_id}_global_FBD.png"
        generate_global_fbd(state, subs_dict, filepath=global_img_path)
        
        blocks.append({
            "type": "text", 
            "content": "First, I drew the global Free Body Diagram to understand the overall loads and supports:"
        })
        blocks.append({
            "type": "image",
            "name": f"{task_id}_global_FBD",
            "content_base64": encode_image_to_base64(global_img_path)
        })
    # ==========================================
    # 2. DEDUPLICATE THE SOLVER TRACE
    # ==========================================
    # We use the exact same deduplication logic from your markdown_generator 
    # to protect against redundant solver calls.
    filtered_trace = []
    seen_steps = set()
    
    for entry in state.trace:
        step_signature = (entry.get("step_type"), entry.get("target"))
        if step_signature not in seen_steps:
            filtered_trace.append(entry)
            seen_steps.add(step_signature)

    # ==========================================
    # 3. WALK THE CHRONOLOGICAL TRACE
    # ==========================================
    for i, trace_entry in enumerate(filtered_trace, start=1):
        step_type = trace_entry.get("step_type", "Calculation")
        target = trace_entry.get("target", "Unknown")
        
        # --- Image Injection: Method of Joints ---
        if step_type == "Method of Joints":
            node_id = target
            node_img_path = output_dir / f"{task_id}_node_{node_id}_FBD.png"
            generate_node_fbd(state, node_id, subs_dict, filepath=node_img_path)
            
            blocks.append({
                "type": "text",
                "content": f"To solve for the forces at Joint {node_id}, I isolated the joint:"
            })
            blocks.append({
                "type": "image",
                "name": f"{task_id}_node_{node_id}_FBD",
                "content_base64": encode_image_to_base64(node_img_path)
            })
            
        # --- Image Injection: Method of Sections ---
        elif step_type == "Method of Sections":
            # Attempt to pull cut members and target from state. 
            # If not explicitly defined, we gracefully skip drawing the cut.
            cut_elements = getattr(state, 'mos_cut_elements', []) 
            target_node = getattr(state, 'mos_target_node', target)
            
            if cut_elements:
                section_img_path = output_dir / f"{task_id}_section_{target_node}_FBD.png"
                generate_section_fbd(state, cut_elements, target_node, subs_dict, filepath=section_img_path)
                
                blocks.append({
                    "type": "text",
                    "content": f"I took a section cut through {', '.join(cut_elements)} and analyzed the sub-truss:"
                })
                blocks.append({
                    "type": "image",
                    "name": f"{task_id}_section_FBD",
                    "content_base64": encode_image_to_base64(section_img_path)
                })

        # --- Text Injection: Render the Math ---
        # We rely on your existing robust rendering function to output the formatted equations
        step_md = render_trace_step(i, trace_entry, unit)
        blocks.append({
            "type": "text",
            "content": step_md
        })

    return blocks


def render_formative_prompt(problem_schema: dict, sampled_vars: dict, student_query: str) -> str:
    """
    Constructs the minimalist system instructions for the LLM, combining the 
    base problem statement and the specific student question.
    
    Args:
        problem_schema (dict): The PROBLEM_SCHEMA dictionary from a task.
        sampled_vars (dict): The numerical values extracted for this specific instance.
        student_query (str): The specific question the student is asking (the posture).
        
    Returns:
        str: The clean text prompt ready to be prepended to the student's work.
    """
    # Render the underlying problem statement using the existing utility
    base_problem = render_prompt(problem_schema, sampled_vars)
    
    # Assemble the minimal prompt
    prompt = (
        f"**Problem Statement:**\n{base_problem}\n\n"
        f"**Student Question:**\n\"{student_query}\""
    )
    
    return prompt


def compile_formative_payload(task: BenchmarkTask, figs_dir: Path, output_dir: Path, posture_type: str) -> dict:
    """
    Compiles all multimodal components of a single formative assessment task 
    into a self-contained, JSON-serializable dictionary.
    """
    # 1. Sample the ground truth variables
    sampled_vars = sample_variables(task.problem_schema["variables"])
    
    # 2. Locate and encode the base problem schematic
    base_image_path = figs_dir / f"{task.benchmark_id}.png"
    base_image_base64 = encode_image_to_base64(base_image_path)
    
    # 3. Assemble the system prompt text
    student_query = get_student_posture(posture_type)
    system_prompt = render_formative_prompt(task.problem_schema, sampled_vars, student_query)
    
    # 4. Programmatically execute the solver recipe to generate the TrussState
    state = None
    error_injected = False
    injected_step_number = None
    
    for step_id, step_data in sorted(task.solution_recipe.items()):
        func = step_data["function"]
        is_corrupted_tool = False
        
        # --- Tool Swapping based on Posture ---
        if posture_type == "posture_2":
            if func.__name__ == "solve_global_reactions_tool":
                func = corrupted_solve_global_reactions_tool
                is_corrupted_tool = True
            elif func.__name__ == "autonomous_moj_macro":
                func = corrupted_autonomous_moj_macro
                is_corrupted_tool = True
        
        elif posture_type == "posture_3":
            if func.__name__ == "solve_method_of_sections_tool":
                func = corrupted_solve_method_of_sections_tool
                is_corrupted_tool = True
        # -----------------------------------

        if step_id == 1:
            state = func(sampled_vars=sampled_vars, **step_data.get("inputs", {}))
        else:
            # Capture how many steps the "student" has written down BEFORE this tool runs
            pre_trace_len = len(state.trace)
            
            # Execute the tool
            func(state, **step_data.get("inputs", {}))
            
            # --- Trace-Based Error Tracking ---
            # If this was the first corrupted tool to run, the error appears in the 
            # very next step added to the trace. We use 1-based indexing for the LLM.
            if is_corrupted_tool and not error_injected:
                injected_step_number = pre_trace_len + 1
                error_injected = True
            # ---------------------------------------
            
    # --- Robust Drawing Substitutions ---
    drawing_subs = sampled_vars.copy()
    
    if hasattr(task, 'drawing_variables'):
        drawing_subs.update(task.drawing_variables)
        
    if state and hasattr(state, 'nodes'):
        for nx, ny in state.nodes.values():
            for coord in (nx, ny):
                try:
                    expr = sp.sympify(coord)
                    for sym in getattr(expr, 'free_symbols', []):
                        if sym.name not in drawing_subs:
                            drawing_subs[sym.name] = 1.0
                except Exception:
                    pass
    # -----------------------------------------
            
    # 5. Generate the multimodal student work blocks (Markdown + localized FBDs)
    fbd_temp_dir = output_dir / "temp_fbds" / task.benchmark_id
    student_blocks = generate_formative_blocks(
        task=task,
        state=state,
        subs_dict=drawing_subs,  
        output_dir=fbd_temp_dir
    )
    
    # 6. Retrieve the strict JSON evaluation schema
    structured_schema = get_formative_schema()
    
    # --- Dynamic Ground Truth Logic ---
    # The work is correct if no error was successfully injected
    is_work_correct = not error_injected
    
    ground_truth = {
        "is_work_correct": is_work_correct,
        "error_step_number": injected_step_number if not is_work_correct else None
    }
    # ---------------------------------------
    
    # 7. Assemble the final payload
    payload = {
        "benchmark_id": task.benchmark_id,
        "posture_type": posture_type,
        "system_prompt": system_prompt,
        "base_image_base64": base_image_base64,
        "student_work": student_blocks,
        "structured_schema": structured_schema,
        "ground_truth_variables": sampled_vars,
        "ground_truth": ground_truth
    }
    
    return payload


def task_to_prompt_formative_single(task: BenchmarkTask, figs_dir: Path, output_dir: Path, posture_type: str) -> Path:
    """
    Compiles the formative payload for a single benchmark task and saves it to disk.
    
    Args:
        task (BenchmarkTask): The loaded task object.
        figs_dir (Path): Directory containing the base schematic .png files.
        output_dir (Path): Directory where the final JSON file will be saved.
        posture_type (str): The specific student question to inject.
        
    Returns:
        Path: The absolute path to the saved JSON file.
    """
    payload = compile_formative_payload(task, figs_dir, output_dir, posture_type)
    
    # Create a unique filename for the formative assessment payload
    file_name_id = f"{task.benchmark_id}_formative_{posture_type}"
    saved_path = save_payload(payload, output_dir, file_name_id)
    
    return saved_path


def task_to_prompt_formative_all(examples_dir: Path, figs_dir: Path, output_dir: Path, posture_type: str = "posture_1") -> None:
    """
    Iterates through all benchmark tasks in a directory, compiles their formative 
    payloads, and saves them to disk.
    
    Args:
        examples_dir (Path): Directory containing the example_TRUSS_xxxx.py files.
        figs_dir (Path): Directory containing the generated .png schematics.
        output_dir (Path): Directory where the final JSON files will be saved.
        posture_type (str): The specific student question to inject for this batch.
    """
    print(f"Loading tasks from {examples_dir}...")
    registry = TaskLoader.load_directory(str(examples_dir))
    
    if not registry:
        print("No tasks found. Exiting.")
        return

    print(f"Found {len(registry)} tasks. Compiling formative payloads...")
    success_count = 0
    
    for benchmark_id, task in registry.items():
        try:
            saved_path = task_to_prompt_formative_single(task, figs_dir, output_dir, posture_type)
            print(f"  [SUCCESS] Saved {benchmark_id} -> {saved_path.name}")
            success_count += 1
        except Exception as e:
            print(f"  [ERROR] Failed to compile {benchmark_id}: {e}")

    print(f"\nAll done! Successfully generated {success_count}/{len(registry)} formative payloads in: {output_dir}")


