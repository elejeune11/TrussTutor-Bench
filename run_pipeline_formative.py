import os
import sys
import json
import asyncio
from pathlib import Path

# Add the project root and src to the Python path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

# Import the formative modules
from trusstutor_bench.task_to_prompt_formative import task_to_prompt_formative_all
from llm_api.run_gemini_mwe import run_gemini_mwe_formative

# Import the evaluation modules
from trusstutor_bench.evaluate_outputs import evaluate_outputs_formative, cross_evaluate_2x2_matrix


async def process_payload_async(payload_path: Path, outputs_dir: Path):
    """Wraps the synchronous Gemini API call in an async thread."""
    benchmark_id = payload_path.stem.replace("_payload", "")
    output_file = outputs_dir / f"{benchmark_id}_response.json"
    
    # Idempotency check: skip if we already have the answer
    if output_file.exists():
        print(f"  [SKIP] Response for {benchmark_id} already exists.")
        return True
        
    print(f"  [API] Starting {benchmark_id}...")
    try:
        # Run the blocking API call in a background thread so it doesn't freeze the loop
        result = await asyncio.to_thread(run_gemini_mwe_formative, str(payload_path))
        
        # Save the result
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
            
        print(f"  [SUCCESS] Finished {benchmark_id}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed {benchmark_id}: {e}")
        return False

async def run_all_concurrently(payload_files: list, outputs_dir: Path):
    """Executes all API requests simultaneously."""
    # Create a list of async tasks
    tasks = [process_payload_async(path, outputs_dir) for path in payload_files]
    # Gather and run them all at once
    results = await asyncio.gather(*tasks)
    return sum(results)

def main():
    print("=== Starting TrussTutor-Bench Formative Pipeline (Async) ===")
    
    # 1. Define Paths (Routed to distinct formative directories)
    tasks_dir = PROJECT_ROOT / "tasks"
    figs_dir = tasks_dir / "figs"
    prompts_dir = PROJECT_ROOT / "prompts_formative" 
    outputs_dir = PROJECT_ROOT / "llm_outputs_formative"
    
    # Define Report Paths
    formative_report_path = PROJECT_ROOT / "formative_evaluation_report.json"
    baseline_report_path = PROJECT_ROOT / "evaluation_report.json"
    
    prompts_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Generate Prompt Payloads
    print("\n--- Step 1: Generating Formative Prompts from Tasks ---")
    if not tasks_dir.exists():
        print(f"[ERROR] Tasks directory not found at {tasks_dir}")
        return
        
    # Generate the payloads using the first student posture
    task_to_prompt_formative_all(tasks_dir, figs_dir, prompts_dir, posture_type="posture_1")
    
    # 3. Execute LLM API Evaluations Concurrently
    print("\n--- Step 2: Running LLM API Evaluations Concurrently ---")
    if not os.environ.get("GEMINI_API_KEY"):
        print("[WARNING] GEMINI_API_KEY is not set. API calls may fail.")
        
    payload_files = list(prompts_dir.glob("*_payload.json"))
    if not payload_files:
        print(f"[ERROR] No prompt payloads found in {prompts_dir}. Exiting.")
        return
        
    # Kick off the async event loop
    success_count = asyncio.run(run_all_concurrently(payload_files, outputs_dir))
    
    print(f"\n  [INFO] ({success_count}/{len(payload_files)} API calls succeeded)")

    # 4. Grade the Outputs and Generate Report
    print("\n--- Step 3: Evaluating LLM Outputs ---")
    evaluate_outputs_formative(prompts_dir, outputs_dir, formative_report_path)
    
    # 5. Generate 2x2 Cross-Evaluation Matrix
    print("\n--- Step 4: Generating Cross-Evaluation Summary ---")
    if baseline_report_path.exists() and formative_report_path.exists():
        cross_evaluate_2x2_matrix(baseline_report_path, formative_report_path)
    else:
        print("[WARNING] Could not generate 2x2 matrix. Ensure both baseline and formative reports exist.")
    
    print("\n=== Pipeline Complete! ===")

if __name__ == "__main__":
    main()