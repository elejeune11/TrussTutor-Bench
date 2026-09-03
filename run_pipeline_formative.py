import os
import sys
import json
import argparse
import asyncio
from pathlib import Path

# Add the project root and src to the Python path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))

# Tasks live in the sibling TrussTutor-support-code repo, not in this repo.
# Its src/ must be on sys.path so task files' own `from trusstutor_support_code...`
# imports resolve (mirrors the pattern AutoMultiverse uses for its sibling repos).
SUPPORT_CODE_DIR = Path("/projectnb/lejlab2/Evangelos_Practice/TrussTutor-support-code")
sys.path.append(str(SUPPORT_CODE_DIR / "src"))

# Import the formative modules
from trusstutor_bench.task_to_prompt_formative import task_to_prompt_formative_all
from llm_api.run_gemini_mwe import run_gemini_mwe_formative
from llm_api.run_anthropic_client import run_anthropic_formative

# Import the evaluation modules
from trusstutor_bench.evaluate_outputs import evaluate_outputs_formative, cross_evaluate_2x2_matrix

# List of available models for the pipeline
available_models = ["gemini-3.1-pro", "claude-opus-5", "claude-sonnet-5"]

async def process_payload_async(payload_path: Path, outputs_dir: Path, llm_model: str):
    """Wraps the synchronous Gemini API call in an async thread, or calls Claude natively."""
    benchmark_id = payload_path.stem.replace("_payload", "")
    output_file = outputs_dir / f"{benchmark_id}_response.json"

    # Idempotency check: skip if we already have the answer
    if output_file.exists():
        print(f"  [SKIP] Response for {benchmark_id} already exists.")
        return True

    print(f"  [API] Starting {benchmark_id}...")
    try:
        if llm_model == "gemini-3.1-pro":
            # Run the blocking API call in a background thread so it doesn't freeze the loop
            result = await asyncio.to_thread(run_gemini_mwe_formative, str(payload_path))
        elif llm_model in ["claude-opus-5", "claude-sonnet-5"]:
            result = await run_anthropic_formative(llm_model, payload_path)
        else:
            raise ValueError(f"Unsupported LLM model: {llm_model}")

        # Save the result
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

        print(f"  [SUCCESS] Finished {benchmark_id}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed {benchmark_id}: {e}")
        return False


async def run_all_concurrently(payload_files: list, outputs_dir: Path, llm_model_id: str, max_concurrent: int = 10):
    """Executes API requests with at most max_concurrent in flight at once."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded(path):
        async with semaphore:
            return await process_payload_async(path, outputs_dir, llm_model_id)

    tasks = [bounded(path) for path in payload_files]
    results = await asyncio.gather(*tasks)
    return sum(results)


def print_pipeline_summary(posture_1_path: Path, posture_2_path: Path, posture_3_path: Path):
    """Parses the generated reports and prints a unified dashboard to the terminal."""
    print("\n" + "="*70)
    print(" TRUSSTUTOR-BENCH: FINAL FORMATIVE SUMMARY")
    print("="*70)
    
    dashboard_rows = [
        ("Posture 1 (Flawless Work: No Error)", posture_1_path), 
        ("Posture 2 (Flawed Work: Global Reactions)", posture_2_path),
        ("Posture 3 (Flawed Work: Method of Sections)", posture_3_path)
    ]
    
    for name, path in dashboard_rows:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f).get("aggregate_metrics", {})
            tasks = data.get("total_tasks", 0)
            passed = data.get("total_passed", 0)
            acc = data.get("accuracy_percentage", 0.0)
            print(f" {name:<45} | {passed}/{tasks} Passed ({acc:.2f}%)")
        else:
            print(f" {name:<45} | Report Not Found")
    print("="*70)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the TrussTutor-Bench formative pipeline.")
    parser.add_argument("-m", "--model", required=True, type=str,
                            choices=available_models,
                            help="The LLM model to use for evaluation.")

    return parser.parse_args()



def main():
    print("=== Starting TrussTutor-Bench Formative Pipeline (Async) ===")

    args = parse_args()
    llm_model_id = args.model

    # 1. Define Paths (Routed to distinct formative directories)
    # SMOKETEST: pointed at the local 5-example set (tasks/, prompts_formative/)
    # instead of the full 200-task set, to cheaply verify the nullable-schema
    # fix before spending on a full rerun. Output paths are the real per-model
    # ones, so the 5 verified tasks land directly in the real dataset.
    tasks_dir = PROJECT_ROOT / "tasks"
    figs_dir = tasks_dir / "figs"

    prompts_dir = PROJECT_ROOT / "prompts_formative"
    outputs_dir = PROJECT_ROOT / f"llm_outputs_formative_{llm_model_id}"
    report_posture_1 = PROJECT_ROOT / f"formative_evaluation_report_posture_1_{llm_model_id}.json"
    report_posture_2 = PROJECT_ROOT / f"formative_evaluation_report_posture_2_{llm_model_id}.json"
    report_posture_3 = PROJECT_ROOT / f"formative_evaluation_report_posture_3_{llm_model_id}.json"
    baseline_report_path = PROJECT_ROOT / f"evaluation_report_{llm_model_id}.json"

    prompts_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 2. Generate Prompt Payloads
    print("\n--- Step 1: Generating Formative Prompts from Tasks ---")
    if not tasks_dir.exists():
        print(f"[ERROR] Tasks directory not found at {tasks_dir}")
        return

    # Generate the payloads using all student postures
    postures = ["posture_1", "posture_2", "posture_3"]
    for posture in postures:
        print(f"  Generating payloads for: {posture}...")
        task_to_prompt_formative_all(tasks_dir, figs_dir, prompts_dir, posture_type=posture)
    
    # 3. Execute LLM API Evaluations Concurrently
    print("\n--- Step 2: Running LLM API Evaluations Concurrently ---")
    if llm_model_id == "gemini-3.1-pro":
        if not os.environ.get("GEMINI_API_KEY"):
            print("[WARNING] GEMINI_API_KEY is not set. API calls may fail.")
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("[WARNING] ANTHROPIC_API_KEY is not set. API calls may fail.")

    all_payload_files = list(prompts_dir.glob("*_payload.json"))
    if not all_payload_files:
        print(f"[ERROR] No prompt payloads found in {prompts_dir}. Exiting.")
        return
        
    # --- Smart API Skipping Logic ---
    valid_payload_files = []
    for path in all_payload_files:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            
        posture = payload.get("posture_type")
        is_correct = payload.get("ground_truth", {}).get("is_work_correct")
        
        # If we are testing error detection (Postures 2 & 3), but no error was 
        # actually injected by the compiler, do not waste an API call.
        if posture in ["posture_2", "posture_3"] and is_correct is True:
            print(f"  [FILTERED] Skipping API call for {path.name} (No error injected)")
            continue
            
        valid_payload_files.append(path)
        
    # Kick off the async event loop with ONLY the valid payloads
    success_count = asyncio.run(run_all_concurrently(valid_payload_files, outputs_dir, llm_model_id))
    
    print(f"\n  [INFO] ({success_count}/{len(valid_payload_files)} API calls succeeded)")

    # 4. Grade the Outputs and Generate Report
    print("\n--- Step 3: Evaluating LLM Outputs ---")
    
    evaluate_outputs_formative(prompts_dir, outputs_dir, "posture_1", report_posture_1)
    evaluate_outputs_formative(prompts_dir, outputs_dir, "posture_2", report_posture_2)
    evaluate_outputs_formative(prompts_dir, outputs_dir, "posture_3", report_posture_3)
    
    # Print the clean, combined dashboard for all postures
    print_pipeline_summary(report_posture_1, report_posture_2, report_posture_3)
    
    # 5. Generate 2x2 Cross-Evaluation Matrix
    print("\n--- Step 4: Generating Cross-Evaluation Summary ---")
    
    # We pass report_posture_1 because the baseline evaluation should be 
    # compared against the LLM's ability to detect an actual error.
    if baseline_report_path.exists() and report_posture_1.exists():
        cross_evaluate_2x2_matrix(baseline_report_path, report_posture_1)
    else:
        print("[WARNING] Could not generate 2x2 matrix. Ensure both the baseline solver report and posture 2 report exist.")
    
    print("\n=== Pipeline Complete! ===")

if __name__ == "__main__":
    main()