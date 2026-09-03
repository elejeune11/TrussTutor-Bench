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

# Import the modules
from trusstutor_bench.task_to_prompt import task_to_prompt_all
from llm_api.run_gemini_mwe import run_gemini_mwe
from trusstutor_bench.evaluate_outputs import evaluate_outputs_minimal
from llm_api.run_anthropic_client import run_anthropic

# List of available models for the pipeline
available_models = ["gemini-3.1-pro", "claude-opus-5", "claude-sonnet-5"]


async def process_payload_async(payload_path: Path, outputs_dir: Path, llm_model: str):
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
        if llm_model == "gemini-3.1-pro":
            result = await asyncio.to_thread(run_gemini_mwe, str(payload_path))
        elif llm_model in ["claude-opus-5", "claude-sonnet-5"]:
            result = await run_anthropic(llm_model, payload_path)
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

def parse_args():
    parser = argparse.ArgumentParser(description="Run the TrussTutor-Bench baseline pipeline.")
    parser.add_argument("-m", "--model", required=True, type=str,
                        choices=available_models,
                        help="The LLM model to use for evaluation.")
    return parser.parse_args()


def main():
    print("=== Starting TrussTutor-Bench Pipeline (Async) ===")

    args = parse_args()
    llm_model_id = args.model

    # 1. Define Paths
    tasks_dir = SUPPORT_CODE_DIR / "examples"
    figs_dir = tasks_dir / "figs"


    prompts_dir = PROJECT_ROOT / "prompts_200_examples"
    outputs_dir = PROJECT_ROOT / f"llm_outputs_{llm_model_id}"
    report_path = PROJECT_ROOT / f"evaluation_report_{llm_model_id}.json"

    prompts_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 2. Generate Prompt Payloads
    print("\n--- Step 1: Generating Prompts from Tasks ---")
    if not tasks_dir.exists():
        print(f"[ERROR] Tasks directory not found at {tasks_dir}")
        return

    task_to_prompt_all(tasks_dir, figs_dir, prompts_dir)
    
    # 3. Execute LLM API Evaluations Concurrently
    print("\n--- Step 2: Running LLM API Evaluations Concurrently ---")
    if llm_model_id == "gemini-3.1-pro":
        if not os.environ.get("GEMINI_API_KEY"):
            print("[WARNING] GEMINI_API_KEY is not set. API calls may fail.")
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("[WARNING] ANTHROPIC_API_KEY is not set. API calls may fail.")
        
    payload_files = list(prompts_dir.glob("*_payload.json"))
    if not payload_files:
        print(f"[ERROR] No prompt payloads found in {prompts_dir}. Exiting.")
        return
        
    # Kick off the async event loop
    success_count = asyncio.run(run_all_concurrently(payload_files, outputs_dir, llm_model_id))
    
    print(f"\n  [INFO] ({success_count}/{len(payload_files)} API calls succeeded)")

    # 4. Grade the Outputs and Generate Report
    print("\n--- Step 3: Evaluating LLM Outputs ---")
    evaluate_outputs_minimal(prompts_dir, outputs_dir, report_path)
    
    print("\n=== Pipeline Complete! ===")

if __name__ == "__main__":
    main()