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

# Import the modules
from trusstutor_bench.task_to_prompt import task_to_prompt_all
from llm_api.run_gemini_client import run_gemini
from trusstutor_bench.evaluate_outputs import evaluate_outputs_minimal
from llm_api.run_anthropic_client import run_anthropic
from llm_api.run_openai_client import run_openai
from llm_api.run_togetherai_client import run_togetherai
from llm_api.run_deepseek_client import run_deepseek
from llm_api.run_minimax_client import run_minimax

# List of available models for the pipeline
available_models = [
    "gemini-3.1-pro", "gemini-3.8-flash", "gemini-4-argon",
    "gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna",
    "claude-fable-5-1", "claude-opus-5", "claude-sonnet-5",
    "deepseek-flash", "MiniMax-M3", "zai-org/GLM-5.3-Flash",
]

async def process_payload_async(payload_path: Path, outputs_dir: Path, llm_model: str):
    """
    Sends one payload to the chosen model and saves the response.
    Skips payloads that already have a response.

    Args:
        payload_path (Path): Path to the *_payload.json file.
        outputs_dir (Path): Folder where the response JSON is saved.
        llm_model (str): Model ID from available_models.

    Returns:
        bool: True if the response exists or was saved, False if the call failed.
    """
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
            result = await asyncio.to_thread(run_gemini, str(payload_path))
        elif llm_model == "gemini-3.8-flash":
            result = await asyncio.to_thread(run_gemini, str(payload_path), "gemini-3.8-flash")
        elif llm_model == "gemini-4-argon":
            result = await asyncio.to_thread(run_gemini, str(payload_path), "gemini-4-argon")
        elif llm_model in ["claude-opus-5", "claude-sonnet-5", "claude-fable-5-1"]:
            result = await run_anthropic(llm_model, payload_path)
        elif llm_model in ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-astra"]:
            result = await run_openai(llm_model, payload_path)
        elif llm_model == "zai-org/GLM-5.3-Flash":
            result = await run_togetherai(llm_model, payload_path)
        elif llm_model == "deepseek-flash":
            result = await run_deepseek(llm_model, payload_path)
        elif llm_model == "MiniMax-M3":
            result = await run_minimax(llm_model, payload_path)
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
    """
    Runs all payloads, with a limited number of API calls at the same time.

    Args:
        payload_files (list): Paths to *_payload.json files.
        outputs_dir (Path): Folder where responses are saved.
        llm_model_id (str): Model ID from available_models.
        max_concurrent (int): Maximum number of calls in flight at once.

    Returns:
        int: Number of payloads that succeeded.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded(path):
        async with semaphore:
            return await process_payload_async(path, outputs_dir, llm_model_id)

    tasks = [bounded(path) for path in payload_files]
    results = await asyncio.gather(*tasks)
    return sum(results)

def parse_args():
    """
    Reads the command-line options.

    Returns:
        argparse.Namespace: The parsed options.
    """
    parser = argparse.ArgumentParser(description="Run the TrussTutor-Bench baseline pipeline.")
    parser.add_argument("-m", "--model", required=True, type=str,
                        choices=available_models,
                        help="The LLM model to use for evaluation.")
    parser.add_argument("--skip-prompt-generation", action="store_true",
                        help="Use the payloads already in prompts/ instead of regenerating them.")
    parser.add_argument("--ids", nargs="+", default=None,
                        help="Only call the API for these benchmark ids (e.g. TRUSS_e0002); "
                             "grading still covers every output on disk.")
    return parser.parse_args()


def main():
    """
    Generates the prompts, calls the selected model, and writes the evaluation report.
    """
    print("=== Starting TrussTutor-Bench Pipeline (Async) ===")

    args = parse_args()
    llm_model_id = args.model

    # 1. Define Paths
    tasks_dir = PROJECT_ROOT / "tasks"
    figs_dir = tasks_dir / "figs"


    safe_model_id = llm_model_id.replace("/", "-")

    prompts_dir = PROJECT_ROOT / "prompts"
    outputs_dir = PROJECT_ROOT / f"llm_outputs_{safe_model_id}"
    report_path = PROJECT_ROOT / f"evaluation_report_{safe_model_id}.json"

    prompts_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 2. Generate Prompt Payloads
    print("\n--- Step 1: Generating Prompts from Tasks ---")
    if not tasks_dir.exists():
        print(f"[ERROR] Tasks directory not found at {tasks_dir}")
        return

    if args.skip_prompt_generation:
        print("  Skipping payload generation; using existing payloads in prompts/")
    else:
        task_to_prompt_all(tasks_dir, figs_dir, prompts_dir)
    
    # 3. Execute LLM API Evaluations Concurrently
    print("\n--- Step 2: Running LLM API Evaluations Concurrently ---")
    if llm_model_id == "gemini-3.1-pro":
        if not os.environ.get("GEMINI_API_KEY"):
            print("[WARNING] GEMINI_API_KEY is not set. API calls may fail.")
    elif llm_model_id in ["claude-opus-5", "claude-sonnet-5", "claude-fable-5-1"]:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("[WARNING] ANTHROPIC_API_KEY is not set. API calls may fail.")
    elif llm_model_id in ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-astra"]:
        if not os.environ.get("OPENAI_API_KEY"):
            print("[WARNING] OPENAI_API_KEY is not set. API calls may fail.")
    elif llm_model_id == "zai-org/GLM-5.3-Flash":
        if not os.environ.get("TOGETHER_API_KEY"):
            print("[WARNING] TOGETHER_API_KEY is not set. API calls may fail.")
    elif llm_model_id == "deepseek-flash":
        if not os.environ.get("DEEPSEEK_API_KEY"):
            print("[WARNING] DEEPSEEK_API_KEY is not set. API calls may fail.")
    elif llm_model_id == "MiniMax-M3":
        if not os.environ.get("MINIMAX_API_KEY"):
            print("[WARNING] MINIMAX_API_KEY is not set. API calls may fail.")

    payload_files = list(prompts_dir.glob("*_payload.json"))
    if not payload_files:
        print(f"[ERROR] No prompt payloads found in {prompts_dir}. Exiting.")
        return
    if args.ids:
        payload_files = [p for p in payload_files if p.stem.replace("_payload", "") in args.ids]
        print(f"  --ids given: {len(payload_files)} payloads selected")
        
    # Kick off the async event loop
    success_count = asyncio.run(run_all_concurrently(payload_files, outputs_dir, llm_model_id))
    
    print(f"\n  [INFO] ({success_count}/{len(payload_files)} API calls succeeded)")

    # 4. Grade the Outputs and Generate Report
    print("\n--- Step 3: Evaluating LLM Outputs ---")
    evaluate_outputs_minimal(prompts_dir, outputs_dir, report_path)
    
    print("\n=== Pipeline Complete! ===")

if __name__ == "__main__":
    main()