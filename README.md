# TrussTutor-Bench Evaluation Pipeline

This repository contains the automated pipeline for evaluating Large Language Models (LLMs) on 2D truss statics problems. Tasks are authored as self-contained Python templates; the pipeline turns them into multimodal prompts, executes concurrent API calls, and rigorously evaluates the model's outputs. Five original example tasks are included, one for each topic: method of joints, method of sections, identifying zero-force members, and finding an unknown load or dimension.

The pipeline tests two distinct LLM capabilities:
1. **Baseline Problem Solving:** Can the model calculate the correct mathematical answer?
2. **Formative Tutoring:** Can the model accurately grade a student's step-by-step scratchpad (interleaved with Free Body Diagrams) and identify errors? This is evaluated across three postures:
   - **Posture 1 (Clean Work, Pressured):** The student's work is correct, but the student insists it is wrong. Tests whether the model resists the pressure and says the work is correct, instead of inventing an error.
   - **Posture 2 (Flawed Work):** Tests whether the model can detect actual mathematical contradictions—specifically, an injected Right-Hand Rule sign error in the support reactions—and correctly localize the exact step where the error occurred.
   - **Posture 3 (Late-Stage Errors):** Tests whether the model can detect conceptual final-step errors, specifically Tension/Compression sign flips in Method of Sections problems, introduced after the core matrix math has been calculated correctly.

Tasks that have no step the corruption can change (for example, a pure zero-force-member task) are skipped in Postures 2 and 3, because the student's work stays correct.

## Table of Contents
- [Setup](#setup)
- [Repository Structure](#repository-structure)
- [Running the Pipeline](#running-the-pipeline)
- [Model Settings](#model-settings)
- [Writing a Task](#writing-a-task)
- [Tutorial: From a New Task to Results](#tutorial-from-a-new-task-to-results)
- [Running Tests](#running-tests)

## Setup

This project uses `uv` for fast dependency management and requires Python 3.12 or newer.

**1. Sync dependencies:**
```bash
uv sync
```

**2. Activate your virtual environment (optional, you can also prefix commands with `uv run`):**
```bash
source .venv/bin/activate
```

**3. Set the API key of the provider you want to use:**

Export the key before running the execution scripts. Each model uses one key:

| Provider | Models | Environment variable |
|---|---|---|
| Google Gemini | `gemini-3.1-pro`, `gemini-3.8-flash`, `gemini-4-argon` | `GEMINI_API_KEY` |
| OpenAI | `gpt-6-astra`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna` | `OPENAI_API_KEY` |
| Anthropic | `claude-fable-5-1`, `claude-opus-5`, `claude-sonnet-5` | `ANTHROPIC_API_KEY` |
| DeepSeek | `deepseek-flash` | `DEEPSEEK_API_KEY` |
| MiniMax | `MiniMax-M3` | `MINIMAX_API_KEY` |
| Together AI | `zai-org/GLM-5.3-Flash` | `TOGETHER_API_KEY` |

```bash
export ANTHROPIC_API_KEY="your_api_key_here"
```

The Gemini models use the legacy `google-generativeai` SDK, which is deprecated and prints a warning on import. It is kept because the Gemini results were produced with it.

## Repository Structure

```
├── tasks/                                          # The five example task templates (example_TRUSS_000X.py)
│   └── figs/                                       # Figure of each task, named <benchmark_id>.png
├── prompts/                                        # Baseline payloads (text, image, ground truth) of the example tasks
├── prompts_formative/                              # Formative payloads (3 postures) of the example tasks
├── llm_api/                                        # One client per provider
│   ├── run_anthropic_client.py
│   ├── run_deepseek_client.py
│   ├── run_gemini_client.py
│   ├── run_minimax_client.py
│   ├── run_openai_client.py
│   └── run_togetherai_client.py
├── src/                                            # Core pipeline and evaluation logic
│   └── trusstutor_bench/                           # Payload compilers, FBD generators, and evaluators
│       ├── corrupted_statics_tools.py              # Intentionally flawed tool variants used for systematic error injection
│       ├── direct_stiffness_method.py              # Direct stiffness method solver, used as an independent check
│       ├── draw_FBDs.py                            # Free body diagram generation
│       ├── draw_truss.py                           # Truss diagram generation
│       ├── evaluate_outputs.py                     # Output evaluation logic
│       ├── markdown_generator.py                   # Markdown rendering of the solver trace
│       ├── math_helpers.py                         # Shared math utilities
│       ├── schemas.py                              # Output schema and allowed metadata values
│       ├── statics_tools.py                        # Statics tools that solve and check trusses
│       ├── symbolic_utils.py                       # Safe parsing of symbolic expressions
│       ├── task_base.py                            # Task data class
│       ├── task_loader.py                          # Task template loader and validator
│       ├── task_to_prompt.py                       # Baseline prompt compiler
│       ├── task_to_prompt_formative.py             # Formative prompt compiler
│       ├── truss_state.py                          # Truss state representation
│       ├── verify_extraction.py                    # Task check: the recipe output matches the output schema
│       └── verify_physics.py                       # Task check: symbolic solution matches the stiffness-method solution
├── tests/                                          # Pytest suite
│   ├── llm_outputs/                                # Fixture responses (the ground-truth answers) for the evaluation tests
│   └── test_*.py                                   # Tests for the code in src and llm_api
├── conftest.py
├── pyproject.toml
├── run_pipeline.py                                 # Executes the baseline solver evaluation
├── run_pipeline_formative.py                       # Executes the formative tutor evaluation (all three postures)
└── verify_tasks.py                                 # Checks that the task templates are correct
```

Running the pipeline also creates the following in the repository root. They are not tracked by git (see `.gitignore`), and `<model>` is the model name with `/` replaced by `-`:

```
├── llm_outputs_<model>/                            # Raw baseline responses
├── evaluation_report_<model>.json                  # Baseline evaluation report
├── llm_outputs_formative_<model>/                  # Raw formative responses
└── formative_evaluation_report_posture_{1,2,3}_<model>.json   # Formative evaluation report of each posture
```

## Running the Pipeline

The framework strictly separates problem definition from execution logic. The pipeline scripts handle the entire lifecycle: converting the task templates in `tasks/` into structured JSON payloads, firing the API calls concurrently (at most 10 at a time), and evaluating the responses.

**To run the baseline evaluation:**
```bash
python run_pipeline.py -m claude-opus-5
```

**To run the formative evaluation:**
```bash
python run_pipeline_formative.py -m claude-opus-5
```
This script automatically executes a triple pass over the tasks in `tasks/`, once for each posture, and writes one report per posture. At the end of the run, a terminal dashboard prints a unified summary across all three postures.

It also cross-references the baseline results to generate a 2x2 diagnostic matrix. This matrix compares baseline solver performance against Posture 1 (clean work, pressured) results, to analyze whether a model that solves a problem outright also holds its ground when a student insists the correct work is wrong. It needs the baseline report of the same model, so run the baseline evaluation first.

**Options** (both scripts):
- `-m`, `--model`: the model to evaluate (required). Pick one from the table in [Setup](#setup).
- `--skip-prompt-generation`: reuse the payloads already in `prompts/` or `prompts_formative/` instead of rebuilding them.
- `--ids TRUSS_0001 TRUSS_0002`: only call the API for these benchmark IDs. Grading still covers every response on disk.

A response that already exists is not requested again, so rerunning a command only retries the tasks that failed.

## Model Settings

Each client file in `llm_api/` defines its settings as constants at the top of the file, for example `CLAUDE_MAX_TOKENS` and `CLAUDE_THINKING_EFFORT` in `run_anthropic_client.py`. The same values apply to every model that client serves, so change them before a run to set the output-token limit, reasoning effort and similar options for the model you evaluate.

The defaults are:

| Client | Settings |
|---|---|
| Anthropic | `max_tokens` 64000, effort `high`, adaptive thinking |
| OpenAI | `max_output_tokens` 20000, reasoning effort `high` |
| DeepSeek | `max_tokens` 60000, reasoning effort `high` |
| MiniMax | `max_completion_tokens` 100000, adaptive thinking |
| Together AI | `max_tokens` 64000, reasoning effort `high` |
| Gemini | temperature 1.0, 900 s timeout, 3 retries on transient errors |

Results are sensitive to these settings, so record them next to any results you report. For example, Opus 5 and Sonnet 5 were run with effort `medium` and Fable 5.1 with `max_tokens` 20000, which differ from the defaults above.

## Writing a Task

A task is one Python file in `tasks/` named `example_TRUSS_<id>.py`, with a figure `tasks/figs/<benchmark_id>.png`. The existing files are complete examples; copy one and change it. (For a hands-on walkthrough, see the [tutorial](#tutorial-from-a-new-task-to-results) below.) A file contains, in this order:

- `METADATA`: `benchmark_id` (it must match the figure name), `topic`, `source_type`, `source_name` and `source_problem_pointer`, among others. `topic`, `source_type` and `source_name` must be values listed in `src/trusstutor_bench/schemas.py`; the loader raises an error for anything else.
- `SYMBOLIC_DEFINITIONS`, `NODES`, `ELEMENTS`, `LOADS` and `SUPPORTS`: the geometry and boundary conditions, with symbols such as `L` and `P` where the problem is parametric.
- `PROBLEM_SCHEMA`: the problem statement `template` (with `{P}`-style placeholders), the `units`, the `variables` with their defaults, and `expected_llm_output`, the JSON schema the model must follow.
- `SOLUTION_RECIPE`: numbered steps that compute the ground truth: first a function that builds the `TrussState`, then the statics tools to apply (for example `solve_global_reactions_tool`, `solve_method_of_sections_tool` or `autonomous_moj_macro`), and last a function that extracts the answer.
- `DRAWING_VARIABLES`: the numeric values used to draw the schematic.

Valid topics are `method_of_joints`, `method_of_sections`, `identify_zfm` and `find_unknown_load_or_dimension`. Running a task file directly solves it and draws its schematic into `tasks/draw_truss/`, which is a quick way to check it:
```bash
uv run python tasks/example_TRUSS_0001.py
```

**Verifying tasks.** Before using a new or changed task, run the verification on the `tasks/` folder:
```bash
uv run python verify_tasks.py
```
It checks every task in four phases and exits with status 1 if any task fails:

| Phase | Check |
|---|---|
| 0 | The task loads and its metadata uses allowed values. |
| 1 | The symbolic solution matches an independent numerical (direct stiffness method) solution on every solved member. |
| 2 | The solution recipe runs to the end and its output matches `expected_llm_output`. |
| 3 | `figs/<benchmark_id>.png` exists. |

Pass another folder to check it instead, for example `uv run python verify_tasks.py path/to/tasks`.

## Tutorial: From a New Task to Results

This walkthrough adds a sixth task, checks it, runs it through the pipeline and reads the results. Do it in a copy of the repository or delete the new files afterwards. Start with the [Setup](#setup) steps.

### 1. Copy the closest example

Pick the example that matches the topic of your problem, and copy it with the next free number:

| Topic | Copy | Answer it asks for |
|---|---|---|
| `method_of_joints` | `example_TRUSS_0001.py` | one number (the maximum compression) |
| `method_of_joints` | `example_TRUSS_0005.py` | every member force and the reactions |
| `method_of_sections` | `example_TRUSS_0002.py` | the forces in chosen members |
| `find_unknown_load_or_dimension` | `example_TRUSS_0003.py` | the largest allowable load and the members that control it |
| `identify_zfm` | `example_TRUSS_0004.py` | the zero-force members |

```bash
cp tasks/example_TRUSS_0001.py tasks/example_TRUSS_0006.py
```

### 2. Edit the copy

For this tutorial, change the equilateral truss of example 1 into an isosceles truss whose height equals its base. Four edits are enough:

```diff
-    "benchmark_id": "TRUSS_0001",
+    "benchmark_id": "TRUSS_0006",
...
-    "C": (sp.Rational(1, 2) * L, sp.Rational(1, 2) * sp.sqrt(3) * L)
+    "C": (sp.Rational(1, 2) * L, L)
...
-        "A statically determinate equilateral truss is supported by a pin at A "
+        "A statically determinate isosceles truss is supported by a pin at A "
...
-        "is applied at the apex (Node C). If the bottom chord length is L, "
-        "determine the maximum compression force in the structure."
+        "is applied at the apex (Node C). The base AB and the height of C are both L. "
+        "Determine the maximum compression force in the structure."
```

Keep in mind:
- `benchmark_id` must match the number in the file name and the figure name.
- `source_name` must be one of the allowed values in `src/trusstutor_bench/schemas.py`. Add your own there if needed.
- Use `sp.Rational` for fractions, and make the problem text agree with `LOADS` and `SUPPORTS`.
- The solution recipe, the answer extraction and the output schema of example 1 still fit this task, so they stay unchanged.

### 3. Run the task file

```bash
uv run python tasks/example_TRUSS_0006.py
```
This solves the task and draws a schematic. The output should be `{'max_compression_magnitude': 27.95..., 'units': 'kN'}`. **Check it by hand**: the reaction at A is P/2 = 25 kN, and the inclined member makes an angle with sin θ = 2/√5, so the force is 25 / 0.894 = 27.95 kN.

### 4. Add the figure

The figure is what the model sees. Use the generated schematic, or your own drawing, but make sure it shows the supports, the loads and the member and joint labels the problem refers to:
```bash
cp tasks/draw_truss/TRUSS_0006_truss_schematic.png tasks/figs/TRUSS_0006.png
```

### 5. Verify the task

```bash
uv run python verify_tasks.py
```
All six tasks should pass. The checks catch the typical mistakes. Without the figure of the previous step, you would see:
```
[FAILED] TRUSS_0006: Missing image asset: 'TRUSS_0006.png' in the 'figs' folder.
```
and a metadata value that is not allowed stops the run with a message such as `Unauthorized source_name '...'. Allowed: [...]`. A task whose symbolic solution disagrees with the numerical one fails with `Mismatch in Element ...`, which usually means a wrong sign in `LOADS` or a wrong member in `connectivity`.

### 6. Preview what the model will receive

You can build the payload without calling any API:
```bash
uv run python - <<'EOF'
from pathlib import Path
import json
import tempfile
from trusstutor_bench.task_to_prompt import task_to_prompt_all

out = Path(tempfile.mkdtemp())
task_to_prompt_all(Path("tasks"), Path("tasks/figs"), out, task_ids=["TRUSS_0006"])
payload = json.load(open(out / "TRUSS_0006_payload.json"))
print(payload["user_prompt"])
print(payload["ground_truth"])
EOF
```
The payload contains the prompt text, the figure, the output schema and the ground truth the answer will be graded against.

### 7. Run the baseline evaluation

Set the key of the provider you use (see the table in [Setup](#setup)) and run the new task first:
```bash
export ANTHROPIC_API_KEY="your_api_key_here"
uv run python run_pipeline.py -m claude-sonnet-5 --ids TRUSS_0006
```
The pipeline rebuilds the payloads, calls the model for the new task, saves the answer in `llm_outputs_claude-sonnet-5/TRUSS_0006_response.json`, grades it and writes `evaluation_report_claude-sonnet-5.json`. If that works, run all tasks (answers already on disk are skipped):
```bash
uv run python run_pipeline.py -m claude-sonnet-5
```

### 8. Run the formative evaluation

```bash
uv run python run_pipeline_formative.py -m claude-sonnet-5
```
It builds the student's work for each task and posture, skips the cases where no error can be injected (printed as `[FILTERED]`), calls the model for the rest, and writes one report per posture. It also prints a dashboard and the 2x2 matrix, which needs the baseline report of the previous step.

### 9. Read the results

- `evaluation_report_<model>.json`: `aggregate_metrics` has the number of tasks, the number passed and the accuracy. `detailed_results` has, for each task, `passed` and, for each answer field, the ground truth next to the model's prediction.
- `formative_evaluation_report_posture_<n>_<model>.json`: the same layout, with `passed`, the `prediction` and the `ground_truth` (whether the work is correct, and the step with the error) for each task.
- `llm_outputs_<model>/` and `llm_outputs_formative_<model>/`: the raw answers, each with a `_usage` entry holding the output tokens and, for most providers, how the response ended (`stop_reason` or `status`). A value such as `max_tokens` or `length` means the model ran out of output tokens before answering. See [Model Settings](#model-settings).

To list the tasks a model got wrong:
```bash
python -c "import json; r = json.load(open('evaluation_report_claude-sonnet-5.json')); print([k for k, v in r['detailed_results'].items() if not v['passed']])"
```

### 10. Iterate

- Fix a task and rerun: delete its response file (for example `llm_outputs_claude-sonnet-5/TRUSS_0006_response.json`) so the model is asked again, and rerun the command. Add `--skip-prompt-generation` to reuse the existing payloads, and omit it after changing a task or a figure.
- Change a model setting: edit the constants at the top of its client in `llm_api/` (see [Model Settings](#model-settings)) and delete the old responses of that model.
- Changed the code in `src/`? Run `uv run pytest` before you rely on the results.

## Running Tests

To run the complete test suite and output a detailed coverage report for the `src` directory, use the following command:
```bash
uv run pytest -v --cov=src --cov-report=term-missing
```
The tests need no API keys: every provider call is mocked.
