# TrussTutor-Bench Evaluation Pipeline

This repository contains the automated pipeline for evaluating Large Language Models (LLMs) on 2D truss statics problems. It is designed to work alongside the TrussTutor-Bench benchmark tasks—which are authored as self-contained Python templates—to generate multimodal prompts, execute concurrent API calls, and rigorously evaluate the model's outputs.

The pipeline tests two distinct LLM capabilities:
1. **Baseline Problem Solving:** Can the model calculate the correct mathematical answer?
2. **Formative Tutoring:** Can the model accurately grade a student's step-by-step scratchpad (interleaved with Free Body Diagrams) and identify errors? This is evaluated across two postures:
   - **Posture 1 (Clean Work):** Tests whether the model correctly validates flawless student work without hallucinating errors that aren't there.
   - **Posture 2 (Flawed Work):** Tests whether the model can detect actual mathematical contradictions—specifically, injected Right-Hand Rule sign flips—and correctly localize the exact step where the error occurred.
   - **Posture 3 (Late-Stage Errors):** Tests whether the model can detect conceptual final-step errors, specifically Tension/Compression sign flips in Method of Sections problems, introduced after the core matrix math has been calculated correctly.

## Table of Contents
- [Setup](#setup)
- [Repository Structure](#repository-structure)
- [Running the Pipeline](#running-the-pipeline)
- [Running Tests](#running-tests)
- [Todo List](#todo-list)

## Setup

This project uses `uv` for fast dependency management.

**1. Activate your virtual environment:**
```bash
source .venv/bin/activate
```

**2. Sync dependencies (if not already installed):**
```bash
uv sync
```

**3. Set your API key:**

The pipeline relies on the Gemini API for model evaluations. You must export your key before running the execution scripts.
```bash
export GEMINI_API_KEY="your_api_key_here"
```

## Repository Structure

```
├── prompts/                                       # Generated baseline payloads (text + images)
├── prompts_formative/                             # Generated formative tutoring payloads
├── llm_api/                                       # API execution wrappers (e.g., run_gemini_mwe.py)
├── src/                                            # Core pipeline and evaluation logic
│   ├── trusstutor_bench/                           # Payload compilers, FBD generators, and evaluators
│   │   ├── direct_stiffness_method.py              # Direct stiffness method solver
│   │   ├── draw_FBDs.py                            # Free body diagram generation
│   │   ├── draw_truss.py                           # Truss diagram generation
│   │   ├── evaluate_outputs.py                     # Output evaluation logic
│   │   ├── markdown_generator.py                   # Markdown report generation
│   │   ├── corrupted_statics_tools.py              # Intentionally flawed macro variants used for systematic error injection
│   │   ├── math_helpers.py                         # Shared math utilities
│   │   ├── schemas.py                              # Data schemas/models
│   │   ├── statics_tools.py                        # Statics analysis utilities
│   │   ├── symbolic_solver.py                      # Symbolic equation solver
│   │   ├── task_base.py                            # Base task class
│   │   ├── task_loader.py                          # Task template loader
│   │   ├── task_to_prompt.py                       # Baseline prompt compiler
│   │   ├── task_to_prompt_formative.py             # Formative prompt compiler
│   │   └── truss_state.py                          # Truss state representation
├── tasks/                                          # Benchmark task templates targeted by the pipeline
│   ├── figs/                                       # Generated schematics for each task
├── tests/                                          # Pytest suite
│   ├── examples/                                   # Example python templates
│   ├── llm_outputs/                                # Saved JSON responses from the baseline API runs
│   ├── llm_outputs_formative/                      # Saved JSON responses from the formative API runs
│   ├── (pytest style tests for all code in src)
├── run_pipeline.py                                 # Executes the baseline solver evaluation
├── run_pipeline_formative.py                       # Executes the formative tutor evaluation (both postures)
├── evaluation_report.json                          # Generated baseline evaluation report
├── formative_evaluation_report_posture_1.json      # Generated Posture 1 (clean work) evaluation report
├── formative_evaluation_report_posture_2.json      # Generated Posture 2 (flawed work) evaluation report
└── formative_evaluation_report_posture_3.json      # Generated Posture 3 (late-stage errors) evaluation report
```

## Running the Pipeline

The framework strictly separates problem definition from execution logic. The pipeline scripts handle the entire lifecycle: converting the task templates into structured JSON payloads, firing the API calls concurrently, and evaluating the responses.

**To run the baseline evaluation:**
```bash
python run_pipeline.py
```

**To run the formative evaluation:**
```bash
python run_pipeline_formative.py
```
This script automatically executes a triple pass over the tasks in `tasks/`, once for each posture, and writes `formative_evaluation_report_posture_1.json`, `formative_evaluation_report_posture_2.json`, and `formative_evaluation_report_posture_3.json`. At the end of the run, a terminal dashboard prints a unified summary across all three postures.

It also cross-references the baseline results to generate a 2x2 diagnostic matrix. This matrix exclusively compares baseline solver performance against Posture 2 (flawed work) results, to analyze the model's ability to solve a problem outright versus its ability to spot a targeted, injected error in a student's solution.

## Running Tests

To run the complete test suite and output a detailed coverage report for the `src` directory, use the following command:
```bash
uv run pytest -v --cov=src --cov-report=term-missing
```

## Todo List
- [ ] Go through and validate what has been done so far
- [ ] Figure out the best way to interface this with the truss-tutor-support-code repository (i.e., how to run all the tasks we created easily)
- [ ] Run all examples with Gemini 3.1 Pro
- [ ] Debug examples (did it fail? why?)
- [ ] Think about preliminary results (e.g., textbook vs. original problems, formative feedback)
- [ ] Decide: do we want to implement more formative feedback testing (e.g., FBD checking, more extreme sycophancy, etc.)? Overall we need 3 results figures for this Letters-style paper, so we don't want to overcomplicate things unless it's helpful. Could imagine adding one more thing, but not much more than that.
- [ ] Implement additional models (Anthropic, OpenAI, one open source)
- [ ] Finalize/figure out paper results and writing
- [ ] More?
