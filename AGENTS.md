# AGENTS.md

This file provides working guidance for AI coding agents in this repository.

## Repository Purpose

`med-self-preference` studies model self-preference bias in medical dialogue tasks by:
- generating medical conversations/responses with multiple LLMs,
- running pairwise LLM-judge evaluations,
- summarizing outputs for analysis.

## Tech Stack

- Python scripts for generation/evaluation
- JSON artifacts for datasets, generations, and eval outputs
- Optional Next.js visualizer in `visualizer/`

## Project Structure

```
config/              # config.yaml
src/
  generation/        # generate_conversations.py, generate_single_turn*.py
  evaluation/        # pairwise_*.py, individual_*.py, generate_eval_report.py
  test_generation.py # Smoke test
example_conversations/   # Multi-turn outputs
meddialog_output/   # Single-turn MedDialog outputs
covid_dialogue_output/  # Single-turn COVID outputs
evals/              # Pairwise evaluation results
```

## Key Entry Points

- `src/generation/generate_conversations.py` - Multi-turn conversation generation.
- `src/generation/generate_single_turn.py` - Single-turn generation for MedDialog-based flow.
- `src/generation/generate_single_turn_covid.py` - Single-turn generation for local `COVID-Dialogue-Dataset-English.txt`.
- `src/evaluation/pairwise_evaluation.py` - Pairwise evaluation for multi-turn outputs.
- `src/evaluation/pairwise_evaluation_single.py` - Pairwise evaluation for single-turn outputs.
- `src/evaluation/individual_evaluation_single.py` - Per-response scoring (single-turn).
- `src/evaluation/generate_eval_report.py` - Human-readable report from pairwise JSON.
- `src/evaluation/compare_individual_scores.py` - Compare individual eval scores for same scenario_ids across two models.
- `src/test_generation.py` - Smoke test (API keys, dataset load).

## Expected Data Contracts

When producing single-turn outputs, keep schema compatible with `pairwise_evaluation_single.py`:
- Required fields in response records:
  - `scenario_id`
  - `patient_query`
  - `generated_response`
  - `generator_model`
- Common additional fields:
  - `id`
  - `reference_doctor_response`
  - `temperature`
  - `max_tokens`
  - `created_at`

Preserve backwards compatibility unless explicitly requested otherwise.

## Runbook

### Setup

1. Install deps:
   - `pip install -r requirements.txt`
2. Configure API keys in `.env` or environment (project root):
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `GOOGLE_API_KEY`
3. Config: `config/config.yaml` (used by generate_conversations.py)

### Typical Commands

- Quick single-turn Covid parse:
  - `python src/generation/generate_single_turn_covid.py --source_file ./COVID-Dialogue-Dataset-English.txt --num_scenarios 20 --parse_only`
- Single-turn Covid generation:
  - `python src/generation/generate_single_turn_covid.py --source_file ./COVID-Dialogue-Dataset-English.txt --num_scenarios 100 --models gpt-4o --output_dir ./covid_dialogue_output`
- Single-turn pairwise eval:
  - `python src/evaluation/pairwise_evaluation_single.py --help`
- Smoke test:
  - `python src/test_generation.py` (or `make test`)
- Compare individual scores (after running individual eval on both models):
  - `python src/evaluation/compare_individual_scores.py --model_a_file evals/covid_march_2/individual_gpt52.json --model_b_file evals/covid_march_2/individual_opus46.json --output evals/covid_march_2/compare_individual.json`

## Coding Conventions

- Prefer small, focused edits over broad refactors.
- Keep scripts runnable as CLIs with clear `argparse` flags.
- Reuse existing helper patterns (client factories, JSON save layout, cleanup helpers).
- Use deterministic sampling controls (`--seed`) when sampling.
- Keep output filenames and folder structure consistent with existing scripts.

## Validation Checklist (Before Claiming Done)

1. Syntax check changed Python files:
   - `python -m py_compile src/<path>/<file>.py`
2. Run a small smoke test (2-5 scenarios).
3. Verify output artifacts exist and schema matches evaluator expectations.
4. Check lints/diagnostics for touched files.

## Safety and Data Hygiene

- Do not hardcode secrets or API keys.
- Do not commit `.env` contents or credentials.
- Avoid destructive git commands unless explicitly requested.
- Do not rewrite or delete unrelated user changes in a dirty worktree.

## Scope Guidance for Agents

- If asked for dataset-specific adaptation, update only data-loading/parsing and keep generation/evaluation interfaces stable.
- If proposing schema changes, explain impact on downstream evaluators before implementation.
- Prefer introducing new scripts (e.g., dataset-specific) over breaking existing baseline scripts unless requested.
