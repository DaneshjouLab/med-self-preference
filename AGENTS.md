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

## Key Entry Points

- `generate_conversations.py`  
  Multi-turn conversation generation.
- `generate_single_turn.py`  
  Single-turn generation for MedDialog-based flow.
- `generate_single_turn_covid.py`  
  Single-turn generation for local `COVID-Dialogue-Dataset-English.txt`.
- `pairwise_evaluation.py`  
  Pairwise evaluation for multi-turn outputs.
- `pairwise_evaluation_single.py`  
  Pairwise evaluation for single-turn outputs.

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
2. Configure API keys in `.env` or environment:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `GOOGLE_API_KEY`

### Typical Commands

- Quick single-turn Covid parse:
  - `python generate_single_turn_covid.py --source_file ./COVID-Dialogue-Dataset-English.txt --num_scenarios 20 --parse_only`
- Single-turn Covid generation:
  - `python generate_single_turn_covid.py --source_file ./COVID-Dialogue-Dataset-English.txt --num_scenarios 100 --models gpt-4o --output_dir ./covid_dialogue_output`
- Single-turn pairwise eval:
  - `python pairwise_evaluation_single.py --help`

## Coding Conventions

- Prefer small, focused edits over broad refactors.
- Keep scripts runnable as CLIs with clear `argparse` flags.
- Reuse existing helper patterns (client factories, JSON save layout, cleanup helpers).
- Use deterministic sampling controls (`--seed`) when sampling.
- Keep output filenames and folder structure consistent with existing scripts.

## Validation Checklist (Before Claiming Done)

1. Syntax check changed Python files:
   - `python -m py_compile <file>.py`
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
