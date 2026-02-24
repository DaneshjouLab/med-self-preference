# Medical Self-Preference 
This project generates multi-turn medical conversations and uses LLM-as-judge evaluation to detect self-preference bias.

## Quick Start

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys** (create `.env` file in project root):
   ```
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GOOGLE_API_KEY=...  
   ```

3. **Verify setup:**
   ```bash
   python test_generation.py
   ```

## Core Tools

### 1. Generate Conversations
Create multi-turn medical dialogues using different LLM models.

```bash
python generate_conversations.py \
  --num_scenarios 100 \
  --turns 2 \
  --models gpt-4 \
  --patient_model gpt-4 \
  --output_dir ./output
```

**Key options:**
- `--num_scenarios`: Number of conversations to generate (default: 100)
- `--turns`: Turns per conversation (default: 8)
- `--models`: List of physician models (e.g., `gpt-4`, `claude-sonnet-4-5-20250929`)
- `--patient_model`: Which model simulates the patient (default: gpt-4)
- `--repair`: Auto-fix turns that violate constraints
- `--output_dir`: Where to save conversations

**Output:** Saves conversations as `{model}_{turns}t_conversations.json` (e.g., `gpt-4_2t_conversations.json`)

### 1b. Generate Single-Turn Covid Dialogue Responses
Create single-turn physician responses from a local Covid dialogue source file.

```bash
python generate_single_turn_covid.py \
  --source_file ./COVID-Dialogue-Dataset-English.txt \
  --num_scenarios 100 \
  --models gpt-4o \
  --output_dir ./covid_dialogue_output
```

**Useful options:**
- `--source_file`: Path to the local `COVID-Dialogue-Dataset-English.txt` file
- `--parse_only`: Parse source file and write `scenarios.json`, skip model generation
- `--shuffle`, `--seed`: Randomized scenario sampling controls

**Output:** `scenarios.json`, `{model}_responses.json`, and `all_responses.json` in your output directory.

### 2. Run Pairwise Evaluation
Compare two models' responses to detect self-preference bias.

```bash
python pairwise_evaluation.py \
  --model_a_file example_conversations/gpt-4_2t_conversations.json \
  --model_b_file example_conversations/claude-sonnet-4-5-20250929_2t_conversations.json \
  --judge_model claude-3-5-sonnet-20241022 \
  --output results/pairwise_claude_judge.json
```

**Key options:**
- `--model_a_file`, `--model_b_file`: Conversation files to compare
- `--judge_model`: Which model evaluates (default: Claude). Use `gpt-4` to test if GPT-4 shows self-preference.
- `--output`: Where to save results

**Output:** JSON with preference scores, win rates, and detailed metric breakdown.

## Evaluation Metrics

Each response is scored 0-5 on:

1. **Faithfulness** - Medical accuracy and appropriateness
2. **Completeness** - Covers diagnosis, medications, follow-up, warning signs
3. **Safety** - Red flag detection, emergency guidance
4. **Clarity** - Communication quality for patient understanding
5. **Conciseness** - Appropriate length, no excessive repetition

## Configuration

Optional: Use `config.yaml` to set defaults instead of command-line args:

```yaml
data:
  num_scenarios: 100

generation:
  turns_per_conversation: 8
  physician_models:
    - gpt-4
    - claude-sonnet-4-5-20250929
  patient_simulator: gpt-4
  physician_temperature: 0.3
  patient_temperature: 0.8
  max_tokens_per_turn: 500
```

## Visualization

Browse conversations in a web UI:

```bash
cd visualizer
npm run dev
```

Then visit: `http://localhost:3000?file=../example_conversations/gpt-4_2t_conversations.json`
