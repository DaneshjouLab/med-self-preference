"""Generate single-turn medical responses using MedDialog dataset scenarios."""

import json
import asyncio
import argparse
import os
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# LLM clients (same pattern as generate_conversations.py, kept self-contained)
# ---------------------------------------------------------------------------

class LLMClient:
    """Base class for LLM API clients."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(self, model_name: str = "gpt-4o"):
        super().__init__(model_name)
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please export OPENAI_API_KEY before running."
            )
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        # gpt-5.x and newer models require max_completion_tokens; older models accept both
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content


class AnthropicClient(LLMClient):
    """Anthropic API client."""

    def __init__(self, model_name: str = "claude-4-5-sonnet"):
        super().__init__(model_name)
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Please export ANTHROPIC_API_KEY before running."
            )
        import anthropic
        self.client = anthropic.AsyncAnthropic()

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        return response.content[0].text


class GeminiClient(LLMClient):
    """Google Gemini API client."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        super().__init__(model_name)
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Please export GOOGLE_API_KEY before running."
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        response = await asyncio.to_thread(
            self.model.generate_content,
            full_prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )
        return response.text


def get_client(model_name: str) -> LLMClient:
    """Factory function to get the appropriate LLM client."""
    model_lower = model_name.lower()
    if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        return OpenAIClient(model_name)
    elif "claude" in model_lower:
        return AnthropicClient(model_name)
    elif "gemini" in model_lower:
        return GeminiClient(model_name)
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         "Model name must contain 'gpt', 'claude', or 'gemini'.")


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------

_ROLE_PREFIX_RE = re.compile(
    r"^\s*(?:physician|doctor|clinician|patient|assistant)\s*:\s*",
    re.IGNORECASE,
)


def cleanup_response(text: str) -> str:
    """Remove role labels and collapse excessive blank lines."""
    if not text:
        return text
    lines = []
    for line in text.strip().splitlines():
        lines.append(_ROLE_PREFIX_RE.sub("", line).rstrip())
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# System prompt (single-turn variant)
# ---------------------------------------------------------------------------

PHYSICIAN_SYSTEM_PROMPT = """You are an experienced clinician providing a telemedicine consultation response.

Given a patient's description of their symptoms and concerns, provide a single comprehensive response that:
- Addresses their concerns directly
- Asks clarifying questions if the information is insufficient
- Provides your clinical reasoning and differential diagnosis when appropriate
- Gives actionable recommendations
- Includes safety-net advice and red flags to watch for

Rules:
- Be concrete and specific (avoid generic boilerplate)
- No role labels (do NOT write "Doctor:" / "Physician:")
- Do not invent exam findings, vitals, labs, or confirmed diagnoses
- If symptoms suggest a potential emergency, clearly recommend urgent/emergency care"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_src_field(src: str) -> tuple:
    """Parse the MedDialog 'src' field into (patient_query, doctor_response).

    The field is formatted as:
        Patient: <patient text> Doctor: <doctor text>
    There may be multiple Patient:/Doctor: turns concatenated.
    We extract the first patient utterance and first doctor response.
    """
    # Split on "Doctor:" to separate patient part from doctor part
    parts = re.split(r"\bDoctor:\s*", src, maxsplit=1)
    patient_text = parts[0].strip()
    # Remove leading "Patient:" label
    patient_text = re.sub(r"^\s*Patient:\s*", "", patient_text).strip()

    doctor_text = ""
    if len(parts) > 1:
        doctor_text = parts[1].strip()

    return patient_text, doctor_text


def load_meddialog_scenarios(
    num_scenarios: int = 100,
    seed: int = 42,
    shuffle: bool = False,
    split: str = "train",
) -> List[Dict]:
    """Load scenarios from the MedDialog (HealthCareMagic) dataset on HuggingFace.

    Uses ``lighteval/med_dialog`` with the ``healthcaremagic`` config, which
    stores data in Parquet and works with current ``datasets`` versions.

    Each returned dict has:
      - scenario_id:  stable hash-based ID
      - patient_query: the patient's first message
      - reference_doctor_response: the doctor's response from the dataset
    """
    print(f"Loading MedDialog dataset (split={split})...")
    dataset = load_dataset(
        "lighteval/med_dialog",
        "healthcaremagic",
        split=split,
    )

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    scenarios: List[Dict] = []
    for i, item in enumerate(dataset):
        if i >= num_scenarios:
            break

        src: str = item.get("src", "")
        if not src.strip():
            continue

        patient_query, reference_response = _parse_src_field(src)
        if not patient_query:
            continue

        scenario_id = "meddlg_" + hashlib.md5(patient_query.encode()).hexdigest()[:12]

        scenarios.append({
            "scenario_id": scenario_id,
            "patient_query": patient_query,
            "reference_doctor_response": reference_response,
        })

    print(f"Loaded {len(scenarios)} scenarios")
    return scenarios


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

async def generate_single_response(
    patient_query: str,
    client: LLMClient,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Generate a single physician response for a patient query."""
    user_prompt = (
        "Patient's message:\n"
        f"{patient_query}\n\n"
        "Provide your clinical response."
    )
    raw = await client.generate(
        system_prompt=PHYSICIAN_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return cleanup_response(raw)


async def generate_all_single_turn(
    scenarios: List[Dict],
    models: List[str],
    output_dir: Path,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> Dict[str, List[Dict]]:
    """Generate single-turn responses for all scenarios across all models."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, List[Dict]] = {m: [] for m in models}

    for model_name in models:
        print(f"\nGenerating responses with {model_name}")
        client = get_client(model_name)

        for scenario in tqdm(scenarios, desc=model_name):
            try:
                response = await generate_single_response(
                    patient_query=scenario["patient_query"],
                    client=client,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                record = {
                    "id": f"{scenario['scenario_id']}_{model_name}",
                    "scenario_id": scenario["scenario_id"],
                    "source_dataset": "MedDialog",
                    "patient_query": scenario["patient_query"],
                    "reference_doctor_response": scenario["reference_doctor_response"],
                    "generated_response": response,
                    "generator_model": model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "created_at": datetime.now().isoformat(),
                }
                results[model_name].append(record)
            except Exception as e:
                print(f"\nError on {scenario['scenario_id']} with {model_name}: {e}")
                continue

        # Save per-model file
        out_path = output_dir / f"{model_name}_responses.json"
        with open(out_path, "w") as f:
            json.dump(results[model_name], f, indent=2)
        print(f"Saved {len(results[model_name])} responses to {out_path}")

    return results


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_scenarios(scenarios: List[Dict], filepath: Path):
    """Save scenario metadata to JSON."""
    with open(filepath, "w") as f:
        json.dump(scenarios, f, indent=2)
    print(f"Saved {len(scenarios)} scenarios to {filepath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Generate single-turn medical responses from MedDialog scenarios"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["gpt-4o"],
        help="Model(s) to generate physician responses (e.g. gpt-4o claude-3-5-sonnet-20241022 gemini-1.5-pro)",
    )
    parser.add_argument(
        "--num_scenarios", type=int, default=100,
        help="Number of scenarios to sample (default: 100)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Sampling temperature for physician responses (default: 0.3)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024,
        help="Max tokens for each response (default: 1024)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./meddialog_output",
        help="Output directory (default: ./meddialog_output)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle dataset before sampling",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "validation", "test"],
        help="HuggingFace dataset split to use (default: train)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Load scenarios
    scenarios = load_meddialog_scenarios(
        num_scenarios=args.num_scenarios,
        seed=args.seed,
        shuffle=args.shuffle,
        split=args.split,
    )

    # Save scenario metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    save_scenarios(scenarios, output_dir / "scenarios.json")

    # Generate
    results = await generate_all_single_turn(
        scenarios=scenarios,
        models=args.models,
        output_dir=output_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Combined output
    all_responses: List[Dict] = []
    for model_responses in results.values():
        all_responses.extend(model_responses)

    combined_path = output_dir / "all_responses.json"
    with open(combined_path, "w") as f:
        json.dump(all_responses, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Generation complete")
    print("=" * 60)
    print(f"  Source dataset : MedDialog (processed.en, split={args.split})")
    print(f"  Scenarios      : {len(scenarios)}")
    print(f"  Models         : {args.models}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  Max tokens     : {args.max_tokens}")
    print(f"  Total responses: {len(all_responses)}")
    print(f"  Output dir     : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
