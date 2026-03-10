"""Generate single-turn counseling-style responses using MentalChat16K scenarios."""

import json
import asyncio
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset

from generate_single_turn import generate_all_single_turn, save_scenarios


def load_mentalchat16k_scenarios(
    num_scenarios: int = 100,
    seed: int = 42,
    shuffle: bool = False,
    split: str = "train",
    start_index: int = 0,
) -> List[Dict]:
    """Load scenarios from ShenLab/MentalChat16K on HuggingFace."""
    print(f"Loading MentalChat16K dataset (split={split})...")
    dataset = load_dataset("ShenLab/MentalChat16K", split=split)

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    scenarios: List[Dict] = []
    collected = 0
    for i, item in enumerate(dataset):
        if i < start_index:
            continue
        if collected >= num_scenarios:
            break

        patient_query = (item.get("input") or "").strip()
        reference_response = (item.get("output") or "").strip()
        if not patient_query:
            continue

        scenario_id = "mentalchat_" + hashlib.md5(patient_query.encode()).hexdigest()[:12]
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "patient_query": patient_query,
                "reference_doctor_response": reference_response,
            }
        )
        collected += 1

    print(f"Loaded {len(scenarios)} scenarios (start_index={start_index})")
    return scenarios


async def main():
    parser = argparse.ArgumentParser(
        description="Generate single-turn responses from MentalChat16K scenarios"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o"],
        help="Model(s) to generate responses (e.g. gpt-5.2 claude-opus-4-6)",
    )
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=100,
        help="Number of scenarios to sample (default: 100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for responses (default: 0.3)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Max tokens for each response (default: 1024)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./mentalchat16k_output",
        help="Output directory (default: ./mentalchat16k_output)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dataset before sampling",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="HuggingFace dataset split to use (default: train)",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Skip this many items from the start (default: 0).",
    )
    parser.add_argument(
        "--parse_only",
        action="store_true",
        help="Only parse and write scenarios.json, then exit",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    scenarios = load_mentalchat16k_scenarios(
        num_scenarios=args.num_scenarios,
        seed=args.seed,
        shuffle=args.shuffle,
        split=args.split,
        start_index=args.start_index,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_scenarios(scenarios, output_dir / "scenarios.json")

    if args.parse_only:
        print("Parse complete (parse-only mode).")
        return

    results = await generate_all_single_turn(
        scenarios=scenarios,
        models=args.models,
        output_dir=output_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        source_dataset="ShenLab/MentalChat16K",
    )

    all_responses: List[Dict] = []
    for model_responses in results.values():
        all_responses.extend(model_responses)

    combined_path = output_dir / "all_responses.json"
    with open(combined_path, "w") as f:
        json.dump(all_responses, f, indent=2)

    print("\n" + "=" * 60)
    print("Generation complete")
    print("=" * 60)
    print(f"  Source dataset : ShenLab/MentalChat16K (split={args.split})")
    print(f"  Scenarios      : {len(scenarios)}")
    print(f"  Models         : {args.models}")
    print(f"  Temperature    : {args.temperature}")
    print(f"  Max tokens     : {args.max_tokens}")
    print(f"  Total responses: {len(all_responses)}")
    print(f"  Output dir     : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
