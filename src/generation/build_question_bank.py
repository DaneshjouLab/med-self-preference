"""Build a unified, deduplicated question bank across medical dialogue datasets.

Sources:
- HealthCareMagic-100k local JSON
- iCliniq local JSON
- COVID Dialogue local TXT
- MentalChat16K (HuggingFace, same flow as existing script)

Outputs:
- question_bank_<tag>.json
- question_bank_<tag>.csv
- overlap_report.json
- sampling_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_ORDER: List[str] = [
    "healthcaremagic-100k",
    "icliniq",
    "COVID Dialog",
    "MentalHealthChat16K",
]

SOURCE_PRIORITY = {source: index for index, source in enumerate(SOURCE_ORDER)}

CANONICAL_SOURCE_BY_NAME = {
    "healthcaremagic-100k": "healthcaremagic_100k_local_json",
    "icliniq": "icliniq_local_json",
    "COVID Dialog": "covid_dialog_local_txt",
    "MentalHealthChat16K": "mentalchat16k_hf",
}

_PATIENT_RE = re.compile(r"^\s*Patient\s*:\s*", re.IGNORECASE)
_DOCTOR_RE = re.compile(r"^\s*Doctor\s*:\s*", re.IGNORECASE)


def resolve_path(path_str: str) -> Path:
    """Resolve relative paths from project root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def now_iso() -> str:
    """Return UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_question_text(text: str) -> str:
    """Normalize question text for exact overlap detection."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
    normalized = normalized.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    return normalized


def question_hash(question_text: str) -> str:
    """Hash normalized question text for deduplication and overlap checks."""
    normalized = normalize_question_text(question_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def build_question_id(question_hash_normalized: str, canonical_source: str) -> str:
    """Build stable question id for the output bank."""
    raw = f"{question_hash_normalized}::{canonical_source}"
    return "qb_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def normalize_block_text(text: str) -> str:
    """Trim and collapse excessive blank lines while preserving paragraphs."""
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def split_consultation_blocks(raw_text: str) -> List[Tuple[str, str]]:
    """Split COVID TXT into (source_record_id, block_text)."""
    id_matches = list(re.finditer(r"(?mi)^id\s*=\s*(\d+)\s*$", raw_text))
    blocks: List[Tuple[str, str]] = []

    for index, match in enumerate(id_matches):
        start = match.end()
        end = id_matches[index + 1].start() if index + 1 < len(id_matches) else len(raw_text)
        source_record_id = match.group(1)
        block = raw_text[start:end].strip()
        if block:
            blocks.append((source_record_id, block))

    return blocks


def extract_first_turn_from_block(block_text: str) -> Tuple[str, str]:
    """Extract first Patient turn and first Doctor turn from a COVID block."""
    dialogue_match = re.search(r"(?mi)^Dialogue\s*$", block_text)
    relevant = block_text[dialogue_match.end():] if dialogue_match else block_text

    lines = relevant.splitlines()
    line_index = 0
    patient_text = ""
    doctor_text = ""

    while line_index < len(lines):
        if _PATIENT_RE.match(lines[line_index]):
            patient_lines: List[str] = [_PATIENT_RE.sub("", lines[line_index]).strip()]
            line_index += 1
            while line_index < len(lines) and not _DOCTOR_RE.match(lines[line_index]) and not _PATIENT_RE.match(lines[line_index]):
                patient_lines.append(lines[line_index].strip())
                line_index += 1
            patient_text = normalize_block_text("\n".join([line for line in patient_lines if line]))
            break
        line_index += 1

    if not patient_text:
        return "", ""

    while line_index < len(lines) and not _DOCTOR_RE.match(lines[line_index]):
        line_index += 1
    if line_index >= len(lines):
        return patient_text, ""

    doctor_lines: List[str] = [_DOCTOR_RE.sub("", lines[line_index]).strip()]
    line_index += 1
    while line_index < len(lines) and not _PATIENT_RE.match(lines[line_index]):
        doctor_lines.append(lines[line_index].strip())
        line_index += 1
    doctor_text = normalize_block_text("\n".join([line for line in doctor_lines if line]))

    return patient_text, doctor_text


def make_candidate(
    *,
    source_name: str,
    question_text: str,
    ground_truth_response: str,
    source_row_index: int,
    source_split: str,
    source_record_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Create standardized candidate row."""
    question_text = (question_text or "").strip()
    ground_truth_response = (ground_truth_response or "").strip()
    if not question_text or not ground_truth_response:
        return None

    normalized = normalize_question_text(question_text)
    if not normalized:
        return None

    return {
        "source_name": source_name,
        "canonical_source": CANONICAL_SOURCE_BY_NAME[source_name],
        "source_split": source_split,
        "source_row_index": source_row_index,
        "source_record_id": source_record_id,
        "question_text": question_text,
        "ground_truth_response": ground_truth_response,
        "question_hash_normalized": question_hash(question_text),
    }


def load_local_json_candidates(
    *,
    file_path: Path,
    source_name: str,
    answer_field: str,
    source_split: str = "local",
) -> List[Dict[str, Any]]:
    """Load candidates from a local JSON array with `input` + answer field."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected top-level JSON array in {file_path}")

    candidates: List[Dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        candidate = make_candidate(
            source_name=source_name,
            question_text=str(item.get("input", "")),
            ground_truth_response=str(item.get(answer_field, "")),
            source_row_index=index,
            source_split=source_split,
            source_record_id=str(item.get("id")) if item.get("id") is not None else None,
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def load_covid_dialog_candidates(file_path: Path) -> List[Dict[str, Any]]:
    """Load first-turn Patient/Doctor pairs from local COVID dialogue TXT."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        raw_text = file.read()

    blocks = split_consultation_blocks(raw_text)
    candidates: List[Dict[str, Any]] = []

    for index, (source_record_id, block_text) in enumerate(blocks):
        question_text, ground_truth_response = extract_first_turn_from_block(block_text)
        candidate = make_candidate(
            source_name="COVID Dialog",
            question_text=question_text,
            ground_truth_response=ground_truth_response,
            source_row_index=index,
            source_split="local",
            source_record_id=source_record_id,
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def load_mentalchat_candidates(dataset_name: str, split: str) -> List[Dict[str, Any]]:
    """Load candidates from MentalChat16K via HuggingFace datasets."""
    dataset = load_dataset(dataset_name, split=split)
    candidates: List[Dict[str, Any]] = []

    for index, item in enumerate(dataset):
        candidate = make_candidate(
            source_name="MentalHealthChat16K",
            question_text=str(item.get("input", "")),
            ground_truth_response=str(item.get("output", "")),
            source_row_index=index,
            source_split=split,
            source_record_id=str(item.get("id")) if item.get("id") is not None else None,
        )
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def cap_candidates(candidates: List[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
    """Optionally cap candidate rows per source for quick test runs."""
    if cap <= 0:
        return candidates
    return candidates[:cap]


def group_by_question_hash(candidates: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group candidate rows by normalized question hash."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["question_hash_normalized"]].append(candidate)
    return grouped


def compute_overlap_report(
    grouped_by_hash: Dict[str, List[Dict[str, Any]]],
    candidates: Sequence[Dict[str, Any]],
    examples_per_pair: int,
) -> Dict[str, Any]:
    """Compute overlap statistics and pairwise examples."""
    source_candidate_counts = {source: 0 for source in SOURCE_ORDER}
    source_hashes: Dict[str, set[str]] = {source: set() for source in SOURCE_ORDER}

    for candidate in candidates:
        source = candidate["source_name"]
        source_candidate_counts[source] += 1
        source_hashes[source].add(candidate["question_hash_normalized"])

    overlap_hash_count = 0
    pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    pair_examples: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for hash_value, rows in grouped_by_hash.items():
        unique_sources = sorted(
            {row["source_name"] for row in rows},
            key=lambda source: SOURCE_PRIORITY[source],
        )
        if len(unique_sources) > 1:
            overlap_hash_count += 1
        for source_a, source_b in combinations(unique_sources, 2):
            pair_key = (source_a, source_b)
            pair_counts[pair_key] += 1
            if len(pair_examples[pair_key]) < examples_per_pair:
                preview = rows[0]["question_text"][:280]
                pair_examples[pair_key].append(
                    {
                        "question_hash_normalized": hash_value,
                        "question_preview": preview,
                        "sources": unique_sources,
                    }
                )

    pairwise_overlap: List[Dict[str, Any]] = []
    for source_a, source_b in combinations(SOURCE_ORDER, 2):
        pair_key = (source_a, source_b)
        pairwise_overlap.append(
            {
                "source_a": source_a,
                "source_b": source_b,
                "shared_unique_questions": pair_counts.get(pair_key, 0),
                "examples": pair_examples.get(pair_key, []),
            }
        )

    pairwise_overlap.sort(key=lambda item: item["shared_unique_questions"], reverse=True)

    return {
        "generated_at": now_iso(),
        "normalization": "NFKC, lowercase, collapse whitespace, trim spaces before punctuation",
        "source_priority_for_overlap_resolution": SOURCE_ORDER,
        "source_candidate_counts": source_candidate_counts,
        "source_unique_question_counts": {
            source: len(source_hashes[source]) for source in SOURCE_ORDER
        },
        "total_unique_questions_global": len(grouped_by_hash),
        "overlapping_unique_questions_global": overlap_hash_count,
        "overlap_rate_global": (
            overlap_hash_count / len(grouped_by_hash) if grouped_by_hash else 0.0
        ),
        "pairwise_overlap": pairwise_overlap,
    }


def select_representative_rows(
    grouped_by_hash: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Resolve duplicate questions across sources by deterministic priority."""
    representatives: List[Dict[str, Any]] = []

    for rows in grouped_by_hash.values():
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                SOURCE_PRIORITY[row["source_name"]],
                int(row["source_row_index"]),
            ),
        )
        primary = dict(sorted_rows[0])
        sources = sorted(
            {row["source_name"] for row in sorted_rows},
            key=lambda source: SOURCE_PRIORITY[source],
        )
        primary["overlap_sources"] = [source for source in sources if source != primary["source_name"]]
        representatives.append(primary)

    return representatives


def make_equal_targets(total_target: int) -> Dict[str, int]:
    """Distribute target rows equally across configured sources."""
    source_count = len(SOURCE_ORDER)
    base = total_target // source_count
    remainder = total_target % source_count
    return {
        source: base + (1 if index < remainder else 0)
        for index, source in enumerate(SOURCE_ORDER)
    }


def sample_with_backfill(
    representative_rows: Sequence[Dict[str, Any]],
    total_target: int,
    seed: int,
) -> Dict[str, Any]:
    """Sample rows with equal quotas, then backfill shortages proportionally."""
    pools: Dict[str, List[Dict[str, Any]]] = {source: [] for source in SOURCE_ORDER}
    for row in representative_rows:
        pools[row["source_name"]].append(row)

    for source in SOURCE_ORDER:
        pools[source].sort(
            key=lambda row: (int(row["source_row_index"]), row["question_hash_normalized"])
        )
        rng = random.Random(seed + SOURCE_PRIORITY[source] * 10007)
        rng.shuffle(pools[source])

    targets = make_equal_targets(total_target)
    selected_by_source: Dict[str, List[Dict[str, Any]]] = {source: [] for source in SOURCE_ORDER}
    selected_counts: Dict[str, int] = {source: 0 for source in SOURCE_ORDER}
    initial_selected_counts: Dict[str, int] = {source: 0 for source in SOURCE_ORDER}

    for source in SOURCE_ORDER:
        take = min(targets[source], len(pools[source]))
        selected_by_source[source].extend(pools[source][:take])
        selected_counts[source] = take
        initial_selected_counts[source] = take

    backfill_allocations: Dict[str, int] = {source: 0 for source in SOURCE_ORDER}
    deficit = total_target - sum(selected_counts.values())

    while deficit > 0:
        capacities = {
            source: len(pools[source]) - selected_counts[source]
            for source in SOURCE_ORDER
        }
        total_capacity = sum(capacities.values())
        if total_capacity <= 0:
            break

        if deficit >= total_capacity:
            round_alloc = capacities
        else:
            raw_alloc = {
                source: (deficit * capacities[source] / total_capacity)
                for source in SOURCE_ORDER
            }
            round_alloc = {
                source: min(capacities[source], int(math.floor(raw_alloc[source])))
                for source in SOURCE_ORDER
            }
            assigned = sum(round_alloc.values())
            leftover = deficit - assigned

            if leftover > 0:
                ranked = sorted(
                    SOURCE_ORDER,
                    key=lambda source: (
                        raw_alloc[source] - round_alloc[source],
                        -SOURCE_PRIORITY[source],
                    ),
                    reverse=True,
                )
                for source in ranked:
                    if leftover <= 0:
                        break
                    if round_alloc[source] < capacities[source]:
                        round_alloc[source] += 1
                        leftover -= 1

            if sum(round_alloc.values()) == 0:
                for source in SOURCE_ORDER:
                    if capacities[source] > 0:
                        round_alloc[source] = 1
                        break

        for source in SOURCE_ORDER:
            allocation = round_alloc.get(source, 0)
            if allocation <= 0:
                continue
            start = selected_counts[source]
            end = start + allocation
            selected_by_source[source].extend(pools[source][start:end])
            selected_counts[source] = end
            backfill_allocations[source] += allocation

        deficit = total_target - sum(selected_counts.values())

    selected_rows: List[Dict[str, Any]] = []
    for source in SOURCE_ORDER:
        selected_rows.extend(selected_by_source[source])

    return {
        "selected_rows": selected_rows,
        "targets": targets,
        "initial_selected_counts": initial_selected_counts,
        "selected_counts": selected_counts,
        "backfill_allocations": backfill_allocations,
        "pool_sizes": {source: len(pools[source]) for source in SOURCE_ORDER},
        "unfilled_target": max(0, total_target - len(selected_rows)),
    }


def enrich_final_rows(rows: Sequence[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """Attach final metadata fields required by the question bank schema."""
    created_at = now_iso()
    final_rows: List[Dict[str, Any]] = []

    for row in rows:
        overlap_sources = row.get("overlap_sources", [])
        if not isinstance(overlap_sources, list):
            overlap_sources = []
        final_row = {
            "question_id": build_question_id(
                row["question_hash_normalized"],
                row["canonical_source"],
            ),
            "question_text": row["question_text"],
            "ground_truth_response": row["ground_truth_response"],
            "source_name": row["source_name"],
            "canonical_source": row["canonical_source"],
            "source_split": row["source_split"],
            "source_row_index": row["source_row_index"],
            "source_record_id": row.get("source_record_id"),
            "question_hash_normalized": row["question_hash_normalized"],
            "overlap_sources": overlap_sources,
            "sampling_seed": seed,
            "created_at": created_at,
        }
        final_rows.append(final_row)

    return final_rows


def write_json(path: Path, payload: Any) -> None:
    """Write JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """Write final question bank rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "question_text",
        "ground_truth_response",
        "source_name",
        "canonical_source",
        "source_split",
        "source_row_index",
        "source_record_id",
        "question_hash_normalized",
        "overlap_sources",
        "sampling_seed",
        "created_at",
    ]

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = dict(row)
            out_row["overlap_sources"] = "|".join(out_row.get("overlap_sources", []))
            writer.writerow(out_row)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build deduplicated 10k question bank with overlap reporting."
    )
    parser.add_argument(
        "--target_total",
        type=int,
        default=10000,
        help="Target number of rows in the final question bank (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./question_bank_output",
        help="Output directory for artifacts (default: ./question_bank_output).",
    )
    parser.add_argument(
        "--healthcaremagic_file",
        type=str,
        default="./HealthCareMagic-100k.json",
        help="Path to HealthCareMagic local JSON file.",
    )
    parser.add_argument(
        "--icliniq_file",
        type=str,
        default="./iCliniq.json",
        help="Path to iCliniq local JSON file.",
    )
    parser.add_argument(
        "--covid_file",
        type=str,
        default="./COVID-Dialogue-Dataset-English.txt",
        help="Path to COVID dialogue local TXT file.",
    )
    parser.add_argument(
        "--mental_dataset",
        type=str,
        default="ShenLab/MentalChat16K",
        help="HuggingFace dataset ID for mental health source.",
    )
    parser.add_argument(
        "--mental_split",
        type=str,
        default="train",
        help="Dataset split for mental source (default: train).",
    )
    parser.add_argument(
        "--overlap_examples_per_pair",
        type=int,
        default=5,
        help="Max overlap examples stored per source pair (default: 5).",
    )
    parser.add_argument(
        "--source_cap",
        type=int,
        default=0,
        help=(
            "Optional cap on loaded candidates per source for quick testing. "
            "Use 0 to disable (default)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Build the question bank and all supporting artifacts."""
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    healthcaremagic_path = resolve_path(args.healthcaremagic_file)
    icliniq_path = resolve_path(args.icliniq_file)
    covid_path = resolve_path(args.covid_file)

    print("Loading source candidates...")
    healthcaremagic_candidates = cap_candidates(
        load_local_json_candidates(
            file_path=healthcaremagic_path,
            source_name="healthcaremagic-100k",
            answer_field="output",
        ),
        args.source_cap,
    )
    icliniq_candidates = cap_candidates(
        load_local_json_candidates(
            file_path=icliniq_path,
            source_name="icliniq",
            answer_field="answer_icliniq",
        ),
        args.source_cap,
    )
    covid_candidates = cap_candidates(
        load_covid_dialog_candidates(covid_path),
        args.source_cap,
    )
    mentalchat_candidates = cap_candidates(
        load_mentalchat_candidates(args.mental_dataset, args.mental_split),
        args.source_cap,
    )

    candidates = (
        healthcaremagic_candidates
        + icliniq_candidates
        + covid_candidates
        + mentalchat_candidates
    )

    source_candidate_counts = {
        "healthcaremagic-100k": len(healthcaremagic_candidates),
        "icliniq": len(icliniq_candidates),
        "COVID Dialog": len(covid_candidates),
        "MentalHealthChat16K": len(mentalchat_candidates),
    }
    print("Loaded candidate counts by source:")
    for source in SOURCE_ORDER:
        print(f"  - {source}: {source_candidate_counts[source]}")

    if not candidates:
        raise RuntimeError("No valid candidates loaded from any source.")

    grouped_by_hash = group_by_question_hash(candidates)
    overlap_report = compute_overlap_report(
        grouped_by_hash,
        candidates,
        args.overlap_examples_per_pair,
    )
    representatives = select_representative_rows(grouped_by_hash)

    sampling_result = sample_with_backfill(
        representative_rows=representatives,
        total_target=args.target_total,
        seed=args.seed,
    )
    final_rows = enrich_final_rows(sampling_result["selected_rows"], args.seed)

    unique_hashes = {row["question_hash_normalized"] for row in final_rows}
    if len(unique_hashes) != len(final_rows):
        raise RuntimeError("Duplicate question hashes detected in final output.")

    bank_tag = "10k" if args.target_total == 10000 else str(args.target_total)
    question_bank_json_path = output_dir / f"question_bank_{bank_tag}.json"
    question_bank_csv_path = output_dir / f"question_bank_{bank_tag}.csv"
    overlap_report_path = output_dir / "overlap_report.json"
    sampling_manifest_path = output_dir / "sampling_manifest.json"

    write_json(question_bank_json_path, final_rows)
    write_csv(question_bank_csv_path, final_rows)
    write_json(overlap_report_path, overlap_report)

    manifest = {
        "generated_at": now_iso(),
        "target_total": args.target_total,
        "final_total": len(final_rows),
        "unfilled_target": sampling_result["unfilled_target"],
        "sampling_seed": args.seed,
        "source_priority_for_overlap_resolution": SOURCE_ORDER,
        "source_inputs": {
            "healthcaremagic_file": str(healthcaremagic_path),
            "icliniq_file": str(icliniq_path),
            "covid_file": str(covid_path),
            "mental_dataset": args.mental_dataset,
            "mental_split": args.mental_split,
        },
        "candidate_counts_by_source": source_candidate_counts,
        "unique_pool_sizes_after_dedup_by_source": sampling_result["pool_sizes"],
        "equal_targets_by_source": sampling_result["targets"],
        "selected_counts_after_initial_pass": sampling_result["initial_selected_counts"],
        "backfill_allocations_by_source": sampling_result["backfill_allocations"],
        "final_selected_counts_by_source": sampling_result["selected_counts"],
        "output_files": {
            "question_bank_json": str(question_bank_json_path),
            "question_bank_csv": str(question_bank_csv_path),
            "overlap_report_json": str(overlap_report_path),
            "sampling_manifest_json": str(sampling_manifest_path),
        },
    }
    write_json(sampling_manifest_path, manifest)

    print("\nBuild complete")
    print(f"  Final rows             : {len(final_rows)}")
    print(f"  Target rows            : {args.target_total}")
    print(f"  Unfilled target        : {sampling_result['unfilled_target']}")
    print(f"  Unique final questions : {len(unique_hashes)}")
    print(f"  Output dir             : {output_dir}")
    print(f"  Question bank JSON     : {question_bank_json_path.name}")
    print(f"  Question bank CSV      : {question_bank_csv_path.name}")
    print(f"  Overlap report         : {overlap_report_path.name}")
    print(f"  Sampling manifest      : {sampling_manifest_path.name}")


if __name__ == "__main__":
    main()
