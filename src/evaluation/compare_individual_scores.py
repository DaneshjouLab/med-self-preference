#!/usr/bin/env python3
"""
Compare individual evaluation scores across two models.

Takes two individual evaluation JSON outputs (from individual_evaluation_single.py),
matches entries by scenario_id, and produces a head-to-head comparison with
win counts per metric and per-scenario results.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

METRICS = ["faithfulness", "completeness", "safety", "clarity", "conciseness", "overall"]


def load_individual_eval(path: str) -> Tuple[Dict[str, dict], str, str]:
    """
    Load individual eval JSON and index scores by scenario_id.

    Returns:
        (index: {scenario_id: score_dict}, model_name, judge_model)
    """
    with open(path, "r") as f:
        data = json.load(f)

    scores = data.get("scores", [])
    if not scores:
        raise ValueError(f"No scores in {path}")

    model_name = scores[0].get("generator_model", "unknown")
    judge_model = data.get("metadata", {}).get("judge_model", "unknown")

    index = {}
    for s in scores:
        sid = s.get("scenario_id")
        if sid:
            index[sid] = s

    return index, model_name, judge_model


def compare_scores(
    index_a: Dict[str, dict],
    index_b: Dict[str, dict],
    model_a: str,
    model_b: str,
) -> Tuple[List[dict], dict]:
    """
    Compare scores for matched scenario_ids.

    Returns:
        (comparisons: list of per-scenario comparison dicts, summary: aggregate stats)
    """
    common = sorted(set(index_a.keys()) & set(index_b.keys()))
    if not common:
        raise ValueError("No overlapping scenario_ids between the two files")

    if len(common) < min(len(index_a), len(index_b)):
        print(f"Warning: Only {len(common)} scenarios overlap (file A: {len(index_a)}, file B: {len(index_b)})")

    by_metric: Dict[str, Dict[str, int]] = {
        m: {"model_a_wins": 0, "model_b_wins": 0, "ties": 0} for m in METRICS
    }
    delta_sums: Dict[str, float] = {m: 0.0 for m in METRICS}

    comparisons = []
    for sid in common:
        sa = index_a[sid]
        sb = index_b[sid]

        by_metric_result = {}
        for m in METRICS:
            va = sa.get(m, 0.0)
            vb = sb.get(m, 0.0)
            delta = va - vb
            delta_sums[m] += delta

            if va > vb:
                winner = "A"
                by_metric[m]["model_a_wins"] += 1
            elif vb > va:
                winner = "B"
                by_metric[m]["model_b_wins"] += 1
            else:
                winner = "tie"
                by_metric[m]["ties"] += 1
            by_metric_result[m] = winner

        comparisons.append({
            "scenario_id": sid,
            "model_a": model_a,
            "model_b": model_b,
            "a_overall": sa.get("overall"),
            "b_overall": sb.get("overall"),
            "winner_overall": by_metric_result["overall"],
            "by_metric": by_metric_result,
        })

    n = len(comparisons)
    avg_deltas = {m: delta_sums[m] / n for m in METRICS}

    summary = {
        "by_metric": by_metric,
        "avg_deltas": avg_deltas,
    }

    return comparisons, summary


def generate_report(
    comparisons: List[dict],
    summary: dict,
    metadata: dict,
) -> str:
    """Generate human-readable text report."""
    model_a = metadata.get("models", ["A", "B"])[0]
    model_b = metadata.get("models", ["A", "B"])[1]
    n = len(comparisons)

    lines = [
        "=" * 70,
        "INDIVIDUAL SCORE COMPARISON (SAME ENTRIES)",
        "=" * 70,
        "",
        f"Input files: {metadata.get('input_files', [])}",
        f"Models: {model_a} vs {model_b}",
        f"Matched scenarios: {n}",
        f"Timestamp: {metadata.get('timestamp', '')}",
        "",
        "WIN COUNTS BY METRIC (A vs B):",
        "-" * 40,
    ]

    for m in METRICS:
        bm = summary["by_metric"][m]
        lines.append(
            f"  {m}: {model_a} wins {bm['model_a_wins']}  |  "
            f"{model_b} wins {bm['model_b_wins']}  |  Ties {bm['ties']}"
        )

    lines.append("")
    lines.append("AVERAGE SCORE DELTAS (A - B):")
    lines.append("-" * 40)
    for m in METRICS:
        d = summary["avg_deltas"][m]
        sign = "+" if d >= 0 else ""
        lines.append(f"  {m}: {sign}{d:.3f}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare individual evaluation scores for same scenario_ids across two models"
    )
    parser.add_argument(
        "--model_a_file",
        required=True,
        help="Path to first model's individual eval JSON",
    )
    parser.add_argument(
        "--model_b_file",
        required=True,
        help="Path to second model's individual eval JSON",
    )
    parser.add_argument(
        "--output",
        default="compare_individual_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing _summary.txt report",
    )
    args = parser.parse_args()

    index_a, model_a, judge_a = load_individual_eval(args.model_a_file)
    index_b, model_b, judge_b = load_individual_eval(args.model_b_file)

    print(f"Loaded {len(index_a)} scores from {args.model_a_file} ({model_a}, judge: {judge_a})")
    print(f"Loaded {len(index_b)} scores from {args.model_b_file} ({model_b}, judge: {judge_b})")

    comparisons, summary = compare_scores(index_a, index_b, model_a, model_b)

    metadata = {
        "comparison_type": "individual_scores",
        "input_files": [args.model_a_file, args.model_b_file],
        "models": [model_a, model_b],
        "judge_models": [judge_a, judge_b],
        "matched_scenarios": len(comparisons),
        "timestamp": datetime.now().isoformat(),
    }

    output_data = {
        "metadata": metadata,
        "summary": summary,
        "comparisons": comparisons,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if not args.no_report:
        report = generate_report(comparisons, summary, metadata)
        report_file = out_path.stem + "_summary.txt"
        report_path = out_path.parent / report_file
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Report saved to {report_path}")
        print("\n" + report)


if __name__ == "__main__":
    main()
