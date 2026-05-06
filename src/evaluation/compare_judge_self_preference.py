"""
Compare scores from two judges across the same set of (question, generator)
responses to detect LLM self-preference.

Inputs are the JSON files produced by individual_evaluation_question_bank.py.
You pass:
  --judge_a_files  one or more files where judge A scored responses (any generator)
  --judge_b_files  one or more files where judge B scored responses (any generator)
  --judge_a_label / --judge_b_label  human-friendly labels (default 'a' / 'b')

For every (question_id, generator_model) present in BOTH judges' outputs,
we compute the paired bias:

    bias_i = score_judge_a(i) - score_judge_b(i)        (per criterion)

Aggregating bias per generator gives:
  - mean_bias_on_X  -> "judge A scores generator X this much higher than judge B does"

Self-Preference Index (SPI), per criterion:
  SPI = mean_bias[A's family generator] - mean_bias[B's family generator]
  -- positive SPI: judges show mutual self-preference (each judge inflates its
     own family's outputs relative to the other judge).
  -- A and B are matched to generators by name pattern (gpt -> OpenAI family,
     claude -> Anthropic family). Override with --judge_a_family / --judge_b_family
     and --gen_family_<name> if names don't match.

Usage:

    python src/evaluation/compare_judge_self_preference.py \
        --judge_a_label gpt --judge_a_family openai \
        --judge_a_files \
            question_bank_output/evaluations/judge_gpt_gen_gpt.json \
            question_bank_output/evaluations/judge_gpt_gen_claude.json \
        --judge_b_label claude --judge_b_family anthropic \
        --judge_b_files \
            question_bank_output/evaluations/judge_claude_gen_gpt.json \
            question_bank_output/evaluations/judge_claude_gen_claude.json \
        --output question_bank_output/evaluations/self_preference_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

CRITERIA = ("faithfulness", "completeness", "safety", "clarity", "conciseness", "overall")


def load_scores_file(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        return payload.get("scores", []) or []
    if isinstance(payload, list):
        return payload
    return []


def index_scores(scores: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for s in scores:
        if not isinstance(s, dict):
            continue
        qid = str(s.get("question_id", ""))
        gen = str(s.get("generator_model", ""))
        if not qid or not gen:
            continue
        out[(qid, gen)] = s
    return out


def avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stderr(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    return statistics.stdev(values) / math.sqrt(n)


def detect_family(generator_name: str, family_overrides: Dict[str, str]) -> str:
    if generator_name in family_overrides:
        return family_overrides[generator_name]
    n = generator_name.lower()
    if "gpt" in n or n.startswith(("o1", "o3", "o4")):
        return "openai"
    if "claude" in n:
        return "anthropic"
    if "gemini" in n:
        return "google"
    return "unknown"


def parse_family_overrides(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in items or []:
        if "=" not in raw:
            continue
        k, v = raw.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def build_per_judge_summary(scores: List[Dict[str, Any]], judge_label: str) -> Dict[str, Dict[str, float]]:
    by_gen: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for s in scores:
        gen = str(s.get("generator_model", ""))
        for c in CRITERIA:
            try:
                by_gen[gen][c].append(float(s[c]))
            except (KeyError, TypeError, ValueError):
                continue
    summary: Dict[str, Dict[str, float]] = {}
    for gen, by_c in by_gen.items():
        row: Dict[str, float] = {"count": float(len(by_c.get("overall", [])))}
        for c in CRITERIA:
            row[f"avg_{c}"] = avg(by_c.get(c, []))
            row[f"sem_{c}"] = stderr(by_c.get(c, []))
        summary[gen] = row
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two judges' scores; detect self-preference.")
    parser.add_argument("--judge_a_label", default="a")
    parser.add_argument("--judge_a_family", default=None,
                        help="Generator-family label for judge A (e.g. openai). Used for SPI.")
    parser.add_argument("--judge_a_files", nargs="+", required=True)
    parser.add_argument("--judge_b_label", default="b")
    parser.add_argument("--judge_b_family", default=None,
                        help="Generator-family label for judge B (e.g. anthropic).")
    parser.add_argument("--judge_b_files", nargs="+", required=True)
    parser.add_argument("--gen_family_override", action="append", default=[],
                        help="Map generator_model name to family, e.g. --gen_family_override gpt-5.4=openai")
    parser.add_argument("--output", required=True, help="Output JSON report")
    args = parser.parse_args()

    family_overrides = parse_family_overrides(args.gen_family_override)

    judge_a_scores: List[Dict[str, Any]] = []
    for path in args.judge_a_files:
        judge_a_scores.extend(load_scores_file(Path(path)))
    judge_b_scores: List[Dict[str, Any]] = []
    for path in args.judge_b_files:
        judge_b_scores.extend(load_scores_file(Path(path)))

    print(f"Loaded judge_{args.judge_a_label}: {len(judge_a_scores)} scores")
    print(f"Loaded judge_{args.judge_b_label}: {len(judge_b_scores)} scores")

    a_index = index_scores(judge_a_scores)
    b_index = index_scores(judge_b_scores)
    paired_keys = sorted(set(a_index.keys()) & set(b_index.keys()))
    print(f"Paired (same question + generator scored by both judges): {len(paired_keys)}")

    summary_a = build_per_judge_summary(judge_a_scores, args.judge_a_label)
    summary_b = build_per_judge_summary(judge_b_scores, args.judge_b_label)

    print("\n=== Average judge scores by (judge, generator) ===")
    print(f"{'judge':<15} {'generator':<28} {'n':<6} {'overall':<10}")
    for gen, row in sorted(summary_a.items()):
        print(f"{args.judge_a_label:<15} {gen:<28} {int(row['count']):<6} {row['avg_overall']:.3f}")
    for gen, row in sorted(summary_b.items()):
        print(f"{args.judge_b_label:<15} {gen:<28} {int(row['count']):<6} {row['avg_overall']:.3f}")

    # Paired bias = judge_a - judge_b, grouped by generator
    bias_by_gen: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for key in paired_keys:
        _, gen = key
        a = a_index[key]
        b = b_index[key]
        for c in CRITERIA:
            try:
                bias_by_gen[gen][c].append(float(a[c]) - float(b[c]))
            except (KeyError, TypeError, ValueError):
                continue

    print(f"\n=== Paired bias: (judge_{args.judge_a_label}) - (judge_{args.judge_b_label}) ===")
    print("positive => judge A more generous; negative => judge B more generous.")
    bias_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for gen, per_c in sorted(bias_by_gen.items()):
        bias_summary[gen] = {}
        n = len(per_c.get("overall", []))
        family = detect_family(gen, family_overrides)
        print(f"\nGenerator: {gen}  (family={family}, n={n})")
        for c in CRITERIA:
            vals = per_c.get(c, [])
            mean = avg(vals)
            sem = stderr(vals)
            ci95 = 1.96 * sem
            bias_summary[gen][c] = {
                "mean_bias": mean,
                "sem": sem,
                "ci95": ci95,
                "n": len(vals),
                "family": family,
            }
            print(f"  {c:<14} mean_bias={mean:+.3f}  ±{ci95:.3f} (95% CI)  n={len(vals)}")

    # Self-Preference Index
    spi: Dict[str, Dict[str, Any]] = {}
    if args.judge_a_family and args.judge_b_family and args.judge_a_family != args.judge_b_family:
        a_family = args.judge_a_family
        b_family = args.judge_b_family
        gens_a = [g for g in bias_by_gen if detect_family(g, family_overrides) == a_family]
        gens_b = [g for g in bias_by_gen if detect_family(g, family_overrides) == b_family]

        if not gens_a or not gens_b:
            print(f"\n[warn] cannot compute SPI: no generator in family={a_family} (got {gens_a}) "
                  f"or family={b_family} (got {gens_b}).")
        else:
            print(f"\n=== Self-Preference Index (judge_a_family={a_family}, judge_b_family={b_family}) ===")
            print("SPI = mean_bias(A-family generator) - mean_bias(B-family generator)")
            print("positive SPI => mutual self-preference; ~0 => judges agree.")
            for c in CRITERIA:
                a_vals: List[float] = []
                for g in gens_a:
                    a_vals.extend(bias_by_gen[g].get(c, []))
                b_vals: List[float] = []
                for g in gens_b:
                    b_vals.extend(bias_by_gen[g].get(c, []))
                if not a_vals or not b_vals:
                    continue
                a_mean = avg(a_vals)
                b_mean = avg(b_vals)
                a_sem = stderr(a_vals)
                b_sem = stderr(b_vals)
                spi_val = a_mean - b_mean
                spi_sem = math.sqrt(a_sem ** 2 + b_sem ** 2)
                spi_ci95 = 1.96 * spi_sem
                spi[c] = {
                    "spi": spi_val,
                    "ci95": spi_ci95,
                    "bias_a_family_gen": a_mean,
                    "bias_b_family_gen": b_mean,
                    "a_family": a_family,
                    "b_family": b_family,
                    "a_family_generators": gens_a,
                    "b_family_generators": gens_b,
                    "n_a": len(a_vals),
                    "n_b": len(b_vals),
                }
                print(f"  {c:<14} bias[A-fam]={a_mean:+.3f}  bias[B-fam]={b_mean:+.3f}  "
                      f"SPI={spi_val:+.3f}  ±{spi_ci95:.3f} (95% CI)")
    else:
        print("\n[note] --judge_a_family / --judge_b_family not provided; skipping SPI.")

    # Per-judge intra-comparison: which generator does each judge prefer?
    print("\n=== Per-judge generator preference (overall score gap, judge's own view) ===")
    print("judge gap = mean(A-fam gen overall) - mean(B-fam gen overall) on this judge.")
    judge_gaps: Dict[str, Dict[str, Any]] = {}
    if args.judge_a_family and args.judge_b_family:
        for label, summary in [(args.judge_a_label, summary_a), (args.judge_b_label, summary_b)]:
            gens_a = [g for g in summary if detect_family(g, family_overrides) == args.judge_a_family]
            gens_b = [g for g in summary if detect_family(g, family_overrides) == args.judge_b_family]
            if gens_a and gens_b:
                a_overall = avg([summary[g]["avg_overall"] for g in gens_a])
                b_overall = avg([summary[g]["avg_overall"] for g in gens_b])
                gap = a_overall - b_overall
                judge_gaps[label] = {
                    "a_family_overall": a_overall,
                    "b_family_overall": b_overall,
                    "gap_a_minus_b": gap,
                }
                print(f"  judge={label:<15} A-fam={a_overall:.3f}  B-fam={b_overall:.3f}  gap={gap:+.3f}")

    report = {
        "judge_a": {
            "label": args.judge_a_label,
            "family": args.judge_a_family,
            "files": args.judge_a_files,
            "n_scores": len(judge_a_scores),
            "summary_by_generator": summary_a,
        },
        "judge_b": {
            "label": args.judge_b_label,
            "family": args.judge_b_family,
            "files": args.judge_b_files,
            "n_scores": len(judge_b_scores),
            "summary_by_generator": summary_b,
        },
        "paired_count": len(paired_keys),
        "paired_bias_by_generator": bias_summary,
        "self_preference_index": spi,
        "per_judge_generator_gap": judge_gaps,
        "family_overrides": family_overrides,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {output_path}")


if __name__ == "__main__":
    main()
