#!/usr/bin/env python3
"""
Generate a human-readable evaluation report from opus_judge.json and model responses.
Output is formatted for easy pasting into a document for PI review.
"""

import json
from pathlib import Path


def load_json(path: str) -> dict | list:
    with open(path, "r") as f:
        return json.load(f)


def main():
    base = Path(__file__).parent
    eval_dir = base / "meddialog_output" / "feb_19_eval"
    responses_dir = base / "meddialog_output" / "feb_19"

    judge = load_json(eval_dir / "opus_judge.json")
    claude_responses = load_json(responses_dir / "claude-opus-4-6_responses.json")
    gpt_responses = load_json(responses_dir / "gpt-5.2_responses.json")

    # Index by id for quick lookup
    claude_by_id = {r["id"]: r for r in claude_responses}
    gpt_by_id = {r["id"]: r for r in gpt_responses}

    lines = []
    lines.append("=" * 80)
    lines.append("PAIRWISE EVALUATION REPORT: Claude Opus 4 vs GPT-5.2")
    lines.append("Judge: claude-opus-4-6 (identity-blind)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 40)
    s = judge["summary"]
    lines.append(f"A wins: {s['A_wins']}  |  B wins: {s['B_wins']}  |  Ties: {s['ties']}")
    lines.append(f"A avg score: {s['A_avg_score']:.2f}  |  B avg score: {s['B_avg_score']:.2f}")
    lines.append("")
    lines.append("")

    for i, comp in enumerate(judge["comparisons"], 1):
        scenario_id = comp["scenario_id"]
        response_a_id = comp["response_a_id"]
        response_b_id = comp["response_b_id"]

        # Get scenario content from either response file (patient_query and reference are same)
        if response_a_id in claude_by_id:
            scenario_data = claude_by_id[response_a_id]
        else:
            scenario_data = gpt_by_id[response_a_id]

        patient_query = scenario_data.get("patient_query", "[N/A]")
        ground_truth = scenario_data.get("reference_doctor_response", "[N/A]")

        # Get model responses
        claude_r = claude_by_id.get(f"{scenario_id}_claude-opus-4-6", {})
        gpt_r = gpt_by_id.get(f"{scenario_id}_gpt-5.2", {})

        claude_resp = claude_r.get("generated_response", "[No response]")
        gpt_resp = gpt_r.get("generated_response", "[No response]")

        # Map A/B to actual models for scores
        if comp["response_a_model"] == "claude-opus-4-6":
            claude_scores = {
                "faithfulness": comp["a_faithfulness"],
                "completeness": comp["a_completeness"],
                "safety": comp["a_safety"],
                "clarity": comp["a_clarity"],
                "conciseness": comp["a_conciseness"],
                "overall": comp["a_overall"],
            }
            gpt_scores = {
                "faithfulness": comp["b_faithfulness"],
                "completeness": comp["b_completeness"],
                "safety": comp["b_safety"],
                "clarity": comp["b_clarity"],
                "conciseness": comp["b_conciseness"],
                "overall": comp["b_overall"],
            }
        else:
            gpt_scores = {
                "faithfulness": comp["a_faithfulness"],
                "completeness": comp["a_completeness"],
                "safety": comp["a_safety"],
                "clarity": comp["a_clarity"],
                "conciseness": comp["a_conciseness"],
                "overall": comp["a_overall"],
            }
            claude_scores = {
                "faithfulness": comp["b_faithfulness"],
                "completeness": comp["b_completeness"],
                "safety": comp["b_safety"],
                "clarity": comp["b_clarity"],
                "conciseness": comp["b_conciseness"],
                "overall": comp["b_overall"],
            }

        winner = comp["preference"]
        if winner == "A":
            winner_str = comp["response_a_model"]
        elif winner == "B":
            winner_str = comp["response_b_model"]
        else:
            winner_str = "Tie"

        lines.append("=" * 80)
        lines.append(f"EVALUATION {i} of {len(judge['comparisons'])}")
        lines.append(f"Scenario ID: {scenario_id}")
        lines.append("=" * 80)
        lines.append("")
        lines.append("SCENARIO (Patient Query)")
        lines.append("-" * 40)
        lines.append(patient_query)
        lines.append("")
        lines.append("GROUND TRUTH (Reference Doctor Response)")
        lines.append("-" * 40)
        lines.append(ground_truth)
        lines.append("")
        lines.append("CLAUDE OPUS 4-6 RESPONSE")
        lines.append("-" * 40)
        lines.append(claude_resp if claude_resp else "[Empty]")
        lines.append("")
        lines.append("GPT-5.2 RESPONSE")
        lines.append("-" * 40)
        lines.append(gpt_resp if gpt_resp else "[Empty]")
        lines.append("")
        lines.append("LLM JUDGEMENT")
        lines.append("-" * 40)
        lines.append(f"Preference: {winner_str}")
        lines.append(f"Confidence: {comp['confidence']:.2f}")
        lines.append("")
        lines.append("Reasoning:")
        lines.append(comp["reasoning"])
        lines.append("")
        lines.append("SCORES")
        lines.append("-" * 40)
        lines.append("Claude Opus 4-6:")
        lines.append(
            f"  Faithfulness: {claude_scores['faithfulness']}  |  Completeness: {claude_scores['completeness']}  |  "
            f"Safety: {claude_scores['safety']}  |  Clarity: {claude_scores['clarity']}  |  "
            f"Conciseness: {claude_scores['conciseness']}  |  Overall: {claude_scores['overall']}"
        )
        lines.append("GPT-5.2:")
        lines.append(
            f"  Faithfulness: {gpt_scores['faithfulness']}  |  Completeness: {gpt_scores['completeness']}  |  "
            f"Safety: {gpt_scores['safety']}  |  Clarity: {gpt_scores['clarity']}  |  "
            f"Conciseness: {gpt_scores['conciseness']}  |  Overall: {gpt_scores['overall']}"
        )
        lines.append("")
        lines.append("")

    report = "\n".join(lines)

    out_path = base / "meddialog_output" / "feb_19_eval" / "eval_report_for_review.txt"
    with open(out_path, "w") as f:
        f.write(report)

    print(f"Report written to: {out_path}")
    print(f"Total evaluations: {len(judge['comparisons'])}")


if __name__ == "__main__":
    main()
