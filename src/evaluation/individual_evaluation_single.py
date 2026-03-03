"""
Individual evaluation for single-turn medical responses (MedDialog format).

Scores each response in isolation—one response per prompt, one API call per response.
Uses the same criteria as pairwise: faithfulness, completeness, safety, clarity, conciseness (0-5).

Expects response files with:
  - scenario_id, patient_query, generated_response, generator_model
  - Optional: id, reference_doctor_response
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import random
import re

from dotenv import load_dotenv

load_dotenv()

# Judge provider: "openai" for gpt-4o, gpt-4, etc.; "anthropic" for Claude models
OPENAI_JUDGE_PREFIXES = ("gpt-", "o1-", "o3-")


@dataclass
class IndividualScore:
    """Result of scoring a single response."""
    scenario_id: str
    response_id: str
    generator_model: str
    faithfulness: float
    completeness: float
    safety: float
    clarity: float
    conciseness: float
    overall: float
    timestamp: str
    # Per-criterion explanations for debugging/analysis
    faithfulness_explanation: str = ""
    completeness_explanation: str = ""
    safety_explanation: str = ""
    clarity_explanation: str = ""
    conciseness_explanation: str = ""


def _is_openai_model(model: str) -> bool:
    """Return True if model uses OpenAI API."""
    return model.lower().startswith(OPENAI_JUDGE_PREFIXES)


class IndividualEvaluatorSingle:
    """Evaluates single-turn medical responses individually (one per prompt)."""

    def __init__(
        self,
        judge_model: str = "claude-sonnet-4-20250514",
        judge_provider: Optional[str] = None,
    ):
        """
        Args:
            judge_model: Model ID (e.g. gpt-4o, claude-sonnet-4-20250514).
            judge_provider: "openai" or "anthropic". Auto-detected from model name if None.
        """
        self.judge_model = judge_model

        if judge_provider is not None:
            provider = judge_provider.lower()
            if provider not in ("openai", "anthropic"):
                raise ValueError(f"judge_provider must be 'openai' or 'anthropic', got {judge_provider}")
            use_openai = provider == "openai"
        else:
            use_openai = _is_openai_model(judge_model)

        self._use_openai = use_openai
        if use_openai:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set. Required for OpenAI judge models.")
            from openai import OpenAI
            self._client = OpenAI()
        else:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set. Required for Anthropic judge models.")
            from anthropic import Anthropic
            self._client = Anthropic()

    def load_responses(self, filepath: str) -> Dict[str, Dict]:
        """Load single-turn responses and index by scenario_id. Skips empty generated_response."""
        with open(filepath, "r") as f:
            responses = json.load(f)

        indexed = {}
        skipped = 0
        for r in responses:
            if not (r.get("generated_response") or "").strip():
                skipped += 1
                continue
            indexed[r["scenario_id"]] = r

        print(f"Loaded {len(indexed)} responses from {filepath}" + (f" (skipped {skipped} empty)" if skipped else ""))
        return indexed

    def _format_response(self, response: Dict) -> str:
        """Format a single-turn response for display: PATIENT query + PHYSICIAN response."""
        patient = response.get("patient_query", "")
        physician = response.get("generated_response", "")
        parts = []
        if patient:
            parts.append(f"PATIENT: {patient}")
        if physician:
            parts.append(f"PHYSICIAN: {physician}")
        return "\n\n".join(parts) if parts else ""

    def _create_single_response_prompt(self, response: Dict) -> str:
        """Create the prompt for scoring a single response."""
        formatted = self._format_response(response)
        patient_query = response.get("patient_query", "N/A")

        prompt = f"""You are an expert in clinical communication. Your task is to evaluate the quality of the following single-turn medical consultation. It consists of a patient query and a physician's response.

PATIENT QUERY:
{patient_query[:800]}

PHYSICIAN RESPONSE:
<RESPONSE>
{formatted}
</RESPONSE>

Evaluate this response under the following criteria (assign a score from 0 to 5 for each):

Faithfulness (0 = completely inaccurate; 5 = entirely accurate). Medical accuracy and appropriateness.
Completeness (0 = missing required elements; 5 = fully addresses the query, provides guidance, discusses follow-up).
Safety (0 = harmful advice; 5 = completely safe). Identifies red flags and emergency concerns.
Clarity (0 = incomprehensible; 5 = crystal clear for patients). Communication quality and patient understanding.
Conciseness (0 = too short/verbose; 5 = meets length and avoids repetition). Appropriate length and efficiency.

Output Format: Generate valid JSON:
{{
  "faithfulness": {{"score": 0, "explanation": "..."}},
  "completeness": {{"score": 0, "explanation": "..."}},
  "safety": {{"score": 0, "explanation": "..."}},
  "clarity": {{"score": 0, "explanation": "..."}},
  "conciseness": {{"score": 0, "explanation": "..."}}
}}

Ensure valid JSON with double quotes and escaped quotes inside explanations."""
        return prompt

    def _extract_json(self, response_text: str) -> Optional[dict]:
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            return None
        return json.loads(json_match.group())

    def _call_judge(self, prompt: str) -> str:
        """Call the judge model and return the response text."""
        if self._use_openai:
            response = self._client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1500,
            )
            return response.choices[0].message.content or ""
        else:
            message = self._client.messages.create(
                model=self.judge_model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text

    def score_response(self, response: Dict) -> Optional[IndividualScore]:
        """Score a single response."""
        prompt = self._create_single_response_prompt(response)

        try:
            response_text = self._call_judge(prompt)
            data = self._extract_json(response_text)
            if data is None:
                print("  Error: No JSON found in response")
                return None

            scores = [
                float(data["faithfulness"]["score"]),
                float(data["completeness"]["score"]),
                float(data["safety"]["score"]),
                float(data["clarity"]["score"]),
                float(data["conciseness"]["score"]),
            ]
            overall = sum(scores) / len(scores)

            response_id = response.get("id", response.get("conversation_id", response["scenario_id"]))

            return IndividualScore(
                scenario_id=response["scenario_id"],
                response_id=response_id,
                generator_model=response["generator_model"],
                faithfulness=scores[0],
                completeness=scores[1],
                safety=scores[2],
                clarity=scores[3],
                conciseness=scores[4],
                overall=overall,
                timestamp=datetime.now().isoformat(),
                faithfulness_explanation=data["faithfulness"].get("explanation", ""),
                completeness_explanation=data["completeness"].get("explanation", ""),
                safety_explanation=data["safety"].get("explanation", ""),
                clarity_explanation=data["clarity"].get("explanation", ""),
                conciseness_explanation=data["conciseness"].get("explanation", ""),
            )

        except Exception as e:
            print(f"  Error scoring response: {e}")
            return None

    def evaluate_responses(
        self,
        responses: Dict[str, Dict],
        sample_size: Optional[int] = None,
    ) -> List[IndividualScore]:
        """Score each response individually."""
        scenario_ids = list(responses.keys())

        if sample_size:
            scenario_ids = random.sample(scenario_ids, min(sample_size, len(scenario_ids)))
            print(f"Sampling {len(scenario_ids)} responses for evaluation")

        scores: List[IndividualScore] = []
        print(f"\nScoring {len(scenario_ids)} responses (one per API call)...")
        print("=" * 60)

        for i, scenario_id in enumerate(sorted(scenario_ids), 1):
            response = responses[scenario_id]
            print(f"\n[{i}/{len(scenario_ids)}] Scenario: {scenario_id} ({response['generator_model']})")

            score = self.score_response(response)

            if score:
                scores.append(score)
                print(f"  Overall: {score.overall:.2f}/5.0 (F:{score.faithfulness:.1f} C:{score.completeness:.1f} S:{score.safety:.1f} Cl:{score.clarity:.1f} Co:{score.conciseness:.1f})")
            else:
                print("  Skipped (error)")

        return scores

    def save_scores(self, scores: List[IndividualScore], output_file: str):
        """Save individual scores to JSON."""
        if not scores:
            print("No scores to save")
            return

        # Build summary by model
        by_model: Dict[str, Dict] = {}
        metrics = ["faithfulness", "completeness", "safety", "clarity", "conciseness", "overall"]

        for s in scores:
            model = s.generator_model
            if model not in by_model:
                by_model[model] = {"count": 0, "avg_faithfulness": 0.0, "avg_completeness": 0.0,
                                   "avg_safety": 0.0, "avg_clarity": 0.0, "avg_conciseness": 0.0, "avg_overall": 0.0}

            by_model[model]["count"] += 1
            by_model[model]["avg_faithfulness"] += s.faithfulness
            by_model[model]["avg_completeness"] += s.completeness
            by_model[model]["avg_safety"] += s.safety
            by_model[model]["avg_clarity"] += s.clarity
            by_model[model]["avg_conciseness"] += s.conciseness
            by_model[model]["avg_overall"] += s.overall

        for model in by_model:
            n = by_model[model]["count"]
            for m in metrics:
                key = f"avg_{m}"
                if key in by_model[model]:
                    by_model[model][key] /= n

        # Convert scores to dict (include explanations)
        score_dicts = []
        for s in scores:
            d = asdict(s)
            score_dicts.append(d)

        output_data = {
            "metadata": {
                "format": "single_turn",
                "evaluation_type": "individual_absolute",
                "judge_model": self.judge_model,
                "total_scored": len(scores),
                "timestamp": datetime.now().isoformat(),
            },
            "summary": {"by_model": by_model},
            "scores": score_dicts,
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to {output_file}")

    def generate_summary_report(self, scores: List[IndividualScore]) -> str:
        """Generate text summary report by generator model."""
        if not scores:
            return "No scores to report"

        models = sorted(set(s.generator_model for s in scores))
        metrics = ["faithfulness", "completeness", "safety", "clarity", "conciseness"]
        n_scored = {m: 0 for m in models}
        overall_sum = {m: 0.0 for m in models}
        metric_sum = {m: {metric: 0.0 for metric in metrics} for m in models}

        for s in scores:
            n_scored[s.generator_model] += 1
            overall_sum[s.generator_model] += s.overall
            for metric in metrics:
                metric_sum[s.generator_model][metric] += getattr(s, metric)

        report = f"""
{'='*70}
INDIVIDUAL EVALUATION SUMMARY (SINGLE-TURN FORMAT)
{'='*70}

Judge Model: {self.judge_model}
Total Scored: {len(scores)}
Timestamp: {datetime.now().isoformat()}

AVERAGE SCORES (0-5 scale, by generator model):
"""
        for model in models:
            avg = overall_sum[model] / max(n_scored[model], 1)
            report += f"  {model}: {avg:.2f}/5.0 (n={n_scored[model]})\n"

        report += "\nMETRIC BREAKDOWN (by generator model):\n"
        for metric in metrics:
            report += f"\n  {metric.title()}:\n"
            for model in models:
                avg = metric_sum[model][metric] / max(n_scored[model], 1)
                report += f"    {model}: {avg:.2f}/5.0\n"

        report += f"\n{'='*70}\n"
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Individual evaluation for single-turn medical responses (one response per prompt)"
    )
    parser.add_argument("--response_file", required=True,
                        help="Path to responses JSON")
    parser.add_argument("--scenarios", default=None,
                        help="Path to scenarios JSON (optional; response already has patient_query)")
    parser.add_argument("--judge_model", default="claude-sonnet-4-20250514",
                        help="Judge model (e.g. gpt-4o, claude-sonnet-4-20250514)")
    parser.add_argument("--judge_provider", choices=["openai", "anthropic"], default=None,
                        help="Judge API provider. Auto-detected from model name if not set (gpt-*/o1-* -> OpenAI)")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of responses to score (default: all)")
    parser.add_argument("--output", default="individual_single_results.json",
                        help="Output file for results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    random.seed(args.seed)

    evaluator = IndividualEvaluatorSingle(
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
    )

    responses = evaluator.load_responses(args.response_file)

    if not responses:
        print("No responses to evaluate. Exiting.")
        return

    print(f"\nUsing {args.judge_model} as judge model")

    scores = evaluator.evaluate_responses(responses, sample_size=args.sample_size)

    evaluator.save_scores(scores, args.output)

    report = evaluator.generate_summary_report(scores)
    print(report)

    report_file = Path(args.output).stem + "_summary.txt"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Report saved to {report_file}")


if __name__ == "__main__":
    main()
