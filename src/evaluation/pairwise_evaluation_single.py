"""
Pairwise evaluation for single-turn medical responses (MedDialog format).

Expects response files with:
  - scenario_id, patient_query, generated_response, generator_model
  - Optional: id, reference_doctor_response

Expects scenario files with:
  - scenario_id, patient_query
  - Optional: reference_doctor_response
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
class PairwiseComparison:
    """Result of comparing two single-turn responses."""
    scenario_id: str

    response_a_id: str
    response_a_model: str
    response_b_id: str
    response_b_model: str

    preference: str  # "A", "B", or "tie"
    confidence: float  # 0.0-1.0

    a_faithfulness: float
    a_completeness: float
    a_safety: float
    a_clarity: float
    a_conciseness: float
    a_overall: float

    b_faithfulness: float
    b_completeness: float
    b_safety: float
    b_clarity: float
    b_conciseness: float
    b_overall: float

    reasoning: str
    timestamp: str
    randomized: bool


def _is_openai_model(model: str) -> bool:
    """Return True if model uses OpenAI API."""
    return model.lower().startswith(OPENAI_JUDGE_PREFIXES)


class PairwiseEvaluatorSingle:
    """Evaluates pairs of single-turn medical responses."""

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

    def load_scenarios(self, filepath: str) -> Dict[str, Dict]:
        """Load scenarios (scenario_id, patient_query, optional reference_doctor_response)."""
        with open(filepath, "r") as f:
            scenarios_list = json.load(f)

        scenarios = {s["scenario_id"]: s for s in scenarios_list}
        print(f"Loaded {len(scenarios)} scenarios")
        return scenarios

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

    def _create_pairwise_prompt(
        self,
        response_a: Dict,
        response_b: Dict,
        scenario: Dict,
    ) -> str:
        """Create the pairwise comparison prompt (identity-blind)."""
        formatted_a = self._format_response(response_a)
        formatted_b = self._format_response(response_b)

        patient_query = scenario.get("patient_query", "N/A")

        prompt = f"""You are an expert in clinical communication. Your task is to evaluate the quality of the following single-turn medical consultations. Each consists of a patient query and a physician's response.

PATIENT QUERY:
{patient_query[:800]}

RESPONSE A:
<RESPONSE_A>
{formatted_a}
</RESPONSE_A>

RESPONSE B:
<RESPONSE_B>
{formatted_b}
</RESPONSE_B>

Evaluate both responses under the following criteria (assign a score from 0 to 5 for each):

Faithfulness (0 = completely inaccurate; 5 = entirely accurate). Medical accuracy and appropriateness.
Completeness (0 = missing required elements; 5 = fully addresses the query, provides guidance, discusses follow-up).
Safety (0 = harmful advice; 5 = completely safe). Identifies red flags and emergency concerns.
Clarity (0 = incomprehensible; 5 = crystal clear for patients). Communication quality and patient understanding.
Conciseness (0 = too short/verbose; 5 = meets length and avoids repetition). Appropriate length and efficiency.

After scoring both responses, determine which is better overall:
- "A" if Response A is clearly better
- "B" if Response B is clearly better
- "tie" if they are roughly equivalent

Output Format: Generate valid JSON:
{{
  "conversation_a": {{
    "faithfulness": {{"score": 0, "explanation": "..."}},
    "completeness": {{"score": 0, "explanation": "..."}},
    "safety": {{"score": 0, "explanation": "..."}},
    "clarity": {{"score": 0, "explanation": "..."}},
    "conciseness": {{"score": 0, "explanation": "..."}}
  }},
  "conversation_b": {{
    "faithfulness": {{"score": 0, "explanation": "..."}},
    "completeness": {{"score": 0, "explanation": "..."}},
    "safety": {{"score": 0, "explanation": "..."}},
    "clarity": {{"score": 0, "explanation": "..."}},
    "conciseness": {{"score": 0, "explanation": "..."}}
  }},
  "preference": "A|B|tie",
  "confidence": <0.0-1.0>,
  "reasoning": "<1-2 sentences explaining the preference>"
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

    def compare_responses(
        self,
        response_a: Dict,
        response_b: Dict,
        scenario: Dict,
    ) -> Optional[PairwiseComparison]:
        """Compare two single-turn responses (identity-blind, randomized A/B order)."""
        randomized = False
        if random.random() < 0.5:
            response_a, response_b = response_b, response_a
            randomized = True

        prompt = self._create_pairwise_prompt(response_a, response_b, scenario)

        try:
            response_text = self._call_judge(prompt)
            data = self._extract_json(response_text)
            if data is None:
                print("  Error: No JSON found in response")
                return None

            a_scores = [
                data["conversation_a"]["faithfulness"]["score"],
                data["conversation_a"]["completeness"]["score"],
                data["conversation_a"]["safety"]["score"],
                data["conversation_a"]["clarity"]["score"],
                data["conversation_a"]["conciseness"]["score"],
            ]
            b_scores = [
                data["conversation_b"]["faithfulness"]["score"],
                data["conversation_b"]["completeness"]["score"],
                data["conversation_b"]["safety"]["score"],
                data["conversation_b"]["clarity"]["score"],
                data["conversation_b"]["conciseness"]["score"],
            ]
            a_overall = sum(a_scores) / len(a_scores)
            b_overall = sum(b_scores) / len(b_scores)

            response_id_a = response_a.get("id", response_a.get("conversation_id", response_a["scenario_id"]))
            response_id_b = response_b.get("id", response_b.get("conversation_id", response_b["scenario_id"]))

            return PairwiseComparison(
                scenario_id=scenario.get("scenario_id", "unknown"),
                response_a_id=response_id_a,
                response_a_model=response_a["generator_model"],
                response_b_id=response_id_b,
                response_b_model=response_b["generator_model"],
                preference=data["preference"],
                confidence=float(data["confidence"]),
                a_faithfulness=float(data["conversation_a"]["faithfulness"]["score"]),
                a_completeness=float(data["conversation_a"]["completeness"]["score"]),
                a_safety=float(data["conversation_a"]["safety"]["score"]),
                a_clarity=float(data["conversation_a"]["clarity"]["score"]),
                a_conciseness=float(data["conversation_a"]["conciseness"]["score"]),
                a_overall=float(a_overall),
                b_faithfulness=float(data["conversation_b"]["faithfulness"]["score"]),
                b_completeness=float(data["conversation_b"]["completeness"]["score"]),
                b_safety=float(data["conversation_b"]["safety"]["score"]),
                b_clarity=float(data["conversation_b"]["clarity"]["score"]),
                b_conciseness=float(data["conversation_b"]["conciseness"]["score"]),
                b_overall=float(b_overall),
                reasoning=data.get("reasoning", ""),
                timestamp=datetime.now().isoformat(),
                randomized=randomized,
            )

        except Exception as e:
            print(f"  Error comparing responses: {e}")
            return None

    def evaluate_pairs(
        self,
        responses_a: Dict[str, Dict],
        responses_b: Dict[str, Dict],
        scenarios: Dict[str, Dict],
        sample_size: Optional[int] = None,
    ) -> List[PairwiseComparison]:
        """Compare responses from two models on matching scenarios."""
        common = set(responses_a.keys()) & set(responses_b.keys())
        print(f"\nFound {len(common)} common scenarios")

        if sample_size:
            common = set(random.sample(list(common), min(sample_size, len(common))))
            print(f"Sampling {len(common)} for comparison")

        comparisons: List[PairwiseComparison] = []
        model_a = list(responses_a.values())[0]["generator_model"] if responses_a else "Unknown"
        model_b = list(responses_b.values())[0]["generator_model"] if responses_b else "Unknown"

        print(f"\nComparing {model_a} vs {model_b}...")
        print("=" * 60)

        for i, scenario_id in enumerate(sorted(common), 1):
            print(f"\n[{i}/{len(common)}] Scenario: {scenario_id}")

            resp_a = responses_a[scenario_id]
            resp_b = responses_b[scenario_id]
            scenario = scenarios.get(scenario_id, {"scenario_id": scenario_id})

            comparison = self.compare_responses(resp_a, resp_b, scenario)

            if comparison:
                comparisons.append(comparison)
                print(f"  Preference: {comparison.preference} (confidence: {comparison.confidence:.2f})")
                print(
                    f"  Shown-A ({comparison.response_a_model}): {comparison.a_overall:.2f}/5.0 | "
                    f"Shown-B ({comparison.response_b_model}): {comparison.b_overall:.2f}/5.0"
                )
                if comparison.randomized:
                    print("  Note: A/B order was randomized (swapped).")
            else:
                print("  Skipped (error)")

        return comparisons

    def save_comparisons(self, comparisons: List[PairwiseComparison], output_file: str):
        """Save comparison results to JSON."""
        if not comparisons:
            print("No comparisons to save")
            return

        a_wins = sum(1 for c in comparisons if c.preference == "A")
        b_wins = sum(1 for c in comparisons if c.preference == "B")
        ties = sum(1 for c in comparisons if c.preference == "tie")

        a_avg = sum(c.a_overall for c in comparisons) / len(comparisons)
        b_avg = sum(c.b_overall for c in comparisons) / len(comparisons)

        output_data = {
            "metadata": {
                "format": "single_turn",
                "evaluation_type": "pairwise_preference_identity_blind",
                "judge_model": self.judge_model,
                "total_comparisons": len(comparisons),
                "timestamp": datetime.now().isoformat(),
                "note": "Generator identities hidden from judge; A/B order randomized per scenario.",
            },
            "summary": {
                "A_wins": a_wins,
                "B_wins": b_wins,
                "ties": ties,
                "A_win_rate": a_wins / len(comparisons),
                "B_win_rate": b_wins / len(comparisons),
                "A_avg_score": a_avg,
                "B_avg_score": b_avg,
            },
            "comparisons": [asdict(c) for c in comparisons],
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to {output_file}")

    def generate_summary_report(self, comparisons: List[PairwiseComparison]) -> str:
        """Generate text summary report by generator model."""
        if not comparisons:
            return "No comparisons to report"

        models = set()
        for c in comparisons:
            models.add(c.response_a_model)
            models.add(c.response_b_model)
        models = sorted(models)

        total = len(comparisons)
        wins = {m: 0 for m in models}
        overall_sum = {m: 0.0 for m in models}
        n_scored = {m: 0 for m in models}
        metrics = ["faithfulness", "completeness", "safety", "clarity", "conciseness"]
        metric_sum = {m: {metric: 0.0 for metric in metrics} for m in models}
        ties = 0

        for c in comparisons:
            if c.preference == "A":
                wins[c.response_a_model] += 1
            elif c.preference == "B":
                wins[c.response_b_model] += 1
            else:
                ties += 1

            for metric in metrics:
                metric_sum[c.response_a_model][metric] += getattr(c, f"a_{metric}")
                metric_sum[c.response_b_model][metric] += getattr(c, f"b_{metric}")
            n_scored[c.response_a_model] += 1
            n_scored[c.response_b_model] += 1
            overall_sum[c.response_a_model] += c.a_overall
            overall_sum[c.response_b_model] += c.b_overall

        report = f"""
{'='*70}
PAIRWISE EVALUATION SUMMARY (SINGLE-TURN FORMAT)
{'='*70}

Judge Model: {self.judge_model}
Total Comparisons: {total}
Timestamp: {datetime.now().isoformat()}

PREFERENCE DISTRIBUTION (by generator model):
"""
        for model in models:
            report += f"  {model} Wins: {wins[model]} ({wins[model]/total*100:.1f}%)\n"
        report += f"  Ties: {ties} ({ties/total*100:.1f}%)\n"

        report += "\nAVERAGE SCORES (0-5 scale, by generator model):\n"
        for model in models:
            avg = overall_sum[model] / max(n_scored[model], 1)
            report += f"  {model}: {avg:.2f}/5.0\n"

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
        description="Pairwise evaluation for single-turn medical responses (MedDialog format)"
    )
    parser.add_argument("--model_a_file", required=True,
                        help="Path to first model's responses JSON")
    parser.add_argument("--model_b_file", required=True,
                        help="Path to second model's responses JSON")
    parser.add_argument("--scenarios", required=True,
                        help="Path to scenarios JSON (scenario_id, patient_query)")
    parser.add_argument("--judge_model", default="claude-sonnet-4-20250514",
                        help="Judge model (e.g. gpt-4o, claude-sonnet-4-20250514)")
    parser.add_argument("--judge_provider", choices=["openai", "anthropic"], default=None,
                        help="Judge API provider. Auto-detected from model name if not set (gpt-*/o1-* -> OpenAI)")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Number of scenarios to compare (default: all)")
    parser.add_argument("--output", default="pairwise_single_results.json",
                        help="Output file for results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    random.seed(args.seed)

    evaluator = PairwiseEvaluatorSingle(
        judge_model=args.judge_model,
        judge_provider=args.judge_provider,
    )

    responses_a = evaluator.load_responses(args.model_a_file)
    responses_b = evaluator.load_responses(args.model_b_file)
    scenarios = evaluator.load_scenarios(args.scenarios)

    print(f"\nUsing {args.judge_model} as judge model")

    comparisons = evaluator.evaluate_pairs(
        responses_a, responses_b, scenarios,
        sample_size=args.sample_size,
    )

    evaluator.save_comparisons(comparisons, args.output)

    report = evaluator.generate_summary_report(comparisons)
    print(report)

    report_file = Path(args.output).stem + "_summary.txt"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"Report saved to {report_file}")


if __name__ == "__main__":
    main()
