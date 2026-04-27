"""Generate LLM answers for rows in question_bank_10k.csv.

This script is provider-aware and supports:
- OpenAI (`openai:gpt-5.4`, `gpt-5.4`, etc.)
- Anthropic (`anthropic:claude-opus-4-6`, `claude-opus-4-6`, etc.)
- Gemini (`gemini:gemini-3.1-pro`, `gemini-3.1-pro`, etc.)
- Moonshot/Kimi (`moonshot:kimi-k2`, `kimi-k2`, etc.)
- Other OpenAI-compatible endpoints (`openrouter:...`, `together:...`, etc.)

Typical usage:
    python src/generation/generate_question_bank_answers.py \
      --input_csv ./question_bank_output/question_bank_10k.csv \
      --model gpt-5.4 \
      --output_dir ./question_bank_output/generated_answers
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

# Always load project-root .env, even if invoked outside repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


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


_ROLE_PREFIX_RE = re.compile(
    r"^\s*(?:physician|doctor|clinician|patient|assistant)\s*:\s*",
    re.IGNORECASE,
)

SUPPORTED_PROVIDER_PREFIXES = {
    "openai",
    "anthropic",
    "gemini",
    "moonshot",
    "openrouter",
    "together",
    "groq",
    "xai",
    "deepinfra",
    "fireworks",
    "compatible",
}

OPENAI_COMPAT_PROVIDER_CONFIG = {
    "moonshot": {
        "api_key_env": "MOONSHOT_API_KEY",
        "base_url_env": "MOONSHOT_BASE_URL",
        "default_base_url": "https://api.moonshot.ai/v1",
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url_env": "OPENROUTER_BASE_URL",
        "default_base_url": "https://openrouter.ai/api/v1",
    },
    "together": {
        "api_key_env": "TOGETHER_API_KEY",
        "base_url_env": "TOGETHER_BASE_URL",
        "default_base_url": "https://api.together.xyz/v1",
    },
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url_env": "GROQ_BASE_URL",
        "default_base_url": "https://api.groq.com/openai/v1",
    },
    "xai": {
        "api_key_env": "XAI_API_KEY",
        "base_url_env": "XAI_BASE_URL",
        "default_base_url": "https://api.x.ai/v1",
    },
    "deepinfra": {
        "api_key_env": "DEEPINFRA_API_KEY",
        "base_url_env": "DEEPINFRA_BASE_URL",
        "default_base_url": "https://api.deepinfra.com/v1/openai",
    },
    "fireworks": {
        "api_key_env": "FIREWORKS_API_KEY",
        "base_url_env": "FIREWORKS_BASE_URL",
        "default_base_url": "https://api.fireworks.ai/inference/v1",
    },
}


class QuotaExhaustedError(RuntimeError):
    """Raised when API quota/credits are exhausted."""


def _increase_csv_field_size_limit() -> None:
    """Raise csv field size limit defensively for long text rows."""
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            return
        except OverflowError:
            max_int = max_int // 10


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


def sanitize_filename(value: str) -> str:
    """Create filesystem-safe filename stem."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return safe or "model"


def stable_question_id_from_text(question_text: str) -> str:
    """Fallback question id when csv row is missing one."""
    digest = hashlib.sha256(question_text.strip().encode("utf-8")).hexdigest()[:16]
    return f"qb_{digest}"


def parse_model_spec(model_spec: str) -> Tuple[Optional[str], str]:
    """Parse model spec format `provider:model_name`."""
    if ":" not in model_spec:
        return None, model_spec
    prefix, model_name = model_spec.split(":", 1)
    prefix = prefix.strip().lower()
    model_name = model_name.strip()
    if prefix in SUPPORTED_PROVIDER_PREFIXES and model_name:
        return prefix, model_name
    return None, model_spec


def infer_provider(model_name: str) -> str:
    """Infer provider from model name when no explicit prefix is provided."""
    name = model_name.lower()
    if "claude" in name:
        return "anthropic"
    if "gemini" in name:
        return "gemini"
    if "kimi" in name:
        return "moonshot"
    if (
        "gpt" in name
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
    ):
        return "openai"
    return "openai"


class LLMClient:
    """Base class for LLM API clients."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """OpenAI API client (official endpoint by default)."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please export OPENAI_API_KEY before running."
            )
        from openai import AsyncOpenAI

        base_url = os.getenv("OPENAI_BASE_URL") or None
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class OpenAICompatibleClient(LLMClient):
    """OpenAI-compatible chat-completions client (OpenRouter, Kimi, etc.)."""

    def __init__(self, model_name: str, api_key: str, base_url: str):
        super().__init__(model_name)
        if not api_key:
            raise RuntimeError("Missing API key for OpenAI-compatible provider.")
        if not base_url:
            raise RuntimeError("Missing base URL for OpenAI-compatible provider.")

        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


class AnthropicClient(LLMClient):
    """Anthropic API client."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Please export ANTHROPIC_API_KEY before running."
            )
        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        return response.content[0].text if response.content else ""


class GeminiClient(LLMClient):
    """Google Gemini API client."""

    def __init__(self, model_name: str):
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
        temperature: float = 0.3,
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
        return response.text or ""


def build_client(
    provider: str,
    model_name: str,
    compatible_api_key_env: str,
    compatible_base_url: str,
) -> LLMClient:
    """Create provider client from provider + model."""
    provider = provider.lower()

    if provider == "openai":
        return OpenAIClient(model_name)
    if provider == "anthropic":
        return AnthropicClient(model_name)
    if provider == "gemini":
        return GeminiClient(model_name)

    if provider == "compatible":
        api_key = os.getenv(compatible_api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"{compatible_api_key_env} is not set for compatible provider."
            )
        if not compatible_base_url:
            raise RuntimeError(
                "--compatible_base_url is required for provider prefix `compatible:`."
            )
        return OpenAICompatibleClient(
            model_name=model_name,
            api_key=api_key,
            base_url=compatible_base_url,
        )

    config = OPENAI_COMPAT_PROVIDER_CONFIG.get(provider)
    if not config:
        raise ValueError(f"Unsupported provider: {provider}")

    api_key = os.getenv(config["api_key_env"], "")
    if not api_key:
        raise RuntimeError(
            f"{config['api_key_env']} is not set for provider `{provider}`."
        )
    base_url = os.getenv(config["base_url_env"], "") or config["default_base_url"]

    return OpenAICompatibleClient(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )


def load_question_bank_rows(
    csv_path: Path,
    start_index: int = 0,
    num_questions: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load question bank rows from csv."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    _increase_csv_field_size_limit()

    with open(csv_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if start_index < 0:
        raise ValueError("--start_index must be >= 0")

    selected = rows[start_index:]
    if num_questions is not None and num_questions > 0:
        selected = selected[:num_questions]

    normalized_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(selected):
        question_text = (row.get("question_text") or "").strip()
        if not question_text:
            continue

        question_id = (row.get("question_id") or "").strip()
        if not question_id:
            question_id = stable_question_id_from_text(question_text)

        normalized_rows.append(
            {
                "question_id": question_id,
                "question_text": question_text,
                "ground_truth_response": (row.get("ground_truth_response") or "").strip(),
                "source_name": (row.get("source_name") or "").strip(),
                "canonical_source": (row.get("canonical_source") or "").strip(),
                "source_split": (row.get("source_split") or "").strip(),
                "source_row_index": row.get("source_row_index"),
                "source_record_id": row.get("source_record_id"),
                "question_hash_normalized": (row.get("question_hash_normalized") or "").strip(),
                "row_position": start_index + i,
            }
        )

    return normalized_rows


def build_user_prompt(question_text: str) -> str:
    """Build user prompt from question text."""
    return (
        "Patient's message:\n"
        f"{question_text}\n\n"
        "Provide your clinical response."
    )


def is_quota_exhaustion_error(error: Exception) -> bool:
    """Best-effort detection for credit/quota exhaustion errors."""
    error_text = str(error).lower()
    quota_markers = [
        "insufficient_quota",
        "quota exceeded",
        "exceeded your current quota",
        "out of credits",
        "insufficient credits",
        "billing",
        "payment required",
        "credit balance",
        "resource has been exhausted",
    ]
    if any(marker in error_text for marker in quota_markers):
        return True

    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)

    if status_code == 402:
        return True
    if status_code == 429 and (
        "quota" in error_text or "credit" in error_text or "insufficient" in error_text
    ):
        return True
    return False


async def generate_single_answer(
    *,
    client: LLMClient,
    question_text: str,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> str:
    """Generate one answer with retry policy."""
    attempt = 0
    while True:
        try:
            raw = await client.generate(
                system_prompt=PHYSICIAN_SYSTEM_PROMPT,
                user_prompt=build_user_prompt(question_text),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return cleanup_response(raw)
        except Exception as error:
            if is_quota_exhaustion_error(error):
                raise QuotaExhaustedError(str(error)) from error
            attempt += 1
            if attempt > retries:
                raise
            await asyncio.sleep(min(2 ** attempt, 8))


async def generate_record_for_row(
    *,
    row: Dict[str, Any],
    model_spec: str,
    provider: str,
    model_name: str,
    client: LLMClient,
    temperature: float,
    max_tokens: int,
    retries: int,
) -> Dict[str, Any]:
    """Generate one output record with row-position metadata."""
    question_id = row["question_id"]
    row_position = int(row.get("row_position", -1))
    csv_line_number = (row_position + 2) if row_position >= 0 else None

    try:
        generated_response = await generate_single_answer(
            client=client,
            question_text=row["question_text"],
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
        )
        record = {
            "id": f"{question_id}_{sanitize_filename(model_spec)}",
            "question_id": question_id,
            "question_text": row["question_text"],
            "ground_truth_response": row["ground_truth_response"],
            "source_name": row["source_name"],
            "canonical_source": row["canonical_source"],
            "source_split": row["source_split"],
            "source_row_index": row["source_row_index"],
            "source_record_id": row["source_record_id"],
            "question_hash_normalized": row["question_hash_normalized"],
            "generated_response": generated_response,
            "generator_model": model_name,
            "provider": provider,
            "model_spec": model_spec,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "created_at": datetime.now().isoformat(),
            "row_position": row_position,
            "input_csv_line_number": csv_line_number,
        }
        return {
            "record": record,
            "error_message": None,
            "is_quota_exhausted": False,
            "row_position": row_position,
            "csv_line_number": csv_line_number,
        }
    except QuotaExhaustedError as error:
        return {
            "record": None,
            "error_message": str(error),
            "is_quota_exhausted": True,
            "row_position": row_position,
            "csv_line_number": csv_line_number,
        }
    except Exception as error:
        return {
            "record": None,
            "error_message": str(error),
            "is_quota_exhausted": False,
            "row_position": row_position,
            "csv_line_number": csv_line_number,
        }


def load_existing_records(output_path: Path) -> List[Dict[str, Any]]:
    """Load existing model output for resume mode."""
    if not output_path.exists():
        return []
    with open(output_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        return []
    return payload


def save_json(path: Path, data: Any) -> None:
    """Write json file with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


async def generate_for_model(
    *,
    model_spec: str,
    rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    temperature: float,
    max_tokens: int,
    retries: int,
    save_every: int,
    batch_size: int,
    concurrency: int,
    stop_on_quota_exhausted: bool,
    resume: bool,
    compatible_api_key_env: str,
    compatible_base_url: str,
) -> Dict[str, Any]:
    """Generate model answers for all rows and save per-model output."""
    explicit_provider, parsed_model_name = parse_model_spec(model_spec)
    provider = explicit_provider or infer_provider(parsed_model_name)
    model_name = parsed_model_name

    client = build_client(
        provider=provider,
        model_name=model_name,
        compatible_api_key_env=compatible_api_key_env,
        compatible_base_url=compatible_base_url,
    )

    safe_file_stem = sanitize_filename(model_spec)
    output_path = output_dir / f"{safe_file_stem}_responses.json"

    records: List[Dict[str, Any]] = []
    completed_ids = set()
    if resume:
        records = load_existing_records(output_path)
        completed_ids = {str(item.get("question_id")) for item in records}
        print(
            f"Resuming {model_spec}: loaded {len(records)} existing rows from {output_path.name}"
        )

    pending_rows = [row for row in rows if row["question_id"] not in completed_ids]
    progress = tqdm(total=len(pending_rows), desc=model_spec)

    failed_count = 0
    processed_count = 0
    quota_exhausted = False
    stopped_reason = "completed"
    quota_error_sample = ""
    last_success_row_position = -1
    last_success_csv_line_number: Optional[int] = None
    quota_trigger_row_position: Optional[int] = None
    quota_trigger_csv_line_number: Optional[int] = None

    batch_size = max(1, batch_size)
    concurrency = max(1, min(concurrency, batch_size))
    since_last_save = 0

    for batch_start in range(0, len(pending_rows), batch_size):
        if quota_exhausted and stop_on_quota_exhausted:
            break

        batch_rows = pending_rows[batch_start : batch_start + batch_size]
        semaphore = asyncio.Semaphore(concurrency)

        async def _worker(row: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await generate_record_for_row(
                    row=row,
                    model_spec=model_spec,
                    provider=provider,
                    model_name=model_name,
                    client=client,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    retries=retries,
                )

        tasks = [asyncio.create_task(_worker(row)) for row in batch_rows]

        for future in asyncio.as_completed(tasks):
            outcome = await future
            record = outcome["record"]
            error_message = outcome["error_message"]
            is_quota_exhausted = outcome["is_quota_exhausted"]
            row_position = outcome["row_position"]
            csv_line_number = outcome["csv_line_number"]
            processed_count += 1
            progress.update(1)

            if record is not None:
                records.append(record)
                completed_ids.add(record["question_id"])
                since_last_save += 1
                if row_position > last_success_row_position:
                    last_success_row_position = row_position
                    last_success_csv_line_number = csv_line_number
            else:
                failed_count += 1
                if error_message:
                    line_suffix = (
                        f" (csv line {csv_line_number})"
                        if csv_line_number is not None
                        else ""
                    )
                    print(f"\nError with {model_spec}{line_suffix}: {error_message}")

            if is_quota_exhausted:
                quota_exhausted = True
                quota_error_sample = error_message or "Quota exhausted."
                quota_trigger_row_position = row_position
                quota_trigger_csv_line_number = csv_line_number
                if stop_on_quota_exhausted:
                    stopped_reason = "quota_exhausted"
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    break

            if save_every > 0 and since_last_save >= save_every:
                save_json(output_path, records)
                since_last_save = 0

        # Batch checkpoint save.
        if since_last_save > 0:
            save_json(output_path, records)
            since_last_save = 0

    save_json(output_path, records)
    progress.close()
    if quota_exhausted and stop_on_quota_exhausted:
        print(
            "\nQuota exhausted. Stopping safely with checkpoint saved.\n"
            f"  - Quota hit near CSV line: {quota_trigger_csv_line_number}\n"
            f"  - Last successful CSV line: {last_success_csv_line_number}\n"
            "  - Re-run with --resume after topping up credits."
        )
    print(f"Saved {len(records)} responses to {output_path}")
    return {
        "records": records,
        "status": stopped_reason,
        "failed_count": failed_count,
        "processed_count": processed_count,
        "quota_exhausted": quota_exhausted,
        "quota_error_sample": quota_error_sample,
        "output_file": str(output_path),
        "last_success_row_position": last_success_row_position,
        "last_success_csv_line_number": last_success_csv_line_number,
        "quota_trigger_row_position": quota_trigger_row_position,
        "quota_trigger_csv_line_number": quota_trigger_csv_line_number,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate model answers for question_bank_10k.csv rows."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="./question_bank_output/question_bank_10k.csv",
        help="Input question bank csv path.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Single model for this inference run. You can use `provider:model` "
            "format. Examples: gpt-5.4, anthropic:claude-opus-4-6, moonshot:kimi-k2"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./question_bank_output/generated_answers",
        help="Directory for this model run outputs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature for generation (default: 0.3).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Max completion tokens per response (default: 1024).",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Row start offset in csv (default: 0).",
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=None,
        help="Optional number of rows to process after start_index.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing per-model json files.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries per failed generation request (default: 2).",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="Save checkpoint every N generated rows per model (default: 50).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=25,
        help="Batch size for grouped processing/checkpointing (default: 25).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max in-flight requests within each batch (default: 5).",
    )
    parser.add_argument(
        "--continue_on_quota_exhausted",
        action="store_true",
        help=(
            "By default the script stops the current model on quota/credit exhaustion "
            "and keeps checkpoints. Set this flag to continue attempting requests."
        ),
    )
    parser.add_argument(
        "--compatible_api_key_env",
        type=str,
        default="COMPATIBLE_API_KEY",
        help="API key env var for `compatible:model` mode (default: COMPATIBLE_API_KEY).",
    )
    parser.add_argument(
        "--compatible_base_url",
        type=str,
        default="",
        help="Base URL for `compatible:model` mode (required if using that provider prefix).",
    )
    return parser.parse_args()


async def main() -> None:
    """Entrypoint."""
    args = parse_args()
    input_csv = Path(args.input_csv)
    if not input_csv.is_absolute():
        input_csv = PROJECT_ROOT / input_csv

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_question_bank_rows(
        csv_path=input_csv,
        start_index=args.start_index,
        num_questions=args.num_questions,
    )
    if not rows:
        raise RuntimeError("No rows found to generate.")

    print(f"Loaded {len(rows)} question rows from {input_csv}")

    stop_on_quota_exhausted = not args.continue_on_quota_exhausted
    model_spec = args.model
    print(f"\nGenerating answers with {model_spec}")
    model_result = await generate_for_model(
        model_spec=model_spec,
        rows=rows,
        output_dir=output_dir,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
        save_every=args.save_every,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        stop_on_quota_exhausted=stop_on_quota_exhausted,
        resume=args.resume,
        compatible_api_key_env=args.compatible_api_key_env,
        compatible_base_url=args.compatible_base_url,
    )

    manifest = {
        "created_at": datetime.now().isoformat(),
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "model": model_spec,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "start_index": args.start_index,
        "num_questions": args.num_questions,
        "resume": args.resume,
        "retries": args.retries,
        "save_every": args.save_every,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "stop_on_quota_exhausted": stop_on_quota_exhausted,
        "total_rows_loaded": len(rows),
        "total_responses_written": len(model_result["records"]),
        "run_result": {
            "status": model_result["status"],
            "processed_count": model_result["processed_count"],
            "failed_count": model_result["failed_count"],
            "quota_exhausted": model_result["quota_exhausted"],
            "quota_error_sample": model_result["quota_error_sample"],
            "output_file": model_result["output_file"],
            "responses_written": len(model_result["records"]),
            "last_success_row_position": model_result["last_success_row_position"],
            "last_success_csv_line_number": model_result["last_success_csv_line_number"],
            "quota_trigger_row_position": model_result["quota_trigger_row_position"],
            "quota_trigger_csv_line_number": model_result["quota_trigger_csv_line_number"],
        },
    }
    save_json(output_dir / "generation_manifest.json", manifest)

    print("\n" + "=" * 60)
    print("Generation complete")
    print("=" * 60)
    print(f"  Input CSV        : {input_csv}")
    print(f"  Rows processed   : {len(rows)}")
    print(f"  Model            : {model_spec}")
    print(f"  Temperature      : {args.temperature}")
    print(f"  Max tokens       : {args.max_tokens}")
    print(f"  Responses saved  : {len(model_result['records'])}")
    if model_result["quota_exhausted"]:
        print(f"  Quota hit line   : {model_result['quota_trigger_csv_line_number']}")
        print(f"  Last saved line  : {model_result['last_success_csv_line_number']}")
    print(f"  Model output     : {model_result['output_file']}")
    print(f"  Output directory : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
