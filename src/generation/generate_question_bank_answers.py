"""Generate LLM answers for rows in question_bank_10k.csv.

This script is provider-aware and supports:
- OpenAI (`openai:gpt-5.4`, `gpt-5.4`, etc.)
- Anthropic (`anthropic:claude-opus-4-6`, `claude-opus-4-6`, etc.)
- Gemini (`gemini:gemini-3.1-pro`, `gemini-3.1-pro`, etc.)
- Moonshot/Kimi (`moonshot:kimi-k2`, `kimi-k2`, etc.)
- Other OpenAI-compatible endpoints (`openrouter:...`, `together:...`, etc.)

Typical usage (single line; split across lines in your shell with line continuation if you prefer):

    python src/generation/generate_question_bank_answers.py --input_csv ./question_bank_output/question_bank_10k.csv --model gpt-5.4 --output_dir ./question_bank_output/generated_answers

All CLI flags (defaults in parentheses):

    --input_csv PATH              Question bank CSV (./question_bank_output/question_bank_10k.csv)
    --model MODEL                 Required. e.g. anthropic:claude-opus-4-6, gpt-5.4, gemini:...
    --output_dir PATH             Run output directory (./question_bank_output/generated_answers)
    --temperature FLOAT           Sampling temperature (0.3)
    --max_tokens INT              Max completion tokens per response (1024)
    --start_index INT             CSV row offset (0)
    --num_questions INT           Optional cap on rows after start_index
    --resume                      Resume from existing per-model JSON
    --retries INT                 Retries per failed request (2)
    --save_every INT              Checkpoint every N rows (50)
    --batch_size INT              Batch size for grouping/checkpoints (25)
    --concurrency INT             Max in-flight requests per batch (5); in Anthropic batch mode, max in-flight Anthropic batches
    --anthropic_use_message_batches
                                 Use Anthropic Message Batches API for anthropic models
    --anthropic_batch_poll_interval_seconds FLOAT
                                 Poll interval while waiting for Anthropic batch completion (15.0)
    --continue_on_quota_exhausted Keep trying after quota/credit exhaustion
    --requests_per_minute INT     Optional RPM cap (Gemini 3.1 Pro defaults to 25 if omitted)
    --disable_builtin_rate_limit  Turn off model-specific built-in rate limits
    --compatible_api_key_env VAR  Env var for compatible:... API key (COMPATIBLE_API_KEY)
    --compatible_base_url URL     Base URL for compatible:... provider (required for that mode)

Hidden / deprecated: --gemini31pro_rpm (use --requests_per_minute). Run with --help for argparse text.

Full example (Anthropic Message Batches; omits no-op optional flags — add --continue_on_quota_exhausted, --disable_builtin_rate_limit, or --compatible_* when needed):

    python src/generation/generate_question_bank_answers.py --input_csv ./question_bank_output/question_bank_10k.csv --model anthropic:claude-opus-4-6 --output_dir ./question_bank_output/generated_answers_claude --temperature 0.3 --max_tokens 1024 --start_index 0 --num_questions 10000 --resume --retries 2 --save_every 50 --batch_size 25 --anthropic_use_message_batches --anthropic_batch_poll_interval_seconds 15
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
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

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
_ANTHROPIC_CUSTOM_ID_CLEAN_RE = re.compile(r"[^a-zA-Z0-9_-]+")

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


_QUOTA_ERROR_MARKERS = (
    "insufficient_quota",
    "quota exceeded",
    "exceeded your current quota",
    "out of credits",
    "insufficient credits",
    "billing",
    "payment required",
    "credit balance",
    "resource has been exhausted",
)


class AsyncRateLimiter:
    """Sliding-window async rate limiter."""

    def __init__(self, max_calls: int, period_seconds: float):
        if max_calls <= 0:
            raise ValueError("max_calls must be > 0")
        if period_seconds <= 0:
            raise ValueError("period_seconds must be > 0")
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until one request slot is available."""
        while True:
            wait_seconds = 0.0
            async with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self.period_seconds:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return

                oldest = self._timestamps[0]
                wait_seconds = max(0.0, self.period_seconds - (now - oldest)) + 0.01

            await asyncio.sleep(wait_seconds)


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


def build_anthropic_batch_custom_id(row: Dict[str, Any], local_index: int) -> str:
    """Build a stable custom_id satisfying Anthropic batch constraints."""
    row_position = int(row.get("row_position", -1))
    normalized_row_position = row_position if row_position >= 0 else local_index
    question_id = str(row.get("question_id") or "")
    hash_input = f"{question_id}:{normalized_row_position}:{local_index}"
    digest = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:16]
    custom_id = f"qb_{normalized_row_position}_{digest}"
    custom_id = _ANTHROPIC_CUSTOM_ID_CLEAN_RE.sub("_", custom_id).strip("_")
    if not custom_id:
        custom_id = f"qb_{digest}"
    return custom_id[:64]


def get_row_position_and_csv_line_number(row: Dict[str, Any]) -> Tuple[int, Optional[int]]:
    """Return row position and CSV line number metadata for a row dict."""
    row_position = int(row.get("row_position", -1))
    csv_line_number = (row_position + 2) if row_position >= 0 else None
    return row_position, csv_line_number


def build_output_record(
    *,
    row: Dict[str, Any],
    generated_response: str,
    model_spec: str,
    provider: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Build output record in evaluator-compatible schema."""
    row_position, csv_line_number = get_row_position_and_csv_line_number(row)
    question_id = row["question_id"]
    return {
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


def infer_retry_after_seconds(error_text: str) -> Optional[float]:
    """Extract server-provided retry delay from error text if present."""
    patterns = [
        r"retry in ([0-9]+(?:\.[0-9]+)?)s",
        r"retry after ([0-9]+(?:\.[0-9]+)?)s",
        r"seconds:\s*([0-9]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def is_retryable_rate_limit_error(error: Exception) -> bool:
    """Detect transient 429/rate-limit errors that should be retried."""
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)

    if status_code != 429:
        return False

    error_text = str(error).lower()
    retryable_markers = [
        "retry in",
        "retry_delay",
        "too many requests",
        "rate limit",
        "ratelimit",
        "per minute",
        "generaterequestsperminute",
    ]
    return any(marker in error_text for marker in retryable_markers)


def is_quota_exhaustion_error(error: Exception) -> bool:
    """Best-effort detection for credit/quota exhaustion errors."""
    if is_retryable_rate_limit_error(error):
        return False

    error_text = str(error).lower()
    if any(marker in error_text for marker in _QUOTA_ERROR_MARKERS):
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


def is_quota_exhaustion_message(error_message: str) -> bool:
    """Detect quota exhaustion from plain error text."""
    lowered = (error_message or "").lower()
    return any(marker in lowered for marker in _QUOTA_ERROR_MARKERS)


T = TypeVar("T")


async def run_with_retries(
    *,
    operation: Callable[[], Awaitable[T]],
    retries: int,
    rate_limiter: Optional[AsyncRateLimiter] = None,
) -> T:
    """Run async operation with retry, rate-limit, and quota handling."""
    attempt = 0
    while True:
        try:
            if rate_limiter is not None:
                await rate_limiter.acquire()
            return await operation()
        except Exception as error:
            if is_retryable_rate_limit_error(error):
                wait_seconds = infer_retry_after_seconds(str(error))
                if wait_seconds is None:
                    wait_seconds = min(2 ** attempt, 60)
                await asyncio.sleep(max(wait_seconds, 0.5))
                attempt += 1
                continue
            if is_quota_exhaustion_error(error):
                raise QuotaExhaustedError(str(error)) from error
            attempt += 1
            if attempt > retries:
                raise
            await asyncio.sleep(min(2 ** attempt, 8))


async def generate_single_answer(
    *,
    client: LLMClient,
    question_text: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    rate_limiter: Optional[AsyncRateLimiter] = None,
) -> str:
    """Generate one answer with retry policy."""
    raw = await run_with_retries(
        operation=lambda: client.generate(
            system_prompt=PHYSICIAN_SYSTEM_PROMPT,
            user_prompt=build_user_prompt(question_text),
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        retries=retries,
        rate_limiter=rate_limiter,
    )
    return cleanup_response(raw)


def extract_text_from_anthropic_message(message: Any) -> str:
    """Extract text blocks from Anthropic message content."""
    content_blocks = getattr(message, "content", None) or []
    text_parts: List[str] = []
    for block in content_blocks:
        if getattr(block, "type", None) != "text":
            continue
        text_value = getattr(block, "text", "")
        if text_value:
            text_parts.append(text_value)
    return cleanup_response("\n".join(text_parts).strip())


def format_anthropic_batch_error(error_payload: Any) -> str:
    """Best-effort flattening for Anthropic batch result errors."""
    if error_payload is None:
        return "Anthropic batch request failed."
    nested_error = getattr(error_payload, "error", None)
    error_type = getattr(nested_error, "type", None) or getattr(error_payload, "type", None)
    error_message = getattr(nested_error, "message", None) or getattr(
        error_payload, "message", None
    )
    if error_type and error_message:
        return f"{error_type}: {error_message}"
    if error_message:
        return str(error_message)
    return str(error_payload)


def make_failed_row_outcome(
    row: Dict[str, Any],
    *,
    error_message: str,
    is_quota_exhausted: bool,
) -> Dict[str, Any]:
    """Build failed outcome payload for one row."""
    row_position, csv_line_number = get_row_position_and_csv_line_number(row)
    return {
        "record": None,
        "error_message": error_message,
        "is_quota_exhausted": is_quota_exhausted,
        "row_position": row_position,
        "csv_line_number": csv_line_number,
    }


async def generate_records_for_rows_anthropic_batch(
    *,
    rows: Sequence[Dict[str, Any]],
    model_spec: str,
    model_name: str,
    client: "AnthropicClient",
    temperature: float,
    max_tokens: int,
    retries: int,
    rate_limiter: Optional[AsyncRateLimiter] = None,
    poll_interval_seconds: float = 15.0,
) -> List[Dict[str, Any]]:
    """Generate row outcomes using Anthropic Message Batches API."""
    if not rows:
        return []

    try:
        ordered_custom_ids: List[str] = []
        row_by_custom_id: Dict[str, Dict[str, Any]] = {}
        requests: List[Dict[str, Any]] = []
        for local_index, row in enumerate(rows):
            custom_id = build_anthropic_batch_custom_id(row=row, local_index=local_index)
            duplicate_suffix = 1
            while custom_id in row_by_custom_id:
                custom_id = f"{custom_id[:52]}_{local_index}_{duplicate_suffix}"[:64]
                duplicate_suffix += 1
            ordered_custom_ids.append(custom_id)
            row_by_custom_id[custom_id] = row
            requests.append(
                {
                    "custom_id": custom_id,
                    "params": {
                        "model": model_name,
                        "max_tokens": max(1, max_tokens),
                        "system": PHYSICIAN_SYSTEM_PROMPT,
                        "messages": [
                            {"role": "user", "content": build_user_prompt(row["question_text"])}
                        ],
                        "temperature": temperature,
                    },
                }
            )

        message_batch = await run_with_retries(
            operation=lambda: client.client.messages.batches.create(requests=requests),
            retries=retries,
            rate_limiter=rate_limiter,
        )

        batch_id = message_batch.id
        status = getattr(message_batch, "processing_status", "in_progress")
        print(f"\nSubmitted Anthropic message batch {batch_id} with {len(requests)} requests.")

        while status != "ended":
            await asyncio.sleep(max(1.0, poll_interval_seconds))
            message_batch = await run_with_retries(
                operation=lambda: client.client.messages.batches.retrieve(batch_id),
                retries=retries,
                rate_limiter=rate_limiter,
            )
            status = getattr(message_batch, "processing_status", "in_progress")

        result_stream = await run_with_retries(
            operation=lambda: client.client.messages.batches.results(batch_id),
            retries=retries,
            rate_limiter=rate_limiter,
        )
        outcomes_by_custom_id: Dict[str, Dict[str, Any]] = {}
        async for result in result_stream:
            custom_id = str(getattr(result, "custom_id", ""))
            row = row_by_custom_id.get(custom_id)
            if row is None:
                continue

            result_payload = getattr(result, "result", None)
            result_type = getattr(result_payload, "type", None)
            if result_type == "succeeded":
                message = getattr(result_payload, "message", None)
                generated_response = extract_text_from_anthropic_message(message)
                record = build_output_record(
                    row=row,
                    generated_response=generated_response,
                    model_spec=model_spec,
                    provider="anthropic",
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                row_position, csv_line_number = get_row_position_and_csv_line_number(row)
                outcomes_by_custom_id[custom_id] = {
                    "record": record,
                    "error_message": None,
                    "is_quota_exhausted": False,
                    "row_position": row_position,
                    "csv_line_number": csv_line_number,
                }
                continue

            if result_type == "errored":
                error_message = format_anthropic_batch_error(
                    getattr(result_payload, "error", None)
                )
            elif result_type == "canceled":
                error_message = "Anthropic batch request was canceled."
            elif result_type == "expired":
                error_message = "Anthropic batch request expired."
            else:
                error_message = f"Unknown Anthropic batch result type: {result_type}"

            outcomes_by_custom_id[custom_id] = make_failed_row_outcome(
                row,
                error_message=error_message,
                is_quota_exhausted=is_quota_exhaustion_message(error_message),
            )

        outcomes: List[Dict[str, Any]] = []
        for custom_id in ordered_custom_ids:
            row = row_by_custom_id[custom_id]
            outcome = outcomes_by_custom_id.get(custom_id)
            if outcome is None:
                outcome = make_failed_row_outcome(
                    row,
                    error_message=(
                        "Anthropic batch result was missing for this request custom_id."
                    ),
                    is_quota_exhausted=False,
                )
            outcomes.append(outcome)
        return outcomes
    except QuotaExhaustedError as error:
        return [
            make_failed_row_outcome(
                row,
                error_message=str(error),
                is_quota_exhausted=True,
            )
            for row in rows
        ]
    except Exception as error:
        error_message = str(error)
        return [
            make_failed_row_outcome(
                row,
                error_message=error_message,
                is_quota_exhausted=is_quota_exhaustion_message(error_message),
            )
            for row in rows
        ]


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
    rate_limiter: Optional[AsyncRateLimiter] = None,
) -> Dict[str, Any]:
    """Generate one output record with row-position metadata."""
    row_position, csv_line_number = get_row_position_and_csv_line_number(row)

    try:
        generated_response = await generate_single_answer(
            client=client,
            question_text=row["question_text"],
            temperature=temperature,
            max_tokens=max_tokens,
            retries=retries,
            rate_limiter=rate_limiter,
        )
        record = build_output_record(
            row=row,
            generated_response=generated_response,
            model_spec=model_spec,
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
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


def build_builtin_rate_limiter(
    provider: str,
    model_name: str,
    requests_per_minute: Optional[int],
    disable_builtin_rate_limit: bool,
) -> Optional[AsyncRateLimiter]:
    """Return a built-in rate limiter for known models/providers."""
    if disable_builtin_rate_limit:
        return None

    provider_lower = provider.lower()
    model_lower = model_name.lower()

    # Explicit override: apply RPM cap to any model/provider.
    if requests_per_minute is not None:
        if requests_per_minute > 0:
            return AsyncRateLimiter(max_calls=requests_per_minute, period_seconds=60.0)
        return None

    # Default safety cap for Gemini 3.1 Pro family only.
    if provider_lower == "gemini" and "gemini-3.1-pro" in model_lower:
        return AsyncRateLimiter(max_calls=25, period_seconds=60.0)
    return None


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
    requests_per_minute: Optional[int],
    disable_builtin_rate_limit: bool,
    compatible_api_key_env: str,
    compatible_base_url: str,
    anthropic_use_message_batches: bool,
    anthropic_batch_poll_interval_seconds: float,
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
    rate_limiter = build_builtin_rate_limiter(
        provider=provider,
        model_name=model_name,
        requests_per_minute=requests_per_minute,
        disable_builtin_rate_limit=disable_builtin_rate_limit,
    )
    if rate_limiter is not None:
        print(
            f"Built-in rate limit enabled for {model_spec}: "
            f"{rate_limiter.max_calls} requests per {int(rate_limiter.period_seconds)}s."
        )
    use_anthropic_message_batches = (
        provider == "anthropic" and anthropic_use_message_batches
    )
    if use_anthropic_message_batches:
        print(
            "Anthropic Message Batches mode enabled. "
            "Batches are submitted asynchronously and results are matched by custom_id."
        )
        print(
            "Using --concurrency as Anthropic in-flight batch parallelism: "
            f"{max(1, concurrency)}"
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
    request_concurrency = max(1, min(concurrency, batch_size))
    anthropic_batch_parallelism = max(1, concurrency)
    since_last_save = 0

    def _consume_outcomes(outcomes: Sequence[Dict[str, Any]]) -> bool:
        nonlocal failed_count
        nonlocal processed_count
        nonlocal quota_exhausted
        nonlocal stopped_reason
        nonlocal quota_error_sample
        nonlocal last_success_row_position
        nonlocal last_success_csv_line_number
        nonlocal quota_trigger_row_position
        nonlocal quota_trigger_csv_line_number
        nonlocal since_last_save

        stop_after_quota = False
        for outcome in outcomes:
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
                    stop_after_quota = True
                    if not use_anthropic_message_batches:
                        break

            if save_every > 0 and since_last_save >= save_every:
                save_json(output_path, records)
                since_last_save = 0
        return stop_after_quota

    if use_anthropic_message_batches:
        batch_rows_groups = [
            pending_rows[batch_start : batch_start + batch_size]
            for batch_start in range(0, len(pending_rows), batch_size)
        ]
        batch_semaphore = asyncio.Semaphore(anthropic_batch_parallelism)

        async def _batch_worker(batch_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
            async with batch_semaphore:
                return await generate_records_for_rows_anthropic_batch(
                    rows=batch_rows,
                    model_spec=model_spec,
                    model_name=model_name,
                    client=client,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                    retries=retries,
                    rate_limiter=rate_limiter,
                    poll_interval_seconds=anthropic_batch_poll_interval_seconds,
                )

        batch_tasks = [asyncio.create_task(_batch_worker(rows_group)) for rows_group in batch_rows_groups]
        stop_after_quota = False
        for future in asyncio.as_completed(batch_tasks):
            outcomes = await future
            stop_after_quota = _consume_outcomes(outcomes)

            # Completed batch checkpoint save.
            if since_last_save > 0:
                save_json(output_path, records)
                since_last_save = 0

            if stop_after_quota:
                for task in batch_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*batch_tasks, return_exceptions=True)
                break
    else:
        for batch_start in range(0, len(pending_rows), batch_size):
            if quota_exhausted and stop_on_quota_exhausted:
                break

            batch_rows = pending_rows[batch_start : batch_start + batch_size]
            semaphore = asyncio.Semaphore(request_concurrency)

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
                        rate_limiter=rate_limiter,
                    )

            tasks = [asyncio.create_task(_worker(row)) for row in batch_rows]
            outcomes: List[Dict[str, Any]] = []
            stop_after_quota = False
            for future in asyncio.as_completed(tasks):
                outcome = await future
                outcomes.append(outcome)
                if outcome["is_quota_exhausted"] and stop_on_quota_exhausted:
                    stop_after_quota = True
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    break
            if not stop_after_quota:
                await asyncio.gather(*tasks, return_exceptions=True)

            stop_after_quota = _consume_outcomes(outcomes)

            # Batch checkpoint save.
            if since_last_save > 0:
                save_json(output_path, records)
                since_last_save = 0

            if stop_after_quota:
                break

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
        "builtin_rate_limit_rpm": rate_limiter.max_calls if rate_limiter else None,
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
        help=(
            "Max in-flight requests within each batch (default: 5). "
            "When Anthropic Message Batches mode is enabled, this becomes "
            "the max in-flight Anthropic batches."
        ),
    )
    parser.add_argument(
        "--anthropic_use_message_batches",
        action="store_true",
        help=(
            "Use Anthropic Message Batches API for provider `anthropic`. "
            "This submits each local --batch_size group as one async Anthropic batch job."
        ),
    )
    parser.add_argument(
        "--anthropic_batch_poll_interval_seconds",
        type=float,
        default=15.0,
        help=(
            "Polling interval when waiting for Anthropic batch completion "
            "(default: 15.0 seconds)."
        ),
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
        "--requests_per_minute",
        type=int,
        default=None,
        help=(
            "Optional generic RPM cap for this run (applies to any model). "
            "If omitted, Gemini 3.1 Pro uses default 25 RPM."
        ),
    )
    parser.add_argument(
        "--gemini31pro_rpm",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--disable_builtin_rate_limit",
        action="store_true",
        help="Disable built-in model-specific rate limiting.",
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
    effective_requests_per_minute = args.requests_per_minute
    if effective_requests_per_minute is None and args.gemini31pro_rpm is not None:
        effective_requests_per_minute = args.gemini31pro_rpm
        print(
            "Warning: --gemini31pro_rpm is deprecated; use --requests_per_minute instead."
        )
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
        requests_per_minute=effective_requests_per_minute,
        disable_builtin_rate_limit=args.disable_builtin_rate_limit,
        compatible_api_key_env=args.compatible_api_key_env,
        compatible_base_url=args.compatible_base_url,
        anthropic_use_message_batches=args.anthropic_use_message_batches,
        anthropic_batch_poll_interval_seconds=args.anthropic_batch_poll_interval_seconds,
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
        "anthropic_use_message_batches": args.anthropic_use_message_batches,
        "anthropic_batch_poll_interval_seconds": args.anthropic_batch_poll_interval_seconds,
        "stop_on_quota_exhausted": stop_on_quota_exhausted,
        "requests_per_minute": effective_requests_per_minute,
        "deprecated_gemini31pro_rpm": args.gemini31pro_rpm,
        "disable_builtin_rate_limit": args.disable_builtin_rate_limit,
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
            "builtin_rate_limit_rpm": model_result["builtin_rate_limit_rpm"],
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
    if model_result["builtin_rate_limit_rpm"] is not None:
        print(f"  Built-in RPM cap : {model_result['builtin_rate_limit_rpm']}")
    if model_result["quota_exhausted"]:
        print(f"  Quota hit line   : {model_result['quota_trigger_csv_line_number']}")
        print(f"  Last saved line  : {model_result['last_success_csv_line_number']}")
    print(f"  Model output     : {model_result['output_file']}")
    print(f"  Output directory : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
