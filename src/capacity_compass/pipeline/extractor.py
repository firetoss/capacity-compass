"""Prompted extractor for (model, precision, context_len) via OpenRouter (Qwen3-8B).

This module provides a small helper to call an OpenAI-compatible endpoint
and parse a strict JSON triple from free-form user text.

It reuses the same runtime settings as llm_summary, with a dedicated prompt
path defaulting to prompts/extract_request_zh.txt.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

try:  # pragma: no cover - optional extra
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled via runtime guard
    OpenAI = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTRACT_PROMPT = PROJECT_ROOT / "prompts" / "extract_request_zh.txt"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-8b"


@dataclass
class ExtractResult:
    model: Optional[str]
    precision: Optional[str]
    context_len: Optional[int]


class ExtractorError(RuntimeError):
    pass


def extract_fields(
    user_text: str,
    *,
    prompt_path: Path = DEFAULT_EXTRACT_PROMPT,
    model: str = DEFAULT_MODEL,
    base_url: Optional[str] = None,
    timeout_seconds: float = 20.0,
    http_proxy: Optional[str] = None,
) -> ExtractResult:
    """Call OpenRouter/Qwen3-8B to extract (model, precision, context_len).

    Raises ExtractorError when the OpenAI client is unavailable or the response
    does not contain valid JSON per contract.
    """
    if OpenAI is None:  # pragma: no cover
        raise ExtractorError("openai package is not installed; install with llm extra")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:  # pragma: no cover
        raise ExtractorError("OPENROUTER_API_KEY is not set")

    system_prompt = prompt_path.read_text(encoding="utf-8")
    proxy_url = http_proxy or os.getenv("CAPACITYCOMPASS_HTTP_PROXY")
    transport = httpx.HTTPTransport(proxy=proxy_url) if proxy_url else None
    http_client = httpx.Client(transport=transport, timeout=httpx.Timeout(timeout_seconds))
    client = OpenAI(base_url=base_url or DEFAULT_BASE_URL, api_key=api_key, http_client=http_client)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0,
        )
    finally:
        http_client.close()

    if not resp.choices:  # pragma: no cover
        raise ExtractorError("LLM response missing choices")
    msg = resp.choices[0].message
    if not msg or not msg.content:  # pragma: no cover
        raise ExtractorError("LLM response missing content")

    data = _parse_json(msg.content)
    return ExtractResult(
        model=data.get("model"),
        precision=data.get("precision"),
        context_len=data.get("context_len"),
    )


def _parse_json(text: str) -> Dict[str, Any]:
    """Parse a minimal JSON object from model output.

    Accepts either raw JSON or JSON fenced in code block. Raises ExtractorError
    if parsing fails or keys are missing.
    """
    snippet = text.strip()
    if snippet.startswith("```"):
        # Remove leading/trailing code fences
        lines = [line for line in snippet.splitlines() if not line.strip().startswith("```")]
        snippet = "\n".join(lines).strip()
    try:
        obj = json.loads(snippet)
    except Exception as exc:  # pragma: no cover
        raise ExtractorError(f"invalid JSON from LLM: {exc}")
    for key in ("model", "precision", "context_len"):
        if key not in obj:
            raise ExtractorError(f"missing key in JSON: {key}")
    # Normalize types
    if obj["model"] is not None and not isinstance(obj["model"], str):
        obj["model"] = str(obj["model"])  # pragma: no cover
    if obj["precision"] is not None and not isinstance(obj["precision"], str):
        obj["precision"] = str(obj["precision"])  # pragma: no cover
    if obj["context_len"] is not None:
        try:
            obj["context_len"] = int(obj["context_len"])  # type: ignore[call-overload]
        except Exception as exc:  # pragma: no cover
            raise ExtractorError(f"context_len not integer: {exc}")
    return obj
