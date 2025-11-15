"""Stage 7: optional LLM summary generation via OpenRouter (Qwen3-8B)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

try:  # pragma: no cover - optional extra
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled via runtime guard
    OpenAI = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPT = PROJECT_ROOT / "prompts" / "sales_summary_zh.txt"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-8b"


class LLMProviderError(RuntimeError):
    """Raised when OpenAI client is unavailable or response invalid."""


def generate_llm_summary(
    evaluation: Dict[str, Any],
    disclaimers: Optional[List[str]] = None,
    *,
    prompt_path: Path = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL,
    base_url: Optional[str] = None,
    referer: Optional[str] = None,
    site_title: Optional[str] = None,
    timeout_seconds: float = 30.0,
    http_proxy: Optional[str] = None,
) -> str:
    """Call OpenRouter-compatible endpoint to produce human-readable summary."""

    if OpenAI is None:  # pragma: no cover - depends on extras
        raise LLMProviderError("openai package is not installed; install with llm extra")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMProviderError("OPENROUTER_API_KEY is not set")

    prompt_text = prompt_path.read_text(encoding="utf-8")
    context_payload = _extract_payload(evaluation, disclaimers or [])
    content = json.dumps(context_payload, ensure_ascii=False, indent=2)

    proxy_url = http_proxy or os.getenv("CAPACITYCOMPASS_HTTP_PROXY")
    transport = httpx.HTTPTransport(proxy=proxy_url) if proxy_url else None
    http_client = httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(timeout_seconds),
    )
    client = OpenAI(
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=api_key,
        http_client=http_client,
    )
    headers = {}
    referer = referer or os.getenv("OPENROUTER_REFERER")
    site_title = site_title or os.getenv("OPENROUTER_SITE_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if site_title:
        headers["X-Title"] = site_title

    try:
        response = client.chat.completions.create(
            extra_headers=headers or None,
            model=model,
            messages=[
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": content},
            ],
        )
    finally:
        http_client.close()
    if not response.choices:
        raise LLMProviderError("LLM response missing choices")
    message = response.choices[0].message
    if not message or not message.content:
        raise LLMProviderError("LLM response missing content")
    return message.content.strip()


def _extract_payload(evaluation: Dict[str, Any], disclaimers: List[str]) -> Dict[str, Any]:
    return {
        "quick_answer": evaluation.get("quick_answer"),
        "scenes": evaluation.get("scenes"),
        "model_profile": evaluation.get("raw_evaluation", {}).get("model_profile"),
        "disclaimers": disclaimers,
    }
