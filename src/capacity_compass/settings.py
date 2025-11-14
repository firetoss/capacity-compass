"""Runtime settings derived from environment variables.

Centralizes feature flags (LLM summary switch, prompt overrides, logging) so
pipeline/API layers can remain declarative. Values are cached per process.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "sales_summary_zh.txt"
DEFAULT_LLM_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class RuntimeSettings:
    """Container for runtime flags."""

    enable_llm_summary: bool
    log_level: str
    llm_prompt_path: Path
    llm_base_url: str
    llm_timeout_seconds: float
    http_proxy: Optional[str]


def _env_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _normalize_log_level(value: str | None) -> str:
    if not value:
        return "INFO"
    level = value.strip().upper()
    if level in logging._nameToLevel:
        return level
    return "INFO"


def _env_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_path(value: str | None, default: Path) -> Path:
    if not value:
        return default
    return Path(value).expanduser()


def load_runtime_settings() -> RuntimeSettings:
    """Load settings without caching (useful for tests)."""

    enable_summary = _env_bool(os.getenv("CAPACITYCOMPASS_ENABLE_SUMMARY"), False)
    log_level = _normalize_log_level(os.getenv("LOG_LEVEL"))
    prompt_path = _env_path(os.getenv("CAPACITYCOMPASS_PROMPT_PATH"), DEFAULT_PROMPT_PATH)
    base_url = os.getenv("CAPACITYCOMPASS_LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
    timeout = _env_float(os.getenv("CAPACITYCOMPASS_LLM_TIMEOUT"), 30.0)
    http_proxy = os.getenv("CAPACITYCOMPASS_HTTP_PROXY")
    return RuntimeSettings(
        enable_llm_summary=enable_summary,
        log_level=log_level,
        llm_prompt_path=prompt_path,
        llm_base_url=base_url,
        llm_timeout_seconds=timeout,
        http_proxy=http_proxy,
    )


@lru_cache(maxsize=1)
def get_runtime_settings() -> RuntimeSettings:
    """Cached accessor used by runtime code."""

    return load_runtime_settings()


def reset_runtime_settings_cache() -> None:
    """Testing helper to clear cached settings."""

    get_runtime_settings.cache_clear()
