from pathlib import Path

from capacity_compass.settings import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_PROMPT_PATH,
    get_runtime_settings,
    load_runtime_settings,
    reset_runtime_settings_cache,
)


def test_settings_defaults(monkeypatch):
    reset_runtime_settings_cache()
    monkeypatch.delenv("CAPACITYCOMPASS_ENABLE_SUMMARY", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("CAPACITYCOMPASS_PROMPT_PATH", raising=False)
    monkeypatch.delenv("CAPACITYCOMPASS_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("CAPACITYCOMPASS_LLM_TIMEOUT", raising=False)
    monkeypatch.delenv("CAPACITYCOMPASS_HTTP_PROXY", raising=False)
    settings = load_runtime_settings()
    assert settings.enable_llm_summary is False
    assert settings.log_level == "INFO"
    assert settings.llm_prompt_path == DEFAULT_PROMPT_PATH
    assert settings.llm_base_url == DEFAULT_LLM_BASE_URL
    assert settings.llm_timeout_seconds == 30.0
    assert settings.http_proxy is None


def test_settings_from_env(monkeypatch):
    reset_runtime_settings_cache()
    monkeypatch.setenv("CAPACITYCOMPASS_ENABLE_SUMMARY", "true")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    custom_prompt = Path(__file__).resolve()
    monkeypatch.setenv("CAPACITYCOMPASS_PROMPT_PATH", str(custom_prompt))
    monkeypatch.setenv("CAPACITYCOMPASS_LLM_BASE_URL", "https://custom")
    monkeypatch.setenv("CAPACITYCOMPASS_LLM_TIMEOUT", "45.5")
    monkeypatch.setenv("CAPACITYCOMPASS_HTTP_PROXY", "http://proxy")
    settings = load_runtime_settings()
    assert settings.enable_llm_summary is True
    assert settings.log_level == "DEBUG"
    assert settings.llm_prompt_path == custom_prompt
    assert settings.llm_base_url == "https://custom"
    assert settings.llm_timeout_seconds == 45.5
    assert settings.http_proxy == "http://proxy"


def test_settings_cache(monkeypatch):
    reset_runtime_settings_cache()
    monkeypatch.setenv("CAPACITYCOMPASS_ENABLE_SUMMARY", "1")
    monkeypatch.setenv("LOG_LEVEL", "warning")
    cached = get_runtime_settings()
    assert cached.enable_llm_summary is True
    assert cached.log_level == "WARNING"
