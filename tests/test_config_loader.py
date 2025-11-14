"""Tests for configuration loading helpers."""

from capacity_compass import config_loader


def test_load_hardware_contains_l20_with_fp8_support() -> None:
    gpus = config_loader.load_hardware()
    l20 = next(g for g in gpus if g.id == "nvidia_l20_48g")
    assert l20.precision_support.fp8 is True


def test_load_models_merges_qwen_and_deepseek() -> None:
    models = config_loader.load_models()
    ids = {m.model_name for m in models}
    assert "Qwen/Qwen3-4B-Instruct-2507" in ids
    assert "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" in ids


def test_load_scenarios_returns_known_entries() -> None:
    scenarios = config_loader.load_scenarios()
    assert "chat" in scenarios
    assert scenarios["chat"].default_context_len == 8192


def test_load_estimation_defaults_present() -> None:
    estimation = config_loader.load_estimation()
    assert estimation.dtype_bytes["fp16"] == 2
    assert estimation.overhead_ratio_default == 0.15
