"""Tests for ModelsRegistry lookups and context selection."""

from capacity_compass import config_loader
from capacity_compass.models_registry import ModelsRegistry


def _make_registry() -> ModelsRegistry:
    models = config_loader.load_models()
    return ModelsRegistry(models)


def test_match_is_case_insensitive() -> None:
    registry = _make_registry()
    result = registry.get("qwen3-4b-instruct-2507")
    assert result is not None
    assert result.max_position_embeddings == 262144


def test_select_best_for_context_prefers_long_context_variant() -> None:
    registry = _make_registry()
    selected = registry.select_best_for_context("Qwen3", target_ctx=200_000)
    assert selected is not None
    assert selected.model_name == "Qwen/Qwen3-4B-Instruct-2507"
