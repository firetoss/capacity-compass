from capacity_compass import config_loader
from capacity_compass.models_registry import ModelsRegistry
from capacity_compass.pipeline.normalizer import EvaluationRequest, normalize_request


def _registry() -> ModelsRegistry:
    return ModelsRegistry(config_loader.load_models())


def test_normalizer_switches_to_long_context_instruct() -> None:
    registry = _registry()
    req = EvaluationRequest(model="Qwen3-4B", max_context_len=200_000)
    normalized = normalize_request(req, registry)
    assert normalized.model.model_name == "Qwen/Qwen3-4B-Instruct-2507"
    assert normalized.switch_notice is not None


def test_normalizer_creates_anonymous_model_when_missing() -> None:
    registry = _registry()
    req = EvaluationRequest(model="Custom-LLM", param_count_b=12)
    normalized = normalize_request(req, registry)
    assert normalized.is_anonymous_model is True
    assert normalized.model.param_count_b == 12
