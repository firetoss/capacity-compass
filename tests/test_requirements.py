from capacity_compass import config_loader
from capacity_compass.models_registry import ModelsRegistry
from capacity_compass.pipeline.normalizer import EvaluationRequest, normalize_request
from capacity_compass.pipeline.requirements import estimate_requirements


def test_requirements_clamp_context_to_model_limit() -> None:
    models = ModelsRegistry(config_loader.load_models())
    scenarios = config_loader.load_scenarios()
    estimation = config_loader.load_estimation()

    req = EvaluationRequest(model="Qwen3-4B-Instruct-2507", max_context_len=300_000)
    normalized = normalize_request(req, models)
    preset = scenarios["chat"]
    result = estimate_requirements(normalized, preset, estimation)

    assert result.context_len == 262144  # clamp to model limit
    assert result.total_mem_bytes > 0
