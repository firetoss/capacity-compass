from capacity_compass import config_loader
from capacity_compass.models_registry import ModelsRegistry
from capacity_compass.pipeline.evaluation_builder import build_evaluation
from capacity_compass.pipeline.normalizer import EvaluationRequest, normalize_request
from capacity_compass.pipeline.recommender import RankedCandidate
from capacity_compass.pipeline.requirements import RequirementResult


def _dummy_requirement(context_len: int, clamped: bool = False) -> RequirementResult:
    return RequirementResult(
        context_len=context_len,
        requested_context_len=context_len,
        context_clamped=clamped,
        weights_mem_bytes=10,
        kv_mem_bytes=5,
        total_mem_bytes=15,
        required_compute_Tx=1.0,
        notes=[],
    )


def _dummy_candidate(name: str, price: float | None = 1000.0) -> RankedCandidate:
    return RankedCandidate(
        gpu_id=name.lower(),
        gpu_name=name,
        cards_needed=1,
        cards_mem=1,
        cards_compute=1,
        headroom=0.3,
        total_price=price,
        deploy_support="excellent",
        notes=[],
    )


def test_evaluation_builder_generates_quick_answer():
    models = ModelsRegistry(config_loader.load_models())
    scenarios = config_loader.load_scenarios()

    req = EvaluationRequest(model="Qwen3-4B", max_context_len=16000)
    normalized = normalize_request(req, models)
    preset = scenarios["chat"]
    requirement = _dummy_requirement(16000)
    ranked = [_dummy_candidate("GPU-A"), _dummy_candidate("GPU-B")]

    evaluation = build_evaluation(
        normalized,
        {"chat": preset},
        {"chat": requirement},
        {"chat": ranked},
    )

    chat_summary = evaluation["scenes"]["chat"]["sales_summary"]
    assert chat_summary["primary"]["device"] == "GPU-A"
    assert evaluation["quick_answer"]["primary"]["device"] == "GPU-A"
    assert evaluation["scenes"]["chat"]["guide"]["title"] == "对话问答"
    assert evaluation["disclaimers"]


def test_evaluation_builder_adds_context_clamp_tip():
    models = ModelsRegistry(config_loader.load_models())
    scenarios = config_loader.load_scenarios()

    req = EvaluationRequest(model="Qwen3-4B", max_context_len=200_000)
    normalized = normalize_request(req, models)
    preset = scenarios["chat"]
    requirement = _dummy_requirement(200_000, clamped=True)
    ranked = [_dummy_candidate("GPU-A", price=None)]

    evaluation = build_evaluation(
        normalized,
        {"chat": preset},
        {"chat": requirement},
        {"chat": ranked},
    )

    chat_summary = evaluation["scenes"]["chat"]["sales_summary"]
    assert any("上下文" in tip for tip in chat_summary["tips"])
    assert chat_summary["switch_notice"] is not None


def test_evaluation_builder_adds_disclaimer_when_no_candidates():
    models = ModelsRegistry(config_loader.load_models())
    scenarios = config_loader.load_scenarios()

    req = EvaluationRequest(model="Qwen3-4B")
    normalized = normalize_request(req, models)
    preset = scenarios["chat"]
    requirement = _dummy_requirement(8192)

    evaluation = build_evaluation(
        normalized,
        {"chat": preset},
        {"chat": requirement},
        {"chat": []},
    )

    disclaimers = evaluation["disclaimers"]
    assert any("暂无满足" in text for text in disclaimers)
