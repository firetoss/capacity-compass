from capacity_compass.pipeline.normalizer import EvaluationRequest
from capacity_compass.pipeline.service import build_service_context, evaluate_capacity


def test_service_evaluates_basic_request():
    context = build_service_context()
    request = EvaluationRequest(model="Qwen3-4B", max_context_len=16000)
    result = evaluate_capacity(request, context)

    assert "quick_answer" in result
    assert "scenes" in result
    assert "sales_summary" in result["scenes"]["chat"]
