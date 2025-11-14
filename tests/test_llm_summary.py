from pathlib import Path
from unittest.mock import MagicMock, patch

from capacity_compass.pipeline.llm_summary import _extract_payload, generate_llm_summary


def test_extract_payload_subset():
    evaluation = {
        "quick_answer": {"primary": {"device": "A"}},
        "scenes": {"chat": {}},
        "raw_evaluation": {"model_profile": {"model_name": "demo"}},
    }
    payload = _extract_payload(evaluation, ["提示"])
    assert payload["quick_answer"]["primary"]["device"] == "A"
    assert payload["disclaimers"] == ["提示"]


@patch("capacity_compass.pipeline.llm_summary.OpenAI")
@patch("capacity_compass.pipeline.llm_summary.os.getenv")
def test_generate_llm_summary_calls_openai(mock_getenv, mock_openai):
    mock_getenv.side_effect = lambda key: "dummy-key" if key == "OPENROUTER_API_KEY" else None
    client_instance = MagicMock()
    mock_openai.return_value = client_instance
    client_instance.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="结果"))
    ]

    evaluation = {"quick_answer": {}, "scenes": {}, "raw_evaluation": {"model_profile": {}}}
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "sales_summary_zh.txt"
    result = generate_llm_summary(evaluation, prompt_path=prompt_path)
    assert result == "结果"
    assert client_instance.chat.completions.create.called
