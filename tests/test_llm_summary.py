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
@patch("capacity_compass.pipeline.llm_summary.httpx.Client")
@patch("capacity_compass.pipeline.llm_summary.httpx.HTTPTransport")
@patch("capacity_compass.pipeline.llm_summary.os.getenv")
def test_generate_llm_summary_calls_openai(
    mock_getenv, mock_transport, mock_httpx_client, mock_openai
):
    def _fake_getenv(key):
        if key == "OPENROUTER_API_KEY":
            return "dummy-key"
        return None

    mock_getenv.side_effect = _fake_getenv
    transport_instance = MagicMock()
    mock_transport.return_value = transport_instance
    http_client = MagicMock()
    mock_httpx_client.return_value = http_client
    client_instance = MagicMock()
    mock_openai.return_value = client_instance
    client_instance.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="结果"))
    ]

    evaluation = {"quick_answer": {}, "scenes": {}, "raw_evaluation": {"model_profile": {}}}
    prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "sales_summary_zh.txt"
    result = generate_llm_summary(
        evaluation,
        prompt_path=prompt_path,
        base_url="https://api.example.com",
        timeout_seconds=12.0,
        http_proxy="http://proxy.local:8080",
    )

    assert result == "结果"
    mock_httpx_client.assert_called_once()
    mock_transport.assert_called_once_with(proxy="http://proxy.local:8080")
    args, kwargs = mock_httpx_client.call_args
    assert kwargs["transport"] is transport_instance
    assert kwargs["timeout"].connect == 12.0
    mock_openai.assert_called_once_with(
        base_url="https://api.example.com",
        api_key="dummy-key",
        http_client=http_client,
    )
    client_instance.chat.completions.create.assert_called_once()
    http_client.close.assert_called_once()
