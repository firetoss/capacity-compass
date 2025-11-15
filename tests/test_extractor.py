from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from capacity_compass.pipeline.extractor import ExtractResult, extract_fields


class DummyChoice:
    def __init__(self, content: str) -> None:
        self.message = type("obj", (), {"content": content})


class DummyResp:
    def __init__(self, content: str) -> None:
        self.choices = [DummyChoice(content)]


def _mock_openai_response(payload: Dict[str, Any]) -> Any:
    content = (
        "{"
        + ", ".join(
            [
                f'"model":"{payload.get("model")}"' if payload.get("model") else '"model":null',
                (
                    f'"precision":"{payload.get("precision")}"'
                    if payload.get("precision")
                    else '"precision":null'
                ),
                f'"context_len":{payload.get("context_len") if payload.get("context_len") is not None else "null"}',
            ]
        )
        + "}"
    )
    return DummyResp(content)


@patch("capacity_compass.pipeline.extractor.OpenAI")
@patch("capacity_compass.pipeline.extractor.httpx.Client")
def test_extract_fields_parses_json(mock_client, mock_openai, monkeypatch) -> None:
    # Arrange
    mock_client.return_value = MagicMock()
    mock_openai.return_value = MagicMock()
    mock_openai.return_value.chat.completions.create.return_value = _mock_openai_response(
        {"model": "Qwen3-8B", "precision": "bf16", "context_len": 16000}
    )

    def _getenv(key: str, default=None):
        if key == "OPENROUTER_API_KEY":
            return "sk-test"
        return default

    monkeypatch.setattr("capacity_compass.pipeline.extractor.os.getenv", _getenv)

    # Act
    result = extract_fields(
        "用Qwen3-8B推理，16K上下文，BF16，单卡可以吗？", prompt_path=Path(__file__).resolve()
    )
    # We bypass reading a real prompt by pointing to this file (content unused in mocked call)

    # Assert
    assert isinstance(result, ExtractResult)
    assert result.model == "Qwen3-8B"
    assert result.precision == "bf16"
    assert result.context_len == 16000
