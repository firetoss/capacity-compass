from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from capacity_compass.config_types import EstimationConfig, ModelConfig, ScenarioPreset
from capacity_compass.pipeline.normalizer import NormalizedRequest
from capacity_compass.pipeline.requirements import estimate_requirements

CASES_PATH = Path(__file__).resolve().parent / "data" / "requirements_cases.yaml"


def _load_cases() -> List[Dict[str, Any]]:
    with CASES_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data.get("cases", [])


def _build_normalized_request(case: Dict[str, Any], model: ModelConfig) -> NormalizedRequest:
    payload = case["normalized_request"]
    return NormalizedRequest(
        model=model,
        eval_precision=payload["eval_precision"],
        requested_context_len=payload["requested_context_len"],
        final_context_len=payload["final_context_len"],
        vendor_scope=None,
        switch_notice=None,
        context_clamped=False,
        notes=list(payload.get("notes", [])),
        is_anonymous_model=payload.get("is_anonymous_model", False),
        original_model_name=(
            model.display_name if hasattr(model, "display_name") else model.model_name
        ),
    )


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["id"])
def test_requirement_cases(case: Dict[str, Any]):
    model_payload = dict(case["model"])
    model_payload.setdefault("display_name", model_payload.get("model_name", "unknown"))
    model = ModelConfig.model_validate(model_payload)
    normalized = _build_normalized_request(case, model)
    preset = ScenarioPreset.model_validate(case["scenario_preset"])
    estimation = EstimationConfig.model_validate(case["estimation_config"])

    result = estimate_requirements(normalized, preset, estimation)
    expected = case["expectations"]

    assert result.context_len == expected["context_len"]
    assert result.context_clamped is expected["context_clamped"]
    assert result.weights_mem_bytes == pytest.approx(expected["weights_mem_bytes"])
    assert result.kv_mem_bytes == pytest.approx(expected["kv_mem_bytes"])
    assert result.total_mem_bytes == pytest.approx(expected["total_mem_bytes"])
    assert result.required_compute_Tx == pytest.approx(expected["required_compute_Tx"])

    for text in expected.get("notes_contains", []):
        if not text:
            continue
        assert any(
            text in note for note in result.notes
        ), f"{text} not found in notes {result.notes}"
