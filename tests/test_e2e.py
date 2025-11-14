from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from capacity_compass.pipeline.normalizer import EvaluationRequest
from capacity_compass.pipeline.service import build_service_context, evaluate_capacity

CASES_PATH = Path(__file__).resolve().parent / "data" / "e2e_test_cases.yaml"


def _load_cases() -> List[Dict[str, Any]]:
    with CASES_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or []
    return data


@pytest.fixture(scope="session")
def service_context():
    return build_service_context()


def _build_request(payload: Dict[str, Any]) -> EvaluationRequest:
    return EvaluationRequest(
        model=payload["model_name"],
        param_count_b=payload.get("param_count_b"),
        max_context_len=payload.get("max_context_len"),
        precision=payload.get("precision"),
        vendor_scope=payload.get("vendor_scope"),
    )


def _assert_scenario_expectations(
    result: Dict[str, Any],
    scene_name: str,
    expectation: Dict[str, Any],
) -> None:
    scenes = result["scenes"]
    raw_scenes = result["raw_evaluation"]["scenarios"]
    assert scene_name in scenes, f"scene {scene_name} missing"
    summary = scenes[scene_name]["sales_summary"]
    raw_scene = raw_scenes[scene_name]

    if expectation.get("has_recommendation"):
        assert summary["primary"], f"scene {scene_name} missing primary recommendation"
        assert raw_scene["candidates"], f"scene {scene_name} missing candidate list"
    if "primary_contains_any_gpu_id" in expectation and raw_scene["candidates"]:
        top_gpu_id = raw_scene["candidates"][0]["gpu_id"]
        assert (
            top_gpu_id in expectation["primary_contains_any_gpu_id"]
        ), f"{scene_name} top gpu {top_gpu_id} unexpected"
    if "max_cards_needed" in expectation and raw_scene["candidates"]:
        top_cards = raw_scene["candidates"][0]["cards_needed"]
        assert top_cards <= expectation["max_cards_needed"]
    if expectation.get("modality"):
        model_profile = result["raw_evaluation"]["model_profile"]
        assert model_profile.get("modality") == expectation["modality"]


@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["id"])
def test_end_to_end_cases(service_context, case):
    request = _build_request(case["request"])
    result = evaluate_capacity(request, service_context)

    expectations = case.get("expectations", {})
    for scene_name, scene_expectation in expectations.get("scenarios", {}).items():
        _assert_scenario_expectations(result, scene_name, scene_expectation)

    aggregated_notes: List[str] = []
    for scene in result["raw_evaluation"]["scenarios"].values():
        aggregated_notes.extend(scene["requirements"]["notes"])
    for expected_note in expectations.get("notes", []):
        assert any(expected_note in note for note in aggregated_notes)

    disclaimers = result.get("disclaimers", [])
    assert disclaimers, "disclaimers should not be empty"
