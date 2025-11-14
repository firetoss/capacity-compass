"""High-level orchestration for capacity evaluation pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict

from ..config_loader import load_estimation, load_hardware, load_models, load_scenarios
from ..config_types import EstimationConfig
from ..hardware_registry import HardwareRegistry
from ..models_registry import ModelsRegistry
from ..scenarios_registry import ScenariosRegistry
from ..settings import get_runtime_settings
from .card_sizer import size_cards
from .evaluation_builder import build_evaluation
from .hardware_filter import filter_hardware
from .llm_summary import LLMProviderError, generate_llm_summary
from .normalizer import EvaluationRequest, normalize_request
from .recommender import RankedCandidate, rank_candidates
from .requirements import RequirementResult, estimate_requirements

logger = logging.getLogger(__name__)


class ServiceContext(TypedDict):
    models: ModelsRegistry
    hardware: HardwareRegistry
    scenarios: ScenariosRegistry
    estimation: EstimationConfig


def build_service_context() -> ServiceContext:
    """Load and cache configuration for repeated evaluations."""

    models = ModelsRegistry(load_models())
    hardware = HardwareRegistry(load_hardware())
    scenarios = ScenariosRegistry(load_scenarios())
    estimation = load_estimation()
    return ServiceContext(
        models=models,
        hardware=hardware,
        scenarios=scenarios,
        estimation=estimation,
    )


def evaluate_capacity(
    request: EvaluationRequest,
    context: ServiceContext,
    *,
    generate_summary: bool = False,
    disclaimers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    models: ModelsRegistry = context["models"]
    hardware_registry: HardwareRegistry = context["hardware"]
    scenarios_registry: ScenariosRegistry = context["scenarios"]
    estimation: EstimationConfig = context["estimation"]

    normalized = normalize_request(request, models)
    logger.info(
        "normalized request model=%s precision=%s context=%s",
        normalized.model.display_name,
        normalized.eval_precision,
        normalized.final_context_len,
    )

    requirements: Dict[str, RequirementResult] = {}
    ranked_candidates: Dict[str, List[RankedCandidate]] = {}
    for scene_name, preset in scenarios_registry.presets.items():
        requirement = estimate_requirements(normalized, preset, estimation)
        requirements[scene_name] = requirement

        filtered = filter_hardware(
            registry=hardware_registry,
            vendors=normalized.vendor_scope,
            eval_precision=normalized.eval_precision,
            weights_mem_bytes=requirement.weights_mem_bytes,
        )
        if filtered:
            evaluations = [
                size_cards(
                    gpu=g,
                    eval_precision=normalized.eval_precision,
                    total_mem_bytes=requirement.total_mem_bytes,
                    required_compute_Tx=requirement.required_compute_Tx,
                )
                for g in filtered
            ]
            ranked = rank_candidates(evaluations)
            ranked_candidates[scene_name] = ranked
            if ranked:
                best = ranked[0]
                logger.info(
                    "scene=%s top_gpu=%s cards=%s vendors=%s",
                    scene_name,
                    best.gpu_name,
                    best.cards_needed,
                    normalized.vendor_scope or "ALL",
                )
            else:
                logger.warning("scene=%s has no viable hardware candidates", scene_name)
        else:
            ranked_candidates[scene_name] = []
            logger.warning(
                "scene=%s filtered out all GPUs for vendors=%s", scene_name, normalized.vendor_scope
            )

    evaluation = build_evaluation(
        normalized,
        scenarios_registry.presets,
        requirements,
        ranked_candidates,
    )

    merged_disclaimers = list(evaluation.get("disclaimers", []))
    if disclaimers:
        for item in disclaimers:
            if item not in merged_disclaimers:
                merged_disclaimers.append(item)
        evaluation["disclaimers"] = merged_disclaimers

    if generate_summary:
        runtime = get_runtime_settings()
        try:
            evaluation["llm_summary"] = generate_llm_summary(
                evaluation,
                disclaimers=merged_disclaimers,
                prompt_path=runtime.llm_prompt_path,
                base_url=runtime.llm_base_url,
                timeout_seconds=runtime.llm_timeout_seconds,
                http_proxy=runtime.http_proxy,
            )
        except LLMProviderError as exc:  # pragma: no cover - depends on env
            logger.warning("llm summary failed: %s", exc)

    return evaluation
