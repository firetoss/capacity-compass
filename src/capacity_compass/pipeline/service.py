"""High-level orchestration for capacity evaluation pipeline."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TypedDict

from ..config_loader import load_estimation, load_hardware, load_models, load_scenarios
from ..config_types import EstimationConfig
from ..hardware_registry import HardwareRegistry
from ..models_registry import ModelsRegistry
from ..scenarios_registry import ScenariosRegistry
from .card_sizer import size_cards
from .evaluation_builder import build_evaluation
from .hardware_filter import filter_hardware
from .llm_summary import LLMProviderError, generate_llm_summary
from .normalizer import EvaluationRequest, normalize_request
from .recommender import rank_candidates
from .requirements import estimate_requirements

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
) -> Dict[str, object]:
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

    requirements = {}
    ranked_candidates = {}
    for scene_name, preset in scenarios_registry.presets.items():
        requirement = estimate_requirements(normalized, preset, estimation)
        requirements[scene_name] = requirement

        filtered = filter_hardware(
            registry=hardware_registry,
            vendors=normalized.vendor_scope,
            eval_precision=normalized.eval_precision,
            weights_mem_bytes=requirement.weights_mem_bytes,
        )
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

    evaluation = build_evaluation(
        normalized,
        scenarios_registry.presets,
        requirements,
        ranked_candidates,
    )

    if generate_summary:
        try:
            evaluation["llm_summary"] = generate_llm_summary(
                evaluation,
                disclaimers=disclaimers,
            )
        except LLMProviderError as exc:  # pragma: no cover - depends on env
            logger.warning("llm summary failed: %s", exc)

    return evaluation
