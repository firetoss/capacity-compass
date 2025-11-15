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
from .normalizer import EvaluationRequest, NormalizedRequest, normalize_request
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
    model_suggestions: Dict[str, Dict[str, Any]] = {}
    runtime = get_runtime_settings()
    for scene_name, preset in scenarios_registry.presets.items():
        # Per-scene model (auto-switch if requested context exceeds model limit)
        scene_normalized = normalized
        if runtime.autoswitch_for_scenes:
            target_ctx = preset.default_context_len
            if (
                scene_normalized.model.max_context
                and target_ctx
                and target_ctx > scene_normalized.model.max_context
            ):
                best = context["models"].select_best_for_context(
                    scene_normalized.model.family, target_ctx
                )
                if best and best is not scene_normalized.model:
                    # Clone normalized request with switched model
                    scene_normalized = NormalizedRequest(
                        model=best,
                        eval_precision=normalized.eval_precision,
                        requested_context_len=normalized.requested_context_len,
                        final_context_len=normalized.final_context_len,
                        vendor_scope=normalized.vendor_scope,
                        switch_notice=(
                            f"场景 {scene_name} 默认上下文 {target_ctx} 超过 {normalized.model.display_name} 上限，"
                            f"已切换为 {best.display_name}"
                        ),
                        context_clamped=normalized.context_clamped,
                        notes=list(normalized.notes),
                        is_anonymous_model=normalized.is_anonymous_model,
                        original_model_name=normalized.original_model_name,
                    )

        requirement = estimate_requirements(scene_normalized, preset, estimation)
        requirements[scene_name] = requirement

        # Per-scene model suggestion based on context fit (same family)
        best = _suggest_model_for_scene(
            context["models"],
            normalized.model.family,
            preset.default_context_len,
            estimation.model_suggestion_rules.get(scene_name, {}),
            current=normalized.model,
        )
        if best and best.display_name != normalized.model.display_name:
            model_suggestions[scene_name] = {
                "display_name": best.display_name,
                "model_name": best.model_name,
                "reason": f"更贴合该场景的上下文需求（目标 {preset.default_context_len}）",
            }

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
                    eval_precision=scene_normalized.eval_precision,
                    total_mem_bytes=requirement.total_mem_bytes,
                    required_compute_Tx=requirement.required_compute_Tx,
                    per_session_Tx=requirement.per_session_Tx,
                    tokens_per_sec_session=requirement.tokens_per_sec_session,
                    desired_concurrency=requirement.concurrency,
                    allowed_shard_sizes=estimation.allowed_shard_sizes,
                    tp_require_divisible=estimation.tp_require_divisible,
                    divisible_heads=requirement.divisible_heads,
                    tp_preferred_orders=estimation.tp_preferred_orders,
                )
                for g in filtered
            ]
            ranked = rank_candidates(evaluations)
            ranked_candidates[scene_name] = ranked
            if ranked:
                top_cand = ranked[0]
                logger.info(
                    "scene=%s top_gpu=%s cards=%s vendors=%s",
                    scene_name,
                    top_cand.gpu_name,
                    top_cand.cards_needed,
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
        model_suggestions=model_suggestions or None,
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


def _suggest_model_for_scene(
    registry: ModelsRegistry,
    family: str,
    target_ctx: int,
    rule: Dict[str, Any],
    *,
    current: Any,
):
    """Pick a suggested model per scene using param range rule then context fit.

    This is a soft suggestion to educate users; it does not change evaluation.
    """
    # Filter by family
    candidates = [
        m
        for m in registry._family_index.get(family, [])
        if m.max_context and m.max_context >= target_ctx
    ]
    if not candidates:
        return registry.select_best_for_context(family, target_ctx)
    lo = rule.get("min_params_b")
    hi = rule.get("max_params_b")
    if lo is not None and hi is not None:
        filtered = [
            m for m in candidates if (m.param_count_b or 0) >= lo and (m.param_count_b or 0) <= hi
        ]
        if filtered:
            # Choose closest to midpoint
            mid = (lo + hi) / 2.0
            filtered.sort(key=lambda m: abs((m.param_count_b or 0) - mid))
            return filtered[0]
    # Fallback to context-based selection
    best = registry.select_best_for_context(family, target_ctx)
    return best
