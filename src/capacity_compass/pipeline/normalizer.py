"""Stage 1: request normalization (model match, precision, context clamp)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel, Field

from ..config_types import ModelConfig
from ..models_registry import ModelsRegistry

logger = logging.getLogger(__name__)


class EvaluationRequest(BaseModel):
    """User-facing request body (API schema layer)."""

    model: str = Field(..., description="User provided model name or alias")
    param_count_b: Optional[float] = None
    max_context_len: Optional[int] = None
    precision: Optional[str] = None
    vendor_scope: Optional[List[str]] = None


@dataclass
class NormalizedRequest:
    """Internal normalized request used by downstream pipeline stages."""

    model: ModelConfig
    eval_precision: str
    requested_context_len: Optional[int]
    final_context_len: Optional[int]
    vendor_scope: Optional[List[str]]
    switch_notice: Optional[str] = None
    context_clamped: bool = False
    notes: List[str] = field(default_factory=list)
    is_anonymous_model: bool = False
    original_model_name: str = ""


def normalize_request(
    request: EvaluationRequest,
    registry: ModelsRegistry,
) -> NormalizedRequest:
    """Normalize request per设计文档 §4.1：模型匹配、Instruct 切换、上下文截断。"""

    logger.debug(
        "normalizing request model=%s precision=%s context=%s vendors=%s",
        request.model,
        request.precision,
        request.max_context_len,
        request.vendor_scope,
    )

    model = _resolve_model(request, registry)
    is_anonymous = model.model_name.startswith("__anonymous__")

    matched_model = model
    switch_notice: Optional[str] = None
    target_ctx = request.max_context_len

    if target_ctx and matched_model.max_context and target_ctx > matched_model.max_context:
        best = registry.select_best_for_context(matched_model.family, target_ctx)
        if best and best is not matched_model:
            # 命中基座但上下文超长，自动选“差距最小”的 Instruct 版本（§8.2）
            switch_notice = (
                f"输入上下文 {target_ctx} 超过 {matched_model.display_name} 上限，"
                f"已切换为 {best.display_name}"
            )
            matched_model = best
            logger.info(
                "context %s exceeds %s limit=%s, switched to %s (max=%s)",
                target_ctx,
                model.display_name,
                model.max_context,
                best.display_name,
                best.max_context,
            )

    eval_precision = _normalize_precision(request.precision, matched_model)
    vendor_scope = _normalize_vendor_scope(request.vendor_scope)

    requested_ctx = request.max_context_len
    final_ctx = _clamp_context(requested_ctx, matched_model)
    context_clamped = requested_ctx is not None and final_ctx != requested_ctx

    notes: List[str] = []
    if context_clamped:
        notes.append(
            f"请求上下文 {requested_ctx} 超过 {matched_model.display_name} 上限，已截断为 {final_ctx}"
        )
        logger.warning(
            "requested context %s exceeds %s max_ctx=%s; clamped to %s",
            requested_ctx,
            matched_model.display_name,
            matched_model.max_context,
            final_ctx,
        )

    if is_anonymous:
        notes.append("模型未在配置表中，按 param_count_b 粗略估算")

    logger.debug(
        "normalized model=%s precision=%s context=%s vendor_scope=%s anonymous=%s",
        matched_model.display_name,
        eval_precision,
        final_ctx,
        vendor_scope,
        is_anonymous,
    )

    return NormalizedRequest(
        model=matched_model,
        eval_precision=eval_precision,
        requested_context_len=request.max_context_len,
        final_context_len=final_ctx,
        vendor_scope=vendor_scope,
        switch_notice=switch_notice,
        context_clamped=context_clamped,
        notes=notes,
        is_anonymous_model=is_anonymous,
        original_model_name=request.model,
    )


def _resolve_model(request: EvaluationRequest, registry: ModelsRegistry) -> ModelConfig:
    """Resolve model name/alias; fallback to anonymous if param_count_b provided."""
    target = registry.get(request.model)
    if target:
        logger.debug("model %s resolved directly to %s", request.model, target.model_name)
        return target

    matches = registry.match(request.model)
    if len(matches) == 1:
        logger.debug("model %s matched alias %s", request.model, matches[0].model_name)
        return matches[0]
    if len(matches) > 1:
        logger.error("model %s has multiple matches: %s", request.model, matches)
        raise ValueError(f"模型 `{request.model}` 存在多个匹配，请输入完整名称")

    if request.param_count_b is not None:
        return _build_anonymous_model(request)
    raise ValueError(f"未知模型 `{request.model}` 且缺少 param_count_b 供估算")


def _build_anonymous_model(request: EvaluationRequest) -> ModelConfig:
    """Construct a lightweight placeholder when用户自报参数量（设计文档 §4.1）。"""
    name = f"__anonymous__::{request.model}"
    logger.warning(
        "model `%s` not found; falling back to anonymous profile with params=%sB",
        request.model,
        request.param_count_b,
    )
    return ModelConfig.model_validate(
        {
            "family": "Custom",
            "model_name": name,
            "display_name": request.model,
            "modality": "text",
            "param_count_b": request.param_count_b,
            "is_moe": False,
            "aliases": [],
            "notes": "粗略估算：未命中模型表，使用用户提供的参数量",
        }
    )


def _normalize_precision(requested: Optional[str], model: ModelConfig) -> str:
    """Use用户输入否则模型默认精度；缺省回退到 bf16。"""
    if requested:
        return requested.lower()
    if model.torch_dtype:
        return model.torch_dtype.lower()
    return "bf16"


def _normalize_vendor_scope(vendors: Optional[List[str]]) -> Optional[List[str]]:
    """Remove重复并保持用户给定顺序。"""
    if not vendors:
        return None
    seen: List[str] = []
    lowered: set[str] = set()
    for vendor in vendors:
        if not vendor:
            continue
        key = vendor.lower()
        if key in lowered:
            continue
        lowered.add(key)
        seen.append(vendor)
    return seen


def _clamp_context(context: Optional[int], model: ModelConfig) -> Optional[int]:
    """Clamp上下文到模型上限（§4.1）。"""
    limit = model.max_context
    if context is None or limit is None:
        return context
    return min(context, limit)
