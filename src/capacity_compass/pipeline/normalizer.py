"""Stage 1: request normalization (model match, precision, context clamp)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel, Field

from ..config_types import ModelConfig
from ..models_registry import ModelsRegistry


class EvaluationRequest(BaseModel):
    model: str = Field(..., description="User provided model name or alias")
    param_count_b: Optional[float] = None
    max_context_len: Optional[int] = None
    precision: Optional[str] = None
    vendor_scope: Optional[List[str]] = None


@dataclass
class NormalizedRequest:
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
    """Normalize user request: match model, pick precision, clamp context."""

    model = _resolve_model(request, registry)
    is_anonymous = model.model_name.startswith("__anonymous__")

    matched_model = model
    switch_notice: Optional[str] = None
    target_ctx = request.max_context_len

    if target_ctx and matched_model.max_context and target_ctx > matched_model.max_context:
        best = registry.select_best_for_context(matched_model.family, target_ctx)
        if best and best is not matched_model:
            switch_notice = (
                f"输入上下文 {target_ctx} 超过 {matched_model.display_name} 上限，"
                f"已切换为 {best.display_name}"
            )
            matched_model = best

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
    target = registry.get(request.model)
    if target:
        return target

    matches = registry.match(request.model)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"模型 `{request.model}` 存在多个匹配，请输入完整名称")

    if request.param_count_b is not None:
        return _build_anonymous_model(request)
    raise ValueError(f"未知模型 `{request.model}` 且缺少 param_count_b 供估算")


def _build_anonymous_model(request: EvaluationRequest) -> ModelConfig:
    name = f"__anonymous__::{request.model}"
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
    if requested:
        return requested.lower()
    if model.torch_dtype:
        return model.torch_dtype.lower()
    return "bf16"


def _normalize_vendor_scope(vendors: Optional[List[str]]) -> Optional[List[str]]:
    if not vendors:
        return None
    seen = []
    for vendor in vendors:
        if vendor and vendor.lower() not in {v.lower() for v in seen}:
            seen.append(vendor)
    return seen


def _clamp_context(context: Optional[int], model: ModelConfig) -> Optional[int]:
    limit = model.max_context
    if context is None or limit is None:
        return context
    return min(context, limit)
