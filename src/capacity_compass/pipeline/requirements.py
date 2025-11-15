"""Stage 2: estimate memory & compute (docs §4.2 + 附录 §9)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from ..config_types import EstimationConfig, ModelConfig, ScenarioPreset
from .normalizer import NormalizedRequest

logger = logging.getLogger(__name__)


@dataclass
class RequirementResult:
    context_len: int
    requested_context_len: int
    context_clamped: bool
    weights_mem_bytes: int
    kv_mem_bytes: int
    total_mem_bytes: int
    required_compute_Tx: float
    concurrency: int
    notes: List[str] = field(default_factory=list)
    throughput_tps: float | None = None
    tokens_per_sec_session: float | None = None
    per_session_Tx: float | None = None
    divisible_heads: int | None = None


def estimate_requirements(
    normalized: NormalizedRequest,
    preset: ScenarioPreset,
    estimation: EstimationConfig,
) -> RequirementResult:
    """Compute权重+KV显存与算力需求（附录公式)."""

    model = normalized.model
    requested_ctx = normalized.requested_context_len or 0
    default_ctx = preset.default_context_len
    target_ctx = max(requested_ctx, default_ctx)
    limit = model.max_context
    context_clamped = False
    if limit and target_ctx > limit:
        target_ctx = limit
        context_clamped = True

    dtype_bytes = estimation.dtype_bytes
    eval_bytes = dtype_bytes.get(normalized.eval_precision, 2)
    kv_dtype = model.recommended_kv_dtype or normalized.eval_precision
    kv_bytes = dtype_bytes.get(kv_dtype, eval_bytes)

    num_layers = _infer_num_layers(model)
    num_heads = _infer_num_heads(model)
    kv_heads = model.num_key_value_heads or num_heads
    head_dim = _infer_head_dim(model, num_heads)
    hidden_size = model.hidden_size or 4096
    concurrency = _target_concurrency(model, preset, estimation)

    # 权重显存：P * B_w（§9：参数量 × 精度字节数）
    weights_mem_bytes = int(
        (model.param_count_b or _fallback_params(hidden_size, num_layers)) * 1e9 * eval_bytes
    )

    kv_mem_bytes = 0
    kv_notes: List[str] = []
    if all([num_layers, kv_heads, head_dim]):
        # K/V cache 估算：C * T * L * H_kv * head_dim * 2 * B_kv（docs §9）
        kv_mem_bytes = int(num_layers * kv_heads * head_dim * target_ctx * concurrency)
        kv_mem_bytes *= kv_bytes * 2
    else:
        kv_notes.append("结构字段缺失，仅按权重估算")
        logger.warning(
            "kv cache skipped due to missing fields (layers=%s kv_heads=%s head_dim=%s) for %s",
            num_layers,
            kv_heads,
            head_dim,
            model.display_name,
        )

    total_mem_bytes = int(
        (weights_mem_bytes + kv_mem_bytes) * (1 + estimation.overhead_ratio_default)
    )

    required_compute = _estimate_compute(model, target_ctx, preset, estimation, concurrency)
    # 业务友好的吞吐（tokens/s），以及单会话计算需求
    tokens_per_sec_session = None
    per_session_Tx = None
    throughput_tps = None
    if preset.avg_output_tokens:
        target_latency_s = max(preset.target_latency_ms / 1000.0, 1e-6)
        alpha = estimation.alpha_default
        tokens_per_sec_session = (preset.avg_output_tokens / target_latency_s) * alpha
        throughput_tps = concurrency * tokens_per_sec_session
        # per-session Tx (考虑多模态乘子)
        eff_params_b = model.param_count_b or 8.0
        if model.is_moe:
            factor = estimation.moe_effective_factor.get(
                model.family, estimation.moe_effective_factor.get("default", 1.0)
            )
            eff_params_b *= factor
        modality = model.modality or "text"
        multiplier = preset.compute_multiplier.get(modality, 1.0)
        per_session_Tx = (2 * eff_params_b * 1e9 * tokens_per_sec_session) / 1e12
        per_session_Tx *= multiplier

    notes = normalized.notes + kv_notes
    if context_clamped:
        notes.append("上下文按模型上限估算")
        logger.info(
            "context for %s limited to %s tokens per preset %s",
            model.display_name,
            target_ctx,
            preset.label,
        )

    logger.debug(
        "requirements model=%s preset=%s ctx=%s weights=%.2f GB kv=%.2f GB total=%.2f GB compute=%.2f Tx",
        model.display_name,
        preset.label,
        target_ctx,
        weights_mem_bytes / 1e9,
        kv_mem_bytes / 1e9,
        total_mem_bytes / 1e9,
        required_compute,
    )

    return RequirementResult(
        context_len=target_ctx,
        requested_context_len=target_ctx,
        context_clamped=context_clamped,
        weights_mem_bytes=weights_mem_bytes,
        kv_mem_bytes=kv_mem_bytes,
        total_mem_bytes=total_mem_bytes,
        required_compute_Tx=required_compute,
        concurrency=concurrency,
        notes=notes,
        throughput_tps=throughput_tps,
        tokens_per_sec_session=tokens_per_sec_session,
        per_session_Tx=per_session_Tx,
        divisible_heads=kv_heads if kv_heads else (num_heads or None),
    )


def _fallback_params(hidden_size: int, num_layers: int) -> float:
    """当 param_count 缺失时，用 hidden_size * num_layers 粗估。"""
    return hidden_size * num_layers / 1e9


def _infer_num_layers(model: ModelConfig) -> int:
    """根据参数量区间推断层数（经验值，便于匿名模型估算）。"""
    if model.num_hidden_layers:
        return model.num_hidden_layers
    params = model.param_count_b or 8.0
    if params < 10:
        return 32
    if params < 50:
        return 40
    return 60


def _infer_num_heads(model: ModelConfig) -> int:
    """同上，根据参数量取常见头数。"""
    if model.num_attention_heads:
        return model.num_attention_heads
    params = model.param_count_b or 8.0
    if params < 10:
        return 32
    if params < 50:
        return 40
    return 64


def _infer_head_dim(model: ModelConfig, heads: int) -> int:
    """优先使用配置，否则用 hidden_size/heads 取整。"""
    if model.head_dim:
        return model.head_dim
    if model.hidden_size:
        return max(model.hidden_size // heads, 64)
    return 128


def _target_concurrency(
    model: ModelConfig, preset: ScenarioPreset, estimation: EstimationConfig
) -> int:
    """按照 param_count_b 落在 small/medium/large 决定场景并发。"""
    bucket = _scale_bucket(model.param_count_b or 8.0, estimation)
    return preset.target_concurrency_per_gpu.get(
        bucket, preset.target_concurrency_per_gpu.get("medium", 1)
    )


def _scale_bucket(params_b: float, estimation: EstimationConfig) -> str:
    small_max = estimation.scale_thresholds_b.get("small_max", 10)
    medium_max = estimation.scale_thresholds_b.get("medium_max", 50)
    if params_b < small_max:
        return "small_model"
    if params_b <= medium_max:
        return "medium_model"
    return "large_model"


def _estimate_compute(
    model: ModelConfig,
    context_len: int,
    preset: ScenarioPreset,
    estimation: EstimationConfig,
    concurrency: int,
) -> float:
    """Compute 需求（以生成阶段为主）：
    compute_Tx ≈ 2 * 有效参数 * (并发 × 单会话 tokens/s) / 1e3。

    - 单会话 tokens/s 取自场景的 avg_output_tokens 与目标延迟，再乘以 alpha（保守系数）。
    - 如 avg_output_tokens 缺失，退化到 prefill 上界：context_len / 目标延迟，并记录为保守上界。
    """
    target_latency_s = max(preset.target_latency_ms / 1000.0, 1e-6)
    alpha = estimation.alpha_default
    # Prefer decode-based tokens/s; fallback to prefill upper bound if missing
    if preset.avg_output_tokens:
        tokens_per_sec_session = (preset.avg_output_tokens / target_latency_s) * alpha
    else:
        tokens_per_sec_session = (context_len / target_latency_s) * alpha
        logger.warning(
            "avg_output_tokens missing for preset=%s; using prefill-bound tokens/s upper bound",
            preset.label,
        )
    tokens_per_sec = concurrency * tokens_per_sec_session

    effective_params = model.param_count_b or 8.0
    if model.is_moe:
        factor = estimation.moe_effective_factor.get(
            model.family, estimation.moe_effective_factor.get("default", 1.0)
        )
        effective_params *= factor

    compute_Tx = (2 * effective_params * 1e9 * tokens_per_sec) / 1e12

    modality = model.modality or "text"
    multiplier = preset.compute_multiplier.get(modality, 1.0)
    return compute_Tx * multiplier
