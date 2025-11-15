"""Typed schemas for configuration files (hardware, models, scenarios, estimation)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Pricing(BaseModel):
    currency: Optional[str] = None
    price: Optional[float] = None
    source: Optional[str] = None
    note: Optional[str] = Field(None, alias="notes")


class PerformanceSpec(BaseModel):
    fp16_tflops: Optional[float] = None
    bf16_tflops: Optional[float] = None
    fp8_tflops: Optional[float] = None
    int8_tops: Optional[float] = None


class PrecisionSupport(BaseModel):
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None
    fp8: Optional[bool] = None
    int8: Optional[bool] = None


class GPUConfig(BaseModel):
    id: str
    name: str
    vendor: str
    family: Optional[str] = None
    category: Optional[str] = None
    memory_gb: float
    memory_type: Optional[str] = None
    memory_bw_gbps: Optional[float] = None
    precision_support: PrecisionSupport
    perf: Optional[PerformanceSpec] = None
    power_w: Optional[float] = None
    pricing: Optional[Pricing] = None
    deploy_support: Optional[str] = None
    deploy_note: Optional[str] = None
    notes: Optional[str] = None


class ModelQuantization(BaseModel):
    supports_fp8: Optional[bool] = None
    supports_bf16: Optional[bool] = None
    supports_fp16: Optional[bool] = None
    supports_int8: Optional[bool] = None
    quantization_support: List[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    """Model metadata loaded from YAML (aligned with design doc §2.2)."""

    family: str
    model_name: str
    display_name: str
    aliases: List[str] = Field(default_factory=list)
    modality: Optional[str] = None
    param_count_b: Optional[float] = None
    is_moe: bool = False
    vocab_size: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    tie_word_embeddings: Optional[bool] = None
    num_experts: Optional[int] = None
    top_k: Optional[int] = None
    expert_intermediate_size: Optional[int] = None
    torch_dtype: Optional[str] = None
    use_cache: Optional[bool] = None
    recommended_kv_dtype: Optional[str] = None
    notes: Optional[str] = None

    # Embedding quantization fields
    supports_fp8: Optional[bool] = None
    supports_bf16: Optional[bool] = None
    supports_fp16: Optional[bool] = None
    supports_int8: Optional[bool] = None
    quantization_support: List[str] = Field(default_factory=list)

    @property
    def max_context(self) -> Optional[int]:
        return self.max_position_embeddings


class ScenarioPreset(BaseModel):
    label: str
    description: Optional[str] = None
    default_context_len: int
    target_latency_ms: int
    target_concurrency_per_gpu: Dict[str, int]
    compute_multiplier: Dict[str, float]
    avg_output_tokens: Optional[int] = None


class EstimationConfig(BaseModel):
    dtype_bytes: Dict[str, int]
    overhead_ratio_default: float = 0.15
    alpha_default: float = 0.25
    moe_effective_factor: Dict[str, float] = Field(default_factory=dict)
    scale_thresholds_b: Dict[str, float] = Field(default_factory=dict)
    kv_dtype_fallback_order: List[str] = Field(default_factory=list)
    avg_output_tokens_default: int | None = None
    allowed_shard_sizes: List[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    # Tensor parallel soft constraints (可配置关闭)
    tp_require_divisible: bool = True
    tp_check_kv_heads_first: bool = True
    tp_preferred_orders: List[int] = Field(default_factory=lambda: [8, 4, 2, 1])
    # Optional per-scene model suggestion rules (non-binding)
    # example:
    #   model_suggestion_rules:
    #     chat: { min_params_b: 7, max_params_b: 9 }
    #     rag: { min_params_b: 14, max_params_b: 32 }
    #     writer: { min_params_b: 14, max_params_b: 32 }
    model_suggestion_rules: Dict[str, Dict[str, float]] = Field(default_factory=dict)
