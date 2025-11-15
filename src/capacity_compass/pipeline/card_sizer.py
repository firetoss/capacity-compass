"""Stage 4: determine card counts per GPU candidate（docs §4.4）。"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

from ..config_types import GPUConfig

PRECISION_FIELD = {
    "fp16": "fp16_tflops",
    "bf16": "bf16_tflops",
    "fp8": "fp8_tflops",
    "int8": "int8_tops",
}


def _resolve_perf_value(gpu: GPUConfig, precision: str) -> tuple[Optional[float], Optional[str]]:
    """Return a usable perf number for the requested precision with safe fallbacks.

    Fallbacks (logged via note):
      - bf16 -> fp16 when bf16 missing
      - fp8  -> fp16 when fp8 missing (conservative)
    We do NOT invent numbers; we reuse provided fp16 as conservative proxy.
    """
    if not gpu.perf:
        return None, None
    field = PRECISION_FIELD.get(precision)
    note: Optional[str] = None
    value: Optional[float] = getattr(gpu.perf, field) if field else None
    if value is not None:
        return value, None
    if precision == "bf16":
        # conservative fallback to fp16
        value = getattr(gpu.perf, "fp16_tflops")
        if value is not None:
            note = "bf16 峰值未提供，按 fp16 估算"
    elif precision == "fp8":
        # conservative fallback to fp16 when fp8 missing
        value = getattr(gpu.perf, "fp16_tflops")
        if value is not None:
            note = "fp8 峰值未提供，按 fp16 估算"
    elif precision == "int8":
        # no safe generic fallback
        value = getattr(gpu.perf, "int8_tops")
    return value, note


logger = logging.getLogger(__name__)


@dataclass
class HardwareEvaluation:
    """Result per GPU：显存/算力卡数、冗余与提示。"""

    gpu: GPUConfig
    cards_mem: int
    cards_compute: Optional[int]
    cards_needed: int
    headroom: float
    total_mem_available: int
    notes: List[str]
    concurrency_per_gpu: int | None = None
    throughput_tokens_per_sec: float | None = None
    shards: int | None = None
    replicas: int | None = None


def size_cards(
    gpu: GPUConfig,
    eval_precision: str,
    total_mem_bytes: int,
    required_compute_Tx: float,
    *,
    per_session_Tx: float | None = None,
    tokens_per_sec_session: float | None = None,
    desired_concurrency: int | None = None,
    allowed_shard_sizes: list[int] | None = None,
    tp_require_divisible: bool | None = None,
    divisible_heads: int | None = None,
    tp_preferred_orders: list[int] | None = None,
) -> HardwareEvaluation:
    """Take显存与算力上界，向上取整卡数，见设计 §4.4。"""

    notes: List[str] = []
    mem_per_card = int(gpu.memory_gb * 1e9)
    cards_mem = max(1, math.ceil(total_mem_bytes / mem_per_card))

    cards_compute: Optional[int] = None
    perf_value, perf_note = _resolve_perf_value(gpu, eval_precision)
    if perf_value:
        cards_compute = max(1, math.ceil(required_compute_Tx / perf_value))
    if cards_compute is None:
        notes.append("算力数据缺失，仅按显存估算")
        logger.warning(
            "compute spec missing for gpu=%s precision=%s; relying on memory-bound sizing",
            gpu.name,
            eval_precision,
        )
    elif perf_note:
        notes.append(perf_note)

    cards_needed = max(cards_mem, cards_compute or 0)
    total_mem_available = cards_needed * mem_per_card
    headroom = 0.0
    if total_mem_available:
        headroom = (total_mem_available - total_mem_bytes) / total_mem_available

    logger.debug(
        "size_cards gpu=%s precision=%s cards_mem=%s cards_compute=%s cards_needed=%s headroom=%.2f",
        gpu.name,
        eval_precision,
        cards_mem,
        cards_compute,
        cards_needed,
        headroom,
    )

    # Compute-limited per-GPU concurrency/throughput
    concurrency_cap = None
    concurrency_used = None
    throughput = None
    perf_value = perf_value
    if per_session_Tx and perf_value:
        if per_session_Tx > 0:
            concurrency_cap = max(1, int(perf_value // per_session_Tx))
    if desired_concurrency:
        concurrency_used = (
            min(desired_concurrency, concurrency_cap) if concurrency_cap else desired_concurrency
        )
    if concurrency_used and tokens_per_sec_session:
        throughput = concurrency_used * tokens_per_sec_session

    # Round to feasible shard sizes for model parallel
    shards = None
    replicas = None
    if allowed_shard_sizes:
        candidates = sorted([s for s in allowed_shard_sizes if s <= cards_needed], reverse=True)
        if not candidates:
            candidates = [min(allowed_shard_sizes)]
        # Apply TP divisibility soft constraint
        picked = None
        ordered = candidates
        if tp_preferred_orders:
            ordered = [s for s in tp_preferred_orders if s in candidates]
            # append rest if not listed
            ordered += [s for s in candidates if s not in ordered]
        for s in ordered:
            if tp_require_divisible and divisible_heads and s > 1:
                if divisible_heads % s != 0:
                    continue
            picked = s
            break
        if picked is None:
            picked = ordered[0]
        shards = picked
        replicas = max(1, math.ceil(cards_needed / shards))

    return HardwareEvaluation(
        gpu=gpu,
        cards_mem=cards_mem,
        cards_compute=cards_compute,
        cards_needed=cards_needed,
        headroom=headroom,
        total_mem_available=total_mem_available,
        notes=notes,
        concurrency_per_gpu=concurrency_used,
        throughput_tokens_per_sec=throughput,
        shards=shards,
        replicas=replicas,
    )
