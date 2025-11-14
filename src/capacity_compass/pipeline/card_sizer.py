"""Stage 4: determine card counts per GPU candidate（docs §4.4）。"""

from __future__ import annotations

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


def size_cards(
    gpu: GPUConfig,
    eval_precision: str,
    total_mem_bytes: int,
    required_compute_Tx: float,
) -> HardwareEvaluation:
    """Take显存与算力上界，向上取整卡数，见设计 §4.4。"""
    notes: List[str] = []
    mem_per_card = int(gpu.memory_gb * 1e9)
    cards_mem = max(1, math.ceil(total_mem_bytes / mem_per_card))

    perf_field = PRECISION_FIELD.get(eval_precision)
    cards_compute: Optional[int] = None
    if perf_field and gpu.perf:
        perf_value = getattr(gpu.perf, perf_field)
        if perf_value:
            cards_compute = max(1, math.ceil(required_compute_Tx / perf_value))
    if cards_compute is None:
        notes.append("算力数据缺失，仅按显存估算")

    cards_needed = max(cards_mem, cards_compute or 0)
    total_mem_available = cards_needed * mem_per_card
    headroom = 0.0
    if total_mem_available:
        headroom = (total_mem_available - total_mem_bytes) / total_mem_available

    return HardwareEvaluation(
        gpu=gpu,
        cards_mem=cards_mem,
        cards_compute=cards_compute,
        cards_needed=cards_needed,
        headroom=headroom,
        total_mem_available=total_mem_available,
        notes=notes,
    )
