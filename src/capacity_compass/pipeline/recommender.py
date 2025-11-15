"""Stage 5: rank hardware candidates (docs §4.5)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from .card_sizer import HardwareEvaluation

DEPLOY_PRIORITY = {
    "native": 0,
    "excellent": 1,
    "good": 2,
    "test": 3,
    "entry_level_only": 4,
    "unknown": 5,
    None: 6,
}

logger = logging.getLogger(__name__)


@dataclass
class RankedCandidate:
    """Serializable representation used by后续阶段/输出。"""

    gpu_id: str
    gpu_name: str
    vendor: str
    cards_needed: int
    cards_mem: int
    cards_compute: Optional[int]
    headroom: float
    total_price: Optional[float]
    deploy_support: Optional[str]
    notes: List[str]
    # Optional per-GPU perf annotations
    concurrency_per_gpu: Optional[int] = None
    throughput_tokens_per_sec: Optional[float] = None
    shards: Optional[int] = None
    replicas: Optional[int] = None
    # Display helpers (optional; filled by ranker for convenience)
    memory_gb: Optional[float] = None


def rank_candidates(evaluations: List[HardwareEvaluation]) -> List[RankedCandidate]:
    """Sort by 卡数→价格→成熟度→冗余，符合设计文档 §4.5。"""
    ranked: List[RankedCandidate] = []
    for evaluation in evaluations:
        gpu = evaluation.gpu
        price = gpu.pricing.price if gpu.pricing else None
        ranked.append(
            RankedCandidate(
                gpu_id=gpu.id,
                gpu_name=gpu.name,
                vendor=gpu.vendor,
                cards_needed=evaluation.cards_needed,
                cards_mem=evaluation.cards_mem,
                cards_compute=evaluation.cards_compute,
                headroom=evaluation.headroom,
                total_price=price,
                deploy_support=gpu.deploy_support,
                notes=evaluation.notes,
                concurrency_per_gpu=evaluation.concurrency_per_gpu,
                throughput_tokens_per_sec=evaluation.throughput_tokens_per_sec,
                shards=evaluation.shards,
                replicas=evaluation.replicas,
                memory_gb=gpu.memory_gb,
            )
        )

    ranked.sort(key=_sort_key)
    if ranked:
        top = ranked[0]
        logger.info(
            "top candidate gpu=%s cards=%s price=%s deploy=%s",
            top.gpu_name,
            top.cards_needed,
            top.total_price,
            top.deploy_support,
        )
    else:
        logger.warning("no ranked candidates available after evaluation")
    return ranked


def _sort_key(candidate: RankedCandidate) -> tuple:
    deploy_rank = DEPLOY_PRIORITY.get(candidate.deploy_support, 6)
    # 1) 卡数最少优先  2) 部署成熟度  3) 冗余越充足越好
    return (candidate.cards_needed, deploy_rank, -candidate.headroom)
