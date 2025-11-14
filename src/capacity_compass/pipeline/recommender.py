"""Stage 5: rank hardware candidates."""

from __future__ import annotations

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


@dataclass
class RankedCandidate:
    gpu_id: str
    cards_needed: int
    cards_mem: int
    cards_compute: Optional[int]
    headroom: float
    total_price: Optional[float]
    deploy_support: Optional[str]
    notes: List[str]


def rank_candidates(evaluations: List[HardwareEvaluation]) -> List[RankedCandidate]:
    ranked: List[RankedCandidate] = []
    for evaluation in evaluations:
        gpu = evaluation.gpu
        price = gpu.pricing.price if gpu.pricing else None
        ranked.append(
            RankedCandidate(
                gpu_id=gpu.id,
                cards_needed=evaluation.cards_needed,
                cards_mem=evaluation.cards_mem,
                cards_compute=evaluation.cards_compute,
                headroom=evaluation.headroom,
                total_price=price,
                deploy_support=gpu.deploy_support,
                notes=evaluation.notes,
            )
        )

    ranked.sort(key=_sort_key)
    return ranked


def _sort_key(candidate: RankedCandidate) -> tuple:
    price = candidate.total_price if candidate.total_price is not None else float("inf")
    deploy_rank = DEPLOY_PRIORITY.get(candidate.deploy_support, 6)
    return (
        candidate.cards_needed,
        price,
        deploy_rank,
        -candidate.headroom,
    )
