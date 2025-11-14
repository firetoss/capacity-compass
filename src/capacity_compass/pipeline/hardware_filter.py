"""Stage 3: hardware candidate filtering."""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from ..config_types import GPUConfig
from ..hardware_registry import HardwareRegistry

logger = logging.getLogger(__name__)


def filter_hardware(
    registry: HardwareRegistry,
    vendors: Optional[Iterable[str]],
    eval_precision: str,
    weights_mem_bytes: int,
) -> List[GPUConfig]:
    """Filter GPU candidates by vendor+precision, retain those fitting权重（docs §4.3）。"""

    vendor_list = list(vendors) if vendors is not None else None
    candidates = registry.filter(vendors=vendor_list, precision=eval_precision)
    eligible = [gpu for gpu in candidates if gpu.memory_gb * 1e9 >= weights_mem_bytes]

    logger.info(
        "hardware filter vendors=%s precision=%s weights=%.2f GB -> %d eligible (raw=%d)",
        vendor_list or "ALL",
        eval_precision,
        weights_mem_bytes / 1e9,
        len(eligible),
        len(candidates),
    )
    if not eligible:
        logger.warning(
            "no GPU meets minimum weight memory requirement for precision=%s", eval_precision
        )
    return eligible
