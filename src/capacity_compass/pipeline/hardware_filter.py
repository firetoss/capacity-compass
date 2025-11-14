"""Stage 3: hardware candidate filtering."""

from __future__ import annotations

from typing import Iterable, List, Optional

from ..config_types import GPUConfig
from ..hardware_registry import HardwareRegistry


def filter_hardware(
    registry: HardwareRegistry,
    vendors: Optional[Iterable[str]],
    eval_precision: str,
    weights_mem_bytes: int,
) -> List[GPUConfig]:
    """Filter GPU candidates by vendor+precision, retain those fitting权重（docs §4.3）。"""

    candidates = registry.filter(vendors=vendors, precision=eval_precision)
    return [gpu for gpu in candidates if gpu.memory_gb * 1e9 >= weights_mem_bytes]
