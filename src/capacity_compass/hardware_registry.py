"""Registry for GPU hardware specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .config_types import GPUConfig, PrecisionSupport


def _normalize_vendor_names(vendors: Optional[Iterable[str]]) -> Optional[set[str]]:
    if vendors is None:
        return None
    return {vendor.strip().lower() for vendor in vendors}


@dataclass
class HardwareRegistry:
    gpus: Sequence[GPUConfig]

    def __post_init__(self) -> None:
        self._by_id: Dict[str, GPUConfig] = {gpu.id: gpu for gpu in self.gpus}

    def get(self, gpu_id: str) -> Optional[GPUConfig]:
        return self._by_id.get(gpu_id)

    def filter(
        self,
        vendors: Optional[Iterable[str]] = None,
        precision: Optional[str] = None,
    ) -> List[GPUConfig]:
        vendor_filter = _normalize_vendor_names(vendors)
        precision_key = precision.lower() if precision else None
        results: List[GPUConfig] = []
        for gpu in self.gpus:
            if vendor_filter and gpu.vendor.lower() not in vendor_filter:
                continue
            if precision_key and not _supports_precision(gpu.precision_support, precision_key):
                continue
            results.append(gpu)
        return results


def _supports_precision(support: PrecisionSupport, precision_key: str) -> bool:
    if precision_key in {"fp16", "bf16", "fp8", "int8"}:
        return bool(getattr(support, precision_key))
    return False
