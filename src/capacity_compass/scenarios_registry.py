"""Registry for scenario presets (chat, rag, writer, ...)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .config_types import ScenarioPreset


@dataclass
class ScenariosRegistry:
    presets: Dict[str, ScenarioPreset]

    def get(self, name: str) -> Optional[ScenarioPreset]:
        return self.presets.get(name)
