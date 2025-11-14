"""Registry helpers for model metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .config_types import ModelConfig


def _norm(text: str) -> str:
    return text.strip().lower()


@dataclass
class ModelsRegistry:
    """In-memory registry for model specs with lookup utilities."""

    models: Sequence[ModelConfig]

    def __post_init__(self) -> None:
        self._by_name: Dict[str, ModelConfig] = {}
        self._family_index: Dict[str, List[ModelConfig]] = {}
        for model in self.models:
            self._register_model(model)

    def _register_model(self, model: ModelConfig) -> None:
        names = {model.model_name, model.display_name, *model.aliases}
        for name in names:
            if not name:
                continue
            key = _norm(name)
            self._by_name.setdefault(key, model)
        self._family_index.setdefault(model.family, []).append(model)

    def match(self, query: str) -> List[ModelConfig]:
        """Return models that match the given name or alias (case-insensitive)."""

        key = _norm(query)
        results = [model for alias, model in self._by_name.items() if alias == key]
        return results

    def get(self, name: str) -> Optional[ModelConfig]:
        """Get a model by exact identifier/alias (case-insensitive)."""

        return self._by_name.get(_norm(name))

    def select_best_for_context(self, family: str, target_ctx: int) -> Optional[ModelConfig]:
        """Pick the model within a family whose context limit best fits the target."""

        candidates = [m for m in self._family_index.get(family, []) if m.max_position_embeddings]
        if not candidates:
            return None

        def score(model: ModelConfig) -> tuple[int, int]:
            max_ctx = model.max_position_embeddings or 0
            fits = 0 if max_ctx >= target_ctx else 1
            diff = abs(max_ctx - target_ctx)
            # Primary sort: whether it fits (0 best), secondary: minimal diff, tertiary: larger ctx
            return (fits, diff, -max_ctx)

        return min(candidates, key=score)
