"""Utilities to load configuration YAML files into typed objects.

Each helper reads a YAML under ``configs/`` and validates it via the
``capacity_compass.config_types`` Pydantic models.  The functions purposely do
not cache results so that callers can decide caching behavior at a higher
level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from .config_types import (
    EstimationConfig,
    GPUConfig,
    ModelConfig,
    ScenarioPreset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigLoaderError(RuntimeError):
    """Raised when a configuration file is missing or malformed."""


def _load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise ConfigLoaderError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigLoaderError(f"Config file {path} should contain a mapping at top level")
    return data


def load_hardware(config_dir: Path | None = None) -> List[GPUConfig]:
    """Load hardware catalog from configs/hardware.yaml."""

    cfg_dir = config_dir or DEFAULT_CONFIG_DIR
    data = _load_yaml(cfg_dir / "hardware.yaml")
    raw_gpus = data.get("gpus", [])
    if not isinstance(raw_gpus, list):
        raise ConfigLoaderError("`gpus` must be a list in hardware.yaml")
    return [GPUConfig.model_validate(item) for item in raw_gpus]


def _load_models_file(path: Path) -> List[ModelConfig]:
    data = _load_yaml(path)
    raw_models = data.get("models")
    if not isinstance(raw_models, list):
        raise ConfigLoaderError(f"`models` must be a list in {path.name}")
    return [ModelConfig.model_validate(item) for item in raw_models]


def load_models(config_dir: Path | None = None) -> List[ModelConfig]:
    """Load and merge Qwen3 + DeepSeek model specs."""

    cfg_dir = config_dir or DEFAULT_CONFIG_DIR
    qwen = _load_models_file(cfg_dir / "models_qwen3.yaml")
    deepseek = _load_models_file(cfg_dir / "models_deepseek.yaml")
    return qwen + deepseek


def load_scenarios(config_dir: Path | None = None) -> Dict[str, ScenarioPreset]:
    cfg_dir = config_dir or DEFAULT_CONFIG_DIR
    data = _load_yaml(cfg_dir / "scenarios.yaml")
    scenarios: Dict[str, ScenarioPreset] = {}
    for name, payload in data.items():
        if not isinstance(payload, dict):  # pragma: no cover - defensive
            raise ConfigLoaderError(f"Scenario `{name}` must be a mapping")
        scenarios[name] = ScenarioPreset.model_validate(payload)
    return scenarios


def load_estimation(config_dir: Path | None = None) -> EstimationConfig:
    cfg_dir = config_dir or DEFAULT_CONFIG_DIR
    data = _load_yaml(cfg_dir / "estimation.yaml")
    return EstimationConfig.model_validate(data)
