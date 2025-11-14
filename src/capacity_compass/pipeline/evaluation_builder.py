"""Stage 6: assemble evaluation output incl. sales summaries."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from ..config_types import ScenarioPreset
from .normalizer import NormalizedRequest
from .recommender import RankedCandidate
from .requirements import RequirementResult

SCENE_GUIDES = {
    "chat": {
        "title": "对话问答",
        "fit": "和模型聊天、问问题，互动频繁，每次内容不长",
        "experience": "反馈较快，日常问答流畅",
        "tip": "内容特别长时，响应可能变慢",
    },
    "rag": {
        "title": "资料问答",
        "fit": "基于资料/知识库回答问题，问题背景更长",
        "experience": "速度中等，能读懂更长的资料",
        "tip": "资料越长，对设备显存和算力要求越高",
    },
    "writer": {
        "title": "长文写作",
        "fit": "写报告/方案/营销稿，单次输出较长",
        "experience": "更稳、更完整的长文生成",
        "tip": "内容越长，生成时间越久",
    },
}

DEFAULT_TIPS = ["本结果为快速粗略估算，仅供参考；实际以压测为准"]
GENERAL_DISCLAIMERS = [
    "本评估为快速预估结果，需结合压测和厂商建议后再定型。",
    "多卡互联、量化策略与软件版本会影响表现，交付前请二次确认。",
]

logger = logging.getLogger(__name__)


def build_evaluation(
    normalized_request: NormalizedRequest,
    presets: Dict[str, ScenarioPreset],
    requirements: Dict[str, RequirementResult],
    ranked_candidates: Dict[str, List[RankedCandidate]],
) -> Dict[str, Any]:
    """Assemble展示层结构，包含 quick_answer / scenes / raw_evaluation。"""

    raw = {
        "model_profile": normalized_request.model.model_dump(),
        "scenarios": {},
    }
    scenes: Dict[str, Any] = {}
    quick_answer = None
    disclaimers: List[str] = list(GENERAL_DISCLAIMERS)
    if normalized_request.is_anonymous_model:
        _append_disclaimer(disclaimers, "模型参数由用户提供，结果仅作粗略估算。")

    for scene, requirement in requirements.items():
        preset = presets.get(scene)
        if not preset:
            continue

        candidates = ranked_candidates.get(scene, [])
        scene_summary = _build_scene_summary(
            scene,
            preset,
            normalized_request,
            requirement,
            candidates,
        )
        scenes[scene] = scene_summary
        raw["scenarios"][scene] = {
            "preset": preset.model_dump(),
            "requirements": asdict(requirement),
            "candidates": [asdict(c) for c in candidates],
        }

        primary = scene_summary["sales_summary"].get("primary")
        if scene == "chat" and primary and not quick_answer:
            quick_answer = _build_quick_answer(scene_summary["sales_summary"])

        if not candidates:
            _append_disclaimer(
                disclaimers,
                f"{preset.label} 场景暂无满足显存与精度要求的硬件，请联系服务商确认。",
            )

        logger.info(
            "scene=%s primary_device=%s cards=%s has_candidates=%s",
            scene,
            primary["device"] if primary else None,
            primary["cards"] if primary else None,
            bool(candidates),
        )

    return {
        "quick_answer": quick_answer,
        "scenes": scenes,
        "raw_evaluation": raw,
        "disclaimers": disclaimers,
    }


def _build_scene_summary(
    scene: str,
    preset: ScenarioPreset,
    normalized_request: NormalizedRequest,
    requirement: RequirementResult,
    candidates: List[RankedCandidate],
) -> Dict[str, Any]:
    """Craft sales summary + guide for单个场景（设计 §4.6）。"""
    primary = candidates[0] if candidates else None
    alternatives = candidates[1:3]

    tips: List[str] = list(DEFAULT_TIPS)
    if requirement.context_clamped:
        tips.append("请求上下文已按模型上限估算")
    if primary and primary.total_price is None:
        tips.append("部分设备价格需与服务商确认")

    sales_summary = {
        "primary": _format_candidate(primary, preset),
        "alternatives": [_format_candidate(c, preset) for c in alternatives],
        "tips": tips,
        "confidence": "low",
        "switch_notice": normalized_request.switch_notice,
    }

    guide = SCENE_GUIDES.get(scene)
    return {
        "sales_summary": sales_summary,
        "guide": guide,
    }


def _experience_label(preset: ScenarioPreset) -> str:
    """Map latency目标到易懂的体验标签。"""
    t = preset.target_latency_ms
    if t <= 1000:
        return "预计响应：快"
    if t <= 2000:
        return "预计响应：中"
    return "预计响应：稳"


def _format_candidate(
    candidate: Optional[RankedCandidate], preset: ScenarioPreset
) -> Optional[Dict[str, Any]]:
    """Convert候选 GPU 为销售可读结构。"""
    if not candidate:
        return None
    reason = _reason_from_candidate(candidate)
    return {
        "device": candidate.gpu_name,
        "cards": candidate.cards_needed,
        "estimate": _experience_label(preset),
        "reason": reason,
    }


def _reason_from_candidate(candidate: RankedCandidate) -> str:
    """Return简短理由，优先显存/成熟度/价格。"""
    support = candidate.deploy_support
    if support in {"native", "excellent"}:
        return "显存合适、推理稳定"
    if support in {"good"}:
        return "显存合适、兼容性较好"
    if candidate.total_price is not None:
        return "价格信息明确"
    if any("算力" in note for note in candidate.notes):
        return "需确认算力与生态"
    return "显存合适"


def _build_quick_answer(summary: Dict[str, Any]) -> Dict[str, Any]:
    """提炼 chat 场景的答案用于“最小响应示例”。"""
    return {
        "primary": summary.get("primary"),
        "alternatives": summary.get("alternatives"),
        "tips": summary.get("tips"),
        "confidence": summary.get("confidence"),
        "switch_notice": summary.get("switch_notice"),
    }


def _append_disclaimer(disclaimers: List[str], text: Optional[str]) -> None:
    """Add text to disclaimers while keeping顺序且避免重复。"""
    if not text:
        return
    if text not in disclaimers:
        disclaimers.append(text)
