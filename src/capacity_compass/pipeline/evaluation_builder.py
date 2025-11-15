"""Stage 6: assemble evaluation output incl. sales summaries."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

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

DOMESTIC_VENDORS = {"huawei", "kunlunxin", "biren", "inspur", "baidu"}

logger = logging.getLogger(__name__)


def build_evaluation(
    normalized_request: NormalizedRequest,
    presets: Dict[str, ScenarioPreset],
    requirements: Dict[str, RequirementResult],
    ranked_candidates: Dict[str, List[RankedCandidate]],
    model_suggestions: Optional[Dict[str, Dict[str, Any]]] = None,
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
        suggestion = (model_suggestions or {}).get(scene) if model_suggestions else None
        scene_summary = _build_scene_summary(
            scene,
            preset,
            normalized_request,
            requirement,
            candidates,
            suggestion,
        )
        scenes[scene] = scene_summary
        raw["scenarios"][scene] = {
            "preset": preset.model_dump(),
            "requirements": asdict(requirement),
            "candidates": [asdict(c) for c in candidates],
        }

        primary = scene_summary["sales_summary"].get("primary")
        if scene == "chat" and primary and not quick_answer:
            quick_answer = _build_quick_answer(scene_summary["sales_summary"], normalized_request)

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

    # 生成整体 Markdown（表格由后端渲染，LLM 仅输出段落）
    markdown = _render_markdown(quick_answer, scenes, disclaimers)
    return {
        "quick_answer": quick_answer,
        "scenes": scenes,
        "raw_evaluation": raw,
        "disclaimers": disclaimers,
        "markdown": markdown,
    }


def _build_scene_summary(
    scene: str,
    preset: ScenarioPreset,
    normalized_request: NormalizedRequest,
    requirement: RequirementResult,
    candidates: List[RankedCandidate],
    suggestion: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Craft sales summary + guide for单个场景（设计 §4.6）。"""
    primary = candidates[0] if candidates else None
    alternatives = candidates[1:3]

    tips: List[str] = list(DEFAULT_TIPS)
    if requirement.context_clamped:
        tips.append("请求上下文已按模型上限估算")
    if primary and primary.total_price is None:
        tips.append("部分设备价格需与服务商确认")

    # Build full ranked list（用于后端表格/LLM 段落）
    ranked_list = [
        _format_candidate(c, preset, requirement, eval_precision=normalized_request.eval_precision)
        for c in candidates
    ]
    table_md = _table_markdown([r for r in ranked_list if r])

    # 如果所有候选的并发/吞吐缺少差异（或完全缺失），显式提示“统一假设”，并标记 uniform_performance
    uniform_performance = False
    if candidates:
        tps_values = [c.throughput_tokens_per_sec for c in candidates]
        if all(v is None for v in tps_values) or len({v for v in tps_values if v is not None}) <= 1:
            tips.append("当前缺少算力数据，吞吐/并发按统一假设估算")
            uniform_performance = True

    sales_summary = {
        "primary": _format_candidate(
            primary, preset, requirement, eval_precision=normalized_request.eval_precision
        ),
        "alternatives": [
            _format_candidate(
                c, preset, requirement, eval_precision=normalized_request.eval_precision
            )
            for c in alternatives
        ],
        "ranked": [r for r in ranked_list if r],
        "table_md": table_md,
        "tips": tips,
        "confidence": "low",
        "uniform_performance": uniform_performance,
        "unified_assumption": {
            "per_session_tokens_per_sec": requirement.tokens_per_sec_session,
            "per_gpu_concurrency": requirement.concurrency,
        },
        "switch_notice": normalized_request.switch_notice,
        "concurrency": requirement.concurrency,
        "target_latency_ms": preset.target_latency_ms,
        "context_len": requirement.context_len,
        "default_context_len": preset.default_context_len,
        "domestic_fallback": _format_candidate(
            _first_domestic_candidate(candidates),
            preset,
            requirement,
            eval_precision=normalized_request.eval_precision,
        ),
        "precision": normalized_request.eval_precision,
        "model": normalized_request.model.display_name,
        "suggested_model": suggestion,
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
    candidate: Optional[RankedCandidate],
    preset: ScenarioPreset,
    requirement: RequirementResult,
    *,
    eval_precision: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Convert候选 GPU 为销售可读结构，并附带并发/上下文信息。"""
    if not candidate:
        return None
    reason = _reason_from_candidate(candidate)
    description = reason
    if candidate.cards_compute is None:
        description += "；算力按显存估算"
    # Friendly ranges
    sessions = candidate.concurrency_per_gpu or requirement.concurrency
    sessions_range = _range_text(sessions, delta=1)
    tps = candidate.throughput_tokens_per_sec or requirement.throughput_tps
    speed_chars_range = _speed_chars_range(tps)
    # 显示用总卡数：若存在“分片×复本”，按乘积展示，避免出现奇数卡数观感
    display_total_cards = None
    if candidate.shards and candidate.replicas:
        display_total_cards = candidate.shards * candidate.replicas
    else:
        display_total_cards = candidate.cards_needed

    # 档位化
    sessions_level = _level_from_ratio(sessions, requirement.concurrency)
    speed_level = _level_from_ratio(tps or 0, requirement.throughput_tps or (tps or 1))

    domestic = _is_domestic_vendor(candidate.vendor)
    vendor_label = f"国产（{candidate.vendor}）" if domestic else f"海外（{candidate.vendor}）"

    return {
        "device": candidate.gpu_name,
        "device_id": candidate.gpu_id,
        "vendor": candidate.vendor,
        "vendor_label": vendor_label,
        "memory_gb": candidate.memory_gb,
        "precision_used": None,  # filled at quick_answer level if需要
        "cards": candidate.cards_needed,
        "card_count": _card_count_text(display_total_cards),
        "layout": (
            f"{candidate.shards}×{candidate.replicas}（共{display_total_cards}）"
            if candidate.shards and candidate.replicas
            else str(candidate.cards_needed)
        ),
        "estimate": _experience_label(preset),
        "reason": description,
        "reason_short": reason,
        "bottleneck": _bottleneck(candidate),
        "concurrency_text": (
            f"单卡预估并发≈{candidate.concurrency_per_gpu} 个会话"
            if candidate.concurrency_per_gpu is not None
            else f"单卡预估并发≈{requirement.concurrency} 个会话"
        ),
        "throughput_text": (
            f"预估吞吐≈{int(candidate.throughput_tokens_per_sec)} tokens/s"
            if candidate.throughput_tokens_per_sec is not None
            else (
                f"预估吞吐≈{int(requirement.throughput_tps)} tokens/s"
                if requirement.throughput_tps is not None
                else None
            )
        ),
        "sessions_range": sessions_range,
        "speed_range_chars_per_s": speed_chars_range,
        "sessions_level": sessions_level,
        "speed_level": speed_level,
        "is_domestic": _is_domestic_vendor(candidate.vendor),
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


def _bottleneck(candidate: RankedCandidate) -> str:
    if candidate.cards_compute is None:
        return "memory"
    return "memory" if candidate.cards_mem >= candidate.cards_compute else "compute"


def _card_count_text(n: int) -> str:
    return "单卡" if n == 1 else f"{n}卡"


def _range_text(base: Optional[int], delta: int = 1) -> Optional[str]:
    if base is None:
        return None
    lo = max(1, base)
    hi = max(lo, base + max(1, delta))
    return f"{lo}–{hi}"


def _speed_chars_range(tps: Optional[float]) -> Optional[str]:
    if not tps or tps <= 0:
        return None
    # 粗略映射：1 token ≈ 1 中文字符；给出 ±20% 范围
    lo = int(max(1, round(tps * 0.8)))
    hi = int(max(lo, round(tps * 1.2)))
    return f"{lo}–{hi}"


def _level_from_ratio(value: Optional[float | int], target: Optional[float | int]) -> Optional[str]:
    if value is None or target is None or target <= 0:
        return None
    ratio = float(value) / float(target)
    if ratio >= 1.0:
        return "高"
    if ratio >= 0.5:
        return "中"
    return "低"


def _build_quick_answer(
    summary: Dict[str, Any], normalized_request: NormalizedRequest
) -> Dict[str, Any]:
    """提炼 chat 场景的答案用于“最小响应示例”。"""
    primary = summary.get("primary")
    alternatives = [dict(alt) for alt in summary.get("alternatives", []) if alt]
    cands_for_domestic: List[Dict[str, Any]] = []
    if primary:
        cands_for_domestic.append(primary)
    cands_for_domestic.extend(alternatives)
    if not _contains_domestic(cands_for_domestic):
        fallback = summary.get("domestic_fallback")
        if fallback:
            if len(alternatives) >= 2:
                alternatives[-1] = dict(fallback)
            else:
                alternatives.append(dict(fallback))

    # Friendly items list for UI/table
    def _mk_item(c: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not c:
            return None
        item = {
            "device_id": c.get("device_id"),
            "device_name": c.get("device") or c.get("device_name"),
            "vendor": c.get("vendor"),
            "vendor_label": c.get("vendor_label"),
            "memory_gb": c.get("memory_gb"),
            "precision_used": normalized_request.eval_precision,
            "card_count": c.get("card_count"),
            "sessions_range": c.get("sessions_range"),
            "speed_range_chars_per_s": c.get("speed_range_chars_per_s"),
            "sessions_level": c.get("sessions_level"),
            "speed_level": c.get("speed_level"),
            "bottleneck": c.get("bottleneck"),
            "reason_short": c.get("reason_short"),
            "is_domestic": c.get("is_domestic"),
        }
        return item

    items = [_mk_item(primary)] + [_mk_item(a) for a in alternatives]
    items = [i for i in items if i]
    # Quick table列出“全部排序设备”
    ranked_full = summary.get("ranked") or []
    table_md = _table_markdown([c for c in ranked_full if c])

    return {
        "primary": primary,
        "alternatives": alternatives,
        "items": items,
        "table_md": table_md,
        "uniform_performance": summary.get("uniform_performance", False),
        "unified_assumption": summary.get("unified_assumption"),
        "tips": summary.get("tips"),
        "confidence": summary.get("confidence"),
        "switch_notice": summary.get("switch_notice"),
        "concurrency": summary.get("concurrency"),
        "target_latency_ms": summary.get("target_latency_ms"),
        "context_len": summary.get("context_len"),
        "default_context_len": summary.get("default_context_len"),
        "ranked": summary.get("ranked"),
    }


def _table_markdown(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    rows = [
        "| 序号 | 设备 | 卡数（方案） | 预计体验 | 简短理由 | 厂商 |",
        "|---|---|---|---|---|---|",
    ]
    for idx, c in enumerate(items, start=1):
        rows.append(
            f"| {idx} | {c.get('device')} | {c.get('card_count')} | {c.get('estimate')} | {c.get('reason_short')} | {c.get('vendor_label')} |"
        )
    return "\n".join(rows)


def _render_markdown(
    quick: Optional[Dict[str, Any]], scenes: Dict[str, Any], disclaimers: List[str]
) -> str:
    parts: List[str] = []
    if quick:
        ua = quick.get("unified_assumption") or {}
        pps = ua.get("per_session_tokens_per_sec")
        conc = ua.get("per_gpu_concurrency")
        head = "**快速结论（表格）**\n"
        if pps and conc:
            head += f"按单会话约{int(pps)}字/秒、单卡并发约{int(conc)} 的统一假设进行估算，推荐设备总表如下：\n\n"
        parts.append(head)
        parts.append(quick.get("table_md", ""))
        parts.append(
            "\n**提示**：本结果为快速粗略估算，仅供参考；部分设备价格需与服务商确认；当前缺少算力数据，吞吐/并发按统一假设估算。\n\n---\n"
        )
    parts.append("**常见场景（分场景表格）**\n")
    order = ["chat", "rag", "writer"]
    for idx, name in enumerate(order, start=1):
        scene = scenes.get(name)
        if not scene:
            continue
        ss = scene.get("sales_summary", {})
        guide = scene.get("guide") or {}
        title = guide.get("title") or name
        parts.append(f"**{idx}. {title}**\n")
        sm = ss.get("suggested_model")
        if sm:
            parts.append(f"建议使用：{sm.get('display_name')}（理由：{sm.get('reason')}）。\n")
        parts.append(ss.get("table_md", ""))
        parts.append("\n")
    parts.append("---\n**补充说明**\n")
    for d in disclaimers:
        parts.append(f"- {d}")
    return "\n".join(parts)


def _append_disclaimer(disclaimers: List[str], text: Optional[str]) -> None:
    """Add text to disclaimers while keeping顺序且避免重复。"""
    if not text:
        return
    if text not in disclaimers:
        disclaimers.append(text)


def _first_domestic_candidate(candidates: List[RankedCandidate]) -> Optional[RankedCandidate]:
    for candidate in candidates:
        if _is_domestic_vendor(candidate.vendor):
            return candidate
    return None


def _contains_domestic(candidates: Sequence[Optional[Dict[str, Any]]]) -> bool:
    for candidate in candidates:
        if candidate and _is_domestic_vendor(candidate.get("vendor")):
            return True
    return False


def _is_domestic_vendor(vendor: Optional[str]) -> bool:
    return bool(vendor and vendor.lower() in DOMESTIC_VENDORS)
