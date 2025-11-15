#!/usr/bin/env python3
"""Batch-run field extraction via OpenRouter with retries and pacing.

Usage examples:
  uv run -- python scripts/extract_batch.py --skip-proxy --timeout 20 \
      --retries 2 --retry-wait 3 --gap 5 --out docs/extract_preview.md

Notes:
  - Requires OPENROUTER_API_KEY in env.
  - --skip-proxy prevents using CAPACITYCOMPASS_HTTP_PROXY.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List

from capacity_compass.pipeline.extractor import ExtractorError, extract_fields

QUESTIONS: List[str] = [
    "用Qwen3-8B推理，16K上下文，BF16，单卡可以吗？",
    "打算先用深度求索V3，fp16 就行，上下文 4k。",
    "普通聊天，不确定用哪个模型，8千字就够了，精度随意。",
    "qwen3 8b instruct，FP8，试试 32K 上下文。",
    "我们要 RAG，考虑 Qwen/Qwen3-14B，bfloat16，文档上下文 64k。",
    "先用 Qwen3-4B 就行，precision 不限，上下文 10万。",
    "DeepSeek-R1，精度 fp16，context 大概 128k。",
    "模型没定，要求 FP16，4k tokens 够吗？",
    "Qwen3-0.6B，int8 量化可以吗？8K 上下文。",
    "用 Llama 3.1 8B，bf16，16k。",
    "请推荐模型，先假定 6k 上下文，bfloat16。",
    "DeepSeek-V3，FP8 有优势吗？我们希望 32k 上下文。",
    "想上 Qwen3-32B，bf16 精度，目标 128k 上下文。",
    "暂不确定模型，先按 12K context，fp16 看。",
    "Qwen/Qwen3-30B-A3B-Instruct-2507，BF16，64K。",
    "Qwen3-8B，精度随便，2万字的输入。",
    "DeepSeek R1，bfloat16，1万字够不够？",
    "Qwen3 4B，fp-8，8K。",
    "模型未知，目标 20k tokens，bf16。",
    "Qwen3-235B-A22B，FP16，上下文 200k。",
]


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs from a .env-like file if present.

    Only sets variables that are not already in os.environ.
    Lines starting with # are ignored.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k and (k not in os.environ):
            os.environ[k] = v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=20.0, help="per-request timeout seconds")
    ap.add_argument("--retries", type=int, default=2, help="retries on failure")
    ap.add_argument("--retry-wait", type=float, default=3.0, help="seconds to wait between retries")
    ap.add_argument("--gap", type=float, default=5.0, help="seconds to wait between requests")
    ap.add_argument(
        "--skip-proxy", action="store_true", help="ignore CAPACITYCOMPASS_HTTP_PROXY during calls"
    )
    ap.add_argument("--out", type=Path, default=Path("docs/extract_preview.md"))
    ap.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env.dev"),
        help="optional .env file to load OPENROUTER_API_KEY, etc.",
    )
    args = ap.parse_args()

    if args.skip_proxy:
        os.environ.pop("CAPACITYCOMPASS_HTTP_PROXY", None)

    # Load API key if not present
    if not os.getenv("OPENROUTER_API_KEY"):
        _load_env_file(args.env_file)

    lines: List[str] = [
        "# 抽取结果预览（在线调用）",
        "",
        f"> 参数：timeout={args.timeout}s, retries={args.retries}, retry_wait={args.retry_wait}s, gap={args.gap}s",
        "",
    ]

    for i, q in enumerate(QUESTIONS, 1):
        result = None
        error = None
        for attempt in range(args.retries + 1):
            try:
                r = extract_fields(q, timeout_seconds=args.timeout, http_proxy=None)
                result = {"model": r.model, "precision": r.precision, "context_len": r.context_len}
                break
            except (ExtractorError, Exception) as e:  # noqa: BLE001
                error = str(e)
                if attempt < args.retries:
                    time.sleep(args.retry_wait)
                    continue
        lines.append(f"{i:02d}. 问题：{q}")
        if result is not None:
            lines.append(f"    提取：{result}")
        else:
            lines.append(f"    提取：{{'error': '{error}'}}")
        time.sleep(args.gap)

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {args.out} ({len(QUESTIONS)} entries)")


if __name__ == "__main__":
    main()
