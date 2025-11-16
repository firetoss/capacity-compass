# CapacityCompass

LLM 推理算力评估与硬件推荐服务。

本项目遵循 AGENTS.md：配置只读、设计文档为真、核心模块必须有日志与注释、单测覆盖关键路径。

## Environment Setup

1) 复制并填写 `.env.dev`（已被忽略）：
- `OPENROUTER_API_KEY`（用于 OpenRouter）
- `OPENROUTER_REFERER` / `OPENROUTER_SITE_TITLE`（可选排名来源）
- `CAPACITYCOMPASS_ENABLE_SUMMARY`（`true/false`，默认 `false`）
- `CAPACITYCOMPASS_LLM_BASE_URL`（默认 `https://openrouter.ai/api/v1`）
- `CAPACITYCOMPASS_LLM_TIMEOUT`（秒，默认 `30`）
- `CAPACITYCOMPASS_PROMPT_PATH`（可选，默认 `prompts/sales_summary_zh.txt`）
- `CAPACITYCOMPASS_HTTP_PROXY`（可选，调用 LLM 时的代理）
- `CAPACITYCOMPASS_AUTOSWITCH_FOR_SCENES`（`true/false`，默认 `true`；场景默认上下文超限时自动切到合适的 Instruct 做实评估）
- `LOG_LEVEL`（`DEBUG/INFO/WARNING`，默认 `INFO`）

2) 启动 API（FastAPI）：
```bash
source .env.dev
uv run uvicorn capacity_compass.api.server:app --reload
```

## What’s Included

- 核心评估流水线：normalizer → requirements → hardware_filter → card_sizer → recommender → evaluation_builder → llm_summary（可选）
- 后端渲染 Markdown 表格（快速结论 + 三场景全量排序），LLM 仅输出段落（更稳健）
- 场景参数采用“中位”目标：chat=1200ms、rag=2500ms、writer=4000ms；RAG 默认上下文 64k；writer 输出 1000 tokens
- 场景级自动 Instruct 切换（开启时）：默认上下文超过当前模型上限时，自动切到更合适的 Instruct 做评估，并写出 switch_notice
- LLM 提示词：`prompts/sales_summary_zh.txt`（销售友好、表格由后端提供）
- 抽取提示词：`prompts/extract_request_zh.txt`（从自然语言中提取 model/precision/context_len，严格 JSON）
- 资料来源与公式：`docs/sources.md`（统一口径，含章节线索与“建议值→当前值”）

## Programmatic Usage

评估（返回结构含 `markdown` 字段，可直接给客户预览）：
```python
from capacity_compass.pipeline.service import build_service_context, evaluate_capacity
from capacity_compass.pipeline.normalizer import EvaluationRequest

ctx = build_service_context()
req = EvaluationRequest(model='Qwen3-4B', max_context_len=16000, precision='fp16')
res = evaluate_capacity(req, ctx, generate_summary=False)
print(res['markdown'])
```

从原话抽取字段（Qwen3‑8B via OpenRouter）：
```python
from capacity_compass.pipeline.extractor import extract_fields
r = extract_fields('用Qwen3-8B推理，16K上下文，BF16')
print(r.model, r.precision, r.context_len)
```

## Batch Extraction (OpenRouter)

脚本：`scripts/extract_batch.py`
- 支持：
  - `--timeout` 每次请求超时（秒，默认 20）
  - `--retries` 失败重试次数（默认 2）、`--retry-wait` 重试间隔（秒，默认 3）
  - `--gap` 请求间隔（秒，默认 5）
  - `--skip-proxy` 跳过代理（忽略 `CAPACITYCOMPASS_HTTP_PROXY`）
  - `--env-file` 加载 `.env.dev` 注入 `OPENROUTER_API_KEY`
  - `--out` 输出 Markdown 文件（默认 `docs/extract_preview.md`）

示例：
```bash
PYTHONPATH=src uv run -- python scripts/extract_batch.py \
  --skip-proxy --env-file .env.dev \
  --timeout 20 --retries 2 --retry-wait 3 --gap 5 \
  --out docs/extract_preview.md
```

## Deploy (Docker Offline, x86_64)

Build offline image tar (linux/amd64) using USTC PyPI mirror:
```bash
bash scripts/build_docker_offline.sh
# outputs: capacity_compass_latest_amd64.tar
```

Load and run on server (port 9050):
```bash
docker load -i capacity_compass_latest_amd64.tar
# prepare runtime envs only (example)
cat > .env.runtime <<'ENV'
LOG_LEVEL=INFO
# OPENROUTER_API_KEY=xxxxx   # only if LLM summary needed
# CAPACITYCOMPASS_ENABLE_SUMMARY=true
# CAPACITYCOMPASS_AUTOSWITCH_FOR_SCENES=true
ENV

docker run -d --name capacity-compass -p 9050:9050 --env-file .env.runtime capacity-compass:latest
```

Smoke test (avoid local proxy):
```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
curl -s http://127.0.0.1:9050/openapi.json | head -n 1
curl -s -X POST http://127.0.0.1:9050/api/llm/capacity/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-4B","max_context_len":16000,"precision":"fp16"}' | jq '.quick_answer.items[0]'
```

## Scenes & Assumptions

- 中位目标：chat=1200ms、rag=2500ms、writer=4000ms；RAG 默认上下文 64k；writer 输出 1000
- 统一假设（当硬件 perf 缺失时）：单会话≈30字/秒、并发按场景预设（chat=12、rag=6、writer=4）。
- 当模型上限或请求超出默认上下文：截断并在文案/notes 中说明；若开启自动切换则改用家族中更合适的 Instruct 实评估。

## Troubleshooting

- OpenRouter 超时：增大 `--timeout`（30–45s）与 `--gap`（6–8s）；确保 `--skip-proxy` 或移除 `CAPACITYCOMPASS_HTTP_PROXY`。
- 提取 model 匹配不上：抽取提示词已内置最小标准化（空格/连字符/中文家族名）；若 `model=null`，可在前端使用默认模型（如 Qwen3‑8B‑Instruct）。
- “并发/吞吐都一样”：未补齐硬件 perf（bf16/fp8/int8）时会走“统一假设”，表格已由后端渲染并在段首提示；待 perf 回填后差异会自然拉开。

## Response Size / Compact Mode（建议）

当前响应包含 `raw_evaluation`、全量 `ranked` 列表与后端渲染 `markdown`，用于销售/技术双场景。若需更精简：

- 在请求体中新增可选开关（建议）：
  - `compact: true` → 仅返回 `quick_answer.items`、`scenes.*.sales_summary.table_md`、可选 `markdown`；不返回 `raw_evaluation`
  - `top_k: 3` → 限制主表与场景表格条目数
  - `include_raw: false`、`include_markdown: true/false`

如需我落地“Compact Mode”，请确认默认行为（是否默认 compact）与 `top_k` 默认值。

## Dev

```bash
uv run pytest -q
```

参考：
- 设计与来源：`docs/sources.md`
- 客户预览示例：`docs/customer_preview.md` / `docs/extract_preview.md`
