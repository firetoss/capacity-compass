# CapacityCompass 计算公式与来源清单（精简版）

目的：统一“计算公式/场景参数/参考链接”的权威来源，避免重复与口径不一致；所有配置改动以本页为依据进行更新与审阅。

更新时间：2025‑11‑15

## 1) 推理估算公式（统一口径）

- Compute 需求（以生成阶段为主）
  - 近似：`compute_Tx ≈ 2 × 有效参数(Params) × tokens_per_sec / 1e3`（单位 TFLOPS）
  - tokens_per_sec 由场景目标推导：`tokens_per_sec = 并发 × (avg_output_tokens / target_latency_s) × alpha`
  - 说明：2× 因素源于乘加（FMA）口径；“有效参数”用于 MoE 稀疏路由的折算（见下）。
- KV Cache 显存
  - 近似：`kv_mem_bytes = L × H_kv × D × C × 2 × B_kv × 并发`，并乘以统一开销（默认 15%）
  - 符号说明：层数 L、KV 头数 H_kv、head_dim D、上下文 C、KV 精度字节 B_kv。
- MoE 有效参数折算
  - 背景：每 token 只激活 top_k 个专家；有效计算量显著低于总参数。
  - 处理：以 `moe_effective_factor`（按模型系）配置控制，例如 DeepSeek‑V3/R1 ≈ 0.06；Qwen3‑A3B ≈ 0.10。

> 以上口径对齐 NVIDIA TensorRT‑LLM / NIM 的实践文档与社区常用推导，用于“快速预估”，并非替代实测。

## 2) 常见场景参数（审校底稿）

- Chat（对话）
  - 目标响应：1–1.5s
  - 平均输出：120–200 tokens
  - 上下文：8k–16k
- RAG（资料问答）
  - 目标响应：2–3s
  - 上下文：16k–32k（随检索条数/分片长度而变）
  - 平均输出：200–300 tokens
- Writer（长文写作）
  - 目标响应：3–5s
  - 平均输出：800–1200 tokens
  - 上下文：8k–16k

配置映射：以上参数对应 `configs/scenarios.yaml` 中 `target_latency_ms / default_context_len / avg_output_tokens`。默认 `alpha_default = 0.15`（见 `configs/estimation.yaml`）。

## 3) 参考链接（去重索引 + 章节级线索）

- NVIDIA 文档
  - TensorRT‑LLM（部署/精度/性能指南）：https://docs.nvidia.com/deeplearning/tensorrt-llm/
    - 章节线索：User Guide → Performance Tuning；Inference Guide → Precision (FP8/BF16/INT8)；KV Cache/Attention Optimization（Paged/Flash 等）
  - NIM LLM 支持与建议（含 H20/L20 等精度）：https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html
    - 章节线索：Supported Models → Precision Availability；Deployment Recommendations（Model-specific profiles）
  - 产品页/数据表（统一仅列一次）
    - L40S: https://www.nvidia.com/en-us/data-center/l40s/
    - L20:  https://www.nvidia.com/en-us/data-center/l20/
    - H100: https://resources.nvidia.com/en-us-tensor-core/h100-datasheet
    - A100: https://resources.nvidia.com/en-us-tensor-core/a100-datasheet
- vLLM 文档
  - https://docs.vllm.ai/
    - 章节线索：Performance Tips；Architecture → PagedAttention；Engine → KV Cache/Memory Usage/Throughput vs Latency
- Hugging Face 生态
  - TGI（Text Generation Inference）：https://github.com/huggingface/text-generation-inference
    - 章节线索：Performance Tuning/Batching/Throughput；Concurrency configuration
  - Transformers（生成与 KV 缓存相关机制）：https://huggingface.co/transformers
    - 章节线索：Generate API → past_key_values；长上下文/RoPE 扩展注意事项
- MoE 参考
  - Shazeer et al., 2017（Mixture‑of‑Experts）：https://arxiv.org/abs/1701.06538
  - DeepSeek‑V3/R1（公开仓库与报告）：
    - https://huggingface.co/deepseek-ai/DeepSeek-V3
    - https://huggingface.co/deepseek-ai/DeepSeek-R1

> 注：如需添加更多链接，请避免重复，保持“一个主题一个权威链接”。

## 4) 与实现的对应关系

- 公式落地位置
  - Compute：`src/capacity_compass/pipeline/requirements.py::_estimate_compute`
  - KV Cache：`src/capacity_compass/pipeline/requirements.py::estimate_requirements`
- 配置口径
  - `configs/estimation.yaml`：`alpha_default`、`overhead_ratio_default`、`moe_effective_factor`
  - `configs/scenarios.yaml`：各场景 `default_context_len`、`target_latency_ms`、`avg_output_tokens`

## 5) 后续动作（只保留不重复的待办）

- 对每条公式补充至少 1 个明确出处（文档章节/页码）；在提交时于 PR 中引用。
- 拿到供应商 perf 数据后，补齐 `configs/hardware.yaml` 的 `perf.*` 字段，驱动并发/吞吐形成差异化。
- 校准场景参数（按来源）；若与当前默认不一致，提交变更说明与对照表。

---

## 附：场景参数对照表（建议值 → 当前值）

说明：建议值为“区间/建议”，当前值为 configs/scenarios.yaml 中的具体配置；alpha=0.15（configs/estimation.yaml）。

- Chat（对话）
  - 目标响应：建议 1–1.5s → 当前 1.0s（1000ms）
  - 平均输出：建议 120–200 tokens → 当前 200
  - 上下文：建议 8k–16k → 当前 8192
  - 单卡并发（small/medium/large）：建议 8–12 / 4–6 / 2–3 → 当前 12 / 6 / 3

- RAG（资料问答）
  - 目标响应：建议 2–3s → 当前 2.0s（2000ms）
  - 平均输出：建议 200–300 tokens → 当前 300
  - 上下文：建议 32k–64k → 当前 65536（默认估算上限 40960，按请求与模型限制取最小）
  - 单卡并发（small/medium/large）：建议 4–6 / 2–3 / 1–2 → 当前 6 / 3 / 2

- Writer（长文写作）
  - 目标响应：建议 3–5s → 当前 3.0s（3000ms）
  - 平均输出：建议 800–1200 tokens → 当前 1000
  - 上下文：建议 8k–16k → 当前 8192
  - 单卡并发（small/medium/large）：建议 2–4 / 1–2 / 1 → 当前 4 / 2 / 1
