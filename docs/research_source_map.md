# Research Source Map (Models + Hardware)

Purpose
- Record authoritative sources used to populate YAML fields in `configs/`.
- Make every important number traceable (config.json → field; vendor doc → precision support).

Notes
- For Qwen3/DeepSeek models: structure comes from Hugging Face `config.json` of each repo.
- For VL models: structure is under `text_config` inside `config.json`.
- For precision/quantization: only mark as supported if the owner org publishes an artifact (e.g., FP8 repo) or the official config includes `quantization_config`.
- For NVIDIA H20/L20 FP8 support: cite NVIDIA NIM official documentation (tokens/s tables list FP8 for H20/L20);
  when a public datasheet URL is unavailable, we keep perf numbers null and rely on NIM as the support signal.

---

## Qwen3 (Text, Base Repos)
- Qwen/Qwen3-0.6B (config.json)
  - https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json
- Qwen/Qwen3-1.7B (config.json)
  - https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json
- Qwen/Qwen3-4B (config.json)
  - https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json
- Qwen/Qwen3-8B (config.json)
  - https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json
- Qwen/Qwen3-14B (config.json)
  - https://huggingface.co/Qwen/Qwen3-14B/raw/main/config.json
- Qwen/Qwen3-32B (config.json)
  - https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json

Key fields used
- hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads
- max_position_embeddings (base repos commonly 40960)
- torch_dtype (often bfloat16)

## Qwen3 (MoE Text)
- Qwen/Qwen3-30B-A3B (config.json)
  - https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json
- Qwen/Qwen3-235B-A22B (config.json)
  - https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json

Notes
- MoE metadata: num_experts, top_k, expert_intermediate_size extracted from config.json
- max_position_embeddings = 40960 (from config.json)

## Qwen3 (VL: Vision + Text)
- Qwen/Qwen3-VL-2B-Instruct (config.json → text_config)
  - https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/raw/main/config.json
- Qwen/Qwen3-VL-4B-Instruct (config.json → text_config)
  - https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/raw/main/config.json
- Qwen/Qwen3-VL-8B-Instruct (config.json → text_config)
  - https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/config.json
- Qwen/Qwen3-VL-30B-A3B-Instruct (config.json → text_config)
  - https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/raw/main/config.json
- Qwen/Qwen3-VL-32B-Instruct (config.json → text_config)
  - https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct/raw/main/config.json
- Qwen/Qwen3-VL-235B-A22B-Instruct (config.json → text_config)
  - https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct/raw/main/config.json

Key fields used (from text_config)
- hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads, head_dim
- max_position_embeddings (commonly 262144 for VL text backbones)

Example (Instruct variant with long context)
- Qwen/Qwen3-4B-Instruct-2507 (config.json)
  - https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/raw/main/config.json
  - max_position_embeddings = 262144

## Qwen3 (Official Quantized Repos)
FP8 (owner: Qwen)
- 0.6B: https://huggingface.co/Qwen/Qwen3-0.6B-FP8
- 1.7B: https://huggingface.co/Qwen/Qwen3-1.7B-FP8
- 4B: https://huggingface.co/Qwen/Qwen3-4B-FP8
- 8B: https://huggingface.co/Qwen/Qwen3-8B-FP8
- 14B: https://huggingface.co/Qwen/Qwen3-14B-FP8
- 32B: https://huggingface.co/Qwen/Qwen3-32B-FP8
- 30B-A3B: https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8
- 235B-A22B: https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8
- VL-2B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-FP8
- VL-4B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-FP8
- VL-8B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-FP8
- VL-30B-A3B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
- VL-32B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8
- VL-235B-A22B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8

AWQ 4-bit (owner: Qwen)
- 4B: https://huggingface.co/Qwen/Qwen3-4B-AWQ
- 8B: https://huggingface.co/Qwen/Qwen3-8B-AWQ
- 14B: https://huggingface.co/Qwen/Qwen3-14B-AWQ
- 32B: https://huggingface.co/Qwen/Qwen3-32B-AWQ

Int8
- No org-hosted Int8 repos confirmed for Qwen3 at the time of writing; leave `supports_int8: null` and capture third-party options only in notes if needed.

---

## DeepSeek
- DeepSeek-V3 (config.json)
  - https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json
  - quantization_config.quant_method = fp8
- DeepSeek-R1 (config.json)
  - https://huggingface.co/deepseek-ai/DeepSeek-R1/raw/main/config.json
  - quantization_config.quant_method = fp8
- R1-Distill-Qwen family (config.json)
  - 1.5B: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/raw/main/config.json
  - 7B:   https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/raw/main/config.json
  - 14B:  https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B/raw/main/config.json
  - 32B:  https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/raw/main/config.json

---

## NVIDIA Hardware
Authoritative FP8 support signal (H20/L20)
- NVIDIA NIM LLM Supported Models
  - https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html
  - Notes: Lists FP8/BF16 throughput & latency for H20 and L20; used to set `precision_support.fp8=true` and upgrade `deploy_support`.

Device product pages (for general specs)
- A100: https://www.nvidia.com/en-us/data-center/a100/
- A10:  https://www.nvidia.com/en-us/data-center/products/a10-gpu/
- T4:   https://www.nvidia.com/en-us/data-center/tesla-t4/
- L20/H20: public datasheet pages are not consistently available; rely on NIM link above for FP8 support; keep perf numbers null unless vendor PDF is provided.

---

## Huawei Ascend / Atlas
- Ascend 910B (product page): https://www.hiascend.com/en/hardware/Ascend910B
- Atlas 300I (product page):  https://www.hiascend.com/en/hardware/atlas-300i
Notes
- These pages do not explicitly list FP8 or detailed INT8/BF16 figures; keep `precision_support` conservative and add `deploy_note` = “需供应商确认”。

---

## Kunlunxin
- Vendor site: https://www.kunlunxin.com/
Notes
- Public, uniform P800 datasheet link not found in this session; keep bandwidth/INT8/FP8 numbers null with a “需供应商确认” note; retain FP16 from commonly cited sources only if an official PDF becomes available.

---

## Estimation Assumptions (Configs)
- `configs/estimation.yaml` holds:
  - dtype_bytes, overhead_ratio_default, alpha_default
  - moe_effective_factor (family-specific), scale_thresholds_b
  - kv_dtype_fallback_order

Traceability policy
- If a field cannot be confirmed via owner org’s config.json or vendor page, leave it null and add `notes` explaining “未公开/需确认”。
- For model matching, prefer exact repo IDs; only use `display_name/aliases` for input convenience, never for merging base and variant entries.

