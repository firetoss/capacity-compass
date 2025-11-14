# SageScale / 智算参谋 —— LLM 推理算力评估与硬件推荐服务设计说明（最终版）

> 本文是对早期设计文档 `capacity_compass_design_spec.md` 的更新与收敛版本：
> - 保留原有分层结构与 7 阶段流水线；
> - 明确了最终使用的 YAML 文件结构与命名；
> - 补齐了 DeepSeek-R1 / DeepSeek-V3 / DeepSeek-VL 等模型规格的约定；
> - 对 MoE、大模型与多模态场景的处理做了更细的说明。

---

## 1. 项目概述

**项目名**：SageScale（中文名：智算参谋）  
**目标**：为 Qwen3 / DeepSeek 系列（以 instruct / thinking / distill 变体为主），在给定推理精度与上下文需求的情况下，估算推理算力与显存需求，并基于一组有限的硬件设备（来自内部 Excel 表）给出推荐的显卡型号与数量，并生成对业务同学友好的说明文档。

整体仍采用「算法层 + 报告层」两级结构：

- **算法层（本项目）**  
  - 接收结构化输入（模型名称 / 参数量 / 精度 / 上下文长度 / 厂商范围）；  
  - 读取 `configs/` 下的 YAML 配置（硬件 / 模型 / 场景）；  
  - 对显存和算力做静态估算，输出带有 raw 计算细节的 JSON 结果，并在各业务场景下给出优选/次选硬件方案。  

- **报告层（上层系统 + Qwen3‑8B）**  
  - 使用算法层输出的 JSON 作为上下文；  
  - 调用大模型生成「给业务看」的中文说明报告；  
  - 报告层不在本项目代码内部实现，只约定输入/输出格式。

---

## 2. 配置与数据文件（最终布局）

所有事实数据统一放在 `configs/` 目录下，设计与约定文档放在 `docs/`：

```text
configs/
  hardware.yaml
  models_qwen3.yaml
  models_deepseek.yaml
  scenarios.yaml
  estimation.yaml
  e2e_test_cases.yaml

docs/
  DESIGN.md        # 本文
  Agent.md         # 给 Codex 的行为&代码规范
```

### 2.1 hardware.yaml（硬件规格）

`configs/hardware.yaml` 描述 Excel 中出现的所有算力设备（NVIDIA / Huawei / Kunlunxin 等）。

关键字段（与之前版本保持一致）：

- `id`: 硬件唯一标识（推荐：小写+下划线，如 `nvidia_h20_96g`）。  
- `name`: 展示名称，例如 `"NVIDIA H20 96GB SXM"`。  
- `vendor`: `"NVIDIA" | "Huawei" | "Kunlunxin" | ...`。  
- `family`: 如 `"Ampere" | "Hopper" | "Ascend 910B"` 等。  
- `category`: `"datacenter" | "consumer" | "domestic"` 等。  
- `memory_gb`: 显存容量（GB）。  
- `memory_type`: `"HBM2" | "HBM3" | "GDDR6X" | ...`。  
- `memory_bw_gbps`: 显存带宽（GB/s），可为 null。  
- `power_w`: 典型功耗（W），可为 null。  

精度与算力：

- `precision_support`:
  - `fp16/bf16/fp8/int8`: `true | false | null`。  
- `perf`:
  - `fp16_tflops`, `bf16_tflops`, `fp8_tflops`, `int8_tops` 等，可为 null。  

价格与部署支持：

- `pricing`:
  - `currency`: `"CNY"` 等；
  - `price`: 单卡参考价格（来自 Excel 或人工补充，可为 null）；
  - `source`: `"internal_sheet" | "estimate" | "vendor_quote"` 等；
  - `note`: 文本说明。  
- `deploy_support`: 针对模型家族的简化支持等级，例如：
  - `"native" | "excellent" | "good" | "test" | "entry_level_only" | "unknown"`。  
- `deploy_note`: 文本说明，例如是否有 Qwen3 / DeepSeek 的官方或社区部署案例。

> 缺失字段填 `null`，并在 `note` 或 `deploy_note` 里标「未公开 / 需补充」。

---

### 2.2 models_qwen3.yaml / models_deepseek.yaml（模型规格）

**本次收敛后的约定：**

- Qwen3 与 DeepSeek 使用**同一套 schema**；  
- 文件名分别为：
  - `configs/models_qwen3.yaml`
  - `configs/models_deepseek.yaml`
- YAML 结构统一为：

```yaml
models:
  - family: "Qwen3" | "DeepSeek-R1" | "DeepSeek-V3" | "DeepSeek-VL" | ...
    model_name: "Qwen/Qwen3-8B"                  # HF/ModelScope 仓库名（唯一主键）
    display_name: "Qwen3-8B-Instruct"            # 给非专业用户看的名称
    aliases: ["Qwen3-8B", "Qwen/Qwen3-8B-Instruct"]
    modality: "text" | "text_vision" | "text_audio" | "omni"
    param_count_b: 8.0                           # 总参数量（B）；无法确定时为 null
    is_moe: false                                # 是否 MoE 架构
    vocab_size: 151936                           # 词表大小；无法确定时为 null

    hidden_size: 4096
    intermediate_size: 12288
    num_hidden_layers: 36
    num_attention_heads: 32
    num_key_value_heads: 8
    head_dim: 128                                 # 对 MLA/特殊注意力模型可视作近似值
    max_position_embeddings: 40960                # 来自 config.json；无法确定时为 null

    rope_scaling: { ... }                         # 若 config 中存在则按原样记录，否则为 null
    tie_word_embeddings: false

    # MoE 相关：对 DeepSeek-V3 / R1 / 大型 MoE 模型重点填写
    num_experts: 256
    top_k: 8
    expert_intermediate_size: 2048

    torch_dtype: "bfloat16"
    use_cache: true

    supports_fp8: true | false | null
    supports_bf16: true | false | null
    supports_fp16: true | false | null
    supports_int8: true | false | null

    recommended_kv_dtype: "bfloat16" | "fp8" | null

    # 可选：量化支持（用于说明官方/组织发布的量化工件）
    quantization_support: ["fp8", "awq_4bit", "gptq_4bit", "int8_weight_only"]

    notes: "文本，说明字段来源及『未公开』情况"
```

#### 2.2.1 DeepSeek 最终版模型列表说明

`configs/models_deepseek.yaml` 的最终版内容为：

- **DeepSeek-R1-Distill-Qwen 系列（蒸馏版）** —— 来自 `models_deepseek_r1.yaml` 的四个条目，保持原始结构参数：fileciteturn1file1  
  - `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`  
  - `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`  
  - `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`  
  - `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`  

- **DeepSeek-R1（全量模型，推理型）** —— 新增条目，结构字段完全来自 HF 官方 `config.json`，参数规模来自公开论文与技术分析文章：fileciteturn1file2turn3file0turn5view0turn1search1turn1search11turn2search6turn2search8turn1search6  

  - `family: "DeepSeek-R1"`  
  - `model_name: "deepseek-ai/DeepSeek-R1"`  
  - `display_name: "DeepSeek-R1"`  
  - `modality: "text"`  
  - `param_count_b: 671.0`（总参数量 671B；激活参数约 37B 写在 `notes` 中说明）  
  - `is_moe: true`  
  - `hidden_size: 7168`  
  - `intermediate_size: 18432`  
  - `num_hidden_layers: 61`  
  - `num_attention_heads: 128`  
  - `num_key_value_heads: 128`  
  - `head_dim: 128`（来自 `qk_nope_head_dim` / `v_head_dim`）  
  - `max_position_embeddings: 163840`（来自 config.json 的 `max_position_embeddings`）  
  - `num_experts: 256`（`n_routed_experts`）  
  - `top_k: 8`（`num_experts_per_tok`）  
  - `expert_intermediate_size: 2048`（`moe_intermediate_size`）  
  - `torch_dtype: "bfloat16"`  
  - `use_cache: true`  
  - `supports_fp8: true`（quantization_config 中 `quant_method: "fp8"`）  
  - 其余 fp16/int8 支持无法从 config 直接确认，保留为 `null`，并在 `notes` 中说明。  

- **DeepSeek-V3（全量 MoE 模型）** —— 与 R1 架构一致，结构参数来自 HF `deepseek-ai/DeepSeek-V3` 的 `config.json` 与技术报告：fileciteturn1file2turn5view0turn2search1turn2search6turn2search8  

  - `family: "DeepSeek-V3"`  
  - `model_name: "deepseek-ai/DeepSeek-V3"`  
  - `display_name: "DeepSeek-V3"`  
  - `modality: "text"`  
  - 结构字段与 DeepSeek-R1 相同（hidden_size / layers / heads / experts 等），`param_count_b` 设为 671.0。  
  - `notes` 中强调：  
    - 总参数量 671B，激活参数约 37B；  
    - 采用 Multi-head Latent Attention（MLA）与 DeepSeekMoE；  
    - KV cache 实际占用和头维度与标准 MHA 存在差异，本服务中仅用于近似估算。  

- **DeepSeek-VL-7B-Instruct（多模态示例）** —— 保留为一个代表性条目：fileciteturn1file2  

  - `family: "DeepSeek-VL"`  
  - `model_name: "deepseek-ai/DeepSeek-VL-7B"`  
  - `display_name: "DeepSeek-VL-7B-Instruct"`  
  - `modality: "text_vision"`  
  - `param_count_b: 7.0`，`encoder_param_b` 等信息只在 `notes` 中做说明；  
  - 结构参数（hidden_size / num_layers 等）暂留 null，并明确标注「需根据官方文档补充」。  

> 说明：  
> - 之前版本中存在 `models_deepseek.yaml` 与 `models_deepseek_r1.yaml` 两套结构，本次收敛为 **单一文件 `models_deepseek.yaml`**，schema 以 `models_deepseek_r1.yaml` 的精细字段为准，并补充了 `modality`。  
> - 旧版本中使用的 `id/base_repo/variant/source` 等字段若确有需求，可在后续迭代中以兼容方式追加；当前实现中不强制依赖。

---

### 2.3 scenarios.yaml（业务场景预设）

`configs/scenarios.yaml` 描述业务场景预设，与前一版设计保持基本一致，但需要支持多模态的算力调整。

示例结构：

```yaml
chat:
  label: "对话型助手"
  description: "用户与模型轮流对话，输入中等篇幅文本，输出多轮回答。"
  default_context_len: 8192
  target_latency_ms: 1000
  target_concurrency_per_gpu:
    small_model: 16
    medium_model: 8
    large_model: 4
  compute_multiplier:
    text: 1.0
    text_vision: 1.1
    omni: 1.2

rag:
  label: "检索增强问答（RAG）"
  description: "先检索文档或向量库，再在较长上下文上进行问答。"
  default_context_len: 32768
  target_latency_ms: 2000
  target_concurrency_per_gpu:
    small_model: 8
    medium_model: 4
    large_model: 2
  compute_multiplier:
    text: 1.0
    text_vision: 1.3
    omni: 1.5

writer:
  label: "长文生成"
  description: "生成报告、总结、营销文案等较长文本。"
  default_context_len: 8192
  target_latency_ms: 3000
  target_concurrency_per_gpu:
    small_model: 4
    medium_model: 2
    large_model: 2
  compute_multiplier:
    text: 1.0
    text_vision: 1.0
    omni: 1.1
```

> 模型规模分档建议保持原方案：  
> - small：param_count_b < 10  
> - medium：10 ≤ param_count_b ≤ 50  
> - large：> 50  

---

### 2.4 e2e_test_cases.yaml（端到端测试）

`configs/e2e_test_cases.yaml` 的使用方式沿用旧方案：定义请求与期望条件，由 `tests/test_e2e.py` 读取并执行断言。

---

### 2.5 estimation.yaml（估算常量）

新增文件 `configs/estimation.yaml`，用于承载估算相关的可调常量，避免在代码中硬编码：

- `dtype_bytes`: 各精度字节数（fp16/bf16/fp8/int8）。
- `overhead_ratio_default`: 显存开销系数（默认 0.15）。
- `alpha_default`: 吞吐估算系数 α（默认 0.25）。
- `moe_effective_factor`: MoE 计算有效参数占比（可按 family 定制）。
- `scale_thresholds_b`: small/medium 分档阈值（B）。
- `kv_dtype_fallback_order`: KV dtype 选择顺序（优先 recommended）。

---

## 3. 服务 API 设计

### 3.1 请求

`POST /api/llm/capacity/evaluate`

```jsonc
{
  "model_name": "Qwen/Qwen3-8B",             // 建议：仓库名（唯一主键）；
  "display_name": "Qwen3-8B-Instruct",       // 允许：用户常用名；也支持 aliases 匹配
  "param_count_b": 8,                         // 可选：模型表有则忽略
  "max_context_len": 8192,                    // 可选：不填则按场景默认值
  "precision": "fp16",                       // 可选：不填则用模型默认精度
  "vendor_scope": ["NVIDIA","Huawei","Kunlunxin"]  // 可选：不填表示全部候选
}
```

### 3.2 响应（结构）

```jsonc
{
  "quick_answer": { /* 可选：销售级一眼答案（默认参考 chat 场景） */ },
  "input": { /* 输入回显 + 归一化后的模型信息 */ },
  "scenarios": {
    "chat": {
      /* chat 场景下的预设与推荐 */
      /* 可选：仅面向展示的销售视图（三行内可读短句） */
      "sales_summary": { /* 主推/备选/提示/置信度/切换说明 */ },
      /* 可选：教育式文案（固定短句）：title/fit/experience/tip */
      "guide": { /* 常见用法：对话问答 */ }
    },
    "rag":  {
      /* rag 场景下的预设与推荐 */
      "sales_summary": { ... },
      "guide": { /* 常见用法：资料问答 */ }
    },
    "writer": {
      /* writer 场景下的预设与推荐 */
      "sales_summary": { ... },
      "guide": { /* 常见用法：长文写作 */ }
    }
  },
  "raw_evaluation": { /* 详细的显存/算力估算与硬件评估结果 */ },
  "llm_summary": "string",   // Qwen3-8B 生成的业务说明文本（本项目只占位）
  "disclaimers": [ "string" ]
}
```

与旧版相比，本版只对字段含义做了细化，没有改变整体结构。

---

## 4. 评估流水线（7 阶段）

流水线模块位于 `src/capacity_compass/pipeline/`，阶段划分保持与旧版一致，做了少量细节收敛。

### 4.1 阶段 1：请求归一化（normalizer）

输入：原始请求 JSON。  
输出：

- `model_profile`: 从 `models_qwen3.yaml` / `models_deepseek.yaml` 解析出的模型规格；  
- `eval_precision`: 实际评估使用的精度；  
- `max_context_len`: 参与评估的上下文长度。

关键行为（收敛与补充）：

1. 模型匹配（支持多入口）：
   - 先按 `model_name` 精确匹配；其次按 `display_name`、再按 `aliases`（区分大小写→不区分→去掉前缀如 `Qwen/`）；
   - 多义项时返回 400 并列出候选；
   - 均未命中且请求包含 `param_count_b` 时，构造“匿名 dense 文本模型”，结构参数留空或保守默认，在 `notes` 标注「粗略估算」。

2. 精度选择：
   - 请求中给出 `precision` 时，直接使用该精度；  
   - 否则优先使用模型的 `torch_dtype` 作为默认精度（例如 Qwen3 通常为 bf16）。

3. 上下文长度：
   - 每个场景内部再与 `scenarios.yaml` 中的 `default_context_len` 结合；  
   - 实现上可以取 `context_len = max(request.max_context_len or default, default)` 或直接以场景默认为主，本版不强制，交由实现通过配置控制。

### 4.2 阶段 2：需求估算（requirements）

输入：`model_profile`, `eval_precision`, `scenarios.yaml`。  
输出：每个场景的 `requirements`：

- `weights_mem_bytes`
- `kv_mem_bytes`
- `total_mem_bytes`
- `required_tflops`
- `overhead_ratio`
- `notes`（记录 MoE / 多模态估算假设）

**显存估算（dense 模型）：**

记：

- `P = param_count_b * 1e9`  
- `B_w`：权重数据类型字节数（fp16/bf16 = 2, int8 = 1, 其它按实际配置）；  
- `L = num_hidden_layers`（缺失时可用经验值或按参数量粗估）；  
- `H_kv = num_key_value_heads`（缺失时可退化为 `num_attention_heads`）；  
- `D_head = head_dim`（缺失时可根据 `hidden_size / num_attention_heads` 估算）；  
- `T = context_len`；  
- `C = target_concurrency_per_gpu`。  

则：

```text
weights_mem_bytes = P * B_w

kv_mem_bytes      = C * T * L * H_kv * D_head * 2 * B_kv
# 2: K/V 两份；B_kv 通常等于 B_w

total_mem_bytes   = (weights_mem_bytes + kv_mem_bytes) * (1 + overhead_ratio)
# overhead_ratio 从 estimation.yaml 读取（默认 0.15，可配置）
```

**算力估算（粗略）：**

1. 目标 token/s（经验值）：

```text
target_latency_s = target_latency_ms / 1000.0
S_tokens         = C * (T / target_latency_s) * alpha    # alpha 为经验常数（0.2–0.3）
```

2. FLOPS 需求（以 dense 模型为近似）：

```text
required_flops   = 2 * P * S_tokens        # 每个 token 约 2P 次浮点运算
required_tflops  = required_flops / 1e12
```

**MoE 模型（DeepSeek-V3 / R1 等）特殊处理：**

- 总参数量 `param_count_b` 反映 671B 规模；  
- 激活参数约 37B，只在 `notes` 中说明；  
- 估算时可以使用一个经验系数 `moe_effective_factor`（例如 0.06–0.1），通过配置或常量控制：  

```text
effective_params_b = param_count_b * moe_effective_factor
```

再将 `P` 替换为 `effective_params_b * 1e9` 进入 FLOPS 估算。  
显存方面默认“专家常驻显存”，以总参数量为准。

**多模态模型（如 DeepSeek-VL）：**

- 对于 `modality != "text"` 的模型，按 `scenarios.yaml.compute_multiplier[modality]` 调整算力需求：  

```text
required_tflops *= compute_multiplier
```

结构字段缺失时（例如 DeepSeek-VL 的 hidden_size / num_layers 目前为 null），可以：

- 显存估算只依据参数量和精度；  
- KV cache 可保守放大估计，或在 `notes` 中说明“仅按权重显存估算”。

### 4.3 阶段 3：硬件候选筛选（hardware_filter）

与旧版相比，这一阶段的逻辑做了一处**明确化**：

- **旧描述**（含糊）：  
  「预估显存：至少能装下单卡（即单卡显存 × N 卡 ≥ total_mem_bytes）」  

- **最终版约定**：  
  阶段 3 **只做“单卡是否有资格参与”过滤**，不直接确定 N 卡：

  1. 过滤 vendor：
     - 若请求带 `vendor_scope`，只保留对应厂商设备。  
  2. 过滤精度：
     - `precision_support[eval_precision] == true`。  
  3. 过滤是否能装下**权重**：
     - `gpu.memory_gb * 1e9 >= weights_mem_bytes`。

  **说明：**  
  - 不在此处引入 `total_mem_bytes` 与 N 卡数量；  
  - 真正的卡数估算在阶段 4 完成。

### 4.4 阶段 4：卡数估算（card_sizer）

对每个 `scenario × gpu` 组合：

```text
M_req = total_mem_bytes
M_gpu = gpu.memory_gb * 1e9

cards_mem     = ceil(M_req / M_gpu)
```

算力：

```text
F_req = required_tflops
F_gpu = 单卡在 eval_precision 下的可用算力（如 fp16_tflops 或 int8_tops）
cards_compute = ceil(F_req / F_gpu)         # 若 F_gpu 缺失则可置为 null
cards_needed  = max(cards_mem, cards_compute or 0)
```

显存冗余：

```text
total_mem_available = cards_needed * M_gpu
headroom            = (total_mem_available - M_req) / total_mem_available
```

若 `F_gpu` 缺失：

- 仍可给出基于显存的 `cards_mem` 与 `cards_needed`；  
- 在 `notes` 与日志中标记「算力数据缺失，仅以显存约束估算」。

### 4.5 阶段 5：排序与推荐（recommender）

排序规则沿用旧版，只是在表达上更明确：

1. `cards_needed` 升序；  
2. `total_price` 升序（价格为 null 的置于有价格之后）；  
3. `deploy_support` 等级优先级（native > excellent > good > test > entry_level_only > unknown）；  
4. `headroom` 降序。

每个场景输出：

```jsonc
{
  "preset": { ... },          // 场景预设
  "requirements": { ... },    // 显存 & 算力需求（阶段 2）
  "evaluation": {             // 每个 GPU 的详细评估
    "gpu_id": {
      "cards_mem": 1,
      "cards_compute": 1,
      "cards_needed": 1,
      "headroom": 0.35,
      "total_price": 60000,
      "deploy_support": "excellent"
    },
    ...
  },
  "primary": ["gpu_id1", "gpu_id2"],
  "secondary": ["gpu_id3", "gpu_id4"]
}
```

### 4.6 阶段 6：raw_evaluation 组装（evaluation_builder）

组装整体结构：

```jsonc
"raw_evaluation": {
  "model_profile": { ... },      // 从模型表 + 请求归一化得到
  "scenarios": {
    "chat":    { /* 如上 */ },
    "rag":     { /* 如上 */ },
    "writer":  { /* 如上 */ }
  }
}
```

同时构建更易读的 `scenarios` 字段（用于前端展示），可加入适量自然语言说明（不依赖 LLM）：

- 场景简介；  
- 并发与延迟假设；  
- 优选 / 次选方案的简短理由与警告（例如 910B 型号差异、P800 生态仍在适配等）。

### 4.7 阶段 7：LLM 报告生成（外部）

保持与旧版一致：本项目只负责为 `llm_summary` 预留位置，可选由外部服务生成。  

推荐做法（可选开关）：
- 参数：`generate_llm_summary: true|false`（实现层自定义，默认 false）。
- 模板：`prompts/sales_summary_zh.txt`（系统提示词，固定三段结构与口径）。
- 上下文：只传“精简子集”，避免把全部 `raw_evaluation` 直接喂给 LLM：
  - `user_input`：模型名/归一化名、输入上下文、截断后上下文、厂商范围；
  - `quick_answer`：主推/备选（≤2 条）、tips、confidence、switch_notice（如发生基座→Instruct/长上下文切换）；
  - `scenes.*.sales_summary`：各场景主推/备选/提示；
  - （可选）`scenes.*.guide`：教育式 1 句用途 + 1 句体验；
  - `disclaimers`：通用提示（≤2 条）。
- 推断参数：`temperature≈0.3`，限制生成长度，统一风格。
- 生成结果写入：`llm_summary`（对客中文说明）。

---

## 5. 错误处理与日志

错误处理原则沿用旧版，仅对返回字段含义做了小幅补充：

- 模型找不到且未提供 `param_count_b`：返回 400 + 错误码 `"unknown_model_and_param_missing"`。  
- 所有硬件都不满足最基本显存需求：返回 200，但在对应场景的 `primary/secondary` 为空，并在 `disclaimers` 与日志中清晰说明原因。  
- YAML 加载或 schema 校验失败：记录 error 级日志，返回 500。  

日志规范详见 `docs/Agent.md`（不在本文件展开），核心要求：

- 统一使用 `logging` 标准库；  
- 关键阶段使用 `info`，中间计算细节用 `debug`，异常但可继续用 `warning`，严重错误用 `error`；  
- 不在日志中打印完整用户业务负载。

---

## 6. 与前一版方案的差异小结

和你上传的上一版设计相比，本最终版的主要变化与收敛点如下：

1. **模型 YAML 统一与补全 DeepSeek**  
   - 旧版：`models_deepseek.yaml` 和 `models_deepseek_r1.yaml` 并存，schema 不同（一个带 `id/base_repo/variant`，一个是精细结构参数）。fileciteturn1file0turn1file2  
   - 最终版：统一为 `configs/models_deepseek.yaml`，采用精细结构参数 schema，并补齐 DeepSeek-R1 / DeepSeek-V3 / DeepSeek-VL-7B 等条目，同时保留全部 R1‑Distill‑Qwen 蒸馏模型。fileciteturn1file1turn1file2  

2. **MoE 模型与多模态模型的说明更具体**  
   - 明确了 DeepSeek-R1 / V3 的 671B 总参数 + 37B 激活参数来源，并在 notes 中说明；  
   - 对 KV cache / MLA 特性加了“仅用于近似估算”的提示，避免误解为精确内存模型。fileciteturn1file2turn1search1turn2search6turn5view0  

3. **阶段 3 筛选逻辑更清晰**  
   - 旧版文字中把 `total_mem_bytes` 和 N 卡放到了“预估显存”里，容易让人误解为在阶段 3 就确定卡数；  
   - 最终版明确：阶段 3 只做“单卡有无资格参与”的过滤（厂商 / 精度 / 能否装下权重），卡数在阶段 4 再算。fileciteturn1file0  

4. **目录与文件命名与当前代码规划对齐**  
   - 明确采用 `configs/` 与 `docs/` 的目录结构，便于上层系统引用；  
   - 设计中引用的 YAML 文件名与你现在整理的实际文件保持一致（hardware.yaml / models_qwen3.yaml / models_deepseek.yaml / scenarios.yaml / e2e_test_cases.yaml）。fileciteturn1file0  

5. **保留原有整体结构，不引入额外复杂度**  
   - API 形状、7 阶段流水线划分、测试策略等保持不变；  
   - 新增字段（如 `modality`、更细 rope_scaling）是向后兼容的增强，不强制前端或上层立刻使用。

---

## 7. 后续扩展方向

与旧版保持一致，不再展开：  
- 训练算力评估模式；  
- 多模态专用场景；  
- 用真实压测结果反向校准参数估算的经验系数；  
- 支持更多类型硬件（AMD GPU、TPU 等）。  

---

## 8. 销售级粗估模式（新增说明）

本节仅规范“对前端/售前友好”的默认行为与输出表达，不改变 API 与配置结构。

### 8.1 用户输入（最小化）
- 必填：`model`（常用名或仓库名均可）。
- 可选：`max_context_len`（不填使用场景默认）、`vendor_scope`（不填=全部）。
- 其余技术项（精度/量化/MoE 等）不暴露给用户，由系统内部默认与回退逻辑处理。

### 8.2 Instruct 补齐与匹配（重要）
- 模型表以“每个仓库一条目”为原则：基座与变体（Instruct/Long 等）不得互相当作别名合并。
- 当用户输入的名称命中“变体/Instruct”时，直接按该仓库条目评估。
- 当用户输入的名称命中“基座”但意图似为 Instruct/长上下文时：
  1) 若同家族存在 Instruct 变体，选择“与用户上下文差距最小”的 Instruct 模型（优先满足 `max_position_embeddings ≥ 用户输入`，否则选择上限最高者）；
  2) 在输出中加入 `switch_notice`：说明已从基座切换为某 Instruct 型号及理由；
  3) 若仍超出模型上限，提示“已截断至模型上限进行估算”。
- 匹配顺序：`model_name` → `display_name` → `aliases`（大小写宽松；去掉前缀如 `Qwen/` ）。多义项返回 400 并列出候选。

### 8.3 估算默认与回退
- 上下文：`context_len = min(用户或场景默认, model.max_position_embeddings)`；若截断，提示“超出上限按上限估算”。
- 精度：无输入 → 使用 `torch_dtype`；当前硬件不支持则自动回退（如 bf16→fp16）并在内部 notes 记录。
- KV 精度：优先 `recommended_kv_dtype`，否则按 `configs/estimation.yaml.kv_dtype_fallback_order`。
- MoE：仅在“算力侧”按 `moe_effective_factor` 折算；显存仍按总参数计算；对外文案避免“稀疏/MoE”术语，统一提示：“该模型采用专家分配机制，推理时仅激活一部分参数，已按经验折算计算需求”。
- 量化能力：`supports_*` 表示能力；`quantization_support` 表示官方已发布的量化工件，两者分开表达，避免混淆“能否”与“是否有发布”。

### 8.4 多卡与互联（提示）
- 说明：本方案的多卡仅作“理论容量与算力线性叠加”的粗略估算，未建模 NVLink/PCIe 等互联与服务器拓扑；实际结果会受硬件与网络影响，请联系服务提供商进行确认或压测评估。

### 8.5 排序与推荐（销售友好）
- 排序：卡数最少 > 有价格优先 > 部署成熟度（native > excellent > good > test > entry_level_only > unknown）> 冗余空间。
- 结果呈现（每场景）：
  - 主推 1 条：设备 + 卡数（例：L20 48GB × 1）、并发/响应（直接回显场景目标）、1 句理由（显存合适/生态成熟/价格合理）。
  - 备选 2–3 条：同上简短格式。
  - 提示：如上下文被截断、切换到了 Instruct 变体、价格未公开需确认等。

### 8.6 置信度与单位（统一）
- 置信度：本工具面向“销售级粗估”，默认标注为“低”（仅供参考，以压测为准）。
  - 经验偏差范围（仅供话术参考）：
    - 显存侧：±20%~±40%（结构字段完整时）；缺 KV 字段或仅按权重估算时可能更大；
    - 卡数侧：±30%~±60%（受实现效率、内核与互联影响）；极端情况下可能达到 1–2 倍误差（如长上下文误配或 MoE 假设不符）。
- 单位与取整：显存按十进制 GB 粗估；卡数向上取整。

### 8.7 可追溯性
- 重要字段来源：见 `docs/research_source_map.md`；输出中可追加“数据来源：官方配置/厂商文档”字样（不强制展示链接）。

### 8.8 输出字段建议（不改现有必选字段）
- 在每个 `scenarios.<scene>` 下可增加只读可选字段：
  - `sales_summary.primary`：主推（设备、卡数、并发/响应、简短理由）。
  - `sales_summary.alternatives`：备选列表（最多 3）。
  - `sales_summary.switch_notice`：如从基座切换为 Instruct/长上下文变体时的说明。
  - `sales_summary.tips`：提示（上下文截断、价格未公开、请联系厂商确认等）。
  - `sales_summary.confidence`：`"low"`（默认）。

---

## 9. 附录：计算公式与变量定义（固化）

通用约定
- 字节换算：显存按十进制 GB 计，`1 GB = 1e9 bytes`。
- 数据类型字节数：从 `configs/estimation.yaml.dtype_bytes` 读取（默认：bf16=2、fp16=2、fp8=1、int8=1）。
- 上下文：`final_context = min(request.max_context_len 或 场景默认, model.max_position_embeddings)`。
- 模型规模分档：`scale_thresholds_b`（默认 small_max=10、medium_max=50），用于选择场景并发 `target_concurrency_per_gpu`。

符号
- `param_count_b`：模型总参数量（B）。
- `P = param_count_b * 1e9`：参数量（个）。
- `B_w`：权重精度字节数；用评估精度（eval_precision）对应的字节数。
- `B_kv`：KV 精度字节数；优先 `model.recommended_kv_dtype`，否则按 `kv_dtype_fallback_order` 选择并取字节数。
- `L = num_hidden_layers`。
- `H_kv = num_key_value_heads`（缺失则用 `num_attention_heads`）。
- `D_head = head_dim`（缺失则用 `hidden_size / num_attention_heads` 取整）。
- `T = final_context`。
- `C`：场景并发（按分档从 `target_concurrency_per_gpu` 选）。
- `overhead_ratio`：从 `estimation.yaml.overhead_ratio_default` 取（默认 0.15）。
- `alpha`：吞吐估算系数，从 `estimation.yaml.alpha_default` 取（默认 0.25）。
- `moe_effective_factor`：MoE 有效参数系数（按 family 或 default，从 `estimation.yaml` 取）。
- `compute_multiplier`：多模态算力乘子（从 `configs/scenarios.yaml` 取）。

显存估算
1) 权重显存
```
weights_mem_bytes = P * B_w
```
2) KV 显存（结构缺失时允许记为 0，并在 notes 标注“仅按权重显存估算”）
```
kv_mem_bytes = C * T * L * H_kv * D_head * 2 * B_kv
# 2 表示 K/V 两份
```
3) 总显存
```
total_mem_bytes = (weights_mem_bytes + kv_mem_bytes) * (1 + overhead_ratio)
```

算力估算（统一以“万亿次/秒”Tx/s 为单位）
1) 目标 tokens/s（经验）
```
target_latency_s = target_latency_ms / 1000.0
S_tokens        = C * (T / target_latency_s) * alpha
```
2) 有效参数（MoE 仅影响算力，不影响显存）
```
effective_params_b = param_count_b * moe_effective_factor   # is_moe 时；否则为 param_count_b 本身
```
3) 需求算力（Tx/s）
```
required_compute_Tx = (2 * effective_params_b * 1e9 * S_tokens) / 1e12
# 2*P 为每 token 近似计算量；单位对齐为 “万亿次/秒”
```
4) 多模态乘子
```
if model.modality != "text":
    required_compute_Tx *= compute_multiplier[model.modality]
```

GPU 单卡算力（与评估精度匹配）
```
if eval_precision in {"bf16","fp16","fp8"}:
    F_gpu = gpu.perf[eval_precision + "_tflops"]    # Tx/s
elif eval_precision == "int8":
    F_gpu = gpu.perf["int8_tops"]                   # Tx/s
else:
    F_gpu = null
```

卡数估算与冗余
```
M_req = total_mem_bytes
M_gpu = gpu.memory_gb * 1e9
cards_mem     = ceil(M_req / M_gpu)

cards_compute = (F_gpu is not null) ? ceil(required_compute_Tx / F_gpu) : null
cards_needed  = max(cards_mem, cards_compute or 0)

total_mem_available = cards_needed * M_gpu
headroom            = (total_mem_available - M_req) / total_mem_available
```

异常与回退
- 结构缺失（VL/Omni 等）：`kv_mem_bytes=0`，notes 标注“仅按权重显存估算”。
- 单卡算力缺失：`cards_compute=null`，仍输出基于显存的 `cards_needed`，notes 标注“仅显存约束”。
- 上下文超限：按模型上限截断并提示。
- Instruct/长上下文切换：选择“上下文差距最小”的变体并提示 `switch_notice`。
