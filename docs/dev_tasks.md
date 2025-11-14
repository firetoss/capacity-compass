# CapacityCompass 开发任务拆分（稳定基线版）

目的
- 固化本期“销售级粗估”方案为可执行任务，不依赖会话上下文。
- 每个任务独立、具备输入/输出与验收标准，便于并行协作与 Code Review。

里程碑与依赖
- M0 脚手架/基础设施 → M1 配置加载/注册表 → M2 流水线各阶段 → M3 结果组装（含 Sales 视图）→ M4 LLM 生成（可选）→ M5 API → M6 测试与验收。

---

## M0 脚手架与规范
1. 目录与依赖
   - 创建 `src/capacity_compass/` 包、`tests/` 目录；添加 `__init__.py`。
   - 依赖：`pydantic>=2`, `pyyaml`, （可选）`fastapi`, `uvicorn`, （可选）`requests`（LLM 网关）。
   - 验收：可以 `import sagescale`；不引入未确认的大型依赖。

2. 日志规范
   - 统一 `logging.getLogger(__name__)`；禁止 `print()`。
   - 验收：关键模块可产生日志；不在库代码中配置 handler。

---

## M1 配置加载与注册表
1. config_loader
   - 文件：`src/capacity_compass/config_loader.py`
   - 能力：读取 `configs/*.yaml`；基本字段校验；抛出清晰异常；记录 error 日志。
   - 接口：
     - `load_hardware() -> HardwareCatalog`
     - `load_models() -> ModelsCatalog`（合并 Qwen3/DeepSeek 两文件）
     - `load_scenarios() -> dict`
     - `load_estimation() -> EstimationConfig`
   - 验收：能成功读入当前仓库的 YAML；字段缺失时抛错或置 null 并告警。

2. models_registry
   - 文件：`src/capacity_compass/models_registry.py`
   - 能力：按 `model_name` / `display_name` / `aliases` 进行匹配；提供“差距最小的上下文匹配”以选择 Instruct/长上下文变体；返回 `ModelProfile`。
   - 接口：
     - `match_model(name: str) -> list[ModelProfile]`（返回候选，解决多义项）
     - `select_best_for_context(family: str, target_ctx: int) -> ModelProfile`（在同家族下选 `max_position_embeddings` ≥ 目标、差距最小；否则选上限最高）
   - 验收：
     - 能命中基座与新增 Instruct（4B/30B‑A3B/235B‑A22B‑Instruct‑2507）。
     - 覆盖大小写、去前缀（如 `Qwen/`）和常见别名。

3. hardware_registry
   - 文件：`src/capacity_compass/hardware_registry.py`
   - 能力：按厂商/精度过滤；读取 `perf` 与 `deploy_support`。
   - 接口：`query_by_vendor_and_precision(vendors: list[str], precision: str) -> list[GPU]`
   - 验收：返回与配置一致的候选集；H20/L20 的 `fp8=true` 生效。

4. scenarios_registry
   - 文件：`src/capacity_compass/scenarios_registry.py`
   - 能力：读取 `configs/scenarios.yaml`，提供场景预设。
   - 接口：`get_default_preset(scene: str) -> ScenarioPreset`
   - 验收：返回 chat/rag/writer 的默认上下文/并发/延迟与乘子。

---

## M2 流水线实现（阶段 1–5）
1. normalizer（阶段 1）
   - 文件：`src/capacity_compass/pipeline/normalizer.py`
   - 输入：用户请求（Pydantic：model/string、max_context_len/int 可选、vendor_scope/list 可选）。
   - 逻辑：
     - 模型匹配：model_name → display_name → aliases→宽松；多义项报错；未命中且给了 `param_count_b` 则构造“匿名 dense”。
     - Instruct/长上下文切换：若命中基座且目标上下文较长，则 `select_best_for_context`；生成 `switch_notice`。
     - 上下文 clamp：`final_context = min(request_or_default, model.max_position_embeddings)`；如截断记录标记。
     - 精度：无输入→`torch_dtype`；KV dtype：推荐→回退顺序。
   - 输出：`NormalizedRequest`（model_profile/eval_precision/final_context/vendor_scope/switch_notice）。
   - 验收：对 4B‑Instruct‑2507 能被选择；超限上下文被截断并产生标记。

2. requirements（阶段 2）
   - 文件：`src/capacity_compass/pipeline/requirements.py`
   - 输入：`NormalizedRequest` + 场景预设 + estimation。
   - 逻辑：显存（权重+KV+overhead）；算力（2·P·S_tokens；MoE 仅 compute 折算）。
   - 输出：`Requirements`（weights/kv/total/required_tflops/notes）。
   - 验收：dense/MoE/VL 路径可用；notes 包含“专家分配机制已折算”。

3. hardware_filter（阶段 3）
   - 文件：`src/capacity_compass/pipeline/hardware_filter.py`
   - 输入：候选 GPU 列表 + 精度 + `weights_mem_bytes`。
   - 输出：可参与评估的 GPU 列表。
   - 验收：过滤 vendor/precision 权限；能装权重。

4. card_sizer（阶段 4）
   - 文件：`src/capacity_compass/pipeline/card_sizer.py`
   - 输入：`Requirements` + GPU。
   - 输出：每 GPU 的 `cards_mem/cards_compute/cards_needed/headroom`。
   - 验收：算力缺失时按显存给卡数并记录标记。

5. recommender（阶段 5）
   - 文件：`src/capacity_compass/pipeline/recommender.py`
   - 输入：各 GPU 评估结果 + 价格/成熟度。
   - 输出：排序后的主推与备选集合。
   - 验收：排序遵循“卡数少>有价>成熟度>冗余”。

---

## M3 结果组装（阶段 6：含 Sales 视图）
1. evaluation_builder
   - 文件：`src/capacity_compass/pipeline/evaluation_builder.py`
   - 能力：组装 `raw_evaluation`；生成 `quick_answer` 与 `scenes.*.sales_summary`；（可选）生成 `scenes.*.guide` 固定短句。
   - 体验标签：将 `target_latency_ms` 映射为 {快/中/稳}。
   - 理由短句库：显存合适/推理稳定/生态成熟/需确认生态/性价比高。
   - 输出结构：遵循文档 3.2 的“可选字段”说明。
   - 验收：最小响应示例可直接由该模块产出，`confidence=low`。

---

## M4 LLM 生成（阶段 7，可选）
1. sales_summary 生成
   - 文件：`src/capacity_compass/pipeline/llm_summary.py`
   - 能力：读取 `prompts/sales_summary_zh.txt` 与“精简 JSON 子集”，调用外部 LLM（OpenRouter/内部网关）生成文本。
   - 开关：`generate_llm_summary`（函数参数或环境变量）。
   - 适配：以最小接口封装 `post_chat_completion()`，便于替换供应商。
   - 验收：当开关打开时，`llm_summary` 非空且符合模板结构要求（人工抽查）。

---

## M5 API（可选）
1. schemas
   - 文件：`src/capacity_compass/api/schemas.py`（Pydantic）
   - 请求：model（string），param_count_b（可选），max_context_len（可选），precision（可选），vendor_scope（可选）。
   - 响应：见文档 3.2，含可选 quick_answer 与 scenes.*。

2. server
   - 文件：`src/capacity_compass/api/server.py`（FastAPI）
   - 路由：POST /api/llm/capacity/evaluate
   - 依赖：加载配置→流水线→（可选）LLM 生成→返回。
   - 验收：能返回结构化结果；在 `generate_llm_summary=false` 时仍可用。

---

## M6 测试与验收
1. 单元测试（pytest）
   - `tests/test_normalizer.py`：匹配/切换/截断。
   - `tests/test_requirements.py`：显存/算力估算（含 MoE 折算）。
   - `tests/test_recommender.py`：排序规则。

2. 端到端测试
   - `tests/test_e2e.py`：读取 `configs/e2e_test_cases.yaml`，验证“存在推荐结果、cards_needed 不大于阈值、场景存在”等。

3. 验收清单
   - 所有测试通过；日志无 print；接口字段与文档一致；可手动构造最小请求拿到 quick_answer。

---

## 补充：对接 OpenRouter（文档）
- Base URL: `https://openrouter.ai/api/v1`; Header: `Authorization: Bearer $OPENROUTER_API_KEY`。
- 建议模型：`qwen/qwen-3-8b-instruct`（以 /models 查询为准）。
- 系统提示：`prompts/sales_summary_zh.txt`；用户消息：精简 JSON 子集；`temperature≈0.3`。
- 失败降级：关掉 `generate_llm_summary`，仅返回结构化结果。
