# Agent 行为规范（面向 GPT-5-Codex）

本项目：**CapacityCompass** —— LLM 推理算力评估与硬件推荐服务。  
你是负责实现代码的智能体，需要严格遵守本文中的约束。

---

## 1. 总体原则

1. **以设计文档为唯一设计真相源**

   - 设计文档位于 `docs/` 目录（例如：`docs/capacity_compass_design_spec.md`）。
   - 在编写或修改代码前，必须先完整阅读并理解设计文档。
   - 任何业务逻辑、数据结构或公式，如与设计文档冲突，以设计文档为准。

2. **配置文件为只读事实源**

   所有硬件与模型规格由 `configs/` 下的 YAML 文件提供（命名示例）：

   - `configs/hardware.yaml`
   - `configs/models_qwen3.yaml`
   - `configs/models_deepseek.yaml`
   - `configs/scenarios.yaml`
   - `configs/e2e_test_cases.yaml`

   约束：

   - 运行时只能读取这些配置，不得在代码中写回或修改 YAML 文件。
   - 禁止在业务逻辑中硬编码具体显卡型号或模型参数值，应通过配置加载。
   - 如确需新增字段，应先更新设计文档与配置 schema，再实现代码。

3. **禁止“脑补业务”**

   - 设计文档没有明确说明的地方，优先选择：
     - 简单、可解释、易回退的实现；
     - 不改变对外接口与文档约定。
   - 如需做必要假设：
     - 在代码注释中说明假设；
     - 能配置的尽量做成配置项，而不是写死在代码。

---

## 2. 代码规范

1. **语言与版本**

   - 使用 Python 3.11+。
   - 推荐依赖管理方式：`uv` / `poetry` / `pip + requirements.txt`，由仓库约定为准。

2. **项目结构（建议）**

   代码目录建议为 `src/capacity_compass/`：

   - `config_loader.py`          —— 统一加载与校验 YAML。
   - `models_registry.py`        —— 模型规格查询封装。
   - `hardware_registry.py`      —— 硬件规格查询封装。
   - `scenarios_registry.py`     —— 场景预设查询封装。
   - `pipeline/normalizer.py`    —— 阶段 1：请求归一化。
   - `pipeline/requirements.py`  —— 阶段 2：显存 & 算力估算。
   - `pipeline/hardware_filter.py` —— 阶段 3：硬件筛选。
   - `pipeline/card_sizer.py`    —— 阶段 4：卡数估算。
   - `pipeline/recommender.py`   —— 阶段 5：排序与推荐。
   - `pipeline/evaluation_builder.py` —— 阶段 6：raw_evaluation 组装。
   - `api/schemas.py`            —— 请求/响应 Pydantic 模型。
   - `api/server.py`             —— FastAPI / Flask Web 接口实现。

   测试目录为 `tests/`，存放单元测试与 e2e 测试。

3. **命名与风格**

   - 模块和文件：小写+下划线，例如 `hardware_registry.py`。
   - 类：PascalCase，例如 `HardwareRegistry`。
   - 函数与变量：小写+下划线，例如 `load_models_config()`。
   - 遵循 PEP8，推荐使用 `black` + `isort` 格式化代码。

4. **类型标注**

   - 所有对外可见函数（被其他模块调用）必须提供类型注解。
   - 数据模型（模型规格、硬件规格、请求/响应结构）优先使用 `pydantic` 或 `dataclasses` 定义。
   - 避免在核心计算逻辑中使用 `Any`，除非确有必要且注释说明原因。

5. **函数职责**

   - 单个函数只做一件事：
     - `estimate_requirements()`：只负责显存和算力需求的计算；
     - `size_cards()`：只负责根据需求与硬件规格计算所需卡数；
     - `rank_candidates()`：只负责排序与推荐逻辑。
   - 尽量保持函数“纯函数”特性（输入 → 输出），减少对全局状态的依赖，方便测试。

6. **注释与文档**

   - 核心模块和公共函数必须有 docstring，简要说明用途、参数和返回值。
   - 对复杂公式或经验系数，需在 docstring 中指明：
     - 对应设计文档的章节；
     - 或外部论文/文档的来源。

7. **测试**

   - 使用 `pytest`（或团队约定框架）作为测试框架。
   - 对每个关键计算模块（`requirements`、`card_sizer`、`recommender`）至少编写一个单元测试文件。
   - 编写 `tests/test_e2e.py`，使用 `configs/e2e_test_cases.yaml` 做端到端测试：
     - 加载一组测试请求；
     - 调用评估接口；
     - 对返回结构中的关键字段进行断言。
   - 强制要求（核心实现必须有单测）：
     - M1–M3 模块（`config_loader`、三个 registry、`normalizer`、`requirements`、`hardware_filter`、`card_sizer`、`recommender`、`evaluation_builder`）的公开函数，均需最小单元测试覆盖；缺失单测的改动不得合并。
     - 单测文件命名：`tests/test_<module>.py`；用例命名清晰、可读、可重复执行；避免依赖网络。
     - 端到端测试必须存在并可通过（允许对数值作区间/存在性断言，不要求精确到常数）。
     - 日志与异常路径需有至少 1 个用例覆盖（例如：模型未命中走“匿名 dense”、上下文被截断的提示）。

8. **提交前检查（必须通过）**

   - 本地运行 `pytest -q` 全部通过；
   - 不使用 `print()`；日志通过 `logging`；
   - 未改动 `configs/*.yaml` 的事实数据（读取只读原则）；
   - 返回结构与 `docs/capacity_compass_design_spec.md` 一致（含可选 `quick_answer`、`scenes.*.sales_summary/guide` 字段的兼容性）；
   - 若接入第七步 LLM 生成，仅读取 `prompts/sales_summary_zh.txt`，并向外部 LLM 仅传“精简 JSON 子集”，不得泄露完整业务负载。

9. **研发任务拆分**

   - 以 `docs/dev_tasks.md` 为执行清单的唯一基线；按 M0→M6 顺序推进；
   - 任何新增模块/依赖，需先更新设计文档/任务清单，再编码。

---

## 3. 日志规范（重点）

统一使用 Python 标准库 `logging`，**禁止在业务代码中使用 `print()`** 输出日志或调试信息。

1. **logger 使用方式**

   在每个模块顶部定义：

   ```python
   import logging
   logger = logging.getLogger(__name__)
	•	日志配置（handler / formatter / level）由应用入口负责；
	•	各模块只负责合理调用 logger。

2.	**日志级别约定**
	•	logger.info：
	•	记录重要阶段事件：
	•	配置加载成功或失败；
	•	每个评估请求的关键摘要（模型名、精度、场景列表等）；
	•	每个场景的推荐结果生成完成（可记录推荐的主力 GPU 型号列表）。
	•	logger.debug：
	•	记录关键中间值（便于排障）：
	•	weights_mem_bytes、kv_mem_bytes、total_mem_bytes；
	•	required_tflops、single_card_tflops；
	•	cards_mem、cards_compute、cards_needed、headroom 等。
	•	避免过于啰嗦的循环级别日志。
	•	logger.warning：
	•	用于非致命异常：
	•	某 GPU 算力数据缺失，只能按显存推荐；
	•	某模型结构字段缺失，仅按参数量粗估；
	•	配置文件存在未知字段，但可以忽略继续执行。
	•	logger.error：
	•	用于严重错误：
	•	YAML 加载失败或 schema 校验失败；
	•	无任何可用硬件满足最小显存需求；
	•	内部异常导致无法完成评估流程。
3.	**隐私与敏感信息**
	•	避免在日志中记录完整业务请求体或用户敏感内容。
	•	当前项目主要记录：
	•	模型名称；
	•	参数量；
	•	上下文长度；
	•	精度与厂商范围；
	•	推荐的硬件 ID/数量。

⸻

## 4. 实现优先级建议
1.	**先实现配置加载与校验**
	•	在 config_loader.py 中实现统一的 YAML 加载逻辑：
	•	负责读取 configs/ 下所有 YAML；
	•	进行基本字段存在性与类型校验；
	•	出错时抛出清晰异常，并记录 logger.error。
	•	在 models_registry.py / hardware_registry.py / scenarios_registry.py 中封装常用查询接口。
2.	**实现评估流水线（7 阶段）**
	•	按设计文档划分阶段模块：
	•	normalizer：请求归一化；
	•	requirements：显存 & 算力需求估算；
	•	hardware_filter：候选硬件筛选；
	•	card_sizer：卡数估算；
	•	recommender：排序与推荐；
	•	evaluation_builder：组装 raw_evaluation 结果。
	•	每个阶段模块提供一个清晰的主入口函数，便于单独测试。
3.	**实现 API 层**
	•	使用 FastAPI（或团队约定框架）实现 HTTP 接口：
	•	POST /api/llm/capacity/evaluate
	•	在 api/schemas.py 中使用 Pydantic 定义请求和响应模型。
	•	在 api/server.py 中组装路由、依赖注入和异常处理。
4.	**实现并行的单元测试和 e2e 测试**
	•	每实现完一个阶段模块，立即补上单元测试；
	•	在整体串联完成后，补充 test_e2e.py，确保主要场景跑通。

⸻

## 5. 行为边界
1.	不要修改设计文档中已经约定好的对外 API 形状和字段命名。
2.	不要添加新的必填字段，除非设计文档已经更新。
3.	允许添加可选字段或内部辅助结构，但必须保证向后兼容。
4.	如需引入新的第三方依赖，应在代码注释中说明用途，并尽量控制依赖数量。

在整个开发过程中，请始终遵守本 Agent 规范与设计文档约定，避免自发更改对外接口与行为。
