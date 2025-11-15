# 抽取结果预览（在线调用）

> 参数：timeout=20.0s, retries=2, retry_wait=3.0s, gap=5.0s

01. 问题：用Qwen3-8B推理，16K上下文，BF16，单卡可以吗？
    提取：{'model': 'Qwen3-8B', 'precision': 'bf16', 'context_len': 16000}
02. 问题：打算先用深度求索V3，fp16 就行，上下文 4k。
    提取：{'model': 'DeepSeek-V3', 'precision': 'fp16', 'context_len': 4000}
03. 问题：普通聊天，不确定用哪个模型，8千字就够了，精度随意。
    提取：{'model': None, 'precision': None, 'context_len': 8000}
04. 问题：qwen3 8b instruct，FP8，试试 32K 上下文。
    提取：{'model': 'Qwen3-8B-Instruct', 'precision': 'fp8', 'context_len': 32000}
05. 问题：我们要 RAG，考虑 Qwen/Qwen3-14B，bfloat16，文档上下文 64k。
    提取：{'model': 'Qwen-Qwen3-14B', 'precision': 'bf16', 'context_len': 64000}
06. 问题：先用 Qwen3-4B 就行，precision 不限，上下文 10万。
    提取：{'model': 'Qwen3-4B', 'precision': None, 'context_len': 100000}
07. 问题：DeepSeek-R1，精度 fp16，context 大概 128k。
    提取：{'model': 'DeepSeek-R1', 'precision': 'fp16', 'context_len': 128000}
08. 问题：模型没定，要求 FP16，4k tokens 够吗？
    提取：{'model': None, 'precision': 'fp16', 'context_len': 4000}
09. 问题：Qwen3-0.6B，int8 量化可以吗？8K 上下文。
    提取：{'model': 'Qwen3-0.6B', 'precision': 'int8', 'context_len': 8000}
10. 问题：用 Llama 3.1 8B，bf16，16k。
    提取：{'model': None, 'precision': 'bf16', 'context_len': 16000}
11. 问题：请推荐模型，先假定 6k 上下文，bfloat16。
    提取：{'model': None, 'precision': 'bf16', 'context_len': 6000}
12. 问题：DeepSeek-V3，FP8 有优势吗？我们希望 32k 上下文。
    提取：{'model': 'DeepSeek-V3', 'precision': 'fp8', 'context_len': 32000}
13. 问题：想上 Qwen3-32B，bf16 精度，目标 128k 上下文。
    提取：{'model': 'Qwen3-32B', 'precision': 'bf16', 'context_len': 128000}
14. 问题：暂不确定模型，先按 12K context，fp16 看。
    提取：{'model': None, 'precision': 'fp16', 'context_len': 12000}
15. 问题：Qwen/Qwen3-30B-A3B-Instruct-2507，BF16，64K。
    提取：{'model': 'Qwen-Qwen3-30B-A3B-Instruct-2507', 'precision': 'bf16', 'context_len': 64000}
16. 问题：Qwen3-8B，精度随便，2万字的输入。
    提取：{'model': 'Qwen3-8B', 'precision': None, 'context_len': 20000}
17. 问题：DeepSeek R1，bfloat16，1万字够不够？
    提取：{'model': 'DeepSeek-R1', 'precision': 'bf16', 'context_len': 10000}
18. 问题：Qwen3 4B，fp-8，8K。
    提取：{'model': 'Qwen3-4B', 'precision': 'fp8', 'context_len': 8000}
19. 问题：模型未知，目标 20k tokens，bf16。
    提取：{'model': None, 'precision': 'bf16', 'context_len': 20000}
20. 问题：Qwen3-235B-A22B，FP16，上下文 200k。
    提取：{'model': 'Qwen3-235B-A22B', 'precision': 'fp16', 'context_len': 200000}
