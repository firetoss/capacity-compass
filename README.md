# CapacityCompass

## Environment Setup

1. Copy `.env.example` to `.env.dev` (already gitignored) and fill in:
   - `OPENROUTER_API_KEY`
   - `OPENROUTER_REFERER` / `OPENROUTER_SITE_TITLE` (optional telemetry headers)
   - `CAPACITYCOMPASS_ENABLE_SUMMARY` (`true/false`, default `false`)
   - `CAPACITYCOMPASS_PROMPT_PATH` (optional, defaults to `prompts/sales_summary_zh.txt`)
   - `CAPACITYCOMPASS_LLM_BASE_URL` (default `https://openrouter.ai/api/v1`)
   - `CAPACITYCOMPASS_LLM_TIMEOUT` (seconds, default `30`)
   - `CAPACITYCOMPASS_HTTP_PROXY` (optional HTTP proxy for outbound LLM calls)
   - `LOG_LEVEL` (`DEBUG/INFO/WARNING`, default `INFO`)
2. Before running the API/server, load the env file:

```bash
source .env.dev
uv run uvicorn capacity_compass.api.server:app --reload
```

The FastAPI layer reads the env switches and decides whether to call Qwen3‑8B (OpenRouter) and what log verbosity to use. When `generate_summary` is omitted in requests, the environment flag drives the default behavior.
