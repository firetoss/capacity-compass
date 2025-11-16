# syntax=docker/dockerfile:1.6

# Multi-stage build to produce a compact runtime image

FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /build

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy project metadata and sources
COPY pyproject.toml uv.lock* ./
COPY src ./src

# Build wheels for runtime deps (api + llm extras)
RUN python -m pip install --upgrade pip && \
    pip wheel --wheel-dir /wheels .[api,llm]


FROM python:3.11-slim AS runtime
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=9050
WORKDIR /app

# Install wheels produced in builder stage
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application sources and read-only configs/prompts
COPY src ./src
COPY configs ./configs
COPY prompts ./prompts

# Expose service port (default 9050)
EXPOSE 9050

# Healthcheck (simple TCP)
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD python - <<'PY' || exit 1 \
import socket, os
host='127.0.0.1'; port=int(os.environ.get('PORT', '9050'))
s=socket.socket(); s.settimeout(2)
try:
    s.connect((host, port))
    s.close()
    print('ok')
except Exception as e:
    print('fail', e)
    raise SystemExit(1)
PY

# Default: uvicorn serving the FastAPI app
CMD ["uvicorn", "capacity_compass.api.server:app", "--host", "0.0.0.0", "--port", "9050"]

