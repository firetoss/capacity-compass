# syntax=docker/dockerfile:1.6

# Multi-stage build to produce a compact runtime image

FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /build
ARG PIP_INDEX_URL="https://mirrors.ustc.edu.cn/pypi/web/simple"
ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG no_proxy
ARG NO_PROXY
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    http_proxy=${http_proxy} https_proxy=${https_proxy} \
    HTTP_PROXY=${HTTP_PROXY} HTTPS_PROXY=${HTTPS_PROXY} \
    no_proxy=${no_proxy} NO_PROXY=${NO_PROXY}

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

# Healthcheck: simple TCP connect to service port
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD python -c "import socket,os; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', int(os.environ.get('PORT','9050')))); s.close()" || exit 1

# Default: uvicorn serving the FastAPI app
CMD ["uvicorn", "capacity_compass.api.server:app", "--host", "0.0.0.0", "--port", "9050"]
