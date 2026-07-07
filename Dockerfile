# NYXARA — sovereign cognitive architecture. Container image for the API server.
#
# Build:  docker build -t nyxara .
# Run:    docker run -p 8000:8000 \
#             -e NYXARA_SERVER__API_TOKEN=change-me \
#             nyxara
#
# Out of the box NYXARA answers through her deterministic reasoner. The LLM stack is fully
# local (TinyLlama-1.1B via HuggingFace — no API keys); the heavy ML extras (llm/foundry/
# senses) are intentionally omitted to keep the image lean — add them by extending this file
# (pip install -e ".[llm,foundry]") for in-container TinyLlama inference and LoRA tuning.

FROM python:3.11-slim AS base

# Don't write .pyc, flush stdout/stderr (clean container logs), no pip cache.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Persist NYXARA's long-term memory under a mountable volume.
    NYXARA_HOME=/data/.nyxara \
    # Bind all interfaces inside the container; publish with -p on the host.
    NYXARA_SERVER__HOST=0.0.0.0 \
    NYXARA_SERVER__PORT=8000

WORKDIR /app

# Install dependencies first (better layer caching) — copy only metadata, then the source.
COPY pyproject.toml README.md ./
COPY nyxara ./nyxara

# Core + server + LLM providers + reasoning/security/observability. No heavy ML.
RUN pip install --no-cache-dir ".[server,llm,reasoning,security,observe]"

# Run as a non-root user; give it the data dir.
RUN useradd --create-home --uid 10001 nyxara \
    && mkdir -p /data/.nyxara \
    && chown -R nyxara:nyxara /data
USER nyxara
VOLUME ["/data"]

EXPOSE 8000

# Liveness probe hits the unauthenticated health route.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status==200 else 1)"

CMD ["nyxara-serve"]
