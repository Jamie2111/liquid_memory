# =============================================================================
# Liquid Memory, containerised gateway.
# -----------------------------------------------------------------------------
# Two-stage build keeps the runtime image small (~1.2 GB without vLLM,
# ~6 GB with) by separating wheel-build artefacts from the runtime layer.
#
# Build:
#     docker build -t liquidmemory/proxy:1.0.0 -t liquidmemory/proxy:latest .
#
# Run (cloud-only synthesis, the common case):
#     docker run --rm -p 8000:8000 \
#       -e OPENAI_API_KEY=sk-... \
#       -e SYNTHESIS_MODEL=gpt-4.1 \
#       liquidmemory/proxy:latest
#
# Run with GPU (full self-hosted stack: local extractor + cloud synthesis):
#     docker run --rm --gpus all -p 8000:8000 \
#       -e OPENAI_API_KEY=sk-... \
#       liquidmemory/proxy:gpu
#     (Use the cuda tag below by building with --build-arg BASE=nvidia/cuda:...)
#
# Health check from the host:
#     curl http://localhost:8000/healthz
# =============================================================================

ARG BASE=python:3.11-slim
ARG VLLM_EXTRA=""

# -----------------------------------------------------------------------------
# Stage 1: build the wheel from the working source.
# -----------------------------------------------------------------------------
FROM ${BASE} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /src

# System deps needed to compile any wheels that don't ship binaries
# for slim. Kept minimal so the builder image stays under 300 MB.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the metadata first so docker layer-caches dependency
# installs across code changes.
COPY pyproject.toml README.md ./
COPY liquid_memory ./liquid_memory
COPY dist_public ./dist_public

RUN python -m pip install --upgrade pip build \
    && python -m build --wheel --outdir /wheels

# -----------------------------------------------------------------------------
# Stage 2: runtime image.
# -----------------------------------------------------------------------------
FROM ${BASE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HYBRID_PROXY_HOST=0.0.0.0 \
    HYBRID_PROXY_PORT=8000

# curl is only here so the HEALTHCHECK below can run; nothing else
# in the runtime needs system packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user so the gateway never runs as PID 1 root in production.
RUN useradd --create-home --shell /bin/bash --uid 1000 lm
WORKDIR /home/lm
USER lm

# Copy the wheel from the builder, install with optional vLLM extra.
# Pass VLLM_EXTRA=vllm at build time to bake vLLM into the image; default
# is no vLLM so the image stays small for cloud-only synthesis users.
COPY --from=builder /wheels /tmp/wheels
RUN python -m pip install --user "/tmp/wheels"/*.whl${VLLM_EXTRA:+[$VLLM_EXTRA]} \
    && rm -rf /tmp/wheels

ENV PATH="/home/lm/.local/bin:${PATH}"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

# The CLI entry point is registered by pyproject.toml's [project.scripts].
# Defaults to `start` so `docker run` Just Works; you can override with
# `docker run ... liquid-memory status` for ad-hoc commands.
ENTRYPOINT ["liquid-memory"]
CMD ["start"]
