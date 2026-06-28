# syntax=docker/dockerfile:1
# =============================================================================
# PromptGuard — multi-stage Dockerfile
#
# Two image variants:
#
#   runtime-slim  Heuristic-only (S0+S1+S3). No ML deps. ~180 MB.
#                 Zero model-loading cold start. p95 latency < 5 ms.
#                 Detection: 100 % recall on seed attacks, 14 % red-team evasion.
#
#   runtime-full  Full ML pipeline (adds Backend A embedding classifier).
#                 ~2 GB (PyTorch). Pre-downloaded embedding model + pre-trained
#                 classifier artifact baked in — no network at runtime.
#                 /readyz passes once the model is warm (~10-30 s startup).
#                 Higher recall on synonym/word-split mutations.
#
# Build commands
# --------------
#   docker build --target runtime-slim -t promptguard:slim .
#   docker build --target runtime-full -t promptguard:full .
#
# Secrets — never bake into the image.  Pass at runtime via -e or compose:
#   PROMPTGUARD_API_KEY, PROMPTGUARD_RATE_LIMIT_RPM, etc.
# =============================================================================

ARG PYTHON_VERSION=3.12
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2

# =============================================================================
# Stage: builder
# Installs ML + server deps, pre-downloads the embedding model, and trains
# the logistic-regression head on the seed dataset.  Only referenced by
# runtime-full; the slim target skips this stage entirely.
# =============================================================================
FROM python:${PYTHON_VERSION}-slim AS builder

ARG EMBEDDING_MODEL

# Build-time tools (some ML wheels need gcc)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy the full Python source tree
COPY python/ /build/

# Install ML + server extras.  The pip cache is NOT purged here so layer
# caching works well for the final-stage re-install.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "/build[ml,server]"

# ── Pre-download embedding model ──────────────────────────────────────────────
# Model weights are saved to /build/models/ and copied into runtime-full.
# After this the image needs NO network access at startup.
ENV SENTENCE_TRANSFORMERS_HOME=/build/models
RUN python3 -c "\
from sentence_transformers import SentenceTransformer; \
model='${EMBEDDING_MODEL}'; \
st = SentenceTransformer(model, cache_folder='/build/models'); \
print(f'  ok  {model!r} downloaded ({st.get_sentence_embedding_dimension()} dims)')"

# ── Train the classifier artifact ─────────────────────────────────────────────
# Produces /build/train/artifacts/classifier_a.pkl using the seed dataset.
# Swap in real public injection datasets (see train/datasets/README.md) and
# rebuild for higher recall.
RUN python3 -m train.train \
        --data-dir  train/datasets \
        --output-dir train/artifacts \
        --encoder   "${EMBEDDING_MODEL}" \
    && ls -lh train/artifacts/

# =============================================================================
# Stage: runtime-slim
# Heuristic-only image — no ML deps, minimal size.
# =============================================================================
FROM python:${PYTHON_VERSION}-slim AS runtime-slim

# curl: needed for the Dockerfile HEALTHCHECK and the CI smoke test.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (uid 1001 avoids collision with typical host uids)
RUN groupadd -r promptguard \
    && useradd -r -g promptguard -u 1001 --no-create-home promptguard

WORKDIR /app

# Copy the Python package source
COPY python/ /app/

# Install only the [server] extra (fastapi + uvicorn + pydantic-settings).
# ML deps are intentionally absent; EmbeddingClassifier falls back to
# HeuristicClassifier automatically.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "/app[server]" \
    && pip cache purge

# Ownership before switching user
RUN chown -R promptguard:promptguard /app

USER promptguard

# These can all be overridden at runtime via -e / compose environment:
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PROMPTGUARD_HOST=0.0.0.0 \
    PROMPTGUARD_PORT=8000

EXPOSE 8000

# Liveness: hits /healthz which always returns 200 while the process is alive.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -sf "http://localhost:${PROMPTGUARD_PORT:-8000}/healthz" || exit 1

CMD ["python3", "-m", "promptguard.server"]

# =============================================================================
# Stage: runtime-full
# Full ML image with pre-baked model weights and classifier artifact.
# =============================================================================
FROM python:${PYTHON_VERSION}-slim AS runtime-full

ARG EMBEDDING_MODEL

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r promptguard \
    && useradd -r -g promptguard -u 1001 --no-create-home promptguard

WORKDIR /app

# Copy source
COPY python/ /app/

# Install ML + server deps
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "/app[ml,server]" \
    && pip cache purge

# ── Bake in model weights (zero network at startup) ───────────────────────────
COPY --from=builder /build/models/ /app/models/

# ── Bake in trained classifier artifact ──────────────────────────────────────
COPY --from=builder /build/train/artifacts/ /app/train/artifacts/

RUN chown -R promptguard:promptguard /app

USER promptguard

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PROMPTGUARD_HOST=0.0.0.0 \
    PROMPTGUARD_PORT=8000 \
    # Tell sentence-transformers to load from the baked-in path
    SENTENCE_TRANSFORMERS_HOME=/app/models \
    # Tell the server which artifact to warm up on startup
    PROMPTGUARD_CLASSIFIER_MODEL_PATH=/app/train/artifacts/classifier_a.pkl

EXPOSE 8000

# Readiness: /readyz returns 503 until the embedding model is warm.
# Start period is generous because PyTorch model loading can take ~10-30 s.
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=5 \
    CMD curl -sf "http://localhost:${PROMPTGUARD_PORT:-8000}/readyz" || exit 1

CMD ["python3", "-m", "promptguard.server"]
