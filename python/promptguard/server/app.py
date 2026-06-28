"""FastAPI application factory for the PromptGuard server.

The app is intentionally structured as a factory (``create_app()``) so that
tests can spin up isolated instances with custom settings without touching
module-level globals or env vars.

Routes
──────
  GET  /healthz   Always 200. Used by load balancers to check liveness.
  GET  /readyz    200 once the classifier model is warm; 503 otherwise.
                  Use this as the Kubernetes readinessProbe target.
  POST /inspect   Inspect messages; return Verdict JSON.
  POST /protect   Inspect + sanitize; return Verdict + safe messages.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from promptguard import AppProfile, PromptGuard
from promptguard.sanitize import sanitize_messages

from .auth import RateLimiter, make_api_key_dep, make_rate_limit_dep
from .config import Settings, get_settings
from .models import (
    HealthResponse,
    InspectRequest,
    InspectResponse,
    Message,
    ProtectRequest,
    ProtectResponse,
    ReadyResponse,
    VerdictOut,
)

# ---------------------------------------------------------------------------
# Server state (one instance per worker process)
# ---------------------------------------------------------------------------


class _ServerState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.guard = PromptGuard(profile=settings.to_app_profile())
        self.limiter: RateLimiter | None = (
            RateLimiter(settings.rate_limit_rpm) if settings.rate_limit_rpm > 0 else None
        )
        self.classifier_backend: str = "heuristic"
        self.ready: bool = False

    async def initialize(self) -> None:
        """Warm up the classifier (if configured) and flip the readiness flag."""
        path = self.settings.classifier_model_path_obj
        if path and path.exists():
            try:
                from promptguard.stages.classifier import EmbeddingClassifier

                clf = EmbeddingClassifier(model_path=path)
                clf.predict("warmup")  # triggers lazy load
                self.classifier_backend = "ml"
            except Exception:
                pass  # fall back to heuristic silently
        self.ready = True


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and return a configured FastAPI instance.

    Args:
        settings: If None, settings are loaded from env / .env file.
    """
    cfg = settings or get_settings()
    state = _ServerState(cfg)

    # ── Lifespan ───────────────────────────────────────────────────────────
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await state.initialize()
        yield

    app = FastAPI(
        title="PromptGuard",
        description="Runtime prompt-injection defense API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Request size limit middleware ──────────────────────────────────────
    @app.middleware("http")
    async def _body_size_limit(request: Request, call_next: Any) -> Response:
        cl = request.headers.get("content-length")
        if cl and int(cl) > cfg.max_request_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": (
                        f"Request body exceeds limit of {cfg.max_request_bytes} bytes. "
                        "Split your messages or increase PROMPTGUARD_MAX_REQUEST_BYTES."
                    )
                },
            )
        return await call_next(request)

    # ── Shared dependencies ────────────────────────────────────────────────
    auth_dep = make_api_key_dep(cfg)
    rate_dep = make_rate_limit_dep(state.limiter)

    # ── Routes ─────────────────────────────────────────────────────────────

    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    async def healthz() -> HealthResponse:
        """Liveness probe — always returns 200 while the process is alive."""
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=ReadyResponse, tags=["ops"])
    async def readyz() -> ReadyResponse:
        """Readiness probe — 503 until the classifier model is warm."""
        if not state.ready:
            raise HTTPException(status_code=503, detail="classifier model not yet loaded")
        return ReadyResponse(status="ready", classifier=state.classifier_backend)

    @app.post("/inspect", response_model=InspectResponse, tags=["detection"])
    async def inspect(
        body: InspectRequest,
        _auth: None = Depends(auth_dep),
        _rate: None = Depends(rate_dep),
    ) -> InspectResponse:
        """Inspect messages for prompt-injection signals.

        Inspects the **last user-role message** in the conversation.
        Returns the full Verdict including per-rule findings and intent score.

        The caller decides what to do with the verdict; no input is modified.
        """
        guard = _resolve_guard(state, body.profile)
        text = body.last_user_text()
        verdict = guard.inspect(text)

        return InspectResponse(
            verdict=VerdictOut.from_verdict(verdict),
            request_id=str(uuid.uuid4()),
            inspected_text_preview=text[:80],
        )

    @app.post("/protect", response_model=ProtectResponse, tags=["detection"])
    async def protect(
        body: ProtectRequest,
        _auth: None = Depends(auth_dep),
        _rate: None = Depends(rate_dep),
    ) -> ProtectResponse:
        """Inspect and sanitize messages.

        - ``action == "allow"`` → safe_messages == original messages.
        - ``action == "sanitize"`` → safe_messages has spotlight markers injected
          and the SPOTLIGHT_SYSTEM_NOTE prepended to the system message.
        - ``action == "flag"`` → safe_messages == original messages (caller may
          want to add a warning or lower downstream tool permissions).
        - ``action == "block"`` → HTTP 400 (when block_on_block=true, default) or
          200 with verdict (when block_on_block=false); safe_messages is empty.
        """
        profile = _resolve_profile(state, body.profile)
        guard = _resolve_guard(state, body.profile)

        text = body.last_user_text()
        verdict = guard.inspect(text)

        if verdict.blocked and body.block_on_block:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "PromptGuard blocked this input",
                    "verdict": VerdictOut.from_verdict(verdict).model_dump(),
                },
            )

        # Build safe messages
        if verdict.blocked:
            safe_messages = []  # nothing forwarded when blocked (even without HTTP 400)
        elif verdict.action == "sanitize":
            raw_messages = [{"role": m.role, "content": m.content} for m in body.messages]
            sanitized, _ = sanitize_messages(raw_messages, profile)
            safe_messages = [Message(role=m["role"], content=m["content"]) for m in sanitized]
        else:
            safe_messages = list(body.messages)

        return ProtectResponse(
            verdict=VerdictOut.from_verdict(verdict),
            request_id=str(uuid.uuid4()),
            safe_messages=safe_messages,
        )

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_profile(state: _ServerState, profile_in: Any) -> AppProfile:
    """Return an AppProfile: per-request override if provided, else server default."""
    if profile_in is not None:
        return profile_in.to_app_profile()
    return state.settings.to_app_profile()


def _resolve_guard(state: _ServerState, profile_in: Any) -> PromptGuard:
    """Return a PromptGuard: fresh instance for per-request profile, shared otherwise."""
    if profile_in is not None:
        return PromptGuard(profile=profile_in.to_app_profile())
    return state.guard


# ---------------------------------------------------------------------------
# Module-level instance for uvicorn direct invocation
# e.g.  uvicorn promptguard.server.app:app
# ---------------------------------------------------------------------------

app = create_app()
