"""API-key auth and per-IP rate limiting (both off by default)."""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

from fastapi import Depends, HTTPException, Request
from fastapi.security.api_key import APIKeyHeader

if TYPE_CHECKING:
    from promptguard.server.config import Settings

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# ---------------------------------------------------------------------------
# API-key authentication
# ---------------------------------------------------------------------------


def make_api_key_dep(settings: Settings):
    """Return a FastAPI dependency that enforces API-key auth when configured."""

    async def _check(api_key: str | None = Depends(_API_KEY_HEADER)) -> None:
        if not settings.api_key:
            return  # auth disabled
        if api_key != settings.api_key:
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid X-API-Key header",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    return _check


# ---------------------------------------------------------------------------
# In-memory sliding-window rate limiter
# ---------------------------------------------------------------------------
# Suitable for single-process deployments.  For multi-worker or distributed
# setups, move rate limiting to nginx / a Redis-backed middleware.


class RateLimiter:
    """Per-IP sliding-window rate limiter (thread-safe for asyncio single-worker)."""

    def __init__(self, requests_per_minute: int) -> None:
        self.rpm = requests_per_minute
        # ip → deque of request timestamps (epoch seconds)
        self._windows: dict[str, deque[float]] = {}

    def is_allowed(self, ip: str) -> bool:
        now = time.monotonic()
        window = self._windows.setdefault(ip, deque())
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= self.rpm:
            return False
        window.append(now)
        return True

    def remaining(self, ip: str) -> int:
        window = self._windows.get(ip, deque())
        now = time.monotonic()
        cutoff = now - 60.0
        active = sum(1 for t in window if t >= cutoff)
        return max(0, self.rpm - active)


def make_rate_limit_dep(limiter: RateLimiter | None):
    """Return a FastAPI dependency that enforces per-IP rate limits when configured."""

    async def _check(request: Request) -> None:
        if limiter is None:
            return
        ip = _get_client_ip(request)
        if not limiter.is_allowed(ip):
            remaining = limiter.remaining(ip)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({limiter.rpm} req/min). Retry after 60s.",
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(limiter.rpm),
                    "X-RateLimit-Remaining": str(remaining),
                },
            )

    return _check


def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For when set."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
