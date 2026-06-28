"""Structured logging of detection verdicts (stub)."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from promptguard.types import Verdict


def _redact(text: str, preview_chars: int = 40) -> str:
    return text[:preview_chars] + ("…" if len(text) > preview_chars else "")


def _salted_hash(text: str, salt: str = "") -> str:
    return hashlib.sha256((salt + text).encode()).hexdigest()[:16]


def log_verdict(
    verdict: Verdict,
    request_id: str,
    raw_input: str,
    profile_name: str,
    *,
    log_raw: bool = False,
    salt: str = "",
    sink: Any = None,
) -> dict[str, Any]:
    """Build a structured log record for a verdict.

    By default, raw input is replaced with a salted hash + short redacted preview.
    Set log_raw=True only for local debugging — never in production.
    """
    record: dict[str, Any] = {
        "timestamp": time.time(),
        "request_id": request_id,
        "profile": profile_name,
        "verdict": verdict.action,
        "score": verdict.score,
        "latency_ms": verdict.latency_ms,
        "rule_ids": [f.id for f in verdict.findings],
        "input_hash": _salted_hash(raw_input, salt),
        "input_preview": _redact(raw_input),
    }
    if log_raw:
        record["input_raw"] = raw_input

    # TODO: route to pluggable sinks (file, webhook, Prometheus counter)
    if sink is None:
        print(json.dumps(record))

    return record
