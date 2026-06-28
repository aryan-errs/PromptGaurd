"""Anthropic SDK transparent wrapper (stub)."""

from __future__ import annotations

from typing import Any

from promptguard.pipeline import Pipeline


class GuardedAnthropic:
    """Wraps an anthropic.Anthropic client and inspects messages before every completion."""

    def __init__(self, client: Any, pipeline: Pipeline) -> None:
        self._client = client
        self._pipeline = pipeline

    # TODO: proxy .messages.create and run pipeline on messages
