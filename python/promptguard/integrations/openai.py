"""OpenAI SDK transparent wrapper (stub)."""

from __future__ import annotations

from typing import Any

from promptguard.pipeline import Pipeline


class GuardedOpenAI:
    """Wraps an openai.OpenAI client and inspects messages before every chat completion."""

    def __init__(self, client: Any, pipeline: Pipeline) -> None:
        self._client = client
        self._pipeline = pipeline

    # TODO: proxy .chat.completions.create and run pipeline on messages
