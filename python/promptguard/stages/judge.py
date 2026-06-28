"""Stage 4 — LLM-as-judge fallback (stub, optional)."""

from __future__ import annotations

from typing import Any

from promptguard.types import Finding


class JudgeStage:
    """Invokes a cheap LLM call only when stages 0-3 produce an uncertain score band."""

    def run(self, text: str, context: dict[str, Any]) -> list[Finding]:
        # TODO: gated on uncertain_band config; call LLM with tight rubric
        return []
