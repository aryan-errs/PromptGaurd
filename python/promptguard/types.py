"""Core data types for the PromptGuard pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass
class Finding:
    """A single detection result from any pipeline stage.

    `structural=True` means the finding targets the message structure itself
    (fake turns, delimiter breakout, special tokens) rather than semantic content.
    Structural findings are blocked regardless of app profile.
    """

    id: str
    category: str
    score: float  # [0.0, 1.0]
    structural: bool
    source_stage: str
    detail: str = ""


VerdictAction = Literal["allow", "sanitize", "flag", "block"]


@dataclass
class Verdict:
    """Aggregate decision after all stages have run.

    When action == "sanitize", sanitized_text and transformations are populated
    by the pipeline so callers can pass the clean version to the LLM and log
    exactly what was changed.
    """

    action: VerdictAction
    score: float  # [0.0, 1.0] — highest-severity finding or weighted aggregate
    findings: list[Finding]
    latency_ms: float
    # Populated when action == "sanitize"
    sanitized_text: str | None = None
    transformations: list[Transformation] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.action == "block"

    @property
    def safe(self) -> bool:
        return self.action == "allow"


RiskTier = Literal["low", "medium", "high"]


@dataclass
class AppProfile:
    """Describes what the host application legitimately does.

    Semantic detection thresholds are adjusted per profile; structural findings
    are always blocked regardless of profile settings.
    """

    name: str
    allow_security_discussion: bool = False
    risk_tier: RiskTier = "medium"
    template_delimiters: list[str] = field(default_factory=list)
    tools_enabled: bool = False


@dataclass
class Transformation:
    """Records one sanitization operation applied to the input."""

    kind: str  # e.g. "spotlight", "delimiter_escape", "zero_width_strip"
    description: str
    original_fragment: str
    transformed_fragment: str
    stage: str = "sanitize"


@runtime_checkable
class Stage(Protocol):
    """A single detection stage.

    `run` receives the (possibly already normalized) text and a shared context
    dict that carries the AppProfile, obfuscation signals, etc.  It returns
    zero or more Findings.
    """

    def run(self, text: str, context: dict[str, Any]) -> list[Finding]: ...
