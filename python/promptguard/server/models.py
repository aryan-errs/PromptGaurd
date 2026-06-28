"""Pydantic request/response models for the PromptGuard server API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """OpenAI-style chat message."""

    role: str
    content: str


class ProfileInput(BaseModel):
    """Per-request AppProfile override.

    When omitted, the server's default profile (configured via env vars) is used.
    Only the fields provided are overridden; all others fall back to the server default.
    """

    name: str = "default"
    allow_security_discussion: bool = False
    risk_tier: Literal["low", "medium", "high"] = "medium"
    template_delimiters: list[str] = Field(default_factory=list)
    tools_enabled: bool = False

    def to_app_profile(self) -> AppProfile:  # noqa: F821
        from promptguard import AppProfile

        return AppProfile(
            name=self.name,
            allow_security_discussion=self.allow_security_discussion,
            risk_tier=self.risk_tier,
            template_delimiters=self.template_delimiters,
            tools_enabled=self.tools_enabled,
        )


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class InspectRequest(BaseModel):
    """POST /inspect — inspect messages for injection signals."""

    messages: list[Message] = Field(..., min_length=1)
    profile: ProfileInput | None = None

    @model_validator(mode="after")
    def _has_user_message(self) -> InspectRequest:
        if not any(m.role == "user" for m in self.messages):
            raise ValueError("at least one message with role='user' is required")
        return self

    def last_user_text(self) -> str:
        """Return the content of the last user-role message."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return ""  # unreachable after validator


class ProtectRequest(BaseModel):
    """POST /protect — inspect and sanitize messages."""

    messages: list[Message] = Field(..., min_length=1)
    profile: ProfileInput | None = None
    block_on_block: bool = Field(
        True, description="Return HTTP 400 when action == 'block'; False returns 200 with verdict"
    )

    @model_validator(mode="after")
    def _has_user_message(self) -> ProtectRequest:
        if not any(m.role == "user" for m in self.messages):
            raise ValueError("at least one message with role='user' is required")
        return self

    def last_user_text(self) -> str:
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return ""


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class FindingOut(BaseModel):
    id: str
    category: str
    score: float
    structural: bool
    source_stage: str
    detail: str


class TransformationOut(BaseModel):
    kind: str
    description: str
    original_fragment: str
    transformed_fragment: str
    stage: str


class VerdictOut(BaseModel):
    action: Literal["allow", "sanitize", "flag", "block"]
    score: float
    findings: list[FindingOut]
    latency_ms: float
    blocked: bool
    safe: bool
    # Only present when action == "sanitize"
    sanitized_text: str | None = None
    transformations: list[TransformationOut] = Field(default_factory=list)

    @classmethod
    def from_verdict(cls, v: Verdict) -> VerdictOut:  # noqa: F821
        return cls(
            action=v.action,
            score=v.score,
            findings=[
                FindingOut(
                    id=f.id,
                    category=f.category,
                    score=f.score,
                    structural=f.structural,
                    source_stage=f.source_stage,
                    detail=f.detail,
                )
                for f in v.findings
            ],
            latency_ms=v.latency_ms,
            blocked=v.blocked,
            safe=v.safe,
            sanitized_text=v.sanitized_text,
            transformations=[
                TransformationOut(
                    kind=t.kind,
                    description=t.description,
                    original_fragment=t.original_fragment,
                    transformed_fragment=t.transformed_fragment,
                    stage=t.stage,
                )
                for t in v.transformations
            ],
        )


class InspectResponse(BaseModel):
    verdict: VerdictOut
    request_id: str
    inspected_text_preview: str = Field(
        description="First 80 chars of the inspected text (for logging; never the full input)"
    )


class ProtectResponse(BaseModel):
    verdict: VerdictOut
    request_id: str
    # Messages to forward to the LLM (sanitized when action=="sanitize", else original)
    safe_messages: list[Message]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    classifier: str = Field(description="'ml' if Backend A loaded, 'heuristic' otherwise")
