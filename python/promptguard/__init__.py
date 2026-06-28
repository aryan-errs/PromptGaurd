"""PromptGuard — runtime prompt-injection defense middleware."""

from promptguard.pipeline import Pipeline, build_pipeline
from promptguard.sanitize import (
    SPOTLIGHT_SYSTEM_NOTE,
    DeeplyObfuscatedError,
    sanitize,
    sanitize_messages,
)
from promptguard.types import (
    AppProfile,
    Finding,
    Stage,
    Transformation,
    Verdict,
    VerdictAction,
)


class PromptGuard:
    """High-level facade: create once, call inspect() or protect() per message.

    Example::

        guard = PromptGuard(profile=AppProfile(
            name="security-chatbot",
            allow_security_discussion=True,
            risk_tier="low",
        ))

        verdict = guard.inspect(user_text)
        if verdict.blocked:
            raise PermissionError("blocked")

        safe_text = guard.protect(user_text)   # raises on block
    """

    def __init__(self, profile: AppProfile | None = None) -> None:
        self._pipeline = build_pipeline(profile)

    @property
    def profile(self) -> AppProfile:
        return self._pipeline.profile

    def inspect(self, text: str) -> Verdict:
        """Run the full detection pipeline and return a Verdict."""
        return self._pipeline.run(text)

    def protect(self, text: str) -> str:
        """Run detection; raise PermissionError if blocked, else return text."""
        verdict = self.inspect(text)
        if verdict.blocked:
            rule_ids = [f.id for f in verdict.findings if f.structural] or [
                f.id for f in verdict.findings
            ]
            raise PermissionError(
                f"PromptGuard blocked input (score={verdict.score:.2f}, rules={rule_ids})"
            )
        return text


__all__ = [
    "SPOTLIGHT_SYSTEM_NOTE",
    "AppProfile",
    "DeeplyObfuscatedError",
    "Finding",
    "Pipeline",
    "PromptGuard",
    "Stage",
    "Transformation",
    "Verdict",
    "VerdictAction",
    "build_pipeline",
    "sanitize",
    "sanitize_messages",
]
