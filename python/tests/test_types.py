"""Tests for core types in promptguard.types."""

import pytest

from promptguard.types import (
    AppProfile,
    Finding,
    Stage,
    Transformation,
    Verdict,
)

# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------


def test_finding_required_fields() -> None:
    f = Finding(
        id="RULE-001",
        category="instruction_override",
        score=0.9,
        structural=False,
        source_stage="signatures",
    )
    assert f.id == "RULE-001"
    assert f.category == "instruction_override"
    assert f.score == 0.9
    assert f.structural is False
    assert f.source_stage == "signatures"
    assert f.detail == ""


def test_finding_with_detail() -> None:
    f = Finding(
        id="RULE-002",
        category="role_injection",
        score=1.0,
        structural=True,
        source_stage="signatures",
        detail="fake <|im_start|> token detected",
    )
    assert f.structural is True
    assert f.detail == "fake <|im_start|> token detected"


def test_finding_score_range() -> None:
    for score in (0.0, 0.5, 1.0):
        f = Finding(id="x", category="y", score=score, structural=False, source_stage="s0")
        assert 0.0 <= f.score <= 1.0


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def test_verdict_allow() -> None:
    v = Verdict(action="allow", score=0.1, findings=[], latency_ms=1.2)
    assert v.safe is True
    assert v.blocked is False


def test_verdict_block() -> None:
    f = Finding(id="R", category="c", score=1.0, structural=True, source_stage="s1")
    v = Verdict(action="block", score=1.0, findings=[f], latency_ms=0.5)
    assert v.blocked is True
    assert v.safe is False


@pytest.mark.parametrize("action", ["allow", "sanitize", "flag", "block"])
def test_verdict_all_actions(action: str) -> None:
    v = Verdict(action=action, score=0.5, findings=[], latency_ms=0.0)  # type: ignore[arg-type]
    assert v.action == action


# ---------------------------------------------------------------------------
# AppProfile
# ---------------------------------------------------------------------------


def test_app_profile_defaults() -> None:
    p = AppProfile(name="default")
    assert p.allow_security_discussion is False
    assert p.risk_tier == "medium"
    assert p.template_delimiters == []
    assert p.tools_enabled is False


def test_app_profile_security_chatbot() -> None:
    p = AppProfile(
        name="security-chatbot",
        allow_security_discussion=True,
        risk_tier="low",
        template_delimiters=["</system>", "<|im_start|>"],
        tools_enabled=False,
    )
    assert p.allow_security_discussion is True
    assert len(p.template_delimiters) == 2


def test_app_profile_risk_tiers() -> None:
    for tier in ("low", "medium", "high"):
        p = AppProfile(name="x", risk_tier=tier)  # type: ignore[arg-type]
        assert p.risk_tier == tier


# ---------------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------------


def test_transformation_fields() -> None:
    t = Transformation(
        kind="spotlight",
        description="wrapped user input with datamarker",
        original_fragment="ignore previous instructions",
        transformed_fragment="[DATA]ignore previous instructions[/DATA]",
    )
    assert t.stage == "sanitize"
    assert t.kind == "spotlight"


# ---------------------------------------------------------------------------
# Stage protocol
# ---------------------------------------------------------------------------


def test_stage_protocol_satisfied_by_class() -> None:
    """Any class with a run(text, context) method satisfies Stage."""

    class NoopStage:
        def run(self, text: str, context: dict) -> list[Finding]:
            return []

    s = NoopStage()
    assert isinstance(s, Stage)


def test_stage_protocol_not_satisfied_without_run() -> None:
    class BrokenStage:
        pass

    assert not isinstance(BrokenStage(), Stage)
