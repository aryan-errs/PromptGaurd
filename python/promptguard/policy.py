"""Decision policy — aggregates Stage findings into a Verdict.

Threat model enforced here
──────────────────────────
Hard rule (overrides all thresholds):
  Any finding with structural=True → block, regardless of profile, tier, or score.
  Rationale: structural matches (fake turns, delimiter breakout, special tokens)
  are never legitimate user content; the use-vs-mention problem does not apply.

Semantic findings (structural=False) are scored against per-tier thresholds:

  risk_tier="high"   (tools-enabled / financial)
    → fails safe: low bars; sanitize/flag/block most things
  risk_tier="medium" (default)
    → balanced: reasonable bars; borderline gets sanitized or flagged
  risk_tier="low"    (pure chat, no tools)
    → fails open: high bars; only high-confidence attacks trigger action

When profile.allow_security_discussion=True, ALL semantic thresholds are raised
by SECURITY_DISCUSSION_BOOST.  This is the coarse-grained proxy for the
use-vs-mention problem until Stage 3 (intent) is implemented.  Effect:

  A security chatbot (low + allow_security_discussion=True) can discuss
  "ignore previous instructions" attacks freely because the semantic score
  (~0.85) stays below the raised sanitize threshold (0.70 + 0.20 = 0.90).

  The same profile still hard-blocks any structural finding, so an actual
  delimiter breakout or fake-turn injection is caught regardless.

Verdicts carry their full findings list so every decision is explainable.
"""

from __future__ import annotations

from dataclasses import dataclass

from promptguard.types import AppProfile, Finding, Verdict, VerdictAction

# ---------------------------------------------------------------------------
# Threshold tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TierThresholds:
    """Thresholds for semantic findings only.

    Evaluated in order: score >= block → block; >= flag → flag; >= sanitize → sanitize; else allow.
    Structural findings bypass all of these (always block).
    """

    sanitize: float
    flag: float
    block: float


# Keep these values consistent with the "key scenario" documented at the top:
# low-risk security chatbot with allow_security_discussion, semantic score 0.85 → allow.
#   sanitize = 0.70 + 0.20 boost = 0.90 > 0.85  ✓
_TIERS: dict[str, _TierThresholds] = {
    "high": _TierThresholds(sanitize=0.30, flag=0.50, block=0.80),
    "medium": _TierThresholds(sanitize=0.45, flag=0.65, block=0.88),
    "low": _TierThresholds(sanitize=0.70, flag=0.82, block=0.93),
}

# Raise all semantic thresholds by this much when allow_security_discussion=True.
SECURITY_DISCUSSION_BOOST: float = 0.20

# Maximum additional threshold raise granted by Stage 3 intent signal.
# Applied as: intent_boost = mention_score x INTENT_MAX_BOOST.
# High mention_score (≈1.0) → full boost; clear USE (≈0.0) → no boost.
# Calibrated so that for high-risk (banking) profiles:
#   - clear mention (score=1.0): intent_boost=0.15 → t_block=0.95 (flags, not blocks)
#   - clear use    (score=0.0): intent_boost=0.00 → t_block=0.80 (still blocks)
INTENT_MAX_BOOST: float = 0.15


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _intent_boost(findings: list[Finding]) -> float:
    """Extract the per-input mention boost contributed by IntentStage (S3).

    Returns a value in [0.0, INTENT_MAX_BOOST].
    """
    intent = [f for f in findings if f.category == "intent"]
    if not intent:
        return 0.0
    mention_score = max(f.score for f in intent)
    return mention_score * INTENT_MAX_BOOST


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decide(findings: list[Finding], profile: AppProfile, latency_ms: float) -> Verdict:
    """Convert aggregated findings into an explainable Verdict.

    Args:
        findings:   All findings emitted by the pipeline stages.
        profile:    The active AppProfile (determines thresholds).
        latency_ms: Wall-clock time for the pipeline run; stored in the Verdict.

    Returns:
        Verdict whose .action is one of allow/sanitize/flag/block and whose
        .findings carries the full list so the caller can explain the decision.
    """
    # ── Hard rule: structural → block immediately ──────────────────────────
    structural = [f for f in findings if f.structural]
    if structural:
        score = max(f.score for f in findings)
        return Verdict(
            action="block",
            score=score,
            findings=findings,
            latency_ms=round(latency_ms, 3),
        )

    # ── No findings at all → allow ─────────────────────────────────────────
    if not findings:
        return Verdict(
            action="allow",
            score=0.0,
            findings=[],
            latency_ms=round(latency_ms, 3),
        )

    # ── Semantic path: apply tier + profile + intent thresholds ───────────
    # Exclude intent findings from the score (they are adjustments, not threats).
    detection_findings = [f for f in findings if f.category != "intent"]
    if not detection_findings:
        return Verdict(
            action="allow",
            score=0.0,
            findings=findings,
            latency_ms=round(latency_ms, 3),
        )

    semantic_score = max(f.score for f in detection_findings)

    tier = _TIERS.get(profile.risk_tier, _TIERS["medium"])
    profile_boost = SECURITY_DISCUSSION_BOOST if profile.allow_security_discussion else 0.0
    intent_boost = _intent_boost(findings)
    total_boost = profile_boost + intent_boost

    # Clamp boosted thresholds to 1.0 so they never become logically impossible
    t_block = min(tier.block + total_boost, 1.0)
    t_flag = min(tier.flag + total_boost, 1.0)
    t_sanitize = min(tier.sanitize + total_boost, 1.0)

    action: VerdictAction
    if semantic_score >= t_block:
        action = "block"
    elif semantic_score >= t_flag:
        action = "flag"
    elif semantic_score >= t_sanitize:
        action = "sanitize"
    else:
        action = "allow"

    return Verdict(
        action=action,
        score=semantic_score,
        findings=findings,
        latency_ms=round(latency_ms, 3),
    )


# ---------------------------------------------------------------------------
# Threshold introspection (useful for tests / debugging)
# ---------------------------------------------------------------------------


def effective_thresholds(
    profile: AppProfile,
    mention_score: float = 0.0,
) -> _TierThresholds:
    """Return the effective thresholds for a given profile + optional intent signal."""
    tier = _TIERS.get(profile.risk_tier, _TIERS["medium"])
    profile_boost = SECURITY_DISCUSSION_BOOST if profile.allow_security_discussion else 0.0
    intent_boost = mention_score * INTENT_MAX_BOOST
    total = profile_boost + intent_boost
    return _TierThresholds(
        sanitize=min(tier.sanitize + total, 1.0),
        flag=min(tier.flag + total, 1.0),
        block=min(tier.block + total, 1.0),
    )
