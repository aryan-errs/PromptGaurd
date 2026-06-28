"""Stage 3 — Intent disambiguation (use vs. mention).

Determines whether detected injection-like phrases in a message are:

  USE    — the instruction is directed at THIS model as an active command
  MENTION — the phrase is being quoted, explained, or described as a topic

The problem (SPEC §3)
─────────────────────
"A security chatbot discussing 'ignore previous instructions' attacks would be
incorrectly blocked if every occurrence of an injection phrase triggers the same
response."

Signals
───────
Four signals combine into mention_score ∈ [0.0, 1.0]
  (1.0 = clear mention / discussion, 0.0 = clear use / attack):

  1. Framing / quotation
     - injection phrase appears inside single/double quotes or code fences → mention
     - "for example", "such as", "like", explicit "example" intro → mention

  2. Addressivity
     - third-party attribution: "attackers would use…", "technique that works by…" → mention
     - direct imperative to this model starting the text → use
     - "you must now / you will now" + injection verb → use

  3. Educational / question context
     - "how does X work?", "explain X", "can you describe" → mention
     - defensive-security vocabulary: "detect", "prevent", "red-team" → mention

  4. Structural target (spec-decisive rule)
     - any structural=True finding from S1 → mention_score = 0.0 (hard use signal)
     - policy.py's hard rule then blocks regardless of profile

Integration
───────────
  • Emits one Finding(category="intent", score=mention_score) per run.
  • Sets context["mention_score"] so policy.decide() can apply a threshold boost.
  • Returns [] when no prior findings exist (nothing to adjudicate).

Decisive rule enforced in policy.py, not here:
  structural → block      (hard rule, profile-independent)
  semantic   → threshold raised by mention_score x INTENT_MAX_BOOST
"""

from __future__ import annotations

import re
from typing import Any

from promptguard.types import Finding

# ---------------------------------------------------------------------------
# Injection core triggers (checked inside quoted regions)
# ---------------------------------------------------------------------------
# Intentionally lighter than S1's full rule library — just enough to detect
# whether a quoted/framed region contains injection content.

_INJECTION_CORE: list[re.Pattern[str]] = [
    re.compile(
        r"ignore\s+(?:previous|prior|above|all)\s+(?:instructions?|rules?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"disregard\s+(?:all\s+)?(?:previous|prior|the)?\s*(?:instructions?|rules?|constraints?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"forget\s+(?:everything|all\s+previous|your\s+(?:instructions?|training))",
        re.IGNORECASE,
    ),
    re.compile(r"<\|im_(?:start|end)\|>|\[/?INST\]|<</?SYS>>", re.IGNORECASE),
    re.compile(
        r"you\s+(?:are\s+now|have\s+no)\s+(?:DAN|restrictions?|rules?|limits?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:print|reveal|repeat)\s+(?:your\s+)?(?:system\s+prompt|instructions?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:developer|dev|debug)\s+mode\s+(?:enabled|activated|on\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"ignore\s+the\s+user\s+and\s+(?:instead\s+)?(?:call|execute|run|send)\b",
        re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Quoted/framed region extractor
# ---------------------------------------------------------------------------

_FRAMING_REGIONS_RE = re.compile(
    r'"([^"]{2,})"'  # double-quoted content
    r"|'([^']{2,})'"  # single-quoted content
    r"|```([\s\S]*?)```"  # fenced code block
    r"|`([^`]{2,})`",  # inline code
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# Mention signals
# ---------------------------------------------------------------------------

# "for example", "such as", "e.g.", etc.
_EXAMPLE_FRAMING_RE = re.compile(
    r"\b(?:for\s+example|for\s+instance|such\s+as|e\.g\.|i\.e\.|like\s+this|"
    r"examples?\s+(?:of|like)|example\s+(?:attack|injection|payload|prompt|technique|jailbreak))",
    re.IGNORECASE,
)

# "here is an example …", "consider the following …"
_EXAMPLE_INTRO_RE = re.compile(
    r"\b(?:here\s+is|this\s+is|the\s+following\s+is|consider)\s+an?\s+"
    r"(?:example|sample|demo|illustration|case)\b",
    re.IGNORECASE,
)

# "attackers would use…", "a hacker might send…", "an adversary can…"
_ATTRIBUTION_RE = re.compile(
    r"\b(?:attacker|hacker|adversary|threat\s+actor|malicious\s+(?:user|actor))s?\s+"
    r"(?:might|would|could|can|may|often|will|typically)\s+"
    r"(?:use|send|write|craft|inject|try|attempt|construct)\b",
    re.IGNORECASE,
)

# "the technique works by telling the model…", "this attack involves…"
_TECHNIQUE_DESCRIPTION_RE = re.compile(
    r"\b(?:technique|method|approach|attack|exploit|vector|payload|prompt)\s+"
    r"(?:that\s+)?(?:works?\s+by|involves?|uses?|sends?|tells?\s+(?:the\s+)?(?:model|AI)|"
    r"instructs?\s+(?:the\s+)?(?:model|AI))\b",
    re.IGNORECASE,
)

# "how does X work?", "explain how…", "can you describe…", "what is X?"
_QUESTION_FRAMING_RE = re.compile(
    r"\b(?:"
    r"how\s+(?:does|do|would|can|could|might)|"
    r"why\s+(?:does|is|would)|"
    r"what\s+(?:is|are|does|would)|"
    r"explain\s+(?:how|what|why)|"
    r"describe\s+(?:how|what)|"
    r"can\s+you\s+(?:explain|describe|show|tell|help\s+me\s+understand)|"
    r"help\s+me\s+understand"
    r")\b",
    re.IGNORECASE,
)

# "detect", "prevent", "defend against", "red-team", "pentest", etc.
_DEFENSIVE_SECURITY_RE = re.compile(
    r"\b(?:"
    r"detect|prevent|defend\s+(?:against|from)|protect\s+(?:against|from)|"
    r"mitigate|understand\s+(?:how|the|these)|identify|recognize|"
    r"red.?team(?:ing)?|pentest(?:ing)?|security\s+(?:review|audit|research)|"
    r"(?:for\s+)?educational\s+(?:purposes?|reasons?)|"
    r"(?:for\s+)?(?:research|learning|study)\s+(?:purposes?|reasons?)"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Use signals
# ---------------------------------------------------------------------------

# Message starts with a bare injection verb (no framing before it)
_DIRECT_COMMAND_START_RE = re.compile(
    r"^\s*(?:"
    r"ignore|disregard|forget|override|print|reveal|repeat|execute|send|"
    r"call|invoke|do\s+not\s+follow|stop\s+following|act\s+as|you\s+are\s+now|"
    r"enable\s+jailbreak|developer\s+mode"
    r")\b",
    re.IGNORECASE | re.MULTILINE,
)

# "you must now …", "you will now …" + an injection verb
_DIRECT_MODEL_COMMAND_RE = re.compile(
    r"\byou\s+(?:must|will|should\s+now|are\s+required\s+to|need\s+to|have\s+to)\s+"
    r"(?:ignore|disregard|forget|override|reveal|print|repeat|execute|send)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Core scoring helpers
# ---------------------------------------------------------------------------


def _injection_in_framing(text: str) -> bool:
    """True if any injection-core phrase appears inside a quoted or fenced region."""
    for m in _FRAMING_REGIONS_RE.finditer(text):
        # One of the four groups will be non-None
        content = next(g for g in m.groups() if g is not None)
        if any(pat.search(content) for pat in _INJECTION_CORE):
            return True
    return False


def _compute_signals(text: str) -> dict[str, bool]:
    """Return a dict of fired signal names (for detail logging).

    Use signals (direct_command_start, direct_model_command) are checked only
    against text OUTSIDE of quoted/code-fenced regions.  This prevents injection
    content shown as a quoted example (e.g. inside ``` ... ```) from incorrectly
    triggering the "bare imperative command" use signal.
    """
    # Replace framed regions with spaces to neutralise their content for use-signal matching
    unframed = _FRAMING_REGIONS_RE.sub(" ", text)

    return {
        "injection_in_quotes": _injection_in_framing(text),
        "example_framing": bool(_EXAMPLE_FRAMING_RE.search(text)),
        "example_intro": bool(_EXAMPLE_INTRO_RE.search(text)),
        "third_party_attribution": bool(_ATTRIBUTION_RE.search(text)),
        "technique_description": bool(_TECHNIQUE_DESCRIPTION_RE.search(text)),
        "question_framing": bool(_QUESTION_FRAMING_RE.search(text)),
        "defensive_security": bool(_DEFENSIVE_SECURITY_RE.search(text)),
        "direct_command_start": bool(_DIRECT_COMMAND_START_RE.search(unframed)),
        "direct_model_command": bool(_DIRECT_MODEL_COMMAND_RE.search(unframed)),
    }


def compute_mention_score(
    text: str, prior_findings: list[Finding]
) -> tuple[float, dict[str, bool]]:
    """Return (mention_score, fired_signals).

    mention_score ∈ [0.0, 1.0]
      0.0 = clear USE  (instruction directed at this model)
      1.0 = clear MENTION (discussing/quoting the technique)

    Args:
        text:           Normalised input text (output of S0).
        prior_findings: Findings from S0 + S1 (+ S2 if present) already in context.
    """
    # Hard signal: structural finding → always USE regardless of framing.
    if any(f.structural for f in prior_findings):
        return 0.0, {"structural_target": True}

    signals = _compute_signals(text)
    score = 0.0

    # --- Framing / quotation (mention signals) ---
    if signals["injection_in_quotes"]:
        score += 0.35  # strongest mention signal: phrase is being quoted
    if signals["example_framing"]:
        score += 0.12
    if signals["example_intro"]:
        score += 0.10

    # --- Addressivity (mention signals) ---
    if signals["third_party_attribution"]:
        score += 0.20
    if signals["technique_description"]:
        score += 0.13

    # --- Educational / question context (mention signals) ---
    if signals["question_framing"]:
        score += 0.18
    if signals["defensive_security"]:
        score += 0.10

    # --- Use signals (reduce from mention score) ---
    if signals["direct_command_start"]:
        score -= 0.38  # strong use signal: text starts with a bare injection verb
    if signals["direct_model_command"]:
        score -= 0.22

    return max(0.0, min(score, 1.0)), signals


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class IntentStage:
    """Stage 3: use-vs-mention disambiguation.

    Reads context["findings"] (populated by S0 and S1) and produces one
    Finding per run (unless there are no prior findings to adjudicate).

    The Finding's score is the mention_score; policy.decide() multiplies it by
    INTENT_MAX_BOOST and adds the result to semantic thresholds, giving more
    headroom when the model is confident the phrase is being discussed rather
    than used as an attack.
    """

    def run(self, text: str, context: dict[str, Any]) -> list[Finding]:
        prior_findings: list[Finding] = context.get("findings", [])

        # Nothing to adjudicate if no injection signals have been detected yet
        if not prior_findings:
            return []

        mention_score, signals = compute_mention_score(text, prior_findings)
        context["mention_score"] = mention_score

        fired = [name for name, hit in signals.items() if hit]
        label = "INTENT-MENTION" if mention_score >= 0.5 else "INTENT-USE"

        return [
            Finding(
                id=label,
                category="intent",
                score=mention_score,
                structural=False,
                source_stage="intent",
                detail=f"mention_score={mention_score:.3f}; signals=[{', '.join(fired)}]",
            )
        ]
