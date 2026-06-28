"""Stage 1 — Signature engine (deterministic, scored regex rule library).

Rules are loaded from rules.yaml alongside this file.  Each rule produces a
SignatureFinding (a Finding with source_stage="signatures").

Structural rules (fake turns, special tokens, delimiter breakout) are flagged
with structural=True and blocked regardless of app profile.

Semantic rules (override phrasing, extraction phrasing, persona framing) carry
structural=False and are subject to app-profile thresholds downstream.

Delimiter-breakout detection is parameterized: the host app's real
template_delimiters from AppProfile are compiled into structural rules on each
run so that an input containing the live closing delimiter is always caught,
even if it would pass every static rule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from promptguard.types import AppProfile, Finding

_RULES_YAML: Path = Path(__file__).parent / "rules.yaml"
_FLAGS: int = re.IGNORECASE | re.MULTILINE


# ---------------------------------------------------------------------------
# Rule dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    pattern: re.Pattern[str]
    weight: float
    structural: bool


# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------


def _load_static_rules(path: Path = _RULES_YAML) -> list[Rule]:
    """Parse rules.yaml and compile each pattern."""
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    rules: list[Rule] = []
    for entry in data["rules"]:
        rules.append(
            Rule(
                id=str(entry["id"]),
                category=str(entry["category"]),
                pattern=re.compile(str(entry["pattern"]), _FLAGS),
                weight=float(entry["weight"]),
                structural=bool(entry["structural"]),
            )
        )
    return rules


def _make_delimiter_rule(delimiter: str, idx: int) -> Rule:
    """Build a high-weight structural rule for one of the app's live delimiters.

    The delimiter is treated as a literal string (re.escape) because the app's
    actual fence/tag sequences are not regex patterns — they are exact strings
    like '</system>' or '---END---'.
    """
    return Rule(
        id=f"SIG-DELIM-APP-{idx:03d}",
        category="delimiter_breakout",
        pattern=re.compile(re.escape(delimiter), _FLAGS),
        weight=0.98,
        structural=True,
    )


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class SignatureStage:
    """Stage 1: scored signature matching.

    Run all static rules from rules.yaml plus any dynamic delimiter-breakout
    rules derived from the active AppProfile.  Each matched rule emits one
    Finding; multiple matches of the same rule slightly boost the score.

    Static rules are loaded once per instance and cached; this is intentional
    so tests can instantiate a fresh stage to pick up rule file changes.
    """

    def __init__(self) -> None:
        self._static_rules: list[Rule] | None = None

    def _get_static_rules(self) -> list[Rule]:
        if self._static_rules is None:
            self._static_rules = _load_static_rules()
        return self._static_rules

    def run(self, text: str, context: dict[str, Any]) -> list[Finding]:
        profile: AppProfile = context.get("profile", AppProfile(name="default"))

        # Combine static rules with app-specific delimiter rules
        rules = list(self._get_static_rules())
        for idx, delimiter in enumerate(profile.template_delimiters):
            rules.append(_make_delimiter_rule(delimiter, idx))

        findings: list[Finding] = []
        for rule in rules:
            matches = list(rule.pattern.finditer(text))
            if not matches:
                continue

            # Multiple hits of the same rule increase confidence slightly
            score = min(rule.weight + (len(matches) - 1) * 0.03, 1.0)
            snippet = matches[0].group(0)[:80]

            findings.append(
                Finding(
                    id=rule.id,
                    category=rule.category,
                    score=score,
                    structural=rule.structural,
                    source_stage="signatures",
                    detail=f"{len(matches)} match(es); first: {snippet!r}",
                )
            )

        return findings
