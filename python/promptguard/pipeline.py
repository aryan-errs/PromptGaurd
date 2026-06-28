"""Pipeline — orchestrates detection stages and produces a Verdict via policy.py."""

from __future__ import annotations

import time
from typing import Any

from promptguard import policy
from promptguard.types import AppProfile, Finding, Stage, Verdict


class Pipeline:
    """Runs a sequence of Stage objects against input text and returns a Verdict.

    Each stage receives the shared context dict; NormalizeStage (S0) stores its
    output in context["normalized_text"] and subsequent stages receive that text.
    The final Verdict is produced by policy.decide() so all threshold logic lives
    in one place.
    """

    def __init__(
        self,
        stages: list[Stage],
        profile: AppProfile | None = None,
    ) -> None:
        self.stages = stages
        self.profile = profile or AppProfile(name="default")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, text: str) -> Verdict:
        """Run all stages and return a Verdict with per-stage timing."""
        wall_start = time.perf_counter()

        findings: list[Finding] = []
        context: dict[str, Any] = {
            "profile": self.profile,
            "stage_latencies_ms": {},
        }

        current_text = text
        for stage in self.stages:
            stage_start = time.perf_counter()
            stage_findings = stage.run(current_text, context)
            stage_ms = (time.perf_counter() - stage_start) * 1000

            stage_name = type(stage).__name__
            context["stage_latencies_ms"][stage_name] = round(stage_ms, 3)

            findings.extend(stage_findings)
            # Expose running findings so later stages can read earlier signals
            context["findings"] = findings
            # If a stage normalised the text (e.g. S0), feed it to the next stage
            current_text = context.get("normalized_text", current_text)

        latency_ms = (time.perf_counter() - wall_start) * 1000

        verdict = policy.decide(findings, self.profile, latency_ms)

        # When the policy says to sanitize, transform the original input and
        # attach the result so callers can use it directly.
        if verdict.action == "sanitize":
            from promptguard.sanitize import sanitize as _sanitize  # lazy import

            sanitized, transforms = _sanitize(text, self.profile)
            return Verdict(
                action=verdict.action,
                score=verdict.score,
                findings=verdict.findings,
                latency_ms=verdict.latency_ms,
                sanitized_text=sanitized,
                transformations=transforms,
            )

        return verdict


# ---------------------------------------------------------------------------
# Factory — heuristic-only pipeline (S0 + S1, no ML)
# ---------------------------------------------------------------------------


def build_pipeline(profile: AppProfile | None = None) -> Pipeline:
    """Create the default heuristic-only pipeline: S0 Normalize + S1 Signatures + S3 Intent.

    Zero ML dependencies (Milestone 1 + 3 from the spec).
    Stages are imported lazily to keep startup fast and avoid import cycles.
    """
    from promptguard.stages.intent import IntentStage
    from promptguard.stages.normalize import NormalizeStage
    from promptguard.stages.signatures import SignatureStage

    return Pipeline(
        stages=[NormalizeStage(), SignatureStage(), IntentStage()],
        profile=profile,
    )
