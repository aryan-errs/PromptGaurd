"""CI test for the red-team mutation engine — SPEC §6.

Fails when the pipeline evasion rate exceeds EVASION_THRESHOLD.

What is tested
──────────────
18 seed attacks from eval/datasets/seed_attacks.jsonl are mutated with
six strategies (base64, hex, zero_width, synonym_swap, token_split,
translation) to produce 267 variants.  Each variant is fed through the
default heuristic pipeline (S0 + S1 + S3; no ML required) and compared to
the original verdict.

Evasion = original detected AND mutated variant gets "allow".

Expected evasion rates after rule tightening
────────────────────────────────────────────
  base64       0%   (S0 decodes and raises score to 0.62 → sanitize)
  hex          0%   (same)
  zero_width   0%   (S0 strips ZW before S1 runs)
  synonym_swap ~22% (some unusual synonyms still evade; catching all requires ML)
  token_split  ~43% (word-level splits like "i gnore" need embedding-based detection)
  translation  100% (expected; English-only patterns cannot catch foreign text)
  OVERALL      ~14% (well under EVASION_THRESHOLD = 0.40)

If the pipeline regresses, overall evasion will rise and this test fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from eval.red_team import (
    _EXPECTED_HIGH_EVASION,
    MUTATIONS,
    RedTeamReport,
    load_seed_attacks,
    run_red_team,
)

from promptguard.pipeline import build_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEED_FILE = Path(__file__).parents[1] / "eval" / "datasets" / "seed_attacks.jsonl"

# CI fails if the total evasion rate exceeds this value.
# Set conservatively above the current ~14% to allow for rounding noise
# while still catching significant regressions.
EVASION_THRESHOLD: float = 0.40

# Per-mutation-type caps — non-"expected high" types are flagged above this.
MUTATION_EVASION_WARNING: float = 0.50


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def red_team_report() -> RedTeamReport:
    """Run the full red team once per test session and share the result."""
    attacks = load_seed_attacks(_SEED_FILE)
    assert attacks, f"No seed attacks found in {_SEED_FILE}"
    pipeline = build_pipeline()
    return run_red_team(attacks, pipeline=pipeline, threshold=EVASION_THRESHOLD)


# ---------------------------------------------------------------------------
# Overall CI gate
# ---------------------------------------------------------------------------


class TestRedTeamCI:
    def test_seed_file_exists(self) -> None:
        assert _SEED_FILE.exists(), f"Seed attack file not found: {_SEED_FILE}"

    def test_seed_attacks_loadable(self) -> None:
        attacks = load_seed_attacks(_SEED_FILE)
        assert len(attacks) >= 10, (
            f"Seed file should contain at least 10 attacks; found {len(attacks)}"
        )

    def test_mutations_generated(self, red_team_report: RedTeamReport) -> None:
        assert red_team_report.total > 0

    def test_overall_evasion_rate_under_threshold(self, red_team_report: RedTeamReport) -> None:
        """Core CI gate: total evasion rate must not exceed EVASION_THRESHOLD."""
        rate = red_team_report.evasion_rate
        assert rate <= EVASION_THRESHOLD, (
            f"Pipeline evasion rate {rate:.1%} exceeds CI threshold {EVASION_THRESHOLD:.1%}.\n"
            f"{red_team_report.summary()}"
        )

    def test_report_passes(self, red_team_report: RedTeamReport) -> None:
        assert red_team_report.passed, f"Red-team report did not pass:\n{red_team_report.summary()}"


# ---------------------------------------------------------------------------
# Per-mutation-type assertions
# ---------------------------------------------------------------------------


class TestMutationTypes:
    def test_all_mutation_types_exercised(self, red_team_report: RedTeamReport) -> None:
        exercised = set(red_team_report.by_mutation_type())
        expected = set(MUTATIONS)
        missing = expected - exercised
        assert not missing, f"Some mutation types produced no variants: {missing}"

    def test_encoding_mutations_zero_evasion(self, red_team_report: RedTeamReport) -> None:
        """S0 must decode base64/hex and flag them before S1 runs."""
        by_type = red_team_report.by_mutation_type()
        for mt in ("base64", "hex"):
            rate = float(by_type[mt]["evasion_rate"])
            assert rate == 0.0, (
                f"{mt} evasion rate {rate:.1%} > 0%: "
                f"S0 normalisation should catch all encoding mutations."
            )

    def test_zero_width_zero_evasion(self, red_team_report: RedTeamReport) -> None:
        """S0 must strip ZW chars before S1 runs, giving 0% evasion."""
        by_type = red_team_report.by_mutation_type()
        rate = float(by_type["zero_width"]["evasion_rate"])
        assert rate == 0.0, (
            f"zero_width evasion rate {rate:.1%} > 0%: S0 should strip all invisible chars."
        )

    def test_non_expected_mutations_below_warning_threshold(
        self, red_team_report: RedTeamReport
    ) -> None:
        """Non-translation mutation types should not massively evade."""
        by_type = red_team_report.by_mutation_type()
        for mt, stats in by_type.items():
            if mt in _EXPECTED_HIGH_EVASION:
                continue
            rate = float(stats["evasion_rate"])
            assert rate <= MUTATION_EVASION_WARNING, (
                f"{mt} evasion rate {rate:.1%} exceeds warning level "
                f"{MUTATION_EVASION_WARNING:.1%}."
            )

    def test_translation_evasion_is_expected(self, red_team_report: RedTeamReport) -> None:
        """Translation evasion is expected and documented; assert it is tracked."""
        by_type = red_team_report.by_mutation_type()
        assert "translation" in by_type, "translation mutations should be present"
        assert "translation" in _EXPECTED_HIGH_EVASION


# ---------------------------------------------------------------------------
# Evasion detail — ensure evading examples are logged correctly
# ---------------------------------------------------------------------------


class TestEvasionDetail:
    def test_evading_results_have_original_detected(self, red_team_report: RedTeamReport) -> None:
        for r in red_team_report.evading_examples():
            assert r.original_action != "allow", (
                f"Evading result should have original_action != allow; "
                f"got {r.original_action!r} for {r.original!r}"
            )

    def test_evading_results_have_mutated_allowed(self, red_team_report: RedTeamReport) -> None:
        for r in red_team_report.evading_examples():
            assert r.mutated_action == "allow", (
                f"Evading result should have mutated_action == allow; got {r.mutated_action!r}"
            )

    def test_mutation_types_present_in_results(self, red_team_report: RedTeamReport) -> None:
        mutation_types_found = {r.mutation_type for r in red_team_report.results}
        assert mutation_types_found == set(MUTATIONS), (
            f"Not all mutation types found in results: "
            f"missing {set(MUTATIONS) - mutation_types_found}"
        )

    def test_report_summary_contains_pass_fail(self, red_team_report: RedTeamReport) -> None:
        summary = red_team_report.summary()
        assert "PASS" in summary or "FAIL" in summary


# ---------------------------------------------------------------------------
# Regression: individual attacks detected before mutation
# ---------------------------------------------------------------------------


class TestSeedAttacksDetected:
    """Ensure the seed attacks themselves are detected (not already allowing)."""

    def test_most_seed_attacks_detected(self) -> None:
        attacks = load_seed_attacks(_SEED_FILE)
        pipeline = build_pipeline()
        allow_count = sum(1 for a in attacks if pipeline.run(a).action == "allow")
        allow_rate = allow_count / len(attacks)
        assert allow_rate < 0.20, (
            f"Too many seed attacks are not being detected ({allow_rate:.0%} allowed). "
            f"The red-team measurement is only meaningful if the pipeline detects the originals."
        )
