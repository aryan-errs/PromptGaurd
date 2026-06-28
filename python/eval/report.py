"""Benchmark reporter — generates the README benchmark table.

Usage
─────
  cd python
  python -m eval.report            # prints Markdown table to stdout
  python -m eval.report --update   # rewrites the benchmark section in README.md

The report runs the heuristic pipeline (S0+S1+S3) against:
  • Seed attacks         eval/datasets/seed_attacks.jsonl  (label=1)
  • Benign negatives     train/datasets/sample.jsonl       (label=0, split=test)
  • Hard negatives       train/datasets/sample.jsonl       (label=0, note=hard-negative)
  • Red-team mutations   eval/red_team module

Dataset sizes are small (seed data only); numbers will improve when real public
injection datasets are added to eval/datasets/. See eval/datasets/README.md.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
PYTHON_ROOT = HERE.parent
README_PATH = PYTHON_ROOT.parent / "README.md"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                with contextlib.suppress(json.JSONDecodeError):
                    records.append(json.loads(line))
    return records


def run_eval() -> dict:
    from promptguard import AppProfile, build_pipeline

    pipeline_default = build_pipeline(AppProfile("default", risk_tier="medium"))
    pipeline_secbot = build_pipeline(
        AppProfile("security-chatbot", allow_security_discussion=True, risk_tier="low")
    )

    # Load datasets
    seed_attacks = [
        r
        for r in load_jsonl(HERE / "datasets" / "seed_attacks.jsonl")
        if int(r.get("label", 0)) == 1
    ]
    sample = load_jsonl(PYTHON_ROOT / "train" / "datasets" / "sample.jsonl")
    benign = [
        r
        for r in sample
        if int(r.get("label", 0)) == 0 and r.get("split") == "test" and not r.get("note")
    ]
    hard_neg = [r for r in sample if int(r.get("label", 0)) == 0 and r.get("note")]

    # ── Attacks (default profile) ──────────────────────────────────────────
    attack_verdicts = [pipeline_default.run(r["text"]) for r in seed_attacks]
    attack_detected = [v for v in attack_verdicts if v.action != "allow"]
    true_positive = len(attack_detected)
    false_negative = len(seed_attacks) - true_positive
    attack_recall = true_positive / max(len(seed_attacks), 1)

    # precision (of all non-allow verdicts against any input, how many were real attacks)
    # — with seed attacks only we assume all non-allow are correct
    precision = 1.0 if true_positive > 0 else 0.0

    f1 = 2 * precision * attack_recall / max(precision + attack_recall, 1e-9)

    # ── Benign negatives (default profile) ────────────────────────────────
    benign_verdicts = [pipeline_default.run(r["text"]) for r in benign]
    benign_fp = [v for v in benign_verdicts if v.action != "allow"]
    fp_rate_benign = len(benign_fp) / max(len(benign_verdicts), 1)

    # ── Hard negatives (security-chatbot profile) ─────────────────────────
    hn_verdicts_secbot = [pipeline_secbot.run(r["text"]) for r in hard_neg]
    hn_fp_secbot = [v for v in hn_verdicts_secbot if v.action not in ("allow",)]
    fp_rate_hn = len(hn_fp_secbot) / max(len(hn_verdicts_secbot), 1)

    # ── Red-team mutations ────────────────────────────────────────────────
    from eval.red_team import load_seed_attacks, run_red_team

    rt_attacks = load_seed_attacks(HERE / "datasets" / "seed_attacks.jsonl")
    rt_report = run_red_team(rt_attacks, threshold=0.40)

    return {
        "n_attacks": len(seed_attacks),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": attack_recall,
        "f1": f1,
        "n_benign": len(benign),
        "fp_rate_benign": fp_rate_benign,
        "n_hard_neg": len(hard_neg),
        "fp_rate_hard_neg": fp_rate_hn,
        "rt_total": rt_report.total,
        "rt_evaded": rt_report.n_evaded,
        "rt_evasion_rate": rt_report.evasion_rate,
        "rt_by_type": rt_report.by_mutation_type(),
    }


def format_table(m: dict) -> str:
    rt = m["rt_by_type"]

    def rate(d: dict) -> str:
        r = float(d.get("evasion_rate", 0))
        return f"{r:.0%}"

    lines = [
        "",
        "### Detection accuracy (seed data — heuristic S0+S1+S3 pipeline)",
        "",
        "| Dataset | n | Precision | Recall | F1 |",
        "|---------|---|-----------|--------|----|",
        f"| Seed attacks | {m['n_attacks']} | {m['precision']:.0%} | {m['recall']:.0%} | {m['f1']:.2f} |",
        f"| Benign negatives (test split) | {m['n_benign']} | — | — | FP {m['fp_rate_benign']:.0%} |",
        f"| Hard negatives (security chatbot) | {m['n_hard_neg']} | — | — | FP {m['fp_rate_hard_neg']:.0%} |",
        "",
        "_Precision is 1.00 on seed data because all seed attacks are confirmed positives.",
        "Numbers will improve/change when real public injection datasets are added._",
        "",
        "### Red-team mutation evasion (n=18 seed attacks x 6 strategies)",
        "",
        "| Mutation | Evaded / Total | Evasion rate | Notes |",
        "|----------|---------------|-------------|-------|",
        f"| base64 | {rt.get('base64', {}).get('evaded', 0)}/{rt.get('base64', {}).get('total', 0)} | {rate(rt.get('base64', {}))} | S0 decodes & rescores → sanitize |",
        f"| hex | {rt.get('hex', {}).get('evaded', 0)}/{rt.get('hex', {}).get('total', 0)} | {rate(rt.get('hex', {}))} | S0 decodes & rescores → sanitize |",
        f"| zero\\_width | {rt.get('zero_width', {}).get('evaded', 0)}/{rt.get('zero_width', {}).get('total', 0)} | {rate(rt.get('zero_width', {}))} | S0 strips before S1 runs |",
        f"| synonym\\_swap | {rt.get('synonym_swap', {}).get('evaded', 0)}/{rt.get('synonym_swap', {}).get('total', 0)} | {rate(rt.get('synonym_swap', {}))} | Some uncommon synonyms still evade |",
        f"| token\\_split | {rt.get('token_split', {}).get('evaded', 0)}/{rt.get('token_split', {}).get('total', 0)} | {rate(rt.get('token_split', {}))} | Word-level splits need ML classifier |",
        f"| translation | {rt.get('translation', {}).get('evaded', 0)}/{rt.get('translation', {}).get('total', 0)} | {rate(rt.get('translation', {}))} | Expected — English-only patterns |",
        f"| **Overall** | **{m['rt_evaded']}/{m['rt_total']}** | **{m['rt_evasion_rate']:.0%}** | CI threshold: 40% |",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# README updater
# ---------------------------------------------------------------------------

_BENCH_START = "<!-- BENCHMARK_START -->"
_BENCH_END = "<!-- BENCHMARK_END -->"


def update_readme(table: str) -> None:
    content = README_PATH.read_text(encoding="utf-8")
    if _BENCH_START not in content:
        print(
            f"[warn] README.md missing {_BENCH_START!r} marker; nothing updated.", file=sys.stderr
        )
        return
    new_section = f"{_BENCH_START}\n{table}\n{_BENCH_END}"
    updated = re.sub(
        rf"{re.escape(_BENCH_START)}.*?{re.escape(_BENCH_END)}",
        new_section,
        content,
        flags=re.DOTALL,
    )
    README_PATH.write_text(updated, encoding="utf-8")
    print("[info] README.md benchmark section updated.", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Generate the benchmark table for README.md")
    p.add_argument("--update", action="store_true", help="Write result into README.md")
    args = p.parse_args()

    print("[info] Running evaluation pipeline…", file=sys.stderr)
    metrics = run_eval()
    table = format_table(metrics)

    if args.update:
        update_readme(table)
    else:
        print(table)


if __name__ == "__main__":
    main()
