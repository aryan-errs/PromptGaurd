#!/usr/bin/env python3
"""PromptGuard — Classifier Backend A training script.

Trains a sentence-embedding + logistic-regression classifier and saves an
artifact that EmbeddingClassifier loads at inference time.

Usage::

    # From the python/ directory:
    python -m train.train

    # With options:
    python -m train.train \\
        --data-dir   train/datasets \\
        --output-dir train/artifacts \\
        --encoder    all-MiniLM-L6-v2

    # Run self-tests (does training work end-to-end?):
    python -m train.train --self-test

Requires: pip install promptguard[ml]

Dataset format
--------------
One or more *.jsonl files in --data-dir, each line::

    {"text": "...", "label": 0|1, "split": "train"|"test"}

    label 1 = injection / attack
    label 0 = benign (including hard negatives discussing injection topics)
    split   = which partition this example belongs to

Where to add real datasets
--------------------------
See train/datasets/README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load all records from a JSONL file."""
    records = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] {path}:{lineno}: skipping malformed JSON — {exc}", file=sys.stderr)
    return records


def build_dataset(data_dir: Path) -> tuple[list[str], list[int], list[str], list[int]]:
    """Load all *.jsonl files in data_dir and return (train_texts, train_labels, test_texts, test_labels).

    Records without a 'split' field default to 'train'.
    """
    all_records: list[dict[str, Any]] = []
    for jsonl_file in sorted(data_dir.glob("*.jsonl")):
        all_records.extend(load_jsonl(jsonl_file))

    if not all_records:
        raise ValueError(f"No records found in {data_dir}. Check that *.jsonl files exist.")

    train_texts, train_labels, test_texts, test_labels = [], [], [], []
    for rec in all_records:
        text = str(rec.get("text", "")).strip()
        label = int(rec.get("label", 0))
        split = str(rec.get("split", "train")).lower()
        if not text:
            continue
        if split == "test":
            test_texts.append(text)
            test_labels.append(label)
        else:
            train_texts.append(text)
            train_labels.append(label)

    return train_texts, train_labels, test_texts, test_labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_head(
    train_texts: list[str],
    train_labels: list[int],
    encoder_name: str,
) -> tuple[Any, Any]:
    """Encode texts with sentence-transformers and fit a logistic-regression head.

    Returns (embedder, fitted_head).
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    print(f"[train] Loading encoder: {encoder_name}")
    embedder = SentenceTransformer(encoder_name)

    print(f"[train] Encoding {len(train_texts)} training examples…")
    t0 = time.perf_counter()
    X_train = embedder.encode(train_texts, show_progress_bar=False)
    print(f"[train] Encoding done in {(time.perf_counter() - t0)*1000:.0f} ms")

    print("[train] Fitting LogisticRegression head…")
    head = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    head.fit(X_train, train_labels)

    return embedder, head


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    embedder: Any,
    head: Any,
    test_texts: list[str],
    test_labels: list[int],
    train_labels: list[int],
) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1 and compare against majority-class baseline."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    X_test = embedder.encode(test_texts, show_progress_bar=False)
    y_pred = head.predict(X_test)

    accuracy = float(accuracy_score(test_labels, y_pred))
    precision = float(precision_score(test_labels, y_pred, zero_division=0))
    recall = float(recall_score(test_labels, y_pred, zero_division=0))
    f1 = float(f1_score(test_labels, y_pred, zero_division=0))

    # Majority-class baseline: predict the most common label in training data
    from collections import Counter

    majority_label = Counter(train_labels).most_common(1)[0][0]
    baseline_pred = [majority_label] * len(test_labels)
    baseline_acc = float(accuracy_score(test_labels, baseline_pred))
    baseline_f1 = float(f1_score(test_labels, baseline_pred, zero_division=0))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "baseline_accuracy": baseline_acc,
        "baseline_f1": baseline_f1,
        "beats_baseline_accuracy": accuracy > baseline_acc,
        "beats_baseline_f1": f1 > baseline_f1,
        "n_test": len(test_labels),
        "n_train": len(train_labels),
    }


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def save_artifact(
    head: Any,
    embedding_model: str,
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    """Serialise the trained head + metadata to a joblib file."""
    import joblib

    artifact = {
        "model_type": "embedding_lr",
        "embedding_model": embedding_model,
        "head": head,
        "metrics": metrics,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    print(f"[train] Artifact saved → {output_path}")


# ---------------------------------------------------------------------------
# Public entry-point (also importable from tests)
# ---------------------------------------------------------------------------


def run_training(
    data_dir: Path,
    output_dir: Path,
    encoder_name: str = "all-MiniLM-L6-v2",
    artifact_name: str = "classifier_a.pkl",
) -> dict[str, Any]:
    """Train Backend A end-to-end and return a results dict.

    This function is importable from tests so they can trigger training
    programmatically without subprocess overhead.

    Returns:
        {
          "metrics":       {accuracy, precision, recall, f1, …},
          "artifact_path": Path,
          "n_train":       int,
          "n_test":        int,
        }
    """
    train_texts, train_labels, test_texts, test_labels = build_dataset(data_dir)
    print(f"[train] Dataset loaded: {len(train_texts)} train, {len(test_texts)} test examples")

    embedder, head = train_head(train_texts, train_labels, encoder_name)

    if test_texts:
        metrics = evaluate(embedder, head, test_texts, test_labels, train_labels)
        _print_metrics(metrics)
    else:
        print("[train] No test split found; skipping evaluation.")
        metrics = {"n_train": len(train_texts), "n_test": 0}

    artifact_path = output_dir / artifact_name
    save_artifact(head, encoder_name, metrics, artifact_path)

    return {
        "metrics": metrics,
        "artifact_path": artifact_path,
        "n_train": len(train_texts),
        "n_test": len(test_texts),
    }


def _print_metrics(m: dict[str, float]) -> None:
    print("\n── Evaluation results ──────────────────────────────")
    print(f"  Accuracy  : {m['accuracy']:.3f}   (baseline {m['baseline_accuracy']:.3f})")
    print(f"  Precision : {m['precision']:.3f}")
    print(f"  Recall    : {m['recall']:.3f}")
    print(f"  F1        : {m['f1']:.3f}   (baseline {m['baseline_f1']:.3f})")
    beat_acc = "✓" if m["beats_baseline_accuracy"] else "✗"
    beat_f1 = "✓" if m["beats_baseline_f1"] else "✗"
    print(f"  Beats baseline accuracy: {beat_acc}")
    print(f"  Beats baseline F1:       {beat_f1}")
    print("────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).parent
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=here / "datasets",
        help="Directory containing *.jsonl dataset files (default: ./datasets)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=here / "artifacts",
        help="Directory for the model artifact and metrics (default: ./artifacts)",
    )
    p.add_argument(
        "--encoder",
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name (default: all-MiniLM-L6-v2)",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run a quick self-test and exit non-zero on failure",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    try:
        import sklearn  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError:
        print(
            "[error] The [ml] extra is required for training.\n"
            "        Run: pip install promptguard[ml]",
            file=sys.stderr,
        )
        sys.exit(1)

    result = run_training(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        encoder_name=args.encoder,
    )

    if args.self_test:
        m = result["metrics"]
        ok = m.get("beats_baseline_accuracy", False) and m.get("f1", 0) > 0.5
        if ok:
            print("[self-test] PASS")
        else:
            print(
                f"[self-test] FAIL — accuracy={m.get('accuracy', 0):.3f}, "
                f"f1={m.get('f1', 0):.3f}",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
