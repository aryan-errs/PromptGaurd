"""Stage 2 — Pluggable ML classifier.

Three backends behind a common Classifier protocol:

  C  HeuristicClassifier   — zero-setup fallback; pure Python; always works.
  A  EmbeddingClassifier   — sentence-embedding + logistic-regression head.
                             Requires [ml] extra + a trained artifact from train/train.py.
                             Automatically falls back to C when artifact or deps are absent.
  B  EncoderClassifier     — fine-tuned small encoder stub (DeBERTa-v3-small).
                             Requires [ml] extra + a fine-tuned checkpoint.
                             Falls back to C until a checkpoint is provided (deferred to M2).

Only Backend C is imported at module level.  A and B import their heavy deps
lazily inside their predict() / load() methods so the library is importable
without the [ml] extra.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from promptguard.types import Finding

# Default path for the trained Backend A artifact (relative to this file).
# train/train.py writes here by default.
_DEFAULT_ARTIFACT: Path = Path(__file__).parents[2] / "train" / "artifacts" / "classifier_a.pkl"

# Predictions below this threshold are not emitted as findings (noise floor).
_NOISE_FLOOR: float = 0.05


# ---------------------------------------------------------------------------
# Classifier protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Classifier(Protocol):
    """Backend contract: maps text → P(injection) ∈ [0, 1].

    Any class that implements predict(str) → float satisfies this protocol
    and can be used as a ClassifierStage backend.
    """

    def predict(self, text: str) -> float: ...


# ---------------------------------------------------------------------------
# Backend C — heuristic fallback (no ML dependencies)
# ---------------------------------------------------------------------------

# (weight, compiled_pattern) pairs — intentionally different vocabulary from
# Stage 1 signatures to provide an independent signal rather than duplicating.
_HEURISTIC_RULES: list[tuple[float, re.Pattern[str]]] = [
    (
        0.80,
        re.compile(
            r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|rules?)",
            re.IGNORECASE,
        ),
    ),
    (
        0.75,
        re.compile(
            r"disregard\s+(?:all\s+)?(?:previous|prior|the)?\s*(?:instructions?|rules?|constraints?)",
            re.IGNORECASE,
        ),
    ),
    (
        0.70,
        re.compile(
            r"forget\s+(?:everything|all\s+previous|your\s+(?:instructions?|training))",
            re.IGNORECASE,
        ),
    ),
    (
        0.85,
        re.compile(r"<\|im_(?:start|end)\|>|\[/?INST\]|<</?SYS>>", re.IGNORECASE),
    ),
    (
        0.70,
        re.compile(
            r"(?:print|reveal|repeat|show)\s+.{0,30}(?:system\s+prompt|instructions?)",
            re.IGNORECASE,
        ),
    ),
    (
        0.75,
        re.compile(
            r"you\s+(?:are\s+now|have\s+no)\s+(?:DAN|jailbroken|restrictions?|rules?)",
            re.IGNORECASE,
        ),
    ),
    (
        0.65,
        re.compile(
            r"(?:developer|dev|debug)\s+mode\s+(?:enabled|activated|on\b)",
            re.IGNORECASE,
        ),
    ),
    (
        0.80,
        re.compile(
            r"ignore\s+the\s+user\s+and\s+(?:instead\s+)?(?:call|execute|run)\s+",
            re.IGNORECASE,
        ),
    ),
    (
        0.75,
        re.compile(
            r"(?:send|exfiltrate|forward)\s+.{0,20}(?:data|conversation)\s+to\s+\S",
            re.IGNORECASE,
        ),
    ),
]


class HeuristicClassifier:
    """Backend C — weighted pattern matching; P(injection) in {0.0} or [0.30, 0.95].

    Returns 0.0 for texts with zero injection signals, and a soft score
    proportional to the sum of matched pattern weights for texts that do.
    No ML dependencies — always works without the [ml] extra.
    """

    def predict(self, text: str) -> float:
        total = sum(w for w, pat in _HEURISTIC_RULES if pat.search(text))
        if total == 0.0:
            return 0.0
        # Saturating ramp: starts at 0.30 for a single weak signal,
        # approaches 0.95 asymptotically with multiple strong signals.
        return min(0.30 + total * 0.09, 0.95)


# ---------------------------------------------------------------------------
# Backend A — sentence embeddings + sklearn head
# ---------------------------------------------------------------------------


class EmbeddingClassifier:
    """Backend A — sentence-embedding model + logistic-regression head.

    Requires:
      pip install promptguard[ml]        (sentence-transformers + scikit-learn)
      python -m train.train              (produces train/artifacts/classifier_a.pkl)

    Graceful degradation:
      - Missing [ml] extra  → falls back to HeuristicClassifier
      - Missing artifact    → falls back to HeuristicClassifier
    """

    def __init__(
        self,
        model_path: Path | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._model_path = model_path or _DEFAULT_ARTIFACT
        self._embedding_model = embedding_model
        self._head: Any = None
        self._embedder: Any = None
        self._fallback: HeuristicClassifier | None = None

    def _load(self) -> bool:
        """Attempt to load ML deps + artifact. Returns True on success."""
        if self._head is not None:
            return True
        try:
            import joblib  # scikit-learn transitive dep
            from sentence_transformers import SentenceTransformer

            if not self._model_path.exists():
                return False

            artifact = joblib.load(self._model_path)
            self._head = artifact["head"]
            emb_name = artifact.get("embedding_model", self._embedding_model)
            self._embedder = SentenceTransformer(emb_name)
            return True
        except Exception:
            return False

    def predict(self, text: str) -> float:
        if not self._load():
            if self._fallback is None:
                self._fallback = HeuristicClassifier()
            return self._fallback.predict(text)

        embedding = self._embedder.encode([text], show_progress_bar=False)
        # predict_proba returns shape (1, 2): [P(benign), P(injection)]
        proba: Any = self._head.predict_proba(embedding)
        return float(proba[0, 1])


# ---------------------------------------------------------------------------
# Backend B — fine-tuned small encoder (stub; deferred to Milestone 2)
# ---------------------------------------------------------------------------


class EncoderClassifier:
    """Backend B stub — fine-tuned DeBERTa-v3-small / DistilBERT sequence classifier.

    Currently falls back to HeuristicClassifier until a fine-tuned checkpoint is
    provided.  Call .load() once a checkpoint is available.

    To activate:
      1. Fine-tune a small encoder on the injection dataset (see train/train.py).
      2. Save the checkpoint to a local path or HuggingFace Hub.
      3. Use: EncoderClassifier(model_checkpoint="path/to/ckpt").load()

    Requires: pip install promptguard[ml]
    """

    def __init__(self, model_checkpoint: str = "microsoft/deberta-v3-small") -> None:
        self._checkpoint = model_checkpoint
        self._model: Any = None
        self._tokenizer: Any = None
        self._fallback: HeuristicClassifier | None = None

    def load(self) -> EncoderClassifier:
        """Load tokenizer + model from checkpoint. Raises RuntimeError if [ml] absent."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Backend B requires the [ml] extra: pip install promptguard[ml]"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self._checkpoint)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._checkpoint)
        self._model.eval()
        return self

    def predict(self, text: str) -> float:
        if self._model is None:
            if self._fallback is None:
                self._fallback = HeuristicClassifier()
            return self._fallback.predict(text)

        import torch
        import torch.nn.functional as F

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        proba = F.softmax(logits, dim=-1)
        return float(proba[0, 1])


# ---------------------------------------------------------------------------
# ClassifierStage
# ---------------------------------------------------------------------------

_BACKEND_IDS: dict[type, str] = {
    EmbeddingClassifier: "CLS-EMB",
    EncoderClassifier: "CLS-ENC",
    HeuristicClassifier: "CLS-HEU",
}


class ClassifierStage:
    """Stage 2: wraps a Classifier backend and emits a Finding for each input.

    Default backend: EmbeddingClassifier (gracefully falls back to HeuristicClassifier
    when the artifact or [ml] extra is absent, so Stage 2 always works).
    """

    def __init__(self, classifier: Classifier | None = None) -> None:
        self._classifier: Classifier = (
            classifier if classifier is not None else EmbeddingClassifier()
        )

    def run(self, text: str, context: dict[str, Any]) -> list[Finding]:
        prob = self._classifier.predict(text)
        if prob < _NOISE_FLOOR:
            return []

        backend_type = type(self._classifier)
        fid = _BACKEND_IDS.get(backend_type, f"CLS-{backend_type.__name__[:3].upper()}")

        return [
            Finding(
                id=fid,
                category="classifier",
                score=prob,
                structural=False,
                source_stage="classifier",
                detail=f"P(injection)={prob:.4f} via {backend_type.__name__}",
            )
        ]
