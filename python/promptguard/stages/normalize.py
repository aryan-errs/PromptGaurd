"""Stage 0 — Normalization & de-obfuscation.

Pipeline:
  1. NFKC — collapses fullwidth/compatibility variants silently.
  2. Zero-width chars stripped → NORM-ZW finding (score scales with count).
  3. Bidi control chars stripped → NORM-BIDI finding (high fixed score).
  4. Homoglyphs folded to ASCII equivalents → NORM-HOMO finding.
  5. Encoding detection: base64 / hex blobs decoded and recursively re-scanned
     (depth-limited to MAX_DECODE_DEPTH) → NORM-B64 / NORM-HEX findings.

Normalised text is stored in context["normalized_text"] for subsequent stages.
"""

from __future__ import annotations

import base64
import re
import unicodedata
from typing import Any

from promptguard.stages._confusables import CONFUSABLES
from promptguard.types import Finding

# ---------------------------------------------------------------------------
# Character sets
# ---------------------------------------------------------------------------

_ZERO_WIDTH: frozenset[str] = frozenset(
    {
        "­",  # SOFT HYPHEN
        "​",  # ZERO WIDTH SPACE
        "‌",  # ZERO WIDTH NON-JOINER
        "‍",  # ZERO WIDTH JOINER
        "⁠",  # WORD JOINER
        "⁡",  # FUNCTION APPLICATION (invisible)
        "⁢",  # INVISIBLE TIMES
        "⁣",  # INVISIBLE SEPARATOR
        "⁤",  # INVISIBLE PLUS
        "﻿",  # ZERO WIDTH NO-BREAK SPACE / BOM
    }
)

_BIDI_CONTROLS: frozenset[str] = frozenset(
    {
        "\u200e",  # LEFT-TO-RIGHT MARK
        "\u200f",  # RIGHT-TO-LEFT MARK
        "\u202a",  # LEFT-TO-RIGHT EMBEDDING
        "\u202b",  # RIGHT-TO-LEFT EMBEDDING
        "\u202c",  # POP DIRECTIONAL FORMATTING
        "\u202d",  # LEFT-TO-RIGHT OVERRIDE
        "\u202e",  # RIGHT-TO-LEFT OVERRIDE  ← classic attack char
        "\u2066",  # LEFT-TO-RIGHT ISOLATE
        "\u2067",  # RIGHT-TO-LEFT ISOLATE
        "\u2068",  # FIRST STRONG ISOLATE
        "\u2069",  # POP DIRECTIONAL ISOLATE
    }
)

# ---------------------------------------------------------------------------
# Encoding patterns
# ---------------------------------------------------------------------------

# Standard base64: 16+ chars of [A-Za-z0-9+/] with optional = padding.
# Lookbehind/ahead prevent matching in the middle of a longer base64 run.
_B64_RE = re.compile(r"(?<![A-Za-z0-9+/])([A-Za-z0-9+/]{16,}={0,2})(?![A-Za-z0-9+/=])")

# URL-safe base64 uses - and _ instead of + and /
_B64_URLSAFE_RE = re.compile(r"(?<![A-Za-z0-9\-_])([A-Za-z0-9\-_]{16,})(?![A-Za-z0-9\-_=])")

# Hex: optional 0x prefix, minimum 16 hex digits (encodes ≥8 bytes of content).
_HEX_RE = re.compile(r"(?<![0-9a-fA-F])((?:0x)?[0-9a-fA-F]{16,})(?![0-9a-fA-F])")

MAX_DECODE_DEPTH: int = 3

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _strip_charset(text: str, charset: frozenset[str]) -> tuple[str, int]:
    """Remove all chars in charset; return (cleaned_text, removed_count)."""
    cleaned = [c for c in text if c not in charset]
    return "".join(cleaned), len(text) - len(cleaned)


def _fold_confusables(text: str) -> tuple[str, int]:
    """Replace confusable non-ASCII chars with their ASCII equivalents."""
    out: list[str] = []
    subs = 0
    for ch in text:
        mapped = CONFUSABLES.get(ch)
        if mapped is not None:
            out.append(mapped)
            subs += 1
        else:
            out.append(ch)
    return "".join(out), subs


def _try_b64(blob: str) -> bytes | None:
    """Attempt standard and URL-safe base64 decode; return bytes or None."""
    for variant in (blob, blob.replace("-", "+").replace("_", "/")):
        # Add padding so length is a multiple of 4
        padded = variant + "=" * (-len(variant) % 4)
        try:
            data = base64.b64decode(padded, validate=True)
            data.decode("utf-8")  # verify decodable as text
            return data
        except Exception:
            continue
    return None


def _try_hex(blob: str) -> bytes | None:
    """Attempt hex decode; return bytes or None."""
    s = blob[2:] if blob.startswith(("0x", "0X")) else blob
    if len(s) % 2 != 0:
        return None
    try:
        data = bytes.fromhex(s)
        data.decode("utf-8")
        return data
    except Exception:
        return None


def _contains_decodable_natural_language(text: str) -> bool:
    """True if text contains a b64/hex blob that itself decodes to text with spaces.

    Used to detect nested encoding: the outer blob decodes to a string that has
    no spaces itself but contains an inner blob that does.
    """
    for m in _B64_RE.finditer(text):
        data = _try_b64(m.group(1))
        if data:
            try:
                if " " in data.decode("utf-8"):
                    return True
            except UnicodeDecodeError:
                pass
    for m in _HEX_RE.finditer(text):
        data = _try_hex(m.group(1))
        if data:
            try:
                if " " in data.decode("utf-8"):
                    return True
            except UnicodeDecodeError:
                pass
    return False


def _is_plausible_text(data: bytes, depth: int) -> bool:
    """True if decoded bytes look like natural language (not a hash or opaque token).

    Filters out binary blobs, random API tokens, and hashes, while preserving:
    - Natural language (has spaces).
    - Nested encodings (no spaces, but the next decode level produces natural language).
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False

    printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
    if printable / max(len(text), 1) <= 0.8:
        return False

    # Natural language — the common case
    if " " in text:
        return True

    # Nested encoding: the decoded text is itself an encoded blob.
    # One extra decode level is enough to establish suspicion; we don't recurse
    # further here (the actual recursive normalise pass handles deeper levels).
    if depth < MAX_DECODE_DEPTH:
        return _contains_decodable_natural_language(text)

    return False


# ---------------------------------------------------------------------------
# Core recursive normalisation pass
# ---------------------------------------------------------------------------


def _normalize(text: str, findings: list[Finding], depth: int) -> str:
    """One normalisation pass; mutates findings in place, returns cleaned text."""
    if depth > MAX_DECODE_DEPTH:
        return text

    # 1. NFKC: silently collapses fullwidth, ligatures, compat decompositions
    text = unicodedata.normalize("NFKC", text)

    # 2. Zero-width chars
    text, zw_count = _strip_charset(text, _ZERO_WIDTH)
    if zw_count:
        score = min(0.15 + zw_count * 0.07, 0.85)
        findings.append(
            Finding(
                id="NORM-ZW",
                category="obfuscation.zero_width",
                score=score,
                structural=False,
                source_stage="normalize",
                detail=f"stripped {zw_count} zero-width character(s)",
            )
        )

    # 3. Bidi control chars (always high score — deliberate when present)
    text, bidi_count = _strip_charset(text, _BIDI_CONTROLS)
    if bidi_count:
        findings.append(
            Finding(
                id="NORM-BIDI",
                category="obfuscation.bidi_control",
                score=0.55,
                structural=False,
                source_stage="normalize",
                detail=f"stripped {bidi_count} bidi control character(s)",
            )
        )

    # 4. Homoglyph folding
    text, sub_count = _fold_confusables(text)
    if sub_count:
        score = min(0.10 + sub_count * 0.09, 0.80)
        findings.append(
            Finding(
                id="NORM-HOMO",
                category="obfuscation.homoglyph",
                score=score,
                structural=False,
                source_stage="normalize",
                detail=f"folded {sub_count} homoglyph(s) to ASCII equivalents",
            )
        )

    # 5. Encoding detection — only recurse if budget allows
    if depth < MAX_DECODE_DEPTH:
        _detect_encodings(text, findings, depth)

    return text


def _detect_encodings(text: str, findings: list[Finding], depth: int) -> None:
    """Scan for base64/hex blobs, decode them, and recursively re-normalise."""
    seen: set[str] = set()

    # --- Base64 (standard) ---
    for m in _B64_RE.finditer(text):
        blob = m.group(1)
        if blob in seen:
            continue
        data = _try_b64(blob)
        if data and _is_plausible_text(data, depth):
            seen.add(blob)
            decoded = data.decode("utf-8")
            # Score raised to 0.62 at depth=0 so even medium-risk profiles
            # (sanitize threshold 0.45) trigger a sanitize action on encoded attacks.
            score = min(0.62 + depth * 0.15, 0.90)
            findings.append(
                Finding(
                    id="NORM-B64",
                    category="obfuscation.encoding.base64",
                    score=score,
                    structural=False,
                    source_stage="normalize",
                    detail=f"depth={depth} base64 blob ({len(blob)} chars) → {len(decoded)} chars decoded",
                )
            )
            _normalize(decoded, findings, depth + 1)

    # --- Base64 (URL-safe) — only if not already caught by standard pass ---
    for m in _B64_URLSAFE_RE.finditer(text):
        blob = m.group(1)
        if blob in seen:
            continue
        # Standard pass already handles if it has +/; skip pure-alnum overlap
        if _B64_RE.fullmatch(blob):
            continue
        data = _try_b64(blob)
        if data and _is_plausible_text(data, depth):
            seen.add(blob)
            decoded = data.decode("utf-8")
            score = min(0.40 + depth * 0.15, 0.90)
            findings.append(
                Finding(
                    id="NORM-B64",
                    category="obfuscation.encoding.base64",
                    score=score,
                    structural=False,
                    source_stage="normalize",
                    detail=f"depth={depth} url-safe base64 blob ({len(blob)} chars) → {len(decoded)} chars decoded",
                )
            )
            _normalize(decoded, findings, depth + 1)

    # --- Hex ---
    for m in _HEX_RE.finditer(text):
        blob = m.group(1)
        if blob in seen:
            continue
        data = _try_hex(blob)
        if data and _is_plausible_text(data, depth):
            seen.add(blob)
            decoded = data.decode("utf-8")
            score = min(0.62 + depth * 0.15, 0.90)
            findings.append(
                Finding(
                    id="NORM-HEX",
                    category="obfuscation.encoding.hex",
                    score=score,
                    structural=False,
                    source_stage="normalize",
                    detail=f"depth={depth} hex blob ({len(blob)} chars) → {len(decoded)} chars decoded",
                )
            )
            _normalize(decoded, findings, depth + 1)


# ---------------------------------------------------------------------------
# Public stage class
# ---------------------------------------------------------------------------


class NormalizeStage:
    """Stage 0: unicode normalization, invisible-char removal, confusable folding,
    and recursive encoding detection.

    Stores the fully-normalised text in context["normalized_text"] so subsequent
    stages operate on clean input.
    """

    def run(self, text: str, context: dict[str, Any]) -> list[Finding]:
        findings: list[Finding] = []
        normalized = _normalize(text, findings, depth=0)
        context["normalized_text"] = normalized
        return findings
