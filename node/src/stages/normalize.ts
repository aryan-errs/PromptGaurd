/**
 * Stage 0 — Normalization & de-obfuscation.
 * Mirrors stages/normalize.py — keep logic in sync when updating either.
 *
 * Pipeline:
 *   1. NFKC normalization (JS String.prototype.normalize)
 *   2. Zero-width char stripping → NORM-ZW finding
 *   3. Bidi control char stripping → NORM-BIDI finding
 *   4. Homoglyph folding → NORM-HOMO finding
 *   5. Base64/hex blob detection + recursive re-scan → NORM-B64/NORM-HEX findings
 *
 * Stores normalized text in context.normalizedText for downstream stages.
 */

import { CONFUSABLES } from "./confusables.js";
import { type Finding, type PipelineContext, type Stage, makeFinding } from "../types.js";

// ---------------------------------------------------------------------------
// Invisible character sets
// ---------------------------------------------------------------------------

const ZERO_WIDTH = new Set([
  "­", // SOFT HYPHEN
  "​", // ZERO WIDTH SPACE
  "‌", // ZERO WIDTH NON-JOINER
  "‍", // ZERO WIDTH JOINER
  "⁠", // WORD JOINER
  "⁡", // FUNCTION APPLICATION
  "⁢", // INVISIBLE TIMES
  "⁣", // INVISIBLE SEPARATOR
  "⁤", // INVISIBLE PLUS
  "﻿", // ZERO WIDTH NO-BREAK SPACE / BOM
]);

const BIDI_CONTROLS = new Set([
  "‎", // LEFT-TO-RIGHT MARK
  "‏", // RIGHT-TO-LEFT MARK
  "‪", // LEFT-TO-RIGHT EMBEDDING
  "‫", // RIGHT-TO-LEFT EMBEDDING
  "‬", // POP DIRECTIONAL FORMATTING
  "‭", // LEFT-TO-RIGHT OVERRIDE
  "‮", // RIGHT-TO-LEFT OVERRIDE
  "⁦", // LEFT-TO-RIGHT ISOLATE
  "⁧", // RIGHT-TO-LEFT ISOLATE
  "⁨", // FIRST STRONG ISOLATE
  "⁩", // POP DIRECTIONAL ISOLATE
]);

// ---------------------------------------------------------------------------
// Encoding patterns
// ---------------------------------------------------------------------------

// Standard base64: 16+ chars, optional padding.
// Lookbehind/ahead prevent matching mid-run.
const B64_RE = /(?<![A-Za-z0-9+/])([A-Za-z0-9+/]{16,}={0,2})(?![A-Za-z0-9+/=])/g;
const B64_URLSAFE_RE = /(?<![A-Za-z0-9\-_])([A-Za-z0-9\-_]{16,})(?![A-Za-z0-9\-_=])/g;
// Hex: optional 0x prefix, min 16 hex digits.
const HEX_RE = /(?<![0-9a-fA-F])((?:0x)?[0-9a-fA-F]{16,})(?![0-9a-fA-F])/g;

export const MAX_DECODE_DEPTH = 3;

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

function stripCharset(text: string, chars: Set<string>): [string, number] {
  const out: string[] = [];
  for (const ch of text) {
    if (!chars.has(ch)) out.push(ch);
  }
  const cleaned = out.join("");
  return [cleaned, text.length - cleaned.length];
}

function foldConfusables(text: string): [string, number] {
  const out: string[] = [];
  let subs = 0;
  for (const ch of text) {
    const mapped = CONFUSABLES[ch];
    if (mapped !== undefined) {
      out.push(mapped);
      subs++;
    } else {
      out.push(ch);
    }
  }
  return [out.join(""), subs];
}

function tryB64(blob: string): string | null {
  for (const variant of [blob, blob.replace(/-/g, "+").replace(/_/g, "/")]) {
    const pad = variant.length % 4 === 0 ? 0 : 4 - (variant.length % 4);
    const padded = variant + "=".repeat(pad);
    try {
      const buf = Buffer.from(padded, "base64");
      // Validate by re-encoding (Node is lenient; we need round-trip check)
      const reenc = buf.toString("base64").replace(/={0,2}$/, "");
      const orig = padded.replace(/={0,2}$/, "");
      if (reenc !== orig) continue;
      return buf.toString("utf8");
    } catch {
      continue;
    }
  }
  return null;
}

function tryHex(blob: string): string | null {
  const s = /^0x/i.test(blob) ? blob.slice(2) : blob;
  if (s.length % 2 !== 0) return null;
  if (!/^[0-9a-fA-F]+$/.test(s)) return null;
  try {
    const buf = Buffer.from(s, "hex");
    if (buf.length !== s.length / 2) return null;
    return buf.toString("utf8");
  } catch {
    return null;
  }
}

/** True if decoded text looks like natural language (not a hash or opaque token). */
function containsDecodableNaturalLanguage(text: string): boolean {
  // Look for nested b64/hex that decodes to text with spaces
  for (const m of text.matchAll(new RegExp(B64_RE.source, "g"))) {
    const decoded = tryB64(m[1] ?? "");
    if (decoded?.includes(" ")) return true;
  }
  for (const m of text.matchAll(new RegExp(HEX_RE.source, "g"))) {
    const decoded = tryHex(m[1] ?? "");
    if (decoded?.includes(" ")) return true;
  }
  return false;
}

function isPlausibleText(text: string, depth: number): boolean {
  // Must not contain replacement char (indicates invalid UTF-8 after decoding)
  if (text.includes("�")) return false;
  // Must be >80% printable
  const printable = [...text].filter(
    (c) => c >= " " || c === "\n" || c === "\r" || c === "\t",
  ).length;
  if (printable / Math.max(text.length, 1) <= 0.8) return false;
  // Natural language has spaces
  if (text.includes(" ")) return true;
  // Nested encoding: decoded text is itself an encoded blob
  if (depth < MAX_DECODE_DEPTH) return containsDecodableNaturalLanguage(text);
  return false;
}

// ---------------------------------------------------------------------------
// Core recursive normalisation
// ---------------------------------------------------------------------------

function detectEncodings(text: string, findings: Finding[], depth: number): void {
  if (depth >= MAX_DECODE_DEPTH) return;
  const seen = new Set<string>();

  // Base64 (standard)
  for (const m of text.matchAll(new RegExp(B64_RE.source, "g"))) {
    const blob = m[1] ?? "";
    if (seen.has(blob)) continue;
    const decoded = tryB64(blob);
    if (decoded && isPlausibleText(decoded, depth)) {
      seen.add(blob);
      const score = Math.min(0.62 + depth * 0.15, 0.9);
      findings.push(
        makeFinding({
          id: "NORM-B64",
          category: "obfuscation.encoding.base64",
          score,
          structural: false,
          sourceStage: "normalize",
          detail: `depth=${depth} base64 blob (${blob.length} chars) → ${decoded.length} chars decoded`,
        }),
      );
      normalize(decoded, findings, depth + 1);
    }
  }

  // Base64 (URL-safe) — skip if already caught by standard pass
  for (const m of text.matchAll(new RegExp(B64_URLSAFE_RE.source, "g"))) {
    const blob = m[1] ?? "";
    if (seen.has(blob)) continue;
    if (new RegExp(B64_RE.source).test(blob)) continue; // already handled
    const decoded = tryB64(blob);
    if (decoded && isPlausibleText(decoded, depth)) {
      seen.add(blob);
      const score = Math.min(0.62 + depth * 0.15, 0.9);
      findings.push(
        makeFinding({
          id: "NORM-B64",
          category: "obfuscation.encoding.base64",
          score,
          structural: false,
          sourceStage: "normalize",
          detail: `depth=${depth} url-safe base64 blob (${blob.length} chars) → ${decoded.length} chars decoded`,
        }),
      );
      normalize(decoded, findings, depth + 1);
    }
  }

  // Hex
  for (const m of text.matchAll(new RegExp(HEX_RE.source, "g"))) {
    const blob = m[1] ?? "";
    if (seen.has(blob)) continue;
    const decoded = tryHex(blob);
    if (decoded && isPlausibleText(decoded, depth)) {
      seen.add(blob);
      const score = Math.min(0.62 + depth * 0.15, 0.9);
      findings.push(
        makeFinding({
          id: "NORM-HEX",
          category: "obfuscation.encoding.hex",
          score,
          structural: false,
          sourceStage: "normalize",
          detail: `depth=${depth} hex blob (${blob.length} chars) → ${decoded.length} chars decoded`,
        }),
      );
      normalize(decoded, findings, depth + 1);
    }
  }
}

export function normalize(text: string, findings: Finding[], depth: number): string {
  if (depth > MAX_DECODE_DEPTH) return text;

  // 1. NFKC
  text = text.normalize("NFKC");

  // 2. Zero-width chars
  {
    const [cleaned, count] = stripCharset(text, ZERO_WIDTH);
    if (count > 0) {
      findings.push(
        makeFinding({
          id: "NORM-ZW",
          category: "obfuscation.zero_width",
          score: Math.min(0.15 + count * 0.07, 0.85),
          structural: false,
          sourceStage: "normalize",
          detail: `stripped ${count} zero-width character(s)`,
        }),
      );
      text = cleaned;
    }
  }

  // 3. Bidi control chars
  {
    const [cleaned, count] = stripCharset(text, BIDI_CONTROLS);
    if (count > 0) {
      findings.push(
        makeFinding({
          id: "NORM-BIDI",
          category: "obfuscation.bidi_control",
          score: 0.55,
          structural: false,
          sourceStage: "normalize",
          detail: `stripped ${count} bidi control character(s)`,
        }),
      );
      text = cleaned;
    }
  }

  // 4. Homoglyph folding
  {
    const [folded, subs] = foldConfusables(text);
    if (subs > 0) {
      findings.push(
        makeFinding({
          id: "NORM-HOMO",
          category: "obfuscation.homoglyph",
          score: Math.min(0.1 + subs * 0.09, 0.8),
          structural: false,
          sourceStage: "normalize",
          detail: `folded ${subs} homoglyph(s) to ASCII equivalents`,
        }),
      );
      text = folded;
    }
  }

  // 5. Encoding detection
  if (depth < MAX_DECODE_DEPTH) detectEncodings(text, findings, depth);

  return text;
}

// ---------------------------------------------------------------------------
// Stage class
// ---------------------------------------------------------------------------

export class NormalizeStage implements Stage {
  run(text: string, context: PipelineContext): Finding[] {
    const findings: Finding[] = [];
    const normalized = normalize(text, findings, 0);
    context.normalizedText = normalized;
    return findings;
  }
}
