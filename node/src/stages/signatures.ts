/**
 * Stage 1 — Signature engine.
 * Mirrors stages/signatures.py — shares the same rules.yaml source file.
 *
 * Shared rule source
 * ──────────────────
 * rules.yaml lives in python/promptguard/stages/rules.yaml and is loaded at
 * runtime by both the Python and Node implementations.  This is the canonical
 * drift-prevention mechanism: edit one file, both runtimes pick up the change.
 *
 * Tradeoff vs. WASM/native core: no build step; both parsers apply the same
 * regex source string but compile it independently.  If a regex feature is
 * supported by Python `re` but not by JS (or vice versa), the pattern must be
 * written to the lowest common denominator.  Current patterns use only
 * standard `\b`, `\s`, `\d`, `(?:...)`, `{n,m}`, `|`, `^` with IGNORECASE+
 * MULTILINE which are identical in both engines.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import yaml from "js-yaml";

import { type AppProfile, type Finding, type PipelineContext, type Stage, makeFinding } from "../types.js";

const _dir = dirname(fileURLToPath(import.meta.url));
// Resolve relative to this file so the path works from both src/ and dist/
// 3 levels up: stages/ → src/ → node/ → Sentinal/
const RULES_YAML_PATH = resolve(_dir, "..", "..", "..", "python", "promptguard", "stages", "rules.yaml");

// ---------------------------------------------------------------------------
// Rule types
// ---------------------------------------------------------------------------

interface RawRule {
  id: string;
  category: string;
  pattern: string;
  weight: number;
  structural: boolean;
}

interface CompiledRule {
  id: string;
  category: string;
  pattern: RegExp;
  weight: number;
  structural: boolean;
}

// ---------------------------------------------------------------------------
// Rule loading (lazy, cached per instance)
// ---------------------------------------------------------------------------

let _cachedRules: CompiledRule[] | null = null;

function loadStaticRules(): CompiledRule[] {
  if (_cachedRules !== null) return _cachedRules;
  const raw = readFileSync(RULES_YAML_PATH, "utf8");
  const data = yaml.load(raw) as { rules: RawRule[] };
  _cachedRules = data.rules.map((r) => ({
    id: r.id,
    category: r.category,
    // Python uses IGNORECASE|MULTILINE; mirror with 'im' flags.
    pattern: new RegExp(r.pattern, "im"),
    weight: r.weight,
    structural: r.structural,
  }));
  return _cachedRules;
}

function makeDelimiterRule(delimiter: string, idx: number): CompiledRule {
  return {
    id: `SIG-DELIM-APP-${String(idx).padStart(3, "0")}`,
    category: "delimiter_breakout",
    pattern: new RegExp(escapeRegex(delimiter), "im"),
    weight: 0.98,
    structural: true,
  };
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ---------------------------------------------------------------------------
// Stage
// ---------------------------------------------------------------------------

export class SignatureStage implements Stage {
  private _staticRules: CompiledRule[] | null = null;

  private getStaticRules(): CompiledRule[] {
    if (this._staticRules === null) {
      this._staticRules = loadStaticRules();
    }
    return this._staticRules;
  }

  run(text: string, context: PipelineContext): Finding[] {
    const profile = context.profile as AppProfile;
    const rules = [
      ...this.getStaticRules(),
      ...profile.templateDelimiters.map((d, i) => makeDelimiterRule(d, i)),
    ];

    const findings: Finding[] = [];
    for (const rule of rules) {
      // Count all matches (reset lastIndex-safe via non-global re)
      const globalRe = new RegExp(rule.pattern.source, rule.pattern.flags.includes("g") ? rule.pattern.flags : rule.pattern.flags + "g");
      const matches = [...text.matchAll(globalRe)];
      if (matches.length === 0) continue;

      const score = Math.min(rule.weight + (matches.length - 1) * 0.03, 1.0);
      const snippet = (matches[0]?.[0] ?? "").slice(0, 80);
      findings.push(
        makeFinding({
          id: rule.id,
          category: rule.category,
          score,
          structural: rule.structural,
          sourceStage: "signatures",
          detail: `${matches.length} match(es); first: ${JSON.stringify(snippet)}`,
        }),
      );
    }
    return findings;
  }
}

/** Exported for tests */
export { loadStaticRules };
