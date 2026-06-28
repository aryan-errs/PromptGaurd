/**
 * Decision policy — mirrors policy.py exactly.
 *
 * Hard rule: structural finding → block, regardless of profile/tier/boost.
 * Semantic findings use per-tier thresholds raised by profile and intent boosts.
 */

import { type AppProfile, type Finding, Verdict, type VerdictAction } from "./types.js";

// ---------------------------------------------------------------------------
// Thresholds
// ---------------------------------------------------------------------------

interface TierThresholds {
  sanitize: number;
  flag: number;
  block: number;
}

const TIERS: Record<string, TierThresholds> = {
  high: { sanitize: 0.3, flag: 0.5, block: 0.8 },
  medium: { sanitize: 0.45, flag: 0.65, block: 0.88 },
  low: { sanitize: 0.7, flag: 0.82, block: 0.93 },
};

export const SECURITY_DISCUSSION_BOOST = 0.2;
export const INTENT_MAX_BOOST = 0.15;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function intentBoost(findings: Finding[]): number {
  const intentFindings = findings.filter((f) => f.category === "intent");
  if (intentFindings.length === 0) return 0;
  const mentionScore = Math.max(...intentFindings.map((f) => f.score));
  return mentionScore * INTENT_MAX_BOOST;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export function decide(
  findings: Finding[],
  profile: AppProfile,
  latencyMs: number,
): Verdict {
  // ── Hard rule: structural → block ─────────────────────────────────────
  const structural = findings.filter((f) => f.structural);
  if (structural.length > 0) {
    const score = Math.max(...findings.map((f) => f.score));
    return new Verdict("block", score, findings, latencyMs);
  }

  // ── No findings → allow ───────────────────────────────────────────────
  const detectionFindings = findings.filter((f) => f.category !== "intent");
  if (detectionFindings.length === 0) {
    return new Verdict("allow", 0, findings, latencyMs);
  }

  // ── Semantic path ─────────────────────────────────────────────────────
  const semanticScore = Math.max(...detectionFindings.map((f) => f.score));
  const tier = TIERS[profile.riskTier] ?? TIERS["medium"]!;
  const profileBoost = profile.allowSecurityDiscussion ? SECURITY_DISCUSSION_BOOST : 0;
  const totalBoost = profileBoost + intentBoost(findings);

  const tBlock = Math.min(tier.block + totalBoost, 1.0);
  const tFlag = Math.min(tier.flag + totalBoost, 1.0);
  const tSanitize = Math.min(tier.sanitize + totalBoost, 1.0);

  let action: VerdictAction;
  if (semanticScore >= tBlock) action = "block";
  else if (semanticScore >= tFlag) action = "flag";
  else if (semanticScore >= tSanitize) action = "sanitize";
  else action = "allow";

  return new Verdict(action, semanticScore, findings, latencyMs);
}

/** Returns effective thresholds for a profile + optional mention_score (for tests/debugging). */
export function effectiveThresholds(
  profile: AppProfile,
  mentionScore = 0,
): TierThresholds {
  const tier = TIERS[profile.riskTier] ?? TIERS["medium"]!;
  const profileBoost = profile.allowSecurityDiscussion ? SECURITY_DISCUSSION_BOOST : 0;
  const total = profileBoost + mentionScore * INTENT_MAX_BOOST;
  return {
    sanitize: Math.min(tier.sanitize + total, 1.0),
    flag: Math.min(tier.flag + total, 1.0),
    block: Math.min(tier.block + total, 1.0),
  };
}
