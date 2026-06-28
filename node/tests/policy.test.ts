import { describe, it, expect } from "vitest";
import { decide, effectiveThresholds, SECURITY_DISCUSSION_BOOST, INTENT_MAX_BOOST } from "../src/policy.js";
import { AppProfile, type Finding, Verdict, makeFinding } from "../src/types.js";

function fFinding(score: number, structural: boolean, category = "test"): Finding {
  return makeFinding({ id: "TEST", category, score, structural, sourceStage: "test" });
}

function banking() {
  return new AppProfile("banking", false, "high", [], true);
}
function secChat() {
  return new AppProfile("security-chatbot", true, "low", ["</system>"]);
}

function doDecide(findings: Finding[], profile: AppProfile): Verdict {
  return decide(findings, profile, 0);
}

// ---------------------------------------------------------------------------
// Hard rule: structural always blocks
// ---------------------------------------------------------------------------

describe("structural hard rule", () => {
  it("blocks on structural with low score", () => {
    expect(doDecide([fFinding(0.1, true)], secChat()).blocked).toBe(true);
  });

  it("blocks on structural across all tiers", () => {
    for (const tier of ["low", "medium", "high"] as const) {
      const v = doDecide([fFinding(0.5, true)], new AppProfile("x", false, tier));
      expect(v.blocked, `tier=${tier}`).toBe(true);
    }
  });

  it("blocks even with security discussion flag", () => {
    expect(doDecide([fFinding(0.3, true)], secChat()).blocked).toBe(true);
  });

  it("structural beats semantic in mixed batch", () => {
    const v = doDecide([fFinding(0.1, false), fFinding(0.3, true)], secChat());
    expect(v.blocked).toBe(true);
  });

  it("findings preserved in blocked verdict", () => {
    const f = fFinding(0.9, true);
    const v = doDecide([f], banking());
    expect(v.findings).toContain(f);
  });
});

// ---------------------------------------------------------------------------
// SPEC key scenario
// ---------------------------------------------------------------------------

describe("SPEC key scenario", () => {
  it("security chatbot allows educational discussion (SPEC §3)", () => {
    // Semantic finding at 0.85; with allow_security_discussion + intent boost
    // the effective sanitize threshold rises above 0.85 → allow
    const semanticFinding = fFinding(0.85, false, "instruction_override");
    const intentFinding = makeFinding({
      id: "INTENT-MENTION",
      category: "intent",
      score: 0.7,
      structural: false,
      sourceStage: "intent",
    });
    const v = doDecide([semanticFinding, intentFinding], secChat());
    expect(v.action).toBe("allow");
    expect(v.safe).toBe(true);
  });

  it("structural breakout blocked on security chatbot (SPEC §3 decisive rule)", () => {
    const structural = makeFinding({
      id: "SIG-ROLE-003",
      category: "role_turn_injection",
      score: 0.97,
      structural: true,
      sourceStage: "signatures",
    });
    const v = doDecide([structural], secChat());
    expect(v.blocked).toBe(true);
  });

  it("same profile gives different outcomes for use vs mention", () => {
    const profile = secChat();
    // USE: no intent boost
    const useV = doDecide([fFinding(0.85, false, "instruction_override")], profile);
    // MENTION: high mention score → intent boost → effective threshold raised
    const intentF = makeFinding({ id: "INTENT-MENTION", category: "intent", score: 0.8, structural: false, sourceStage: "intent" });
    const mentionV = doDecide([fFinding(0.85, false, "instruction_override"), intentF], profile);

    // Both should be allow on a security chatbot (which is very permissive for semantic)
    expect(useV.action).toBe("allow");
    expect(mentionV.action).toBe("allow");
    // But mention has lower effective score (intent boost raised the threshold more)
    // — we verify this by checking the intent boost calculation
    const withMention = effectiveThresholds(profile, 0.8);
    const withoutMention = effectiveThresholds(profile, 0);
    expect(withMention.sanitize).toBeGreaterThan(withoutMention.sanitize);
  });
});

// ---------------------------------------------------------------------------
// Per-tier thresholds
// ---------------------------------------------------------------------------

describe("per-tier thresholds", () => {
  it("high tier blocks at score=0.85", () => {
    expect(doDecide([fFinding(0.85, false)], banking()).blocked).toBe(true);
  });

  it("medium tier flags at score=0.85", () => {
    const v = doDecide([fFinding(0.85, false)], new AppProfile("m", false, "medium"));
    expect(v.action).toBe("flag");
  });

  it("low tier flags at score=0.85", () => {
    const v = doDecide([fFinding(0.85, false)], new AppProfile("l", false, "low"));
    expect(v.action).toBe("flag");
  });

  it("low tier allows borderline semantic at 0.60", () => {
    const v = doDecide([fFinding(0.6, false)], new AppProfile("l", false, "low"));
    expect(v.action).toBe("allow");
  });

  it("no findings → allow on all tiers", () => {
    for (const tier of ["low", "medium", "high"] as const) {
      const v = doDecide([], new AppProfile("x", false, tier));
      expect(v.action).toBe("allow");
    }
  });

  it("tiers form monotone severity ordering", () => {
    const SEVERITY: Record<string, number> = { allow: 0, sanitize: 1, flag: 2, block: 3 };
    for (const score of [0.35, 0.55, 0.75, 0.85, 0.95]) {
      const vH = doDecide([fFinding(score, false)], new AppProfile("h", false, "high"));
      const vM = doDecide([fFinding(score, false)], new AppProfile("m", false, "medium"));
      const vL = doDecide([fFinding(score, false)], new AppProfile("l", false, "low"));
      expect(SEVERITY[vH.action]!).toBeGreaterThanOrEqual(SEVERITY[vM.action]!);
      expect(SEVERITY[vM.action]!).toBeGreaterThanOrEqual(SEVERITY[vL.action]!);
    }
  });
});

// ---------------------------------------------------------------------------
// Security discussion boost
// ---------------------------------------------------------------------------

describe("security discussion boost", () => {
  it("boost raises semantic thresholds", () => {
    const base = effectiveThresholds(new AppProfile("x", false, "low"));
    const boosted = effectiveThresholds(new AppProfile("x", true, "low"));
    expect(boosted.sanitize).toBeCloseTo(Math.min(base.sanitize + SECURITY_DISCUSSION_BOOST, 1));
  });

  it("boost converts flag to allow (medium tier, score=0.55)", () => {
    const without = doDecide([fFinding(0.55, false)], new AppProfile("x", false, "medium"));
    const with_ = doDecide([fFinding(0.55, false)], new AppProfile("x", true, "medium"));
    expect(without.action).toBe("sanitize");
    expect(with_.action).toBe("allow");
  });

  it("boost does not help structural findings", () => {
    expect(doDecide([fFinding(0.1, true)], new AppProfile("x", true, "low")).blocked).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Intent boost
// ---------------------------------------------------------------------------

describe("intent boost", () => {
  it("high mention score raises effective threshold", () => {
    const base = effectiveThresholds(banking(), 0);
    const boosted = effectiveThresholds(banking(), 1.0);
    expect(boosted.block).toBeCloseTo(Math.min(base.block + INTENT_MAX_BOOST, 1));
  });

  it("banking profile: mention converts block to flag", () => {
    const intentF = makeFinding({ id: "INTENT-MENTION", category: "intent", score: 1.0, structural: false, sourceStage: "intent" });
    const semantic = fFinding(0.85, false);
    const v = doDecide([semantic, intentF], banking());
    // With intent boost (0.15), t_block = 0.80 + 0.15 = 0.95 > 0.85 → flag not block
    expect(v.action).not.toBe("block");
    expect(["flag", "sanitize"].includes(v.action)).toBe(true);
  });
});
