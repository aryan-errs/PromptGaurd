/**
 * Core data types for the PromptGuard Node pipeline.
 * Mirrors promptguard/types.py — keep naming aligned when updating either.
 *
 * Naming convention: Python snake_case → TypeScript camelCase
 *   source_stage → sourceStage
 *   risk_tier    → riskTier
 *   latency_ms   → latencyMs
 */

export type VerdictAction = "allow" | "sanitize" | "flag" | "block";
export type RiskTier = "low" | "medium" | "high";

// ---------------------------------------------------------------------------
// Finding — one detection result from a pipeline stage
// ---------------------------------------------------------------------------

export interface Finding {
  readonly id: string;
  readonly category: string;
  readonly score: number; // [0, 1]
  readonly structural: boolean;
  readonly sourceStage: string;
  readonly detail: string;
}

export function makeFinding(f: Omit<Finding, "detail"> & { detail?: string }): Finding {
  return { ...f, detail: f.detail ?? "" };
}

// ---------------------------------------------------------------------------
// Transformation — one sanitization operation
// ---------------------------------------------------------------------------

export interface Transformation {
  readonly kind: string;
  readonly description: string;
  readonly originalFragment: string;
  readonly transformedFragment: string;
  readonly stage: string;
}

export function makeTransformation(
  t: Omit<Transformation, "stage"> & { stage?: string },
): Transformation {
  return { ...t, stage: t.stage ?? "sanitize" };
}

// ---------------------------------------------------------------------------
// Verdict — aggregate decision
// ---------------------------------------------------------------------------

export class Verdict {
  constructor(
    public readonly action: VerdictAction,
    public readonly score: number,
    public readonly findings: Finding[],
    public readonly latencyMs: number,
    public readonly sanitizedText?: string,
    public readonly transformations: Transformation[] = [],
  ) {}

  get blocked(): boolean {
    return this.action === "block";
  }

  get safe(): boolean {
    return this.action === "allow";
  }
}

// ---------------------------------------------------------------------------
// AppProfile — per-deployment configuration
// ---------------------------------------------------------------------------

export class AppProfile {
  constructor(
    public readonly name: string,
    public readonly allowSecurityDiscussion: boolean = false,
    public readonly riskTier: RiskTier = "medium",
    public readonly templateDelimiters: string[] = [],
    public readonly toolsEnabled: boolean = false,
  ) {}
}

// ---------------------------------------------------------------------------
// Stage protocol — implemented by each pipeline stage
// ---------------------------------------------------------------------------

export interface Stage {
  run(text: string, context: PipelineContext): Finding[];
}

// Shared mutable context dict passed through the pipeline.
// Typed loosely to allow stages to store arbitrary signals.
export type PipelineContext = {
  profile: AppProfile;
  stageLatenciesMs: Record<string, number>;
  findings?: Finding[];
  normalizedText?: string;
  mentionScore?: number;
  [key: string]: unknown;
};
