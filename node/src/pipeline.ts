/**
 * Pipeline — orchestrates stages and produces a Verdict via policy.ts.
 * Mirrors pipeline.py.
 */

import { decide } from "./policy.js";
import { sanitize } from "./sanitize.js";
import { IntentStage } from "./stages/intent.js";
import { NormalizeStage } from "./stages/normalize.js";
import { SignatureStage } from "./stages/signatures.js";
import { AppProfile, type Finding, type PipelineContext, type Stage, Verdict } from "./types.js";

export class Pipeline {
  constructor(
    private readonly stages: Stage[],
    public readonly profile: AppProfile = new AppProfile("default"),
  ) {}

  run(text: string): Verdict {
    const wallStart = performance.now();

    const findings: Finding[] = [];
    const context: PipelineContext = {
      profile: this.profile,
      stageLatenciesMs: {},
    };

    let currentText = text;
    for (const stage of this.stages) {
      const stageStart = performance.now();
      const stageFindings = stage.run(currentText, context);
      const stageMs = performance.now() - stageStart;

      const stageName = stage.constructor.name;
      context.stageLatenciesMs[stageName] = parseFloat(stageMs.toFixed(3));

      findings.push(...stageFindings);
      context.findings = findings;
      if (context.normalizedText !== undefined) {
        currentText = context.normalizedText;
      }
    }

    const latencyMs = performance.now() - wallStart;
    const verdict = decide(findings, this.profile, latencyMs);

    // Wire sanitize path: populate sanitizedText + transformations on verdict
    if (verdict.action === "sanitize") {
      const [sanitizedText, transformations] = sanitize(text, this.profile);
      return new Verdict(
        verdict.action,
        verdict.score,
        verdict.findings,
        verdict.latencyMs,
        sanitizedText,
        transformations,
      );
    }

    return verdict;
  }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/** Default heuristic-only pipeline: S0 + S1 + S3 (no ML required). */
export function buildPipeline(profile?: AppProfile): Pipeline {
  return new Pipeline(
    [new NormalizeStage(), new SignatureStage(), new IntentStage()],
    profile ?? new AppProfile("default"),
  );
}
