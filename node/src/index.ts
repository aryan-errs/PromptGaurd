/**
 * PromptGuard — Node/TypeScript public API.
 *
 * Rule sync note
 * ──────────────
 * Stage 1 patterns are loaded at runtime from
 * python/promptguard/stages/rules.yaml — the single source of truth shared
 * by both the Python and Node implementations.  When you edit a pattern,
 * both runtimes pick it up on the next process start with no manual sync step.
 * Stages 0, 3, policy, and sanitizer logic are ported independently; see the
 * README drift-prevention section for the recommended test strategy.
 */

// Core types
export {
  AppProfile,
  type Finding,
  type PipelineContext,
  type Stage,
  type Transformation,
  Verdict,
  type VerdictAction,
  type RiskTier,
} from "./types.js";

// Pipeline
export { Pipeline, buildPipeline } from "./pipeline.js";

// Policy
export {
  decide,
  effectiveThresholds,
  SECURITY_DISCUSSION_BOOST,
  INTENT_MAX_BOOST,
} from "./policy.js";

// Sanitizer
export {
  sanitize,
  sanitizeMessages,
  SPOTLIGHT_PREFIX,
  SPOTLIGHT_SUFFIX,
  SPOTLIGHT_SYSTEM_NOTE,
  DeeplyObfuscatedError,
  DEEP_OBFUSCATION_THRESHOLD,
  type ChatMessage,
} from "./sanitize.js";

// Stages (for custom pipelines)
export { NormalizeStage } from "./stages/normalize.js";
export { SignatureStage } from "./stages/signatures.js";
export { IntentStage } from "./stages/intent.js";
export { ClassifierStage, HeuristicClassifier } from "./stages/classifier.js";

// Intent helpers (for testing / debugging)
export { computeMentionScore } from "./stages/intent.js";

// Integrations
export {
  createMiddleware,
  type MiddlewareOptions,
  type ExpressMiddleware,
} from "./integrations/express.js";
export {
  wrapOpenAI,
  type OpenAILike,
  type GuardOptions as OpenAIGuardOptions,
} from "./integrations/openai.js";
export {
  wrapAnthropic,
  type AnthropicLike,
  type GuardOptions as AnthropicGuardOptions,
} from "./integrations/anthropic.js";

// ---------------------------------------------------------------------------
// High-level PromptGuard facade (mirrors Python PromptGuard class)
// ---------------------------------------------------------------------------

import { AppProfile } from "./types.js";
import { Pipeline, buildPipeline } from "./pipeline.js";
import { Verdict } from "./types.js";
import { wrapOpenAI, type OpenAILike } from "./integrations/openai.js";
import { wrapAnthropic, type AnthropicLike } from "./integrations/anthropic.js";

export class PromptGuard {
  private readonly _pipeline: Pipeline;

  constructor(profile?: AppProfile) {
    this._pipeline = buildPipeline(profile);
  }

  get profile(): AppProfile {
    return this._pipeline.profile;
  }

  /** Run the pipeline and return a Verdict. */
  inspect(text: string): Verdict {
    return this._pipeline.run(text);
  }

  /**
   * Run the pipeline; throw on block, return sanitized text (or original) otherwise.
   * When action == "sanitize", returns verdict.sanitizedText.
   */
  protect(text: string): string {
    const verdict = this.inspect(text);
    if (verdict.blocked) {
      const ruleIds = verdict.findings
        .filter((f) => f.structural)
        .map((f) => f.id);
      throw new Error(
        `PromptGuard blocked input (score=${verdict.score.toFixed(2)}, rules=[${ruleIds.join(",")}])`,
      );
    }
    if (verdict.action === "sanitize" && verdict.sanitizedText !== undefined) {
      return verdict.sanitizedText;
    }
    return text;
  }

  /** Wrap an OpenAI client with PromptGuard. Requires the 'openai' npm package. */
  wrapOpenAI<T extends OpenAILike>(client: T): T {
    return wrapOpenAI(client, { pipeline: this._pipeline, profile: this.profile });
  }

  /** Wrap an Anthropic client with PromptGuard. Requires '@anthropic-ai/sdk'. */
  wrapAnthropic<T extends AnthropicLike>(client: T): T {
    return wrapAnthropic(client, { pipeline: this._pipeline, profile: this.profile });
  }
}
