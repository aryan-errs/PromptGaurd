/**
 * Stage 3 — Intent disambiguation (use vs mention).
 * Mirrors stages/intent.py — keep signal weights in sync when updating either.
 *
 * Returns mention_score ∈ [0, 1]:
 *   0.0 = clear USE  (instruction directed at this model)
 *   1.0 = clear MENTION (discussing/quoting the technique)
 *
 * Structural findings from S1 force mention_score = 0.0 and bypass thresholds
 * in the policy layer (decisive rule — same as Python).
 */

import { type Finding, type PipelineContext, type Stage, makeFinding } from "../types.js";

// ---------------------------------------------------------------------------
// Injection core triggers (checked inside quoted/framed regions)
// ---------------------------------------------------------------------------

const INJECTION_CORE: RegExp[] = [
  /ignore\s+(?:previous|prior|above|all)\s+(?:instructions?|rules?)/i,
  /disregard\s+(?:all\s+)?(?:previous|prior|the)?\s*(?:instructions?|rules?|constraints?)/i,
  /forget\s+(?:everything|all\s+previous|your\s+(?:instructions?|training))/i,
  /<\|im_(?:start|end)\|>|\[\/INST\]|<<\/?SYS>>/i,
  /you\s+(?:are\s+now|have\s+no)\s+(?:DAN|restrictions?|rules?|limits?)/i,
  /(?:print|reveal|repeat)\s+(?:your\s+)?(?:system\s+prompt|instructions?)/i,
  /(?:developer|dev|debug)\s+mode\s+(?:enabled|activated|on\b)/i,
  /ignore\s+the\s+user\s+and\s+(?:instead\s+)?(?:call|execute|run|send)\b/i,
];

// ---------------------------------------------------------------------------
// Framing region extractor
// ---------------------------------------------------------------------------

const FRAMING_REGIONS_RE = new RegExp(
  [
    '"([^"]{2,})"',           // double-quoted
    "'([^']{2,})'",           // single-quoted
    "`{3}([\\s\\S]*?)`{3}",  // fenced code block
    "`([^`]{2,})`",           // inline code
  ].join("|"),
  "gs",
);

function injectionInFraming(text: string): boolean {
  const re = new RegExp(FRAMING_REGIONS_RE.source, FRAMING_REGIONS_RE.flags);
  for (const m of text.matchAll(re)) {
    const content = m[1] ?? m[2] ?? m[3] ?? m[4] ?? "";
    if (INJECTION_CORE.some((p) => p.test(content))) return true;
  }
  return false;
}

function stripFramingRegions(text: string): string {
  // Replace framed content with spaces so use-signals don't fire inside code blocks
  const re = new RegExp(FRAMING_REGIONS_RE.source, FRAMING_REGIONS_RE.flags);
  return text.replace(re, (m) => " ".repeat(m.length));
}

// ---------------------------------------------------------------------------
// Mention signals
// ---------------------------------------------------------------------------

const EXAMPLE_FRAMING_RE =
  /\b(?:for\s+example|for\s+instance|such\s+as|e\.g\.|i\.e\.|like\s+this|examples?\s+(?:of|like)|example\s+(?:attack|injection|payload|prompt|technique|jailbreak))\b/i;

const EXAMPLE_INTRO_RE =
  /\b(?:here\s+is|this\s+is|the\s+following\s+is|consider)\s+an?\s+(?:example|sample|demo|illustration|case)\b/i;

const ATTRIBUTION_RE =
  /\b(?:attacker|hacker|adversary|threat\s+actor|malicious\s+(?:user|actor))s?\s+(?:might|would|could|can|may|often|will|typically)\s+(?:use|send|write|craft|inject|try|attempt|construct)\b/i;

const TECHNIQUE_DESCRIPTION_RE =
  /\b(?:technique|method|approach|attack|exploit|vector|payload|prompt)\s+(?:that\s+)?(?:works?\s+by|involves?|uses?|sends?|tells?\s+(?:the\s+)?(?:model|AI)|instructs?\s+(?:the\s+)?(?:model|AI))\b/i;

const QUESTION_FRAMING_RE =
  /\b(?:how\s+(?:does|do|would|can|could|might)|why\s+(?:does|is|would)|what\s+(?:is|are|does|would)|explain\s+(?:how|what|why)|describe\s+(?:how|what)|can\s+you\s+(?:explain|describe|show|tell|help\s+me\s+understand)|help\s+me\s+understand)\b/i;

const DEFENSIVE_SECURITY_RE =
  /\b(?:detect|prevent|defend\s+(?:against|from)|protect\s+(?:against|from)|mitigate|understand\s+(?:how|the|these)|identify|recognize|red.?team(?:ing)?|pentest(?:ing)?|security\s+(?:review|audit|research)|(?:for\s+)?educational\s+(?:purposes?|reasons?)|(?:for\s+)?(?:research|learning|study)\s+(?:purposes?|reasons?))\b/i;

// ---------------------------------------------------------------------------
// Use signals
// ---------------------------------------------------------------------------

const DIRECT_COMMAND_START_RE =
  /^\s*(?:ignore|disregard|forget|override|print|reveal|repeat|execute|send|call|invoke|do\s+not\s+follow|stop\s+following|act\s+as|you\s+are\s+now|enable\s+jailbreak|developer\s+mode)\b/im;

const DIRECT_MODEL_COMMAND_RE =
  /\byou\s+(?:must|will|should\s+now|are\s+required\s+to|need\s+to|have\s+to)\s+(?:ignore|disregard|forget|override|reveal|print|repeat|execute|send)\b/i;

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

export interface MentionSignals {
  injectionInQuotes: boolean;
  exampleFraming: boolean;
  exampleIntro: boolean;
  thirdPartyAttribution: boolean;
  techniqueDescription: boolean;
  questionFraming: boolean;
  defensiveSecurity: boolean;
  directCommandStart: boolean;
  directModelCommand: boolean;
  structuralTarget?: boolean;
}

export function computeMentionScore(
  text: string,
  priorFindings: Finding[],
): [number, MentionSignals] {
  // Hard signal: structural finding → always USE
  if (priorFindings.some((f) => f.structural)) {
    return [0.0, { injectionInQuotes: false, exampleFraming: false, exampleIntro: false, thirdPartyAttribution: false, techniqueDescription: false, questionFraming: false, defensiveSecurity: false, directCommandStart: false, directModelCommand: false, structuralTarget: true }];
  }

  const unframed = stripFramingRegions(text);
  const signals: MentionSignals = {
    injectionInQuotes: injectionInFraming(text),
    exampleFraming: EXAMPLE_FRAMING_RE.test(text),
    exampleIntro: EXAMPLE_INTRO_RE.test(text),
    thirdPartyAttribution: ATTRIBUTION_RE.test(text),
    techniqueDescription: TECHNIQUE_DESCRIPTION_RE.test(text),
    questionFraming: QUESTION_FRAMING_RE.test(text),
    defensiveSecurity: DEFENSIVE_SECURITY_RE.test(text),
    directCommandStart: DIRECT_COMMAND_START_RE.test(unframed),
    directModelCommand: DIRECT_MODEL_COMMAND_RE.test(unframed),
  };

  let score = 0.0;
  if (signals.injectionInQuotes) score += 0.35;
  if (signals.exampleFraming) score += 0.12;
  if (signals.exampleIntro) score += 0.10;
  if (signals.thirdPartyAttribution) score += 0.20;
  if (signals.techniqueDescription) score += 0.13;
  if (signals.questionFraming) score += 0.18;
  if (signals.defensiveSecurity) score += 0.10;
  if (signals.directCommandStart) score -= 0.38;
  if (signals.directModelCommand) score -= 0.22;

  return [Math.max(0.0, Math.min(score, 1.0)), signals];
}

// ---------------------------------------------------------------------------
// Stage
// ---------------------------------------------------------------------------

export class IntentStage implements Stage {
  run(text: string, context: PipelineContext): Finding[] {
    const priorFindings = (context.findings ?? []) as Finding[];
    if (priorFindings.length === 0) return [];

    const [mentionScore, signals] = computeMentionScore(text, priorFindings);
    context.mentionScore = mentionScore;

    const fired = (Object.entries(signals) as [string, boolean | undefined][])
      .filter(([, v]) => v === true)
      .map(([k]) => k);

    const label = mentionScore >= 0.5 ? "INTENT-MENTION" : "INTENT-USE";
    return [
      makeFinding({
        id: label,
        category: "intent",
        score: mentionScore,
        structural: false,
        sourceStage: "intent",
        detail: `mention_score=${mentionScore.toFixed(3)}; signals=[${fired.join(", ")}]`,
      }),
    ];
  }
}
