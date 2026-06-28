/**
 * Stage 2 — ML classifier (heuristic-only Backend C for Node).
 *
 * ML tradeoff note (mirrored from SPEC §3 / stages/classifier.py)
 * ─────────────────────────────────────────────────────────────────
 * Backend A (Python): sentence-embeddings + logistic-regression head.
 *   Pros: accurate, runs locally once trained.
 *   Cons: requires PyTorch / sentence-transformers; not trivial to port to Node.
 *
 * Options for Node deployments:
 *   1. HeuristicClassifier (this file) — zero deps, always works, lower recall.
 *   2. HTTP bridge: call the Python model server (e.g. a FastAPI /predict
 *      endpoint) from Node.  Add latency but shares the trained artifact.
 *   3. ONNX export: export the sklearn head + embedding model to ONNX, run
 *      via onnxruntime-node.  Zero network hop; requires an export step.
 *
 * The heuristic fallback is intentionally independent of Stage 1 patterns to
 * provide a complementary signal rather than duplicating them.
 */

import { type Finding, type PipelineContext, type Stage, makeFinding } from "../types.js";

// ---------------------------------------------------------------------------
// Heuristic patterns (Backend C)
// ---------------------------------------------------------------------------

const HEURISTIC_RULES: Array<[number, RegExp]> = [
  [0.80, /ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|rules?)/i],
  [0.75, /disregard\s+(?:all\s+)?(?:previous|prior|the)?\s*(?:instructions?|rules?|constraints?)/i],
  [0.70, /forget\s+(?:everything|all\s+previous|your\s+(?:instructions?|training))/i],
  [0.85, /<\|im_(?:start|end)\|>|\[\/INST\]|<<\/?SYS>>/i],
  [0.70, /(?:print|reveal|repeat)\s+.{0,30}(?:system\s+prompt|instructions?)/i],
  [0.75, /you\s+(?:are\s+now|have\s+no)\s+(?:DAN|jailbroken|restrictions?|rules?)/i],
  [0.65, /(?:developer|dev|debug)\s+mode\s+(?:enabled|activated|on\b)/i],
  [0.80, /ignore\s+the\s+user\s+and\s+(?:instead\s+)?(?:call|execute|run)\s+/i],
  [0.75, /(?:send|exfiltrate|forward)\s+.{0,20}(?:data|conversation)\s+to\s+\S/i],
];

const NOISE_FLOOR = 0.05;

export class HeuristicClassifier {
  predict(text: string): number {
    let total = 0;
    for (const [weight, pattern] of HEURISTIC_RULES) {
      if (pattern.test(text)) total += weight;
    }
    if (total === 0) return 0.0;
    return Math.min(0.3 + total * 0.09, 0.95);
  }
}

// ---------------------------------------------------------------------------
// Stage
// ---------------------------------------------------------------------------

export class ClassifierStage implements Stage {
  private readonly classifier: HeuristicClassifier;

  constructor(classifier?: HeuristicClassifier) {
    this.classifier = classifier ?? new HeuristicClassifier();
  }

  run(text: string, _context: PipelineContext): Finding[] {
    const prob = this.classifier.predict(text);
    if (prob < NOISE_FLOOR) return [];
    return [
      makeFinding({
        id: "CLS-HEU",
        category: "classifier",
        score: prob,
        structural: false,
        sourceStage: "classifier",
        detail: `P(injection)=${prob.toFixed(4)} via HeuristicClassifier`,
      }),
    ];
  }
}
