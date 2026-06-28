/**
 * Express-style middleware integration.
 * Does not import from 'express' directly — uses structural typing so it works
 * with any Express-compatible framework without making express a hard dep.
 */

import { Pipeline, buildPipeline } from "../pipeline.js";
import { AppProfile } from "../types.js";
import { SPOTLIGHT_SYSTEM_NOTE } from "../sanitize.js";

// Minimal structural types matching Express 4/5 Request/Response/NextFunction
export interface MinimalRequest {
  body?: { messages?: Array<{ role: string; content: string }>; [k: string]: unknown };
  headers: Record<string, string | string[] | undefined>;
  [key: string]: unknown;
}

export interface MinimalResponse {
  status(code: number): MinimalResponse;
  json(body: unknown): void;
  setHeader(name: string, value: string): void;
}

export type NextFunction = (err?: unknown) => void;

export type ExpressMiddleware = (
  req: MinimalRequest,
  res: MinimalResponse,
  next: NextFunction,
) => void;

export interface MiddlewareOptions {
  /** AppProfile to use for detection; defaults to a medium-risk profile. */
  profile?: AppProfile;
  /** Pipeline instance; defaults to buildPipeline(profile). */
  pipeline?: Pipeline;
  /**
   * Called when the verdict is "block". Default: 400 + JSON error body.
   * Return true to stop further processing; false to pass through to next().
   */
  onBlock?: (req: MinimalRequest, res: MinimalResponse, verdict: import("../types.js").Verdict) => boolean;
  /**
   * Which request property to inspect for the user message.
   * Defaults to req.body.messages (last user role) or req.body.prompt (string).
   */
  extractText?: (req: MinimalRequest) => string | null;
}

function defaultExtract(req: MinimalRequest): string | null {
  const body = req.body;
  if (!body) return null;
  // OpenAI-style messages array
  if (Array.isArray(body.messages)) {
    const userMsgs = body.messages.filter((m) => m.role === "user");
    return userMsgs.at(-1)?.content ?? null;
  }
  // Simple string prompt
  if (typeof body.prompt === "string") return body.prompt;
  return null;
}

/**
 * Creates an Express-compatible middleware that inspects the request body
 * for injection signals and blocks/sanitizes before calling next().
 *
 * Usage:
 *   app.use(createMiddleware({ profile: new AppProfile('myapp', false, 'medium') }))
 */
export function createMiddleware(options: MiddlewareOptions = {}): ExpressMiddleware {
  const profile = options.profile ?? new AppProfile("default");
  const pipeline = options.pipeline ?? buildPipeline(profile);
  const extractText = options.extractText ?? defaultExtract;
  const onBlock =
    options.onBlock ??
    ((_req, res, verdict) => {
      res.status(400).json({
        error: "PromptGuard blocked this request",
        action: verdict.action,
        score: verdict.score,
      });
      return true;
    });

  return (req, res, next) => {
    const text = extractText(req);
    if (text === null) {
      next();
      return;
    }

    const verdict = pipeline.run(text);

    if (verdict.blocked) {
      const handled = onBlock(req, res, verdict);
      if (handled) return;
    }

    // Attach the verdict to the request for downstream handlers
    (req as Record<string, unknown>)["promptguardVerdict"] = verdict;
    // Add the spotlight system note as a header so API gateways can inject it
    if (verdict.action === "sanitize") {
      res.setHeader("X-PromptGuard-System-Note", SPOTLIGHT_SYSTEM_NOTE.slice(0, 200));
    }
    next();
  };
}
