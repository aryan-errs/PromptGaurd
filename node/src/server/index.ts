/**
 * PromptGuard Express server — mirrors the Python FastAPI service.
 *
 * Routes
 * ──────
 *   GET  /healthz   Liveness probe — always 200.
 *   GET  /readyz    Readiness probe — 200 (heuristic always ready; ML if configured).
 *   POST /inspect   Inspect messages; return Verdict JSON.
 *   POST /protect   Inspect + sanitize; return Verdict + safe messages.
 *
 * Config (environment variables, all prefixed PROMPTGUARD_)
 * ─────────────────────────────────────────────────────────
 *   PORT                   bind port (default: 8001)
 *   HOST                   bind host (default: 0.0.0.0)
 *   PROFILE_NAME           default profile name (default: "default")
 *   RISK_TIER              low | medium | high (default: "medium")
 *   ALLOW_SECURITY_DISCUSSION  true | false (default: false)
 *   TEMPLATE_DELIMITERS    comma-separated (default: "")
 *   TOOLS_ENABLED          true | false (default: false)
 *   API_KEY                shared secret for X-API-Key auth (default: disabled)
 *   RATE_LIMIT_RPM         requests/min per IP; 0 = disabled (default: 0)
 *   MAX_REQUEST_BYTES      body size cap in bytes (default: 65536)
 *
 * Usage
 * ─────
 *   node dist/server/index.js
 *   npx ts-node src/server/index.ts          (dev)
 */

import type { NextFunction, Request, Response } from "express";
import express from "express";

import { AppProfile } from "../types.js";
import { buildPipeline } from "../pipeline.js";
import { sanitizeMessages } from "../sanitize.js";
import type { ChatMessage } from "../sanitize.js";
import type { Verdict } from "../types.js";

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

function env(key: string, fallback: string): string {
  return process.env[`PROMPTGUARD_${key}`] ?? fallback;
}
function envBool(key: string, fallback: boolean): boolean {
  const v = process.env[`PROMPTGUARD_${key}`];
  if (!v) return fallback;
  return v.toLowerCase() === "true" || v === "1";
}
function envInt(key: string, fallback: number): number {
  const v = process.env[`PROMPTGUARD_${key}`];
  return v ? parseInt(v, 10) : fallback;
}

export interface ServerConfig {
  port: number;
  host: string;
  profileName: string;
  riskTier: "low" | "medium" | "high";
  allowSecurityDiscussion: boolean;
  templateDelimiters: string[];
  toolsEnabled: boolean;
  apiKey: string | null;
  rateLimitRpm: number;
  maxRequestBytes: number;
}

export function loadConfig(): ServerConfig {
  const delims = env("TEMPLATE_DELIMITERS", "");
  return {
    port: envInt("PORT", 8001),
    host: env("HOST", "0.0.0.0"),
    profileName: env("PROFILE_NAME", "default"),
    riskTier: env("RISK_TIER", "medium") as "low" | "medium" | "high",
    allowSecurityDiscussion: envBool("ALLOW_SECURITY_DISCUSSION", false),
    templateDelimiters: delims ? delims.split(",").map((s) => s.trim()) : [],
    toolsEnabled: envBool("TOOLS_ENABLED", false),
    apiKey: env("API_KEY", "") || null,
    rateLimitRpm: envInt("RATE_LIMIT_RPM", 0),
    maxRequestBytes: envInt("MAX_REQUEST_BYTES", 65_536),
  };
}

// ---------------------------------------------------------------------------
// Rate limiter (per-IP sliding window, in-process only)
// ---------------------------------------------------------------------------

class RateLimiter {
  private readonly rpm: number;
  private readonly windows = new Map<string, number[]>();

  constructor(requestsPerMinute: number) {
    this.rpm = requestsPerMinute;
  }

  isAllowed(ip: string): boolean {
    const now = Date.now();
    const cutoff = now - 60_000;
    const times = (this.windows.get(ip) ?? []).filter((t) => t >= cutoff);
    if (times.length >= this.rpm) {
      this.windows.set(ip, times);
      return false;
    }
    times.push(now);
    this.windows.set(ip, times);
    return true;
  }
}

// ---------------------------------------------------------------------------
// Request/response serialization
// ---------------------------------------------------------------------------

function serializeVerdict(v: Verdict): object {
  return {
    action: v.action,
    score: v.score,
    findings: v.findings.map((f) => ({
      id: f.id,
      category: f.category,
      score: f.score,
      structural: f.structural,
      source_stage: f.sourceStage,
      detail: f.detail,
    })),
    latency_ms: v.latencyMs,
    blocked: v.blocked,
    safe: v.safe,
    sanitized_text: v.sanitizedText ?? null,
    transformations: v.transformations.map((t) => ({
      kind: t.kind,
      description: t.description,
      original_fragment: t.originalFragment,
      transformed_fragment: t.transformedFragment,
      stage: t.stage,
    })),
  };
}

function parseProfile(raw: unknown): AppProfile | null {
  if (!raw || typeof raw !== "object") return null;
  const p = raw as Record<string, unknown>;
  return new AppProfile(
    typeof p["name"] === "string" ? p["name"] : "default",
    typeof p["allow_security_discussion"] === "boolean"
      ? p["allow_security_discussion"]
      : false,
    (["low", "medium", "high"].includes(p["risk_tier"] as string)
      ? (p["risk_tier"] as "low" | "medium" | "high")
      : "medium"),
    Array.isArray(p["template_delimiters"])
      ? (p["template_delimiters"] as string[])
      : [],
    typeof p["tools_enabled"] === "boolean" ? p["tools_enabled"] : false,
  );
}

function lastUserText(messages: ChatMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m && m.role === "user") return m.content;
  }
  return "";
}

function getClientIp(req: Request): string {
  const forwarded = req.headers["x-forwarded-for"];
  if (typeof forwarded === "string") return forwarded.split(",")[0]?.trim() ?? "unknown";
  return req.socket.remoteAddress ?? "unknown";
}

// ---------------------------------------------------------------------------
// App factory
// ---------------------------------------------------------------------------

export interface AppOptions {
  config?: Partial<ServerConfig>;
}

export function createApp(options: AppOptions = {}): express.Application {
  const cfg: ServerConfig = { ...loadConfig(), ...options.config };
  const defaultProfile = new AppProfile(
    cfg.profileName,
    cfg.allowSecurityDiscussion,
    cfg.riskTier,
    cfg.templateDelimiters,
    cfg.toolsEnabled,
  );
  const pipeline = buildPipeline(defaultProfile);

  const limiter = cfg.rateLimitRpm > 0 ? new RateLimiter(cfg.rateLimitRpm) : null;

  const app = express();
  app.use(express.json({ limit: cfg.maxRequestBytes }));

  // ── Auth middleware ────────────────────────────────────────────────────
  const requireAuth = (req: Request, res: Response, next: NextFunction): void => {
    if (!cfg.apiKey) {
      next();
      return;
    }
    const provided = req.headers["x-api-key"];
    if (provided !== cfg.apiKey) {
      res.status(401).json({ detail: "Missing or invalid X-API-Key header" });
      return;
    }
    next();
  };

  // ── Rate-limit middleware ──────────────────────────────────────────────
  const rateLimit = (req: Request, res: Response, next: NextFunction): void => {
    if (!limiter) {
      next();
      return;
    }
    const ip = getClientIp(req);
    if (!limiter.isAllowed(ip)) {
      res
        .status(429)
        .set({ "Retry-After": "60", "X-RateLimit-Limit": String(cfg.rateLimitRpm) })
        .json({ detail: `Rate limit exceeded (${cfg.rateLimitRpm} req/min). Retry after 60s.` });
      return;
    }
    next();
  };

  // ── Liveness probe ─────────────────────────────────────────────────────
  app.get("/healthz", (_req, res) => {
    res.json({ status: "ok" });
  });

  // ── Readiness probe ────────────────────────────────────────────────────
  // Node port always uses the heuristic classifier (Backend C); readyz is
  // immediately ready.  If an ML model is served via the Python service,
  // readyz will reflect that service's state instead.
  app.get("/readyz", (_req, res) => {
    res.json({ status: "ready", classifier: "heuristic" });
  });

  // ── POST /inspect ──────────────────────────────────────────────────────
  app.post("/inspect", requireAuth, rateLimit, (req: Request, res: Response) => {
    const body = req.body as {
      messages?: unknown;
      profile?: unknown;
    };

    if (!Array.isArray(body.messages) || body.messages.length === 0) {
      res.status(422).json({ detail: "messages must be a non-empty array" });
      return;
    }

    const messages = body.messages as ChatMessage[];
    const hasUser = messages.some((m) => m.role === "user");
    if (!hasUser) {
      res.status(422).json({ detail: "at least one message with role='user' is required" });
      return;
    }

    const profile = parseProfile(body.profile) ?? defaultProfile;
    const pipe = profile === defaultProfile ? pipeline : buildPipeline(profile);

    const text = lastUserText(messages);
    const verdict = pipe.run(text);

    const requestId = crypto.randomUUID();
    res.json({
      verdict: serializeVerdict(verdict),
      request_id: requestId,
      inspected_text_preview: text.slice(0, 80),
    });
  });

  // ── POST /protect ──────────────────────────────────────────────────────
  app.post("/protect", requireAuth, rateLimit, (req: Request, res: Response) => {
    const body = req.body as {
      messages?: unknown;
      profile?: unknown;
      block_on_block?: unknown;
    };

    if (!Array.isArray(body.messages) || body.messages.length === 0) {
      res.status(422).json({ detail: "messages must be a non-empty array" });
      return;
    }

    const messages = body.messages as ChatMessage[];
    const hasUser = messages.some((m) => m.role === "user");
    if (!hasUser) {
      res.status(422).json({ detail: "at least one message with role='user' is required" });
      return;
    }

    const profile = parseProfile(body.profile) ?? defaultProfile;
    const pipe = profile === defaultProfile ? pipeline : buildPipeline(profile);
    const blockOnBlock = body.block_on_block !== false; // default true

    const text = lastUserText(messages);
    const verdict = pipe.run(text);

    if (verdict.blocked && blockOnBlock) {
      res.status(400).json({
        detail: {
          message: "PromptGuard blocked this input",
          verdict: serializeVerdict(verdict),
        },
      });
      return;
    }

    let safeMessages: ChatMessage[];
    if (verdict.blocked) {
      safeMessages = [];
    } else if (verdict.action === "sanitize") {
      const [sanitized] = sanitizeMessages(messages, profile);
      safeMessages = sanitized;
    } else {
      safeMessages = [...messages];
    }

    res.json({
      verdict: serializeVerdict(verdict),
      request_id: crypto.randomUUID(),
      safe_messages: safeMessages,
    });
  });

  return app;
}

// ---------------------------------------------------------------------------
// Standalone entry point
// ---------------------------------------------------------------------------

if (
  process.argv[1] !== undefined &&
  new URL(import.meta.url).pathname === process.argv[1]
) {
  const cfg = loadConfig();
  const app = createApp();
  app.listen(cfg.port, cfg.host, () => {
    console.log(`\n  PromptGuard Node server`);
    console.log(`  ─────────────────────`);
    console.log(`  Listening on http://${cfg.host}:${cfg.port}\n`);
  });
}
