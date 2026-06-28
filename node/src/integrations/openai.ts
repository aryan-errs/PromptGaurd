/**
 * Transparent wrapper around the OpenAI Node SDK client.
 * Intercepts chat.completions.create(), runs the pipeline on user messages,
 * and blocks/sanitizes before the real API call is made.
 *
 * Does NOT import from 'openai' at module level — uses structural typing to
 * avoid making openai a hard dependency.
 */

import { Pipeline, buildPipeline } from "../pipeline.js";
import { AppProfile } from "../types.js";
import { sanitizeMessages } from "../sanitize.js";
import type { ChatMessage } from "../sanitize.js";

// Minimal structural type matching openai.OpenAI.chat.completions.create params
interface ChatCompletionParams {
  messages: ChatMessage[];
  [key: string]: unknown;
}

interface ChatCompletionsLike {
  create(params: ChatCompletionParams): Promise<unknown>;
}

interface ChatLike {
  completions: ChatCompletionsLike;
}

export interface OpenAILike {
  chat: ChatLike;
  [key: string]: unknown;
}

export interface GuardOptions {
  profile?: AppProfile;
  pipeline?: Pipeline;
  /** If true, throw on block instead of returning an error object */
  throwOnBlock?: boolean;
}

/**
 * Wraps an openai.OpenAI client so all chat.completions.create() calls
 * pass through the PromptGuard pipeline first.
 *
 * Usage:
 *   import OpenAI from 'openai';
 *   import { wrapOpenAI } from 'promptguard';
 *
 *   const client = wrapOpenAI(new OpenAI(), { profile });
 *   const response = await client.chat.completions.create({ ... });
 */
export function wrapOpenAI<T extends OpenAILike>(client: T, options: GuardOptions = {}): T {
  const profile = options.profile ?? new AppProfile("default");
  const pipeline = options.pipeline ?? buildPipeline(profile);
  const throwOnBlock = options.throwOnBlock ?? true;

  const guardedCompletions: ChatCompletionsLike = {
    async create(params: ChatCompletionParams) {
      const userMessages = params.messages.filter((m) => m.role === "user");
      const lastUser = userMessages.at(-1);

      if (lastUser) {
        const verdict = pipeline.run(lastUser.content);

        if (verdict.blocked) {
          if (throwOnBlock) {
            throw new Error(
              `PromptGuard blocked input (score=${verdict.score.toFixed(2)}, ` +
              `rules=${verdict.findings.filter((f) => f.structural).map((f) => f.id).join(",")})`,
            );
          }
          // Return a synthetic error response instead of calling the API
          return {
            id: "promptguard-blocked",
            object: "chat.completion",
            choices: [
              {
                index: 0,
                message: { role: "assistant", content: null },
                finish_reason: "content_filter",
              },
            ],
            promptguardVerdict: verdict,
          };
        }

        if (verdict.action === "sanitize" && verdict.sanitizedText !== undefined) {
          // Replace the last user message with the sanitized version
          const [sanitizedMessages] = sanitizeMessages(params.messages, profile);
          params = { ...params, messages: sanitizedMessages };
        }
      }

      return client.chat.completions.create(params);
    },
  };

  // Return a Proxy so all other client properties (models, embeddings, etc.) pass through
  return new Proxy(client, {
    get(target, prop, receiver) {
      if (prop === "chat") {
        return { completions: guardedCompletions };
      }
      return Reflect.get(target, prop, receiver);
    },
  }) as T;
}
