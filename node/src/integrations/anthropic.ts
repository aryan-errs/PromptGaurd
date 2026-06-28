/**
 * Transparent wrapper around the Anthropic Node SDK client.
 * Intercepts messages.create(), runs the pipeline on user messages,
 * and blocks/sanitizes before the real API call is made.
 */

import { Pipeline, buildPipeline } from "../pipeline.js";
import { AppProfile } from "../types.js";
import { sanitizeMessages } from "../sanitize.js";
import type { ChatMessage } from "../sanitize.js";

interface AnthropicMessage {
  role: string;
  content: string | unknown;
  [key: string]: unknown;
}

interface MessagesCreateParams {
  messages: AnthropicMessage[];
  system?: string;
  [key: string]: unknown;
}

interface MessagesLike {
  create(params: MessagesCreateParams): Promise<unknown>;
}

export interface AnthropicLike {
  messages: MessagesLike;
  [key: string]: unknown;
}

export interface GuardOptions {
  profile?: AppProfile;
  pipeline?: Pipeline;
  throwOnBlock?: boolean;
}

/**
 * Wraps an Anthropic client so all messages.create() calls
 * pass through the PromptGuard pipeline first.
 *
 * Usage:
 *   import Anthropic from '@anthropic-ai/sdk';
 *   import { wrapAnthropic } from 'promptguard';
 *
 *   const client = wrapAnthropic(new Anthropic(), { profile });
 *   const response = await client.messages.create({ ... });
 */
export function wrapAnthropic<T extends AnthropicLike>(
  client: T,
  options: GuardOptions = {},
): T {
  const profile = options.profile ?? new AppProfile("default");
  const pipeline = options.pipeline ?? buildPipeline(profile);
  const throwOnBlock = options.throwOnBlock ?? true;

  const guardedMessages: MessagesLike = {
    async create(params: MessagesCreateParams) {
      const userMessages = params.messages.filter(
        (m) => m.role === "user" && typeof m.content === "string",
      );
      const lastUser = userMessages.at(-1);

      if (lastUser && typeof lastUser.content === "string") {
        const verdict = pipeline.run(lastUser.content);

        if (verdict.blocked) {
          if (throwOnBlock) {
            throw new Error(
              `PromptGuard blocked input (score=${verdict.score.toFixed(2)})`,
            );
          }
          return {
            id: "promptguard-blocked",
            type: "message",
            content: [],
            stop_reason: "end_turn",
            promptguardVerdict: verdict,
          };
        }

        if (verdict.action === "sanitize" && verdict.sanitizedText !== undefined) {
          const chatMessages = params.messages.map((m) => ({
            role: m.role,
            content: typeof m.content === "string" ? m.content : "",
          })) as ChatMessage[];
          const [sanitized] = sanitizeMessages(chatMessages, profile);
          // Rebuild Anthropic-format messages from sanitized chat messages
          const sanitizedAnthropic = params.messages.map((m, i) => ({
            ...m,
            content: sanitized[i]?.content ?? m.content,
          }));
          params = { ...params, messages: sanitizedAnthropic };
        }
      }

      return client.messages.create(params);
    },
  };

  return new Proxy(client, {
    get(target, prop, receiver) {
      if (prop === "messages") return guardedMessages;
      return Reflect.get(target, prop, receiver);
    },
  }) as T;
}
