/**
 * OpenAIBridge – TransformerBridge implementation for the OpenAI Chat API.
 *
 * Supports both non-streaming and streaming (SSE) completions.
 * Compatible with any OpenAI-compatible endpoint via the `baseUrl` option.
 */

import { SSMError } from '../errors/SSMError.js';
import type { LlmUsage } from '../telemetry/types.js';
import type { TransformerBridge, BridgeGenerateOptions, BridgeCallInfo } from './TransformerBridge.js';

export interface OpenAIBridgeOptions {
    /** OpenAI API key (or compatible service key). */
    apiKey        : string;
    /** Model to use. Default: 'gpt-4o-mini'. */
    model?        : string;
    /** API base URL. Default: 'https://api.openai.com/v1'. */
    baseUrl?      : string;
    /** Default system prompt sent with every request. */
    systemPrompt? : string;
    /** Default max tokens. Default: 512. */
    maxTokens?    : number;
}

export class OpenAIBridge implements TransformerBridge {
    private _lastCall: BridgeCallInfo | undefined;

    readonly supportsStreaming = true as const;

    private readonly _apiKey      : string;
    private readonly _model       : string;
    private readonly _baseUrl     : string;
    private readonly _systemPrompt: string;
    private readonly _maxTokens   : number;

    constructor(opts: OpenAIBridgeOptions) {
        this._apiKey       = opts.apiKey;
        this._model        = opts.model        ?? 'gpt-4o-mini';
        this._baseUrl      = (opts.baseUrl     ?? 'https://api.openai.com/v1').replace(/\/$/, '');
        this._systemPrompt = opts.systemPrompt ?? '';
        this._maxTokens    = opts.maxTokens    ?? 512;
    }

    /** Provider-reported usage for the last call — see {@link BridgeCallInfo}. */
    get lastCall(): BridgeCallInfo | undefined {
        return this._lastCall;
    }

    async generate(prompt: string, opts: BridgeGenerateOptions = {}): Promise<string> {
        const body = this._buildBody(prompt, opts, false);
        const res  = await this._fetch(body);

        if (!res.ok) {
            const text = await res.text().catch(() => '');
            throw new SSMError(
                'BRIDGE_REQUEST_FAILED',
                `OpenAI API returned ${res.status}: ${text}`,
            );
        }

        const json = await res.json() as Record<string, unknown>;
        const content = (json as any).choices?.[0]?.message?.content;
        if (typeof content !== 'string') {
            throw new SSMError('BRIDGE_RESPONSE_INVALID', 'Unexpected OpenAI response shape.');
        }
        this._lastCall = {
            usage: readOpenAIUsage((json as any).usage, (json as any).model ?? opts.model ?? this._model),
        };
        return content;
    }

    async *stream(prompt: string, opts: BridgeGenerateOptions = {}): AsyncIterable<string> {
        const body = this._buildBody(prompt, opts, true);
        const res  = await this._fetch(body);

        if (!res.ok) {
            const text = await res.text().catch(() => '');
            throw new SSMError(
                'BRIDGE_REQUEST_FAILED',
                `OpenAI streaming API returned ${res.status}: ${text}`,
            );
        }

        if (!res.body) {
            throw new SSMError('BRIDGE_RESPONSE_INVALID', 'OpenAI streaming response has no body.');
        }

        const model = opts.model ?? this._model;
        let usage: LlmUsage | undefined;

        yield* parseOpenAIStream(res.body, (event) => {
            // Only emitted because `_buildBody` sets `stream_options.include_usage`;
            // without it OpenAI streams no usage at all and cost would be a guess.
            const reported = readOpenAIUsage((event as any).usage, (event as any).model ?? model);
            if (reported) usage = reported;
        });

        this._lastCall = usage ? { usage } : {};
    }

    private _buildBody(prompt: string, opts: BridgeGenerateOptions, stream: boolean): string {
        const sys = opts.systemPrompt ?? this._systemPrompt;
        const messages: { role: string; content: string }[] = [];
        if (sys) messages.push({ role: 'system', content: sys });
        messages.push({ role: 'user', content: prompt });

        return JSON.stringify({
            model      : opts.model     ?? this._model,
            messages,
            max_tokens : opts.maxTokens ?? this._maxTokens,
            temperature: opts.temperature ?? 0.7,
            top_p      : opts.topP        ?? 0.9,
            stream,
            // Without this the streaming response carries no usage block, so
            // cost-per-request on streamed calls would silently fall back to an
            // estimate. Non-streaming requests reject the field, hence the guard.
            ...(stream ? { stream_options: { include_usage: true } } : {}),
        });
    }

    private _fetch(body: string): Promise<Response> {
        return fetch(`${this._baseUrl}/chat/completions`, {
            method : 'POST',
            headers: {
                'Content-Type' : 'application/json',
                'Authorization': `Bearer ${this._apiKey}`,
            },
            body,
        });
    }
}

// ── SSE parser ────────────────────────────────────────────────────────────────

async function* parseOpenAIStream(
    body: ReadableStream<Uint8Array>,
    onEvent?: (event: Record<string, unknown>) => void,
): AsyncIterable<string> {
    const reader  = body.getReader();
    const decoder = new TextDecoder();
    let buffer    = '';

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() as string;   // keep the last (possibly partial) line; split() always yields ≥1 element

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed.startsWith('data: ')) continue;

                const data = trimmed.slice(6);
                if (data === '[DONE]') return;

                try {
                    const chunk = JSON.parse(data) as Record<string, unknown>;
                    const delta = (chunk as any).choices?.[0]?.delta?.content;
                    if (typeof delta === 'string' && delta.length > 0) yield delta;
                } catch {
                    // Malformed JSON in stream — skip silently
                }
            }
        }
    } finally {
        reader.releaseLock();
    }
}

/** Maps an OpenAI `usage` object onto the canonical {@link LlmUsage} shape. */
function readOpenAIUsage(usage: unknown, model: string): LlmUsage | undefined {
    if (!usage || typeof usage !== 'object') return undefined;
    const u = usage as Record<string, unknown>;
    const cached = Number((u['prompt_tokens_details'] as Record<string, unknown> | undefined)?.['cached_tokens']) || 0;
    const prompt = Number(u['prompt_tokens']) || 0;
    return {
        model,
        // OpenAI reports `prompt_tokens` INCLUSIVE of cached tokens; the canonical
        // shape keeps them disjoint so the two rates are applied exactly once each.
        inputTokens: Math.max(0, prompt - cached),
        outputTokens: Number(u['completion_tokens']) || 0,
        cachedInputTokens: cached,
    };
}
