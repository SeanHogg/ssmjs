/**
 * AnthropicBridge – TransformerBridge implementation for the Anthropic Messages API.
 *
 * Uses the /v1/messages endpoint.  System prompts are passed as the top-level
 * `system` field (not a message role), per the Anthropic spec.
 */

import { SSMError } from '../errors/SSMError.js';
import type { LlmUsage } from '../telemetry/types.js';
import type { TransformerBridge, BridgeGenerateOptions, BridgeCallInfo } from './TransformerBridge.js';

export interface AnthropicBridgeOptions {
    /** Anthropic API key. */
    apiKey        : string;
    /**
     * Model to use. Default: 'claude-haiku-4-5' (cheapest current model:
     * $1/1M input, $5/1M output). The previous default `claude-3-5-haiku-*`
     * was retired on 2026-02-19 and now 404s.
     */
    model?        : string;
    /** Anthropic API version header. Default: '2023-06-01'. */
    apiVersion?   : string;
    /** Default system prompt. Default: none. */
    systemPrompt? : string;
    /** Default max tokens — required by Anthropic. Default: 1024. */
    maxTokens?    : number;
    /**
     * When true (default), the system prompt is sent as a cacheable content
     * block (`cache_control: {type: 'ephemeral'}`). Prompt caching bills cache
     * reads at ~10% of the input price, so a stable system prefix reused across
     * turns is up to ~90% cheaper on its input tokens. Caching only engages once
     * the cached prefix exceeds the model minimum (~4096 tokens for Haiku 4.5);
     * below that it is a silent no-op, never an error. Set false to opt out.
     */
    cacheSystem?  : boolean;
}

const API_URL = 'https://api.anthropic.com/v1/messages';

export class AnthropicBridge implements TransformerBridge {
    readonly supportsStreaming = true as const;

    private readonly _apiKey      : string;
    private readonly _model       : string;
    private readonly _apiVersion  : string;
    private readonly _systemPrompt: string;
    private readonly _maxTokens   : number;
    private readonly _cacheSystem : boolean;

    private _lastCall: BridgeCallInfo | undefined;

    constructor(opts: AnthropicBridgeOptions) {
        this._apiKey       = opts.apiKey;
        this._model        = opts.model      ?? 'claude-haiku-4-5';
        this._apiVersion   = opts.apiVersion ?? '2023-06-01';
        this._systemPrompt = opts.systemPrompt ?? '';
        this._maxTokens    = opts.maxTokens    ?? 1024;
        this._cacheSystem  = opts.cacheSystem  ?? true;
    }

    /**
     * Provider-reported usage for the last call. Anthropic splits input tokens
     * three ways — fresh, cache read, and cache write — and each is billed at a
     * different rate, so a cost model that only reads `input_tokens` is wrong on
     * exactly the cache-heavy traffic this bridge is tuned to produce.
     */
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
                `Anthropic API returned ${res.status}: ${text}`,
            );
        }

        const json    = await res.json() as Record<string, unknown>;
        const content = (json as any).content?.[0]?.text;
        if (typeof content !== 'string') {
            throw new SSMError('BRIDGE_RESPONSE_INVALID', 'Unexpected Anthropic response shape.');
        }
        this._lastCall = {
            usage: readAnthropicUsage(
                (json as any).usage,
                (json as any).model ?? opts.model ?? this._model,
            ),
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
                `Anthropic streaming API returned ${res.status}: ${text}`,
            );
        }

        if (!res.body) {
            throw new SSMError('BRIDGE_RESPONSE_INVALID', 'Anthropic streaming response has no body.');
        }

        const model = opts.model ?? this._model;
        let input = 0, output = 0, cacheRead = 0, cacheWrite = 0, reportedModel = model;

        yield* parseAnthropicStream(res.body, (event) => {
            // `message_start` carries the input split; `message_delta` the running
            // output count. Both are folded so the final usage is provider-measured.
            const start = (event as any).message?.usage;
            const delta = (event as any).usage;
            const usage = start ?? delta;
            if (!usage) return;
            if (typeof (event as any).message?.model === 'string') reportedModel = (event as any).message.model;
            input      = Math.max(input,      Number(usage.input_tokens) || 0);
            output     = Math.max(output,     Number(usage.output_tokens) || 0);
            cacheRead  = Math.max(cacheRead,  Number(usage.cache_read_input_tokens) || 0);
            cacheWrite = Math.max(cacheWrite, Number(usage.cache_creation_input_tokens) || 0);
        });

        this._lastCall = {
            usage: {
                model: reportedModel,
                inputTokens: input,
                outputTokens: output,
                cachedInputTokens: cacheRead,
                cacheWriteTokens: cacheWrite,
            },
        };
    }

    private _buildBody(prompt: string, opts: BridgeGenerateOptions, stream: boolean): string {
        const sys = opts.systemPrompt ?? this._systemPrompt;
        const body: Record<string, unknown> = {
            model     : opts.model     ?? this._model,
            max_tokens: opts.maxTokens ?? this._maxTokens,
            messages  : [{ role: 'user', content: prompt }],
        };
        if (sys) {
            // Caching is a prefix match: render the stable system prompt as a
            // single cache-marked content block so reads on subsequent turns are
            // billed at ~10% of input price. The volatile user message is sent
            // unmarked after it, so it never enters the cached prefix.
            body['system'] = this._cacheSystem
                ? [{ type: 'text', text: sys, cache_control: { type: 'ephemeral' } }]
                : sys;
        }
        if (stream) body['stream'] = true;
        return JSON.stringify(body);
    }

    private _fetch(body: string): Promise<Response> {
        return fetch(API_URL, {
            method : 'POST',
            headers: {
                'Content-Type'      : 'application/json',
                'x-api-key'         : this._apiKey,
                'anthropic-version' : this._apiVersion,
            },
            body,
        });
    }
}

// ── SSE parser (Anthropic event format) ──────────────────────────────────────

async function* parseAnthropicStream(
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
            buffer = lines.pop() as string; // split() always yields ≥1 element → never undefined

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed.startsWith('data: ')) continue;

                const data = trimmed.slice(6);
                try {
                    const event = JSON.parse(data) as Record<string, unknown>;
                    onEvent?.(event);
                    // content_block_delta events carry the streamed text
                    if (event['type'] === 'content_block_delta') {
                        const text = (event as any).delta?.text;
                        if (typeof text === 'string' && text.length > 0) yield text;
                    }
                } catch {
                    // Skip malformed SSE lines
                }
            }
        }
    } finally {
        reader.releaseLock();
    }
}

/**
 * Maps an Anthropic `usage` object onto the canonical {@link LlmUsage} shape.
 * Absent when the response omits usage (an older gateway shim), in which case
 * `InstrumentedBridge` falls back to an estimate and flags it as such.
 */
function readAnthropicUsage(usage: unknown, model: string): LlmUsage | undefined {
    if (!usage || typeof usage !== 'object') return undefined;
    const u = usage as Record<string, unknown>;
    return {
        model,
        inputTokens: Number(u['input_tokens']) || 0,
        outputTokens: Number(u['output_tokens']) || 0,
        cachedInputTokens: Number(u['cache_read_input_tokens']) || 0,
        cacheWriteTokens: Number(u['cache_creation_input_tokens']) || 0,
    };
}
