/**
 * OpenAIBridge – TransformerBridge implementation for the OpenAI Chat API.
 *
 * Supports both non-streaming and streaming (SSE) completions.
 * Compatible with any OpenAI-compatible endpoint via the `baseUrl` option.
 */

import { SSMError } from '../errors/SSMError.js';
import type { TransformerBridge, BridgeGenerateOptions } from './TransformerBridge.js';

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

        yield* parseOpenAIStream(res.body);
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

async function* parseOpenAIStream(body: ReadableStream<Uint8Array>): AsyncIterable<string> {
    const reader  = body.getReader();
    const decoder = new TextDecoder();
    let buffer    = '';

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() ?? '';   // keep the last (possibly partial) line

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
