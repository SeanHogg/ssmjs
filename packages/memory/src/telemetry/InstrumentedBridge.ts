/**
 * telemetry/InstrumentedBridge.ts — one decorator that makes ANY bridge observable.
 *
 * The alternative — instrumenting Anthropic, OpenAI, Vertex, Fetch, Caching and
 * SemanticCaching bridges each in their own way — guarantees six subtly different
 * definitions of "a call". This is a single decorator over the
 * {@link TransformerBridge} port, so a bridge added tomorrow is observable the
 * moment it satisfies the interface, and the caching bridges are measured at the
 * same seam as the provider bridges (which is how the cache-hit rate becomes a
 * real number rather than an anecdote).
 *
 * Token counts come from the provider when the inner bridge reports them via
 * `lastCall`, and are estimated from characters otherwise — with `estimated: true`
 * carried through to the metric snapshot so a dashboard never mixes measured and
 * inferred spend without saying so.
 */

import type { TransformerBridge, BridgeGenerateOptions } from '../bridges/TransformerBridge.js';
import { estimateTokens } from './pricing.js';
import { Tracer, type Span } from './Tracer.js';
import type { LlmUsage } from './types.js';

export interface InstrumentedBridgeOptions {
    /** Tracer that receives the spans. */
    tracer: Tracer;
    /** Model label used when the inner bridge reports none. Default `'unknown'`. */
    model?: string;
    /** Span name. Default `'llm.generate'`. */
    spanName?: string;
    /**
     * Parent span, so a model call nests under the agent/graph node that made it.
     * A function is accepted for the common case where the parent changes per call.
     */
    parent?: Span | (() => Span | undefined);
    /** Static attributes stamped on every span (tenant, environment, route). */
    attributes?: Record<string, string | number | boolean>;
    /**
     * Token counter, when an exact one is available (e.g. the provider's
     * count_tokens endpoint). Default: ~4 chars per token.
     */
    countTokens?: (text: string) => number;
}

export class InstrumentedBridge implements TransformerBridge {
    readonly supportsStreaming: boolean;

    private readonly _inner: TransformerBridge;
    private readonly _tracer: Tracer;
    private readonly _model: string;
    private readonly _spanName: string;
    private readonly _parent?: Span | (() => Span | undefined);
    private readonly _attributes: Record<string, string | number | boolean>;
    private readonly _count: (text: string) => number;

    private _lastCall: { usage?: LlmUsage; cacheHit?: boolean; cacheTier?: string } | undefined;

    constructor(inner: TransformerBridge, opts: InstrumentedBridgeOptions) {
        this._inner = inner;
        this._tracer = opts.tracer;
        this._model = opts.model ?? 'unknown';
        this._spanName = opts.spanName ?? 'llm.generate';
        if (opts.parent) this._parent = opts.parent;
        this._attributes = opts.attributes ?? {};
        this._count = opts.countTokens ?? estimateTokens;
        this.supportsStreaming = inner.supportsStreaming;
    }

    /** Usage/cache info for the most recent call — makes the decorator chainable. */
    get lastCall(): { usage?: LlmUsage; cacheHit?: boolean; cacheTier?: string } | undefined {
        return this._lastCall;
    }

    async generate(prompt: string, opts: BridgeGenerateOptions = {}): Promise<string> {
        const span = this._startSpan(prompt, opts, false);
        try {
            const reply = await this._inner.generate(prompt, opts);
            this._finish(span, prompt, reply, opts);
            span.end('ok');
            return reply;
        } catch (err) {
            span.fail(err);
            throw err;
        }
    }

    async *stream(prompt: string, opts: BridgeGenerateOptions = {}): AsyncIterable<string> {
        if (!this._inner.stream) {
            throw new Error('InstrumentedBridge: the wrapped bridge does not support streaming.');
        }
        const span = this._startSpan(prompt, opts, true);
        let reply = '';
        let first = true;
        try {
            for await (const chunk of this._inner.stream(prompt, opts)) {
                if (first) {
                    span.recordFirstToken();
                    first = false;
                }
                reply += chunk;
                yield chunk;
            }
            this._finish(span, prompt, reply, opts);
            span.end('ok');
        } catch (err) {
            span.fail(err);
            throw err;
        }
    }

    private _startSpan(prompt: string, opts: BridgeGenerateOptions, streaming: boolean): Span {
        const parent = typeof this._parent === 'function' ? this._parent() : this._parent;
        const span = this._tracer.startSpan(this._spanName, {
            kind: 'llm',
            ...(parent ? { parent } : {}),
        });
        span.setAttributes({
            ...this._attributes,
            'llm.model': opts.model ?? this._model,
            'llm.streaming': streaming,
            'llm.prompt_chars': prompt.length,
            'llm.max_tokens': opts.maxTokens,
            'llm.temperature': opts.temperature,
        });
        return span;
    }

    private _finish(span: Span, prompt: string, reply: string, opts: BridgeGenerateOptions): void {
        const reported = (this._inner as { lastCall?: { usage?: LlmUsage; cacheHit?: boolean; cacheTier?: string } }).lastCall;
        const model = reported?.usage?.model ?? opts.model ?? this._model;

        const usage: LlmUsage = reported?.usage ?? {
            model,
            inputTokens: this._count(prompt),
            outputTokens: this._count(reply),
            estimated: true,
        };
        if (reported?.cacheHit) usage.localCacheHit = true;

        span.recordUsage(usage);
        span.setAttributes({
            'llm.response_chars': reply.length,
            'llm.cache_hit': reported?.cacheHit ?? false,
            'llm.cache_tier': reported?.cacheTier,
        });

        this._lastCall = {
            usage,
            ...(reported?.cacheHit !== undefined ? { cacheHit: reported.cacheHit } : {}),
            ...(reported?.cacheTier !== undefined ? { cacheTier: reported.cacheTier } : {}),
        };
    }
}
