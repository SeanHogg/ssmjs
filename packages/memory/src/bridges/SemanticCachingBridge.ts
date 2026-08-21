/**
 * SemanticCachingBridge – a read-through *semantic* caching decorator for any
 * TransformerBridge. The semantic sibling of CachingBridge: where CachingBridge
 * only reuses byte-identical prompts, this reuses a prior answer when the new
 * prompt is within `threshold` cosine similarity of one already answered.
 *
 *   const bridge = new SemanticCachingBridge(new AnthropicBridge({ apiKey }), {
 *     embed: (t) => runtime.embed(t),        // on-device SSM, free
 *     l2: new FetchSemanticCacheBackend({ baseUrl, apiKey }),  // shared via gateway
 *   });
 *
 * Streaming is delegated straight through and never cached.
 */

import type { TransformerBridge, BridgeGenerateOptions, BridgeCallInfo } from './TransformerBridge.js';
import { SemanticCache, type SemanticCacheOptions } from '../cache/SemanticCache.js';

export interface SemanticCachingBridgeOptions extends Omit<SemanticCacheOptions, never> {
    /** Provide a shared SemanticCache instance instead of constructing one. */
    cache? : SemanticCache;
}

export class SemanticCachingBridge implements TransformerBridge {
    private readonly _inner : TransformerBridge;
    private readonly _cache : SemanticCache;

    private _lastCall: BridgeCallInfo | undefined;

    constructor(inner: TransformerBridge, opts: SemanticCachingBridgeOptions) {
        this._inner = inner;
        this._cache = opts.cache ?? new SemanticCache(opts);
    }

    get supportsStreaming(): boolean {
        return this._inner.supportsStreaming;
    }

    /** The underlying SemanticCache — exposed for stats inspection. */
    get cache(): SemanticCache {
        return this._cache;
    }

    /**
     * Reports which tier answered (`l1` local vector scan, `l2` shared backend, or
     * a miss), so the saving from paraphrase reuse is attributable per tier rather
     * than lumped into one "cache" number.
     */
    get lastCall(): BridgeCallInfo | undefined {
        return this._lastCall;
    }

    async generate(prompt: string, opts: BridgeGenerateOptions = {}): Promise<string> {
        // Match on system + prompt meaning so different system contexts don't
        // cross-hit; partition further by model via the stored meta.
        const queryText = opts.systemPrompt ? `${opts.systemPrompt}\n${prompt}` : prompt;
        const { response, cached, tier } = await this._cache.getOrGenerate(
            queryText,
            () => this._inner.generate(prompt, opts),
            opts.model ? { model: opts.model } : undefined,
        );

        if (cached) {
            this._lastCall = {
                cacheHit : true,
                ...(tier ? { cacheTier: tier } : {}),
                usage    : {
                    model        : opts.model ?? 'unknown',
                    inputTokens  : 0,
                    outputTokens : 0,
                    localCacheHit: true,
                },
            };
        } else {
            const inner = this._inner.lastCall;
            this._lastCall = { cacheHit: false, ...(inner?.usage ? { usage: inner.usage } : {}) };
        }
        return response;
    }

    /** Streaming is delegated unchanged and never cached. */
    stream(prompt: string, opts?: BridgeGenerateOptions): AsyncIterable<string> {
        if (!this._inner.stream) {
            throw new Error('Wrapped bridge does not support streaming.');
        }
        return this._inner.stream(prompt, opts);
    }
}
