/**
 * CachingBridge – a read-through caching decorator for any TransformerBridge.
 *
 * Wraps an inner bridge and memoises `generate()` keyed on the full request
 * shape (model, system, prompt, sampling). Identical completions are served
 * from memory instead of re-billing the provider — the single most effective
 * lever for cutting LLM spend on repeated prompts (distillation passes, retries,
 * fan-out over duplicate inputs).
 *
 * Composes with every bridge, so the caching policy lives in one place rather
 * than being reimplemented per provider:
 *
 *   const bridge = new CachingBridge(new AnthropicBridge({ apiKey }));
 *
 * Streaming is delegated straight through and never cached — a token stream is
 * consumed once and caching it would defeat its purpose.
 */

import type { TransformerBridge, BridgeGenerateOptions, BridgeCallInfo } from './TransformerBridge.js';
import { ResponseCache, buildCacheKey, type ResponseCacheOptions } from './ResponseCache.js';

export interface CachingBridgeOptions extends ResponseCacheOptions {
    /**
     * Provide a shared ResponseCache instance instead of letting the bridge
     * create its own. Use this to share one cache across multiple bridges, or
     * to inspect/clear the cache from outside.
     */
    cache? : ResponseCache;
}

export class CachingBridge implements TransformerBridge {
    private readonly _inner : TransformerBridge;
    private readonly _cache : ResponseCache;

    private _lastCall: BridgeCallInfo | undefined;

    constructor(inner: TransformerBridge, opts: CachingBridgeOptions = {}) {
        this._inner = inner;
        this._cache = opts.cache ?? new ResponseCache(opts);
    }

    /** Mirrors the wrapped bridge so callers can still gate on streaming support. */
    get supportsStreaming(): boolean {
        return this._inner.supportsStreaming;
    }

    /** The underlying cache — exposed for stats inspection and manual eviction. */
    get cache(): ResponseCache {
        return this._cache;
    }

    /**
     * A hit is declared here rather than inferred downstream, which is what turns
     * the cache-hit rate into a measured metric. On a miss the inner bridge's own
     * provider-reported usage is passed through unchanged, so decorating a bridge
     * never downgrades its cost accounting to an estimate.
     */
    get lastCall(): BridgeCallInfo | undefined {
        return this._lastCall;
    }

    async generate(prompt: string, opts: BridgeGenerateOptions = {}): Promise<string> {
        const key = buildCacheKey({
            prompt,
            model       : opts.model,
            systemPrompt : opts.systemPrompt,
            maxTokens   : opts.maxTokens,
            temperature : opts.temperature,
            topP        : opts.topP,
        });

        const cached = this._cache.get(key);
        if (cached !== undefined) {
            this._lastCall = {
                cacheHit : true,
                cacheTier: 'exact',
                usage    : {
                    model            : opts.model ?? 'unknown',
                    inputTokens      : 0,
                    outputTokens     : 0,
                    localCacheHit    : true,
                },
            };
            return cached;
        }

        const value = await this._inner.generate(prompt, opts);
        this._cache.set(key, value, Date.now());
        const inner = this._inner.lastCall;
        this._lastCall = { cacheHit: false, ...(inner?.usage ? { usage: inner.usage } : {}) };
        return value;
    }

    /**
     * Streaming is delegated to the inner bridge unchanged and is never cached.
     * Present only when the inner bridge supports it, so `supportsStreaming`
     * stays an accurate gate.
     */
    stream(prompt: string, opts?: BridgeGenerateOptions): AsyncIterable<string> {
        if (!this._inner.stream) {
            throw new Error('Wrapped bridge does not support streaming.');
        }
        return this._inner.stream(prompt, opts);
    }
}
