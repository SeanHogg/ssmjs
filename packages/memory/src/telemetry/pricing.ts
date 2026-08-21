/**
 * telemetry/pricing.ts — cost-per-request, computed from real token counts.
 *
 * "Cost per request" is an LLM-native metric that cannot be derived from latency
 * or call count: it depends on the model, the input/output split, AND how much of
 * the input was a cache read (cheap) versus a cache write (dear). This module is
 * the ONE place that arithmetic lives, so the bridges, the metrics registry, and
 * the eval harness cannot disagree about what a run cost.
 *
 * Rates are DATA, not code: an enterprise on a negotiated rate card, a partner
 * platform (Bedrock/Vertex bill separately from the first-party API), or a model
 * released after this file was written is handled by passing a {@link PriceBook},
 * never by editing a branch. Unknown models cost `undefined` — reported as an
 * `unpricedCalls` count rather than silently billed at zero, because a silent
 * zero is how cost dashboards lie.
 */

import type { LlmUsage } from './types.js';

export interface ModelRate {
    /** USD per 1M input tokens. */
    inputPerMTok: number;
    /** USD per 1M output tokens. */
    outputPerMTok: number;
    /**
     * Multiplier applied to `inputPerMTok` for tokens READ from the provider's
     * prompt cache. Anthropic bills cache reads at 10% of input. Default 0.1.
     */
    cacheReadMultiplier?: number;
    /**
     * Multiplier applied to `inputPerMTok` for tokens WRITTEN to the prompt
     * cache. Anthropic bills a 5-minute cache write at 125% of input. Default 1.25.
     */
    cacheWriteMultiplier?: number;
}

/** Model id → rate. Keys match exactly, or by longest prefix (see {@link resolveRate}). */
export type PriceBook = Record<string, ModelRate>;

export const DEFAULT_CACHE_READ_MULTIPLIER = 0.1;
export const DEFAULT_CACHE_WRITE_MULTIPLIER = 1.25;

/**
 * First-party Anthropic API rates (USD / 1M tokens), current as of 2026-08-20.
 *
 * Bedrock and Vertex are partner-operated and priced separately — supply those
 * rates via a custom {@link PriceBook} rather than assuming these apply.
 */
export const ANTHROPIC_PRICE_BOOK: PriceBook = {
    'claude-fable-5':    { inputPerMTok: 10, outputPerMTok: 50 },
    'claude-mythos-5':   { inputPerMTok: 10, outputPerMTok: 50 },
    'claude-opus-5':     { inputPerMTok: 5,  outputPerMTok: 25 },
    'claude-opus-4-8':   { inputPerMTok: 5,  outputPerMTok: 25 },
    'claude-opus-4-7':   { inputPerMTok: 5,  outputPerMTok: 25 },
    'claude-opus-4-6':   { inputPerMTok: 5,  outputPerMTok: 25 },
    'claude-sonnet-5':   { inputPerMTok: 3,  outputPerMTok: 15 },
    'claude-sonnet-4-6': { inputPerMTok: 3,  outputPerMTok: 15 },
    'claude-haiku-4-5':  { inputPerMTok: 1,  outputPerMTok: 5  },
};

/**
 * Evermind serves `evermind/<ref>` traffic on-device, so it has no per-token
 * vendor bill. It is priced explicitly at zero rather than left unknown — the
 * whole point of routing to Evermind is that the saving is *measurable*.
 */
export const EVERMIND_PRICE_BOOK: PriceBook = {
    'evermind': { inputPerMTok: 0, outputPerMTok: 0 },
    'ssm':      { inputPerMTok: 0, outputPerMTok: 0 },
};

export const DEFAULT_PRICE_BOOK: PriceBook = { ...ANTHROPIC_PRICE_BOOK, ...EVERMIND_PRICE_BOOK };

/**
 * Resolves a model id to a rate: exact match first, then the longest key the
 * model id starts with. Prefix matching is what makes dated snapshots
 * (`claude-opus-4-5@20251101`) and platform-prefixed ids (`anthropic.claude-opus-5`)
 * resolve without a per-variant entry.
 */
export function resolveRate(model: string, book: PriceBook = DEFAULT_PRICE_BOOK): ModelRate | undefined {
    const exact = book[model];
    if (exact) return exact;

    let best: ModelRate | undefined;
    let bestLen = 0;
    for (const [key, rate] of Object.entries(book)) {
        if (key.length > bestLen && (model.startsWith(key) || model.includes(`/${key}`))) {
            best = rate;
            bestLen = key.length;
        }
    }
    return best;
}

/**
 * Cost in USD for one call, or `undefined` when the model is not in the book.
 *
 * A locally-served call (`localCacheHit`) is free by construction — it never
 * reached a provider — so it costs 0 even for an unpriced model.
 */
export function estimateCostUsd(usage: LlmUsage, book: PriceBook = DEFAULT_PRICE_BOOK): number | undefined {
    if (usage.localCacheHit) return 0;

    const rate = resolveRate(usage.model, book);
    if (!rate) return undefined;

    const cacheRead  = rate.cacheReadMultiplier  ?? DEFAULT_CACHE_READ_MULTIPLIER;
    const cacheWrite = rate.cacheWriteMultiplier ?? DEFAULT_CACHE_WRITE_MULTIPLIER;

    const perToken = (perMTok: number) => perMTok / 1_000_000;

    return (
        usage.inputTokens              * perToken(rate.inputPerMTok) +
        usage.outputTokens             * perToken(rate.outputPerMTok) +
        (usage.cachedInputTokens ?? 0) * perToken(rate.inputPerMTok * cacheRead) +
        (usage.cacheWriteTokens  ?? 0) * perToken(rate.inputPerMTok * cacheWrite)
    );
}

/**
 * Character-based token estimate — the fallback when a provider reports no usage.
 *
 * ~4 characters per token is the standard English approximation. Estimates are
 * flagged (`estimated: true`) all the way through to the metric snapshot so a
 * cost figure never silently mixes measured and guessed numbers.
 */
export function estimateTokens(text: string): number {
    if (!text) return 0;
    return Math.max(1, Math.ceil(text.length / 4));
}
