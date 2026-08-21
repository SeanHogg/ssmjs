/**
 * telemetry/MetricsRegistry.ts — LLM-native metrics, aggregated from spans.
 *
 * The registry is itself a {@link SpanExporter}. That is the DRY move: every
 * subsystem already emits spans for tracing, so metrics are a *projection* of the
 * trace stream rather than a second instrumentation path that can drift out of
 * agreement with it. Attach it to a tracer and tokens/sec, cost-per-request, TTFT
 * and cache-hit rate fall out of the traffic that is already being traced.
 *
 * Metrics that matter for an LLM system, and why the usual web ones do not:
 *   • tokens/sec        — throughput the user actually feels, not requests/sec.
 *   • cost per request  — the only unit that ties spend to a product decision.
 *   • TTFT              — perceived latency for streamed answers; p95 latency
 *                         hides it entirely.
 *   • cache-hit rate    — the lever that moves cost, split local vs provider.
 *   • unpriced calls    — honesty: a model missing from the rate card is counted,
 *                         never billed at a silent zero.
 */

import type { SpanData, SpanExporter, SpanKind } from './types.js';

export interface LlmMetricSnapshot {
    /** Model id, or `'*'` for the all-models roll-up. */
    model: string;
    calls: number;
    errors: number;
    errorRate: number;

    inputTokens: number;
    outputTokens: number;
    cachedInputTokens: number;
    cacheWriteTokens: number;
    totalTokens: number;

    /** Sum of priced calls, USD. Calls on unpriced models contribute 0 here. */
    costUsd: number;
    avgCostPerRequestUsd: number;
    /** Calls whose model was absent from the price book — excluded from `costUsd`. */
    unpricedCalls: number;
    /** Calls whose token counts were estimated rather than provider-reported. */
    estimatedCalls: number;

    totalDurationMs: number;
    avgLatencyMs: number;
    p50LatencyMs: number;
    p95LatencyMs: number;

    /** Output tokens per second of wall-clock spent generating them. */
    tokensPerSecond: number;
    /** Mean time-to-first-token over streamed calls; 0 when none streamed. */
    avgTimeToFirstTokenMs: number;
    streamedCalls: number;

    /** Share of calls served without reaching a provider (exact/semantic cache). */
    localCacheHitRate: number;
    /** Share of input tokens served from the provider's prompt cache. */
    providerCacheHitRate: number;
}

export interface KindMetricSnapshot {
    kind: SpanKind;
    count: number;
    errors: number;
    totalDurationMs: number;
    avgLatencyMs: number;
    p95LatencyMs: number;
}

export interface MetricsRegistryOptions {
    /**
     * Max latency samples retained per series for percentiles. Default 2048 —
     * bounded so a long-lived process cannot grow the registry without limit.
     */
    maxSamples?: number;
}

interface LlmAccumulator {
    calls: number;
    errors: number;
    inputTokens: number;
    outputTokens: number;
    cachedInputTokens: number;
    cacheWriteTokens: number;
    costUsd: number;
    unpricedCalls: number;
    estimatedCalls: number;
    localCacheHits: number;
    totalDurationMs: number;
    ttftTotalMs: number;
    streamedCalls: number;
    latencies: number[];
}

interface KindAccumulator {
    count: number;
    errors: number;
    totalDurationMs: number;
    latencies: number[];
}

export class MetricsRegistry implements SpanExporter {
    private readonly _maxSamples: number;
    private readonly _byModel = new Map<string, LlmAccumulator>();
    private readonly _byKind = new Map<SpanKind, KindAccumulator>();

    constructor(opts: MetricsRegistryOptions = {}) {
        this._maxSamples = Math.max(1, opts.maxSamples ?? 2048);
    }

    /** {@link SpanExporter} — attach this registry directly to a {@link Tracer}. */
    export(spans: readonly SpanData[]): void {
        for (const span of spans) this.record(span);
    }

    /** Folds one finished span into the aggregates. */
    record(span: SpanData): void {
        const duration = span.durationMs ?? 0;
        const failed = span.status === 'error';

        const kindAcc = this._kind(span.kind);
        kindAcc.count += 1;
        if (failed) kindAcc.errors += 1;
        kindAcc.totalDurationMs += duration;
        push(kindAcc.latencies, duration, this._maxSamples);

        if (span.kind !== 'llm' && span.kind !== 'embedding') return;

        const model = span.usage?.model ?? String(span.attributes['llm.model'] ?? 'unknown');
        for (const acc of [this._model(model), this._model('*')]) {
            acc.calls += 1;
            if (failed) acc.errors += 1;
            acc.totalDurationMs += duration;
            push(acc.latencies, duration, this._maxSamples);

            if (span.timeToFirstTokenMs !== undefined) {
                acc.streamedCalls += 1;
                acc.ttftTotalMs += span.timeToFirstTokenMs;
            }

            const usage = span.usage;
            if (!usage) continue;

            acc.inputTokens       += usage.inputTokens;
            acc.outputTokens      += usage.outputTokens;
            acc.cachedInputTokens += usage.cachedInputTokens ?? 0;
            acc.cacheWriteTokens  += usage.cacheWriteTokens ?? 0;
            if (usage.estimated) acc.estimatedCalls += 1;
            if (usage.localCacheHit) acc.localCacheHits += 1;

            if (span.costUsd === undefined) acc.unpricedCalls += 1;
            else acc.costUsd += span.costUsd;
        }
    }

    /** Snapshot for one model, or the `'*'` roll-up when omitted. */
    llm(model = '*'): LlmMetricSnapshot {
        return summarize(model, this._byModel.get(model) ?? emptyLlm());
    }

    /** Snapshots for every model seen, excluding the `'*'` roll-up. */
    models(): LlmMetricSnapshot[] {
        return [...this._byModel.entries()]
            .filter(([model]) => model !== '*')
            .map(([model, acc]) => summarize(model, acc))
            .sort((a, b) => b.costUsd - a.costUsd || b.calls - a.calls);
    }

    /** Latency/error snapshots per span kind — retrieval, ingest, graph, tool. */
    kinds(): KindMetricSnapshot[] {
        return [...this._byKind.entries()]
            .map(([kind, acc]) => ({
                kind,
                count: acc.count,
                errors: acc.errors,
                totalDurationMs: acc.totalDurationMs,
                avgLatencyMs: acc.count ? acc.totalDurationMs / acc.count : 0,
                p95LatencyMs: percentile(acc.latencies, 0.95),
            }))
            .sort((a, b) => b.count - a.count);
    }

    reset(): void {
        this._byModel.clear();
        this._byKind.clear();
    }

    private _model(model: string): LlmAccumulator {
        let acc = this._byModel.get(model);
        if (!acc) {
            acc = emptyLlm();
            this._byModel.set(model, acc);
        }
        return acc;
    }

    private _kind(kind: SpanKind): KindAccumulator {
        let acc = this._byKind.get(kind);
        if (!acc) {
            acc = { count: 0, errors: 0, totalDurationMs: 0, latencies: [] };
            this._byKind.set(kind, acc);
        }
        return acc;
    }
}

function emptyLlm(): LlmAccumulator {
    return {
        calls: 0, errors: 0,
        inputTokens: 0, outputTokens: 0, cachedInputTokens: 0, cacheWriteTokens: 0,
        costUsd: 0, unpricedCalls: 0, estimatedCalls: 0, localCacheHits: 0,
        totalDurationMs: 0, ttftTotalMs: 0, streamedCalls: 0, latencies: [],
    };
}

function summarize(model: string, acc: LlmAccumulator): LlmMetricSnapshot {
    const totalTokens = acc.inputTokens + acc.outputTokens + acc.cachedInputTokens + acc.cacheWriteTokens;
    const seconds = acc.totalDurationMs / 1000;
    const billableInput = acc.inputTokens + acc.cachedInputTokens;

    return {
        model,
        calls: acc.calls,
        errors: acc.errors,
        errorRate: acc.calls ? acc.errors / acc.calls : 0,

        inputTokens: acc.inputTokens,
        outputTokens: acc.outputTokens,
        cachedInputTokens: acc.cachedInputTokens,
        cacheWriteTokens: acc.cacheWriteTokens,
        totalTokens,

        costUsd: acc.costUsd,
        avgCostPerRequestUsd: acc.calls ? acc.costUsd / acc.calls : 0,
        unpricedCalls: acc.unpricedCalls,
        estimatedCalls: acc.estimatedCalls,

        totalDurationMs: acc.totalDurationMs,
        avgLatencyMs: acc.calls ? acc.totalDurationMs / acc.calls : 0,
        p50LatencyMs: percentile(acc.latencies, 0.5),
        p95LatencyMs: percentile(acc.latencies, 0.95),

        tokensPerSecond: seconds > 0 ? acc.outputTokens / seconds : 0,
        avgTimeToFirstTokenMs: acc.streamedCalls ? acc.ttftTotalMs / acc.streamedCalls : 0,
        streamedCalls: acc.streamedCalls,

        localCacheHitRate: acc.calls ? acc.localCacheHits / acc.calls : 0,
        providerCacheHitRate: billableInput ? acc.cachedInputTokens / billableInput : 0,
    };
}

function push(samples: number[], value: number, cap: number): void {
    samples.push(value);
    if (samples.length > cap) samples.shift();
}

/** Nearest-rank percentile over an unsorted sample array. */
export function percentile(samples: readonly number[], q: number): number {
    if (samples.length === 0) return 0;
    const sorted = [...samples].sort((a, b) => a - b);
    const rank = Math.ceil(q * sorted.length);
    const idx = Math.min(sorted.length - 1, Math.max(0, rank - 1));
    return sorted[idx] as number;
}
