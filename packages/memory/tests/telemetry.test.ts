import { describe, expect, it, jest } from '@jest/globals';

import { Tracer } from '../src/telemetry/Tracer.js';
import { MetricsRegistry, percentile } from '../src/telemetry/MetricsRegistry.js';
import { InstrumentedBridge } from '../src/telemetry/InstrumentedBridge.js';
import {
    ConsoleSpanExporter,
    InMemorySpanExporter,
    OtlpHttpSpanExporter,
    toOtlpPayload,
    toOtlpSpan,
} from '../src/telemetry/exporters.js';
import {
    ANTHROPIC_PRICE_BOOK,
    estimateCostUsd,
    estimateTokens,
    resolveRate,
} from '../src/telemetry/pricing.js';
import type { TransformerBridge, BridgeGenerateOptions } from '../src/bridges/TransformerBridge.js';
import type { BridgeCallInfo, SpanData } from '../src/telemetry/types.js';

/** A deterministic clock so span durations are exact, not flaky. */
function fixedClock(startAt = 1_000) {
    let now = startAt;
    return {
        now: () => now,
        advance: (ms: number) => { now += ms; },
    };
}

function deterministicIds() {
    let n = 0;
    return (bytes: number) => `${bytes}-${(n++).toString(16).padStart(4, '0')}`;
}

class StubBridge implements TransformerBridge {
    readonly supportsStreaming = true;
    lastCall: BridgeCallInfo | undefined;

    constructor(
        private readonly reply: string,
        private readonly usage?: BridgeCallInfo,
        private readonly failWith?: Error,
    ) {}

    async generate(_prompt: string, _opts?: BridgeGenerateOptions): Promise<string> {
        if (this.failWith) throw this.failWith;
        this.lastCall = this.usage;
        return this.reply;
    }

    async *stream(): AsyncIterable<string> {
        for (const chunk of this.reply.split(' ')) yield `${chunk} `;
        this.lastCall = this.usage;
    }
}

describe('pricing', () => {
    it('prices input, output, cache reads and cache writes at their own rates', () => {
        const cost = estimateCostUsd({
            model: 'claude-opus-5',
            inputTokens: 1_000_000,
            outputTokens: 1_000_000,
            cachedInputTokens: 1_000_000,
            cacheWriteTokens: 1_000_000,
        });
        // 5 (input) + 25 (output) + 0.5 (cache read @0.1x) + 6.25 (cache write @1.25x)
        expect(cost).toBeCloseTo(36.75, 6);
    });

    it('does not fold cache reads into input — the 10x overstatement guard', () => {
        const split = estimateCostUsd({
            model: 'claude-opus-5', inputTokens: 0, outputTokens: 0, cachedInputTokens: 1_000_000,
        }) as number;
        const naive = estimateCostUsd({
            model: 'claude-opus-5', inputTokens: 1_000_000, outputTokens: 0,
        }) as number;
        expect(naive / split).toBeCloseTo(10, 6);
    });

    it('returns undefined for an unpriced model rather than a silent zero', () => {
        expect(estimateCostUsd({ model: 'some-new-model', inputTokens: 100, outputTokens: 100 }))
            .toBeUndefined();
    });

    it('prices a locally-served call at zero even when the model is unknown', () => {
        expect(estimateCostUsd({
            model: 'mystery', inputTokens: 0, outputTokens: 0, localCacheHit: true,
        })).toBe(0);
    });

    it('resolves dated snapshots and platform prefixes by longest prefix', () => {
        expect(resolveRate('claude-opus-4-6@20260101', ANTHROPIC_PRICE_BOOK))
            .toEqual(ANTHROPIC_PRICE_BOOK['claude-opus-4-6']);
        expect(resolveRate('publishers/anthropic/models/claude-sonnet-5', ANTHROPIC_PRICE_BOOK))
            .toEqual(ANTHROPIC_PRICE_BOOK['claude-sonnet-5']);
        expect(resolveRate('gpt-nope', ANTHROPIC_PRICE_BOOK)).toBeUndefined();
    });

    it('estimates tokens at ~4 characters each, never zero for non-empty text', () => {
        expect(estimateTokens('')).toBe(0);
        expect(estimateTokens('a')).toBe(1);
        expect(estimateTokens('x'.repeat(400))).toBe(100);
    });
});

describe('Tracer', () => {
    it('nests children under one trace id and records durations', () => {
        const clock = fixedClock();
        const tracer = new Tracer({ now: clock.now, newId: deterministicIds() });

        const root = tracer.startSpan('run', { kind: 'graph' });
        clock.advance(5);
        const child = root.child('node', { kind: 'agent' });
        clock.advance(10);
        child.end();
        clock.advance(5);
        root.end();

        const spans = tracer.trace(root.traceId);
        expect(spans).toHaveLength(2);
        expect(spans[0]?.name).toBe('node');
        expect(spans[0]?.parentSpanId).toBe(root.spanId);
        expect(spans[0]?.durationMs).toBe(10);
        expect(spans[1]?.durationMs).toBe(20);
    });

    it('prices usage recorded on a span', () => {
        const tracer = new Tracer({ now: fixedClock().now });
        const span = tracer.startSpan('llm', { kind: 'llm' });
        span.recordUsage({ model: 'claude-haiku-4-5', inputTokens: 1_000_000, outputTokens: 0 });
        const data = span.end();
        expect(data.costUsd).toBeCloseTo(1, 6);
    });

    it('ignores mutation after a span has closed', () => {
        const tracer = new Tracer({ now: fixedClock().now });
        const span = tracer.startSpan('late');
        span.end();
        span.setAttribute('ignored', 1);
        span.addEvent('ignored');
        span.recordUsage({ model: 'claude-opus-5', inputTokens: 5, outputTokens: 5 });
        span.recordFirstToken();

        expect(span.data.attributes['ignored']).toBeUndefined();
        expect(span.data.events).toHaveLength(0);
        expect(span.data.usage).toBeUndefined();
        expect(span.data.status).toBe('ok');
    });

    it('records a failure once and does not overwrite it on a second close', () => {
        const tracer = new Tracer({ now: fixedClock().now });
        const span = tracer.startSpan('boom');
        span.fail(new Error('first'));
        span.fail(new Error('second'));
        expect(span.data.status).toBe('error');
        expect(span.data.error).toBe('first');
    });

    it('withSpan ends ok on success and error on throw', async () => {
        const tracer = new Tracer({ now: fixedClock().now });
        await tracer.withSpan('ok', {}, () => 'value');
        await expect(tracer.withSpan('bad', {}, () => { throw new Error('nope'); }))
            .rejects.toThrow('nope');

        const statuses = tracer.finished().map((s) => s.status);
        expect(statuses).toEqual(['ok', 'error']);
    });

    it('drops a whole trace when sampled out — children inherit, never re-roll', () => {
        const tracer = new Tracer({
            now: fixedClock().now,
            newId: deterministicIds(),
            sampleRate: 0,
            random: () => 0.99,
        });
        const root = tracer.startSpan('dropped');
        const child = root.child('also-dropped');
        child.end();
        root.end();

        // The point of caching the verdict per trace: a dropped trace is dropped
        // whole, so an exported trace is never a tree with holes in it.
        expect(root.recorded).toBe(false);
        expect(child.recorded).toBe(false);
        expect(tracer.trace(root.traceId)).toHaveLength(0);
    });

    it('records a child whose parent verdict is unknown rather than tearing the trace', () => {
        // The parent was decided by a DIFFERENT tracer (a trace crossing a process
        // boundary), so this tracer has no cached verdict for it.
        const upstream = new Tracer({ now: fixedClock().now });
        const parent = upstream.startSpan('upstream-root');

        const local = new Tracer({ now: fixedClock().now, sampleRate: 0, random: () => 0.99 });
        const adopted = local.startSpan('local-work', { parent });
        adopted.end();

        expect(adopted.recorded).toBe(true);
        expect(local.trace(parent.traceId)).toHaveLength(1);
    });

    it('flushes finished spans to the exporter and honours the batch size', async () => {
        const exporter = new InMemorySpanExporter();
        const tracer = new Tracer({ now: fixedClock().now, exporter, maxBatchSize: 2 });

        tracer.startSpan('a').end();
        expect(exporter.spans).toHaveLength(0);
        tracer.startSpan('b').end();
        await Promise.resolve();
        expect(exporter.spans).toHaveLength(2);

        tracer.startSpan('c').end();
        await tracer.shutdown();
        expect(exporter.spans).toHaveLength(3);
    });
});

describe('MetricsRegistry', () => {
    function llmSpan(over: Partial<SpanData> = {}): SpanData {
        return {
            traceId: 't', spanId: 's', name: 'llm.generate', kind: 'llm',
            startedAt: 0, endedAt: 100, durationMs: 100,
            attributes: {}, events: [], status: 'ok',
            ...over,
        };
    }

    it('derives tokens/sec, cost per request and cache rates from spans', () => {
        const registry = new MetricsRegistry();
        registry.export([
            llmSpan({
                durationMs: 1000, costUsd: 0.01,
                usage: { model: 'claude-opus-5', inputTokens: 900, outputTokens: 500, cachedInputTokens: 100 },
            }),
            llmSpan({
                durationMs: 1000, costUsd: 0.03,
                usage: { model: 'claude-opus-5', inputTokens: 1000, outputTokens: 500 },
            }),
        ]);

        const snapshot = registry.llm('claude-opus-5');
        expect(snapshot.calls).toBe(2);
        expect(snapshot.outputTokens).toBe(1000);
        expect(snapshot.tokensPerSecond).toBeCloseTo(500, 6);   // 1000 tokens / 2s
        expect(snapshot.avgCostPerRequestUsd).toBeCloseTo(0.02, 6);
        expect(snapshot.providerCacheHitRate).toBeCloseTo(100 / 2000, 6);
    });

    it('counts unpriced calls instead of billing them at zero', () => {
        const registry = new MetricsRegistry();
        registry.record(llmSpan({ usage: { model: 'unknown-model', inputTokens: 10, outputTokens: 10 } }));
        const snapshot = registry.llm('unknown-model');
        expect(snapshot.unpricedCalls).toBe(1);
        expect(snapshot.costUsd).toBe(0);
    });

    it('tracks errors, TTFT and local cache hits', () => {
        const registry = new MetricsRegistry();
        registry.record(llmSpan({ status: 'error' }));
        registry.record(llmSpan({
            timeToFirstTokenMs: 40,
            usage: { model: 'claude-opus-5', inputTokens: 1, outputTokens: 1, localCacheHit: true },
        }));

        const all = registry.llm();
        expect(all.calls).toBe(2);
        expect(all.errorRate).toBeCloseTo(0.5, 6);
        expect(all.avgTimeToFirstTokenMs).toBe(40);
        expect(all.localCacheHitRate).toBeCloseTo(0.5, 6);
    });

    it('summarises non-LLM spans by kind and ranks models by cost', () => {
        const registry = new MetricsRegistry();
        registry.record({
            traceId: 't', spanId: 'r', name: 'retrieval.hybrid', kind: 'retrieval',
            startedAt: 0, endedAt: 30, durationMs: 30, attributes: {}, events: [], status: 'ok',
        });
        registry.record(llmSpan({ costUsd: 5, usage: { model: 'big', inputTokens: 1, outputTokens: 1 } }));
        registry.record(llmSpan({ costUsd: 1, usage: { model: 'small', inputTokens: 1, outputTokens: 1 } }));

        expect(registry.kinds().find((k) => k.kind === 'retrieval')?.avgLatencyMs).toBe(30);
        expect(registry.models().map((m) => m.model)).toEqual(['big', 'small']);

        registry.reset();
        expect(registry.models()).toHaveLength(0);
    });

    it('computes nearest-rank percentiles', () => {
        expect(percentile([], 0.5)).toBe(0);
        expect(percentile([10, 20, 30, 40], 0.5)).toBe(20);
        expect(percentile([10, 20, 30, 40], 0.95)).toBe(40);
    });
});

describe('InstrumentedBridge', () => {
    it('prefers provider-reported usage over an estimate', async () => {
        const tracer = new Tracer({ now: fixedClock().now });
        const inner = new StubBridge('hello there', {
            usage: { model: 'claude-opus-5', inputTokens: 11, outputTokens: 3, cachedInputTokens: 2 },
        });
        const bridge = new InstrumentedBridge(inner, { tracer });

        await bridge.generate('a prompt');
        const span = tracer.finished()[0] as SpanData;

        expect(span.usage?.inputTokens).toBe(11);
        expect(span.usage?.estimated).toBeUndefined();
        expect(span.costUsd).toBeGreaterThan(0);
    });

    it('falls back to a flagged estimate when the bridge reports nothing', async () => {
        const tracer = new Tracer({ now: fixedClock().now });
        const bridge = new InstrumentedBridge(new StubBridge('four'), { tracer, model: 'claude-haiku-4-5' });

        await bridge.generate('x'.repeat(40));
        const span = tracer.finished()[0] as SpanData;

        expect(span.usage?.estimated).toBe(true);
        expect(span.usage?.inputTokens).toBe(10);
        expect(span.attributes['llm.cache_hit']).toBe(false);
    });

    it('marks a cache hit reported by a caching decorator', async () => {
        const tracer = new Tracer({ now: fixedClock().now });
        const inner = new StubBridge('cached', {
            cacheHit: true, cacheTier: 'l1',
            usage: { model: 'claude-opus-5', inputTokens: 0, outputTokens: 0, localCacheHit: true },
        });
        const bridge = new InstrumentedBridge(inner, { tracer });

        await bridge.generate('paraphrase');
        const span = tracer.finished()[0] as SpanData;

        expect(span.attributes['llm.cache_hit']).toBe(true);
        expect(span.attributes['llm.cache_tier']).toBe('l1');
        expect(span.costUsd).toBe(0);
        expect(bridge.lastCall?.cacheTier).toBe('l1');
    });

    it('records time-to-first-token while streaming', async () => {
        const clock = fixedClock();
        const tracer = new Tracer({ now: clock.now });
        const inner = new StubBridge('one two three');
        const bridge = new InstrumentedBridge(inner, { tracer });

        const chunks: string[] = [];
        for await (const chunk of bridge.stream('go')) {
            clock.advance(7);
            chunks.push(chunk);
        }

        const span = tracer.finished()[0] as SpanData;
        expect(chunks.join('').trim()).toBe('one two three');
        expect(span.timeToFirstTokenMs).toBe(0);
        expect(span.events.some((e) => e.name === 'first-token')).toBe(true);
    });

    it('fails the span and rethrows when the inner bridge throws', async () => {
        const tracer = new Tracer({ now: fixedClock().now });
        const bridge = new InstrumentedBridge(
            new StubBridge('', undefined, new Error('upstream 503')),
            { tracer },
        );
        await expect(bridge.generate('x')).rejects.toThrow('upstream 503');
        expect(tracer.finished()[0]?.status).toBe('error');
    });

    it('rejects streaming when the wrapped bridge cannot stream', async () => {
        const tracer = new Tracer({ now: fixedClock().now });
        const nonStreaming: TransformerBridge = {
            supportsStreaming: false,
            generate: async () => 'x',
        };
        const bridge = new InstrumentedBridge(nonStreaming, { tracer });
        await expect(async () => {
            for await (const _ of bridge.stream('x')) { /* unreachable */ }
        }).rejects.toThrow('does not support streaming');
    });

    it('feeds a MetricsRegistry attached as the tracer exporter', async () => {
        const registry = new MetricsRegistry();
        const tracer = new Tracer({ now: fixedClock().now, exporter: registry, maxBatchSize: 1 });
        const bridge = new InstrumentedBridge(
            new StubBridge('ok', { usage: { model: 'claude-opus-5', inputTokens: 100, outputTokens: 50 } }),
            { tracer },
        );

        await bridge.generate('hello');
        await tracer.flush();

        expect(registry.llm('claude-opus-5').calls).toBe(1);
        expect(registry.llm('claude-opus-5').outputTokens).toBe(50);
    });
});

describe('exporters', () => {
    it('renders one console line per span', () => {
        const lines: string[] = [];
        const exporter = new ConsoleSpanExporter({ write: (l) => lines.push(l), verbose: true });
        exporter.export([{
            traceId: 'abcdef1234', spanId: 's', name: 'llm.generate', kind: 'llm',
            startedAt: 0, endedAt: 12, durationMs: 12,
            attributes: { 'llm.model': 'claude-opus-5' }, events: [], status: 'error', error: 'boom',
            usage: { model: 'claude-opus-5', inputTokens: 3, outputTokens: 4 }, costUsd: 0.5,
        }]);
        expect(lines[0]).toContain('3in/4out');
        expect(lines[0]).toContain('ERROR: boom');
        expect(lines[1]).toContain('claude-opus-5');
    });

    it('maps a span to OTLP with GenAI attributes', () => {
        const otlp = toOtlpSpan({
            traceId: 't', spanId: 's', parentSpanId: 'p', name: 'llm.generate', kind: 'llm',
            startedAt: 1, endedAt: 3, durationMs: 2,
            attributes: { tenant: 'acme', retries: 2, cached: true, ratio: 0.5 },
            events: [{ name: 'first-token', at: 2 }],
            status: 'ok',
            usage: { model: 'claude-opus-5', inputTokens: 7, outputTokens: 9, cachedInputTokens: 1 },
            costUsd: 0.25, timeToFirstTokenMs: 1,
        }) as Record<string, unknown>;

        expect(otlp['kind']).toBe(3);
        expect(otlp['parentSpanId']).toBe('p');
        expect(otlp['startTimeUnixNano']).toBe('1000000');
        const keys = (otlp['attributes'] as Array<{ key: string }>).map((a) => a.key);
        expect(keys).toContain('gen_ai.usage.input_tokens');
        expect(keys).toContain('gen_ai.usage.cost_usd');
        expect(keys).toContain('gen_ai.server.time_to_first_token_ms');
        expect((otlp['status'] as { code: number }).code).toBe(1);
    });

    it('wraps spans in a resource envelope', () => {
        const payload = toOtlpPayload([{
            traceId: 't', spanId: 's', name: 'n', kind: 'internal',
            startedAt: 0, attributes: {}, events: [], status: 'unset',
        }], 'my-service') as { resourceSpans: Array<Record<string, unknown>> };

        const resource = payload.resourceSpans[0]?.['resource'] as { attributes: Array<{ value: { stringValue: string } }> };
        expect(resource.attributes[0]?.value.stringValue).toBe('my-service');
    });

    it('reports OTLP transport failures instead of throwing into the caller', async () => {
        const onError = jest.fn();
        const failing = jest.fn(async () => new Response('nope', { status: 500 })) as unknown as typeof fetch;
        const exporter = new OtlpHttpSpanExporter({ url: 'https://collector/v1/traces', fetchImpl: failing, onError });

        await exporter.export([{
            traceId: 't', spanId: 's', name: 'n', kind: 'internal',
            startedAt: 0, endedAt: 1, attributes: {}, events: [], status: 'ok',
        }]);
        expect(onError).toHaveBeenCalledTimes(1);

        await exporter.export([]);
        expect(onError).toHaveBeenCalledTimes(1);
    });

    it('collects spans per trace in memory', () => {
        const exporter = new InMemorySpanExporter();
        exporter.export([
            { traceId: 'a', spanId: '1', name: 'x', kind: 'internal', startedAt: 0, attributes: {}, events: [], status: 'ok' },
            { traceId: 'b', spanId: '2', name: 'y', kind: 'internal', startedAt: 0, attributes: {}, events: [], status: 'ok' },
        ]);
        expect(exporter.trace('a')).toHaveLength(1);
        exporter.reset();
        expect(exporter.spans).toHaveLength(0);
    });
});
