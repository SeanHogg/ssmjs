/**
 * tests/enterprise-coverage.test.ts — exercises the request builders, parsers and
 * degradation paths of the enterprise layer that the behavioural suites do not
 * reach directly.
 *
 * Dialect wire shapes get particular attention: they cannot be integration-tested
 * here (no vendor credentials), so the emitted request body and the parsing of a
 * recorded response shape ARE the contract under test.
 */

import { describe, expect, it, jest } from '@jest/globals';

import {
    evermindDialect,
    pineconeDialect,
    qdrantDialect,
    vertexAiDialect,
    type DialectContext,
} from '../src/vectorstore/dialects.js';
import { RestVectorStore } from '../src/vectorstore/RestVectorStore.js';
import { MemoryVectorStore } from '../src/vectorstore/MemoryVectorStore.js';
import { EnterpriseRetriever } from '../src/rag/EnterpriseRetriever.js';
import { MetricsRegistry } from '../src/telemetry/MetricsRegistry.js';
import { ConsoleSpanExporter, OtlpHttpSpanExporter } from '../src/telemetry/exporters.js';
import { Tracer } from '../src/telemetry/Tracer.js';
import { InstrumentedBridge } from '../src/telemetry/InstrumentedBridge.js';
import { AgentGraph } from '../src/orchestration/AgentGraph.js';
import { START, END } from '../src/orchestration/types.js';
import type { SpanData } from '../src/telemetry/types.js';
import type { VectorRecord } from '../src/vectorstore/types.js';
import type { TransformerBridge } from '../src/bridges/TransformerBridge.js';

const ctx: DialectContext = {
    collection: 'my kb',
    options: {
        index: 'projects/p/locations/l/indexes/7',
        indexEndpoint: 'projects/p/locations/l/indexEndpoints/9',
        deployedIndexId: 'deployed-1',
    },
};

const rec = (id: string): VectorRecord => ({
    id,
    text: `text ${id}`,
    vector: Float32Array.from([1, 2]),
    metadata: { tenantId: 'acme', acl: ['*'], sensitivity: 2 },
});

describe('evermind dialect wire shape', () => {
    it('builds every request against the collection path', () => {
        expect(evermindDialect.upsert(ctx, [rec('a')]).path).toBe('/collections/my%20kb/upsert');
        expect(evermindDialect.query(ctx, { topK: 3, includeVectors: false }).path).toBe('/collections/my%20kb/query');
        expect(evermindDialect.keywordSearch?.(ctx, { text: 'q', topK: 2 }).path).toBe('/collections/my%20kb/keyword');
        expect(evermindDialect.deleteByIds(ctx, ['a']).body).toEqual({ ids: ['a'] });
        expect(evermindDialect.deleteByFilter?.(ctx, { op: 'eq' }).body).toEqual({ filter: { op: 'eq' } });
        expect(evermindDialect.fetch(ctx, ['a']).body).toEqual({ ids: ['a'] });
        expect(evermindDialect.count?.(ctx, undefined).path).toBe('/collections/my%20kb/count');
    });

    it('reads counts, upsert results and records, defaulting sensibly', () => {
        expect(evermindDialect.parseUpsert({ upserted: 4, skipped: 1 }, 5)).toEqual({ upserted: 4, skipped: 1 });
        expect(evermindDialect.parseUpsert({}, 5)).toEqual({ upserted: 5, skipped: 0 });
        expect(evermindDialect.parseCount?.({ count: 12 })).toBe(12);
        expect(evermindDialect.parseCount?.({})).toBe(0);

        const records = evermindDialect.parseRecords({
            records: [{ id: 'a', text: 't', metadata: { x: 1 }, vector: [1, 2] }, { id: 'b' }],
        });
        expect(records[0]?.vector).toEqual(Float32Array.from([1, 2]));
        expect(records[1]).toEqual({ id: 'b', text: '', metadata: {} });
        expect(evermindDialect.parseRecords({})).toEqual([]);
        expect(evermindDialect.parseMatches({})).toEqual([]);
        expect(evermindDialect.parseMatches({ matches: [{ id: 'a' }] })[0]).toEqual({
            id: 'a', text: '', score: 0, metadata: {},
        });
    });
});

describe('qdrant dialect wire shape', () => {
    it('stores chunk text in the payload under a reserved key', () => {
        const req = qdrantDialect.upsert(ctx, [rec('a')]);
        expect(req.method).toBe('PUT');
        expect((req.body as any).points[0].payload).toEqual({
            tenantId: 'acme', acl: ['*'], sensitivity: 2, __text: 'text a',
        });
        expect(qdrantDialect.parseUpsert({}, 3)).toEqual({ upserted: 3, skipped: 0 });
    });

    it('round-trips payload text back out of matches and records', () => {
        const matches = qdrantDialect.parseMatches({
            result: [{ id: 'a', score: 0.5, payload: { __text: 'hi', tenantId: 'acme' }, vector: [1] }],
        });
        expect(matches[0]).toMatchObject({ id: 'a', text: 'hi', score: 0.5 });
        expect(matches[0]?.metadata).toEqual({ tenantId: 'acme' });
        expect(matches[0]?.vector).toEqual(Float32Array.from([1]));
        expect(qdrantDialect.parseMatches({})).toEqual([]);

        const records = qdrantDialect.parseRecords({
            result: [{ id: 'a', payload: { __text: 'hi' }, vector: [1] }, { id: 'b', payload: {} }],
        });
        expect(records[0]?.text).toBe('hi');
        expect(records[1]?.text).toBe('');
        expect(qdrantDialect.parseRecords({})).toEqual([]);
    });

    it('builds delete, fetch and count requests', () => {
        expect(qdrantDialect.deleteByIds(ctx, ['a']).body).toEqual({ points: ['a'] });
        expect(qdrantDialect.deleteByFilter?.(ctx, { must: [] }).body).toEqual({ filter: { must: [] } });
        expect((qdrantDialect.fetch(ctx, ['a']).body as any).with_payload).toBe(true);
        expect((qdrantDialect.count?.(ctx, undefined).body as any).exact).toBe(true);
        expect(qdrantDialect.parseCount?.({ result: { count: 9 } })).toBe(9);
        expect(qdrantDialect.parseCount?.({})).toBe(0);
    });

    it('maps contains and exists the way qdrant expresses them', () => {
        const contains = qdrantDialect.translateFilter({ op: 'contains', field: 'title', value: 'x' });
        expect(contains.pushed).toEqual({ must: [{ key: 'title', match: { text: 'x' } }] });

        // `is_empty` is the inverse of `exists`, so it stays a local residual.
        const exists = qdrantDialect.translateFilter({ op: 'exists', field: 'title' });
        expect(exists.pushed).toBeUndefined();
        expect(exists.residual).toEqual({ op: 'exists', field: 'title' });

        // A range clause with a non-numeric bound cannot be expressed.
        const badRange = qdrantDialect.translateFilter({ op: 'gt', field: 'title', value: 'abc' });
        expect(badRange.residual).toEqual({ op: 'gt', field: 'title', value: 'abc' });

        const nin = qdrantDialect.translateFilter({ op: 'nin', field: 'tenantId', values: ['x'] });
        expect(nin.pushed).toEqual({ must: [{ key: 'tenantId', match: { except: ['x'] } }] });

        const multi = qdrantDialect.translateFilter({
            op: 'and',
            filters: [
                { op: 'exists', field: 'a' },
                { op: 'ne', field: 'b', value: 1 },
            ],
        });
        expect(multi.residual?.op).toBe('and');
    });
});

describe('pinecone dialect wire shape', () => {
    it('namespaces every request and reports the vendor upsert count', () => {
        expect((pineconeDialect.upsert(ctx, [rec('a')]).body as any).namespace).toBe('my kb');
        expect(pineconeDialect.parseUpsert({ upsertedCount: 3 }, 5)).toEqual({ upserted: 3, skipped: 0 });
        expect(pineconeDialect.parseUpsert({}, 5)).toEqual({ upserted: 5, skipped: 0 });

        expect(pineconeDialect.deleteByIds(ctx, ['a']).body).toEqual({ namespace: 'my kb', ids: ['a'] });
        expect(pineconeDialect.deleteByFilter?.(ctx, { a: 1 }).body).toEqual({ namespace: 'my kb', filter: { a: 1 } });
        expect(pineconeDialect.fetch(ctx, ['a']).body).toEqual({ namespace: 'my kb', ids: ['a'] });

        expect(pineconeDialect.parseMatches({})).toEqual([]);
        expect(pineconeDialect.parseRecords({})).toEqual([]);
    });

    it('translates every comparison operator it supports', () => {
        const ops = ['ne', 'gt', 'gte', 'lt', 'lte'] as const;
        for (const op of ops) {
            const translated = pineconeDialect.translateFilter({ op, field: 'n', value: 1 });
            expect(translated.pushed).toEqual({ n: { [`$${op}`]: 1 } });
        }
        expect(pineconeDialect.translateFilter({ op: 'nin', field: 'n', values: [1] }).pushed)
            .toEqual({ n: { $nin: [1] } });

        const unsupported = pineconeDialect.translateFilter({ op: 'exists', field: 'n' });
        expect(unsupported.pushed).toBeUndefined();
        expect(unsupported.residual).toEqual({ op: 'exists', field: 'n' });
    });
});

describe('vertex-ai dialect wire shape', () => {
    it('emits datapoints with string and numeric restricts', () => {
        const req = vertexAiDialect.upsert(ctx, [rec('a')]);
        expect(req.path).toBe('/v1/projects/p/locations/l/indexes/7:upsertDatapoints');
        const datapoint = (req.body as any).datapoints[0];
        expect(datapoint.datapointId).toBe('a');
        expect(datapoint.restricts).toEqual([
            { namespace: 'tenantId', allowList: ['acme'] },
            { namespace: 'acl', allowList: ['*'] },
        ]);
        expect(datapoint.numericRestricts).toEqual([{ namespace: 'sensitivity', valueDouble: 2 }]);
        expect(vertexAiDialect.parseUpsert({}, 2)).toEqual({ upserted: 2, skipped: 0 });
    });

    it('builds findNeighbors, removeDatapoints and readIndexDatapoints requests', () => {
        const query = vertexAiDialect.query(ctx, {
            vector: Float32Array.from([1, 2]), topK: 4, includeVectors: true,
            filter: { restricts: [{ namespace: 'tenantId', allowList: ['acme'] }] },
        });
        expect(query.path).toBe('/v1/projects/p/locations/l/indexEndpoints/9:findNeighbors');
        expect((query.body as any).deployedIndexId).toBe('deployed-1');
        expect((query.body as any).queries[0].neighborCount).toBe(4);
        expect((query.body as any).queries[0].datapoint.restricts).toHaveLength(1);

        // No pushed filter at all still produces a well-formed datapoint.
        const bare = vertexAiDialect.query(ctx, { topK: 1, includeVectors: false });
        expect((bare.body as any).queries[0].datapoint.restricts).toEqual([]);

        expect(vertexAiDialect.deleteByIds(ctx, ['a']).body).toEqual({ datapointIds: ['a'] });
        expect(vertexAiDialect.fetch(ctx, ['a']).path)
            .toBe('/v1/projects/p/locations/l/indexEndpoints/9:readIndexDatapoints');
    });

    it('reconstructs metadata from restricts when reading datapoints', () => {
        const records = vertexAiDialect.parseRecords({
            datapoints: [{
                datapointId: 'a',
                featureVector: [1, 2],
                restricts: [
                    { namespace: 'acl', allowList: ['a', 'b'] },
                    { namespace: 'tenantId', allowList: ['acme'] },
                    { allowList: ['ignored'] },
                ],
                numericRestricts: [
                    { namespace: 'sensitivity', valueFloat: 3 },
                    { namespace: 'skipped' },
                ],
            }],
        });

        expect(records[0]?.metadata).toEqual({ acl: ['a', 'b'], tenantId: 'acme', sensitivity: 3 });
        expect(records[0]?.vector).toEqual(Float32Array.from([1, 2]));
        expect(vertexAiDialect.parseRecords({})).toEqual([]);
        expect(vertexAiDialect.parseMatches({ nearestNeighbors: [{}] })).toEqual([]);
    });

    it('leaves a boolean equality and a non-string list residual', () => {
        const translated = vertexAiDialect.translateFilter({
            op: 'and',
            filters: [
                { op: 'eq', field: 'archived', value: true },
                { op: 'in', field: 'code', values: [1, 2] },
                { op: 'eq', field: 'rank', value: 5 },
            ],
        });
        const pushed = translated.pushed as { restricts: unknown[]; numericRestricts: unknown[] };
        expect(pushed.restricts).toEqual([]);
        expect(pushed.numericRestricts).toEqual([{ namespace: 'rank', op: 'EQUAL', valueDouble: 5 }]);
        expect(translated.residual?.op).toBe('and');
    });
});

describe('RestVectorStore remaining paths', () => {
    it('accepts a dialect instance and hydrates text on fetch', async () => {
        const impl = (async () => new Response(JSON.stringify({
            datapoints: [{ datapointId: 'chunk-1' }],
        }), { status: 200 })) as unknown as typeof fetch;

        const store = new RestVectorStore({
            dialect: vertexAiDialect,
            baseUrl: 'https://vertex',
            collection: 'idx',
            options: ctx.options,
            fetchImpl: impl,
            textResolver: async (ids) => new Map(ids.map((id) => [id, `hydrated ${id}`])),
        });

        expect(store.name).toBe('rest:vertex-ai');
        const records = await store.fetch(['chunk-1']);
        expect(records[0]?.text).toBe('hydrated chunk-1');
    });

    it('leaves text empty when a vector-only dialect has no resolver', async () => {
        const impl = (async () => new Response(JSON.stringify({
            datapoints: [{ datapointId: 'chunk-1' }],
        }), { status: 200 })) as unknown as typeof fetch;

        const store = new RestVectorStore({
            dialect: 'vertex-ai', baseUrl: 'https://vertex', collection: 'idx',
            options: ctx.options, fetchImpl: impl,
        });
        expect((await store.fetch(['chunk-1']))[0]?.text).toBe('');
    });

    it('pushes count down when the dialect supports it', async () => {
        const impl = (async () => new Response(JSON.stringify({ count: 42 }), { status: 200 })) as unknown as typeof fetch;
        const store = new RestVectorStore({ dialect: 'evermind', baseUrl: 'https://gw', collection: 'kb', fetchImpl: impl });
        expect(await store.count()).toBe(42);
    });

    it('surfaces a network failure after exhausting retries', async () => {
        const impl = (async () => { throw new Error('ECONNRESET'); }) as unknown as typeof fetch;
        const store = new RestVectorStore({
            dialect: 'evermind', baseUrl: 'https://gw', collection: 'kb',
            fetchImpl: impl, maxRetries: 1, sleep: async () => {},
        });
        await expect(store.query({ vector: Float32Array.from([1]) })).rejects.toThrow('ECONNRESET');
    });

    it('tolerates a non-JSON success body', async () => {
        const impl = (async () => new Response('not json', { status: 200 })) as unknown as typeof fetch;
        const store = new RestVectorStore({ dialect: 'evermind', baseUrl: 'https://gw', collection: 'kb', fetchImpl: impl });
        expect(await store.query({ vector: Float32Array.from([1]) })).toEqual([]);
    });

    it('emulates deleteByFilter on a dialect with no bulk delete at all', async () => {
        const impl = ((async (url: string) =>
            new Response(JSON.stringify(
                url.includes('findNeighbors')
                    ? { nearestNeighbors: [{ neighbors: [{ distance: 0, datapoint: { datapointId: 'a' } }] }] }
                    : {},
            ), { status: 200 })) as unknown) as typeof fetch;

        const store = new RestVectorStore({
            dialect: 'vertex-ai', baseUrl: 'https://vertex', collection: 'idx',
            options: ctx.options, fetchImpl: impl,
        });
        expect(await store.deleteByFilter({ op: 'eq', field: 'sourceId', value: 's' })).toBe(1);
    });
});

describe('EnterpriseRetriever degradation paths', () => {
    it('keeps the fused order when candidates carry no vectors', async () => {
        const store = new MemoryVectorStore();
        await store.upsert([
            { id: 'a', text: 'alpha beta', metadata: { tenantId: 't', acl: ['*'] } },
            { id: 'b', text: 'beta gamma', metadata: { tenantId: 't', acl: ['*'] } },
        ]);

        // Records were stored WITHOUT vectors, so MMR has nothing to diversify on.
        const retriever = new EnterpriseRetriever({
            store, embed: async () => Float32Array.from([1, 0]),
        });
        const result = await retriever.retrieve('beta', { scope: { tenantId: 't' }, topK: 1 });

        expect(result.mode).toBe('keyword-only');
        expect(result.passages).toHaveLength(1);
    });

    it('falls back to the source id when a passage has no title or uri', async () => {
        const store = new MemoryVectorStore();
        await store.upsert([{
            id: 'chunk', text: 'body', vector: Float32Array.from([1, 0]),
            metadata: { tenantId: 't', acl: ['*'] },
        }]);

        const retriever = new EnterpriseRetriever({ store, embed: async () => Float32Array.from([1, 0]) });
        const result = await retriever.retrieve('body', { scope: { tenantId: 't' }, topK: 1 });

        expect(result.passages[0]?.sourceId).toBe('chunk');
        expect(result.passages[0]?.title).toBeUndefined();
        expect(result.passages[0]?.uri).toBeUndefined();
    });
});

describe('telemetry edge paths', () => {
    it('writes to console by default and stays terse without verbose', () => {
        const spy = jest.spyOn(console, 'log').mockImplementation(() => {});
        new ConsoleSpanExporter().export([{
            traceId: 'abcdef1234', spanId: 's', name: 'n', kind: 'internal',
            startedAt: 0, endedAt: 1, durationMs: 1,
            attributes: { a: 1 }, events: [], status: 'ok',
        }]);
        expect(spy).toHaveBeenCalledTimes(1);   // no attribute line without verbose
        spy.mockRestore();
    });

    it('omits an attribute line when a verbose span has no attributes', () => {
        const lines: string[] = [];
        new ConsoleSpanExporter({ write: (l) => lines.push(l), verbose: true }).export([{
            traceId: 't', spanId: 's', name: 'n', kind: 'internal',
            startedAt: 0, attributes: {}, events: [], status: 'unset',
        }]);
        expect(lines).toHaveLength(1);
    });

    it('posts a successful OTLP batch and swallows a thrown transport by default', async () => {
        const ok = jest.fn(async () => new Response('{}', { status: 200 }));
        const exporter = new OtlpHttpSpanExporter({
            url: 'https://collector', headers: { 'x-key': 'v' },
            fetchImpl: ok as unknown as typeof fetch,
        });
        await exporter.export([{
            traceId: 't', spanId: 's', name: 'n', kind: 'embedding',
            startedAt: 0, endedAt: 1, attributes: {}, events: [], status: 'unset',
            usage: { model: 'm', inputTokens: 1, outputTokens: 1 },
        }]);
        expect(ok).toHaveBeenCalledTimes(1);

        const throwing = jest.fn(async () => { throw new Error('dns'); });
        const quiet = new OtlpHttpSpanExporter({ url: 'https://collector', fetchImpl: throwing as unknown as typeof fetch });
        await expect(quiet.export([{
            traceId: 't', spanId: 's', name: 'n', kind: 'internal',
            startedAt: 0, attributes: {}, events: [], status: 'ok',
        }])).resolves.toBeUndefined();
    });

    it('attributes a model-less llm span from its attributes, then to unknown', () => {
        const registry = new MetricsRegistry({ maxSamples: 2 });
        const base: SpanData = {
            traceId: 't', spanId: 's', name: 'llm', kind: 'llm',
            startedAt: 0, attributes: {}, events: [], status: 'ok',
        };

        registry.record({ ...base, attributes: { 'llm.model': 'from-attribute' } });
        registry.record(base);

        expect(registry.llm('from-attribute').calls).toBe(1);
        expect(registry.llm('unknown').calls).toBe(1);
        // A span with no durationMs contributes zero rather than NaN.
        expect(registry.llm().tokensPerSecond).toBe(0);
        expect(registry.llm().avgLatencyMs).toBe(0);

        // maxSamples bounds the percentile reservoir.
        for (let i = 0; i < 5; i++) registry.record({ ...base, durationMs: i * 10 });
        expect(registry.llm().p95LatencyMs).toBeGreaterThan(0);
    });

    it('resolves a per-call parent function on an instrumented bridge', async () => {
        const tracer = new Tracer();
        const parent = tracer.startSpan('turn', { kind: 'agent' });
        const inner: TransformerBridge = { supportsStreaming: false, generate: async () => 'ok' };

        const bridge = new InstrumentedBridge(inner, { tracer, parent: () => parent });
        await bridge.generate('hi');
        parent.end();

        const llmSpan = tracer.finished().find((s) => s.kind === 'llm');
        expect(llmSpan?.parentSpanId).toBe(parent.spanId);
    });

    it('sets only defined attributes', () => {
        const tracer = new Tracer();
        const span = tracer.startSpan('x');
        span.setAttributes({ kept: 'yes', dropped: undefined });
        expect(span.data.attributes).toEqual({ kept: 'yes' });
        span.end();
    });
});

describe('AgentGraph edge paths', () => {
    it('reports a non-Error thrown value as a string', async () => {
        const graph = new AgentGraph<{ x: number }>({ failFast: true });
        graph.addNode('bad', () => { throw 'a bare string'; });
        graph.addEdge(START, 'bad');
        graph.addEdge('bad', END);

        const result = await graph.invoke({ x: 0 });
        expect(result.error).toContain('a bare string');
    });

    it('treats a node returning nothing as an empty update', async () => {
        const graph = new AgentGraph<{ x: number }>();
        graph.addNode('noop', () => undefined);
        graph.addEdge(START, 'noop');
        graph.addEdge('noop', END);

        const result = await graph.invoke({ x: 7 });
        expect(result.state.x).toBe(7);
        expect(result.path).toEqual(['noop']);
    });

    it('honours a per-run recursion limit override', async () => {
        const graph = new AgentGraph<{ n: number }>({ recursionLimit: 100 });
        graph.addNode('loop', (s) => ({ n: s.n + 1 }));
        graph.addEdge(START, 'loop');
        graph.addConditionalEdges('loop', () => 'loop');

        const result = await graph.invoke({ n: 0 }, { recursionLimit: 3 });
        expect(result.reason).toBe('recursion-limit');
        expect(result.state.n).toBe(3);
    });

    it('accepts an async conditional edge', async () => {
        const graph = new AgentGraph<{ go: boolean; hit: boolean }>();
        graph.addNode('a', () => ({}));
        graph.addNode('b', () => ({ hit: true }));
        graph.addEdge(START, 'a');
        graph.addConditionalEdges('a', async (s) => (s.go ? 'b' : END));
        graph.addEdge('b', END);

        expect((await graph.invoke({ go: true, hit: false })).state.hit).toBe(true);
        expect((await graph.invoke({ go: false, hit: false })).state.hit).toBe(false);
    });

    it('runs without a checkpointer even when a threadId is supplied', async () => {
        const graph = new AgentGraph<{ x: number }>();
        graph.addNode('a', () => ({ x: 1 }));
        graph.addEdge(START, 'a');
        graph.addEdge('a', END);

        const result = await graph.invoke({ x: 0 }, { threadId: 'no-store' });
        expect(result.checkpointId).toBeUndefined();
        expect(result.state.x).toBe(1);
    });
});
