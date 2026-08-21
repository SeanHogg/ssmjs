import { describe, expect, it, jest } from '@jest/globals';

import { MemoryVectorStore } from '../src/vectorstore/MemoryVectorStore.js';
import { RestVectorStore } from '../src/vectorstore/RestVectorStore.js';
import {
    accessFilter,
    combineFilters,
    filterFields,
    matchesFilter,
} from '../src/vectorstore/filter.js';
import {
    evermindDialect,
    getDialect,
    listDialects,
    pineconeDialect,
    qdrantDialect,
    registerDialect,
    vertexAiDialect,
    type VectorDialect,
} from '../src/vectorstore/dialects.js';
import type { MetadataFilter, MetadataRecord, VectorRecord } from '../src/vectorstore/types.js';

const vec = (...values: number[]) => Float32Array.from(values);

function record(id: string, text: string, metadata: MetadataRecord, vector?: Float32Array): VectorRecord {
    return vector ? { id, text, metadata, vector } : { id, text, metadata };
}

/** Two tenants sharing a store — the setup that makes a leak visible. */
function seeded(): MemoryVectorStore {
    const store = new MemoryVectorStore();
    void store.upsert([
        record('acme#1', 'quarterly revenue rose 12 percent', { tenantId: 'acme', acl: ['*'], sensitivity: 1 }, vec(1, 0)),
        record('acme#2', 'the SSO outage incident report ERR-4021', { tenantId: 'acme', acl: ['sec-team'], sensitivity: 5 }, vec(0.9, 0.1)),
        record('globex#1', 'globex confidential merger memo', { tenantId: 'globex', acl: ['*'] }, vec(1, 0)),
    ]);
    return store;
}

describe('filter algebra', () => {
    const meta: MetadataRecord = {
        tenantId: 'acme', acl: ['sec-team', 'ops'], sensitivity: 3,
        title: 'Incident Report', archived: false, owner: null,
    };

    it('evaluates comparisons, membership and intersection', () => {
        expect(matchesFilter(meta, { op: 'eq', field: 'tenantId', value: 'acme' })).toBe(true);
        expect(matchesFilter(meta, { op: 'ne', field: 'tenantId', value: 'globex' })).toBe(true);
        expect(matchesFilter(meta, { op: 'lte', field: 'sensitivity', value: 3 })).toBe(true);
        expect(matchesFilter(meta, { op: 'lt', field: 'sensitivity', value: 3 })).toBe(false);
        expect(matchesFilter(meta, { op: 'gt', field: 'sensitivity', value: 1 })).toBe(true);
        expect(matchesFilter(meta, { op: 'gte', field: 'sensitivity', value: 3 })).toBe(true);
        expect(matchesFilter(meta, { op: 'in', field: 'tenantId', values: ['acme', 'x'] })).toBe(true);
        expect(matchesFilter(meta, { op: 'nin', field: 'tenantId', values: ['acme'] })).toBe(false);
        expect(matchesFilter(meta, { op: 'anyOf', field: 'acl', values: ['ops'] })).toBe(true);
        expect(matchesFilter(meta, { op: 'anyOf', field: 'acl', values: ['nobody'] })).toBe(false);
        expect(matchesFilter(meta, { op: 'contains', field: 'title', value: 'incident' })).toBe(true);
        expect(matchesFilter(meta, { op: 'exists', field: 'title' })).toBe(true);
        expect(matchesFilter(meta, { op: 'exists', field: 'owner' })).toBe(false);
    });

    it('treats an ordering comparison on a missing or wrong-typed field as false', () => {
        expect(matchesFilter(meta, { op: 'lte', field: 'missing', value: 3 })).toBe(false);
        expect(matchesFilter(meta, { op: 'gt', field: 'title', value: 3 })).toBe(false);
        expect(matchesFilter(meta, { op: 'gt', field: 'archived', value: 1 })).toBe(false);
        expect(matchesFilter(meta, { op: 'lt', field: 'sensitivity', value: null })).toBe(false);
    });

    it('composes and/or/not, and treats an absent filter as match-all', () => {
        expect(matchesFilter(meta, undefined)).toBe(true);
        expect(matchesFilter(meta, {
            op: 'and',
            filters: [
                { op: 'eq', field: 'tenantId', value: 'acme' },
                { op: 'not', filter: { op: 'eq', field: 'archived', value: true } },
            ],
        })).toBe(true);
        expect(matchesFilter(meta, {
            op: 'or',
            filters: [
                { op: 'eq', field: 'tenantId', value: 'globex' },
                { op: 'eq', field: 'sensitivity', value: 3 },
            ],
        })).toBe(true);
    });

    it('compares array values element-wise', () => {
        expect(matchesFilter(meta, { op: 'eq', field: 'acl', value: ['sec-team', 'ops'] })).toBe(true);
        expect(matchesFilter(meta, { op: 'eq', field: 'acl', value: ['ops', 'sec-team'] })).toBe(false);
        expect(matchesFilter(meta, { op: 'eq', field: 'acl', value: 'ops' })).toBe(false);
    });

    it('compiles an access scope that fails CLOSED without an acl', () => {
        const scope = accessFilter({ tenantId: 'acme', principals: ['sec-team'], maxSensitivity: 3 });
        expect(matchesFilter(meta, scope)).toBe(true);

        // No acl at all → matches nothing, which is the whole safety property.
        expect(matchesFilter({ tenantId: 'acme', sensitivity: 1 }, scope)).toBe(false);
        // Wrong tenant with a permissive acl still fails.
        expect(matchesFilter({ tenantId: 'globex', acl: ['*'] }, scope)).toBe(false);
        // Over the sensitivity ceiling.
        expect(matchesFilter({ tenantId: 'acme', acl: ['*'], sensitivity: 9 }, scope)).toBe(false);
    });

    it('lets a public record through to any principal in the tenant', () => {
        const scope = accessFilter({ tenantId: 'acme' });
        expect(matchesFilter({ tenantId: 'acme', acl: ['*'] }, scope)).toBe(true);
        expect(matchesFilter({ tenantId: 'acme', acl: ['sec-team'] }, scope)).toBe(false);
    });

    it('ANDs a caller filter with the scope, keeping either alone', () => {
        const caller: MetadataFilter = { op: 'eq', field: 'kind', value: 'doc' };
        expect(combineFilters(caller, { tenantId: 'acme' })?.op).toBe('and');
        expect(combineFilters(caller, undefined)).toBe(caller);
        expect(combineFilters(undefined, { tenantId: 'acme' })?.op).toBe('and');
        expect(combineFilters(undefined, undefined)).toBeUndefined();
    });

    it('reports every field a filter touches', () => {
        const fields = filterFields({
            op: 'and',
            filters: [
                { op: 'eq', field: 'a', value: 1 },
                { op: 'not', filter: { op: 'or', filters: [{ op: 'exists', field: 'b' }] } },
            ],
        });
        expect([...fields].sort()).toEqual(['a', 'b']);
    });
});

describe('MemoryVectorStore', () => {
    it('scopes a query to one tenant', async () => {
        const store = seeded();
        const hits = await store.query({
            vector: vec(1, 0), topK: 10, scope: { tenantId: 'acme' },
        });
        expect(hits.map((h) => h.id)).toEqual(['acme#1']);
    });

    it('honours principals and the sensitivity ceiling', async () => {
        const store = seeded();

        const asSecurity = await store.query({
            vector: vec(1, 0), topK: 10, scope: { tenantId: 'acme', principals: ['sec-team'] },
        });
        expect(asSecurity.map((h) => h.id).sort()).toEqual(['acme#1', 'acme#2']);

        const capped = await store.query({
            vector: vec(1, 0), topK: 10,
            scope: { tenantId: 'acme', principals: ['sec-team'], maxSensitivity: 2 },
        });
        expect(capped.map((h) => h.id)).toEqual(['acme#1']);
    });

    it('runs BM25 keyword search over the visible corpus only', async () => {
        const store = seeded();
        const hits = await store.keywordSearch({
            text: 'ERR-4021', topK: 5, scope: { tenantId: 'acme', principals: ['sec-team'] },
        });
        expect(hits[0]?.id).toBe('acme#2');
        expect(hits[0]?.score).toBeCloseTo(1, 6);

        const denied = await store.keywordSearch({ text: 'ERR-4021', scope: { tenantId: 'acme' } });
        expect(denied).toHaveLength(0);

        expect(await store.keywordSearch({ text: '   ' })).toHaveLength(0);
    });

    it('lists by filter when no query vector is supplied', async () => {
        const store = seeded();
        const hits = await store.query({ topK: 10, scope: { tenantId: 'globex' } });
        expect(hits.map((h) => h.id)).toEqual(['globex#1']);
    });

    it('counts, fetches, deletes and bulk-deletes', async () => {
        const store = seeded();
        expect(await store.count()).toBe(3);
        expect(await store.count(undefined, { tenantId: 'acme', principals: ['sec-team'] })).toBe(2);

        expect((await store.fetch(['acme#1', 'nope'])).map((r) => r.id)).toEqual(['acme#1']);
        expect(await store.delete(['acme#1', 'nope'])).toBe(1);
        expect(await store.deleteByFilter({ op: 'eq', field: 'tenantId', value: 'globex' })).toBe(1);
        expect(await store.count()).toBe(1);
    });

    it('honours minScore and includeVectors', async () => {
        const store = seeded();
        const hits = await store.query({
            vector: vec(0, 1), topK: 5, minScore: 0.5, scope: { tenantId: 'acme' },
        });
        expect(hits).toHaveLength(0);

        const withVectors = await store.query({ vector: vec(1, 0), topK: 1, includeVectors: true });
        expect(withVectors[0]?.vector).toBeInstanceOf(Float32Array);
    });

    it('replaces a vector under an existing id without returning the stale neighbour', async () => {
        const store = new MemoryVectorStore({ annThreshold: 1 });
        await store.upsert([record('a', 'first', { tenantId: 't', acl: ['*'] }, vec(1, 0))]);
        await store.query({ vector: vec(1, 0), topK: 1 });  // force index construction
        await store.upsert([record('a', 'second', { tenantId: 't', acl: ['*'] }, vec(0, 1))]);

        const hits = await store.query({ vector: vec(0, 1), topK: 1 });
        expect(hits[0]?.text).toBe('second');
    });

    it('uses the ANN path above the threshold and still returns a full filtered page', async () => {
        const store = new MemoryVectorStore({ annThreshold: 8, filterOverfetch: 1 });
        const records: VectorRecord[] = [];
        for (let i = 0; i < 60; i++) {
            const angle = (i / 60) * Math.PI / 2;
            records.push(record(
                `r${i}`,
                `document ${i}`,
                { tenantId: 't', acl: ['*'], rare: i === 59 },
                vec(Math.cos(angle), Math.sin(angle)),
            ));
        }
        await store.upsert(records);

        // A filter so selective the graph's over-fetch cannot fill the page; the
        // store must fall back to an exact scan rather than return short.
        const rare = await store.query({
            vector: vec(1, 0), topK: 3, filter: { op: 'eq', field: 'rare', value: true },
        });
        expect(rare.map((h) => h.id)).toEqual(['r59']);

        const unfiltered = await store.query({ vector: vec(1, 0), topK: 5 });
        expect(unfiltered).toHaveLength(5);
        expect(unfiltered[0]?.id).toBe('r0');
    });

    it('rebuilds the graph once tombstones pass the threshold', async () => {
        const store = new MemoryVectorStore({ annThreshold: 4, rebuildTombstoneRatio: 0.1 });
        const records = Array.from({ length: 10 }, (_, i) =>
            record(`r${i}`, `doc ${i}`, { tenantId: 't', acl: ['*'] }, vec(1, i / 10)));
        await store.upsert(records);
        await store.query({ vector: vec(1, 0), topK: 1 });

        await store.delete(['r0', 'r1', 'r2']);
        const hits = await store.query({ vector: vec(1, 0), topK: 5 });
        expect(hits.map((h) => h.id)).not.toContain('r0');
        expect(store.size).toBe(7);
    });
});

describe('dialects', () => {
    it('passes the filter through unchanged on the evermind contract', () => {
        const filter: MetadataFilter = { op: 'eq', field: 'tenantId', value: 'acme' };
        expect(evermindDialect.translateFilter(filter)).toEqual({ pushed: filter });
    });

    it('translates ACL intersection for qdrant and keeps unsupported clauses residual', () => {
        const translated = qdrantDialect.translateFilter({
            op: 'and',
            filters: [
                { op: 'eq', field: 'tenantId', value: 'acme' },
                { op: 'anyOf', field: 'acl', values: ['sec-team', '*'] },
                { op: 'lte', field: 'sensitivity', value: 3 },
                { op: 'ne', field: 'archived', value: true },
            ],
        });

        expect(translated.pushed).toEqual({
            must: [
                { key: 'tenantId', match: { value: 'acme' } },
                { key: 'acl', match: { any: ['sec-team', '*'] } },
                { key: 'sensitivity', range: { lte: 3 } },
            ],
        });
        expect(translated.residual).toEqual({ op: 'ne', field: 'archived', value: true });
    });

    it('never splits an OR across the wire', () => {
        const or: MetadataFilter = {
            op: 'or',
            filters: [
                { op: 'eq', field: 'a', value: 1 },
                { op: 'eq', field: 'b', value: 2 },
            ],
        };
        const translated = qdrantDialect.translateFilter(or);
        expect(translated.pushed).toBeUndefined();
        expect(translated.residual).toEqual(or);
    });

    it('builds pinecone bodies and reads its response shape', () => {
        const translated = pineconeDialect.translateFilter({
            op: 'and',
            filters: [
                { op: 'eq', field: 'tenantId', value: 'acme' },
                { op: 'anyOf', field: 'acl', values: ['*'] },
            ],
        });
        expect(translated.pushed).toEqual({
            $and: [{ tenantId: { $eq: 'acme' } }, { acl: { $in: ['*'] } }],
        });

        const req = pineconeDialect.upsert(
            { collection: 'ns', options: {} },
            [record('a', 'text', { tenantId: 'acme' }, vec(1, 2))],
        );
        expect(req.path).toBe('/vectors/upsert');
        expect((req.body as any).vectors[0].metadata.__text).toBe('text');

        const matches = pineconeDialect.parseMatches({
            matches: [{ id: 'a', score: 0.7, metadata: { __text: 'text', tenantId: 'acme' }, values: [1, 2] }],
        });
        expect(matches[0]).toMatchObject({ id: 'a', text: 'text', score: 0.7 });
        expect(matches[0]?.metadata['__text']).toBeUndefined();

        const records = pineconeDialect.parseRecords({ vectors: { a: { metadata: { __text: 't' }, values: [1] } } });
        expect(records[0]?.text).toBe('t');
    });

    it('maps the filter algebra onto Vertex restricts and converts distance to similarity', () => {
        const translated = vertexAiDialect.translateFilter({
            op: 'and',
            filters: [
                { op: 'eq', field: 'tenantId', value: 'acme' },
                { op: 'anyOf', field: 'acl', values: ['sec-team', '*'] },
                { op: 'lte', field: 'sensitivity', value: 3 },
                { op: 'contains', field: 'title', value: 'x' },
            ],
        });
        const pushed = translated.pushed as { restricts: unknown[]; numericRestricts: unknown[] };
        expect(pushed.restricts).toEqual([
            { namespace: 'tenantId', allowList: ['acme'] },
            { namespace: 'acl', allowList: ['sec-team', '*'] },
        ]);
        expect(pushed.numericRestricts).toEqual([
            { namespace: 'sensitivity', op: 'LESS_EQUAL', valueDouble: 3 },
        ]);
        expect(translated.residual).toEqual({ op: 'contains', field: 'title', value: 'x' });

        const matches = vertexAiDialect.parseMatches({
            nearestNeighbors: [{
                neighbors: [{
                    distance: 0.25,
                    datapoint: {
                        datapointId: 'a',
                        restricts: [{ namespace: 'acl', allowList: ['*', 'ops'] }],
                        numericRestricts: [{ namespace: 'sensitivity', valueInt: 2 }],
                    },
                }],
            }],
        });
        expect(matches[0]?.score).toBeCloseTo(0.75, 6);
        expect(matches[0]?.metadata['acl']).toEqual(['*', 'ops']);
        expect(matches[0]?.metadata['sensitivity']).toBe(2);
        expect(matches[0]?.text).toBe('');   // Vertex stores no text

        expect(vertexAiDialect.parseMatches({})).toEqual([]);
    });

    it('exposes a registry that a customer can extend', () => {
        expect(listDialects()).toEqual(expect.arrayContaining(['evermind', 'qdrant', 'pinecone', 'vertex-ai']));
        expect(() => getDialect('nope')).toThrow(/Unknown vector dialect/);

        const custom = { ...evermindDialect, id: 'house-db' } as VectorDialect;
        registerDialect(custom);
        expect(getDialect('house-db').id).toBe('house-db');
    });
});

describe('RestVectorStore', () => {
    function stubFetch(responses: unknown[] | ((url: string, init: RequestInit) => unknown)) {
        const calls: Array<{ url: string; body: unknown }> = [];
        let i = 0;
        const impl = (async (url: string, init: RequestInit) => {
            calls.push({ url, body: init.body ? JSON.parse(init.body as string) : undefined });
            const payload = typeof responses === 'function'
                ? responses(url, init)
                : responses[Math.min(i++, responses.length - 1)];
            return new Response(JSON.stringify(payload), { status: 200 });
        }) as unknown as typeof fetch;
        return { impl, calls };
    }

    it('pushes the whole filter down on the evermind dialect', async () => {
        const { impl, calls } = stubFetch([{ matches: [{ id: 'a', text: 'hi', score: 0.9, metadata: {} }] }]);
        const store = new RestVectorStore({
            dialect: 'evermind', baseUrl: 'https://gw.example.com/', collection: 'kb',
            headers: { 'x-api-key': 'k' }, fetchImpl: impl,
        });

        const hits = await store.query({ vector: vec(1, 0), topK: 2, scope: { tenantId: 'acme' } });
        expect(hits).toHaveLength(1);
        expect(calls[0]?.url).toBe('https://gw.example.com/collections/kb/query');
        expect((calls[0]?.body as any).topK).toBe(2);
        expect((calls[0]?.body as any).filter.op).toBe('and');
    });

    it('over-fetches and applies the residual locally so a clause is never dropped', async () => {
        const { impl, calls } = stubFetch([{
            result: [
                { id: 'keep', score: 0.9, payload: { __text: 'keep', tenantId: 'acme', archived: false } },
                { id: 'drop', score: 0.8, payload: { __text: 'drop', tenantId: 'acme', archived: true } },
            ],
        }]);
        const store = new RestVectorStore({
            dialect: 'qdrant', baseUrl: 'https://q', collection: 'kb',
            fetchImpl: impl, residualOverfetch: 3,
        });

        const hits = await store.query({
            vector: vec(1, 0), topK: 2,
            filter: { op: 'ne', field: 'archived', value: true },
            scope: { tenantId: 'acme' },
        });

        expect(hits.map((h) => h.id)).toEqual(['keep']);
        expect((calls[0]?.body as any).limit).toBe(6);   // topK * residualOverfetch
    });

    it('hydrates text for a store that holds vectors only', async () => {
        const { impl } = stubFetch([{
            nearestNeighbors: [{ neighbors: [{ distance: 0.1, datapoint: { datapointId: 'chunk-1' } }] }],
        }]);
        const store = new RestVectorStore({
            dialect: 'vertex-ai', baseUrl: 'https://vertex', collection: 'idx',
            options: { indexEndpoint: 'projects/p/locations/l/indexEndpoints/1', deployedIndexId: 'd' },
            fetchImpl: impl,
            textResolver: async (ids) => new Map(ids.map((id) => [id, `text for ${id}`])),
        });

        const hits = await store.query({ vector: vec(1, 0), topK: 1 });
        expect(hits[0]?.text).toBe('text for chunk-1');
    });

    it('batches upserts and reports the total', async () => {
        const { impl, calls } = stubFetch([{ upserted: 2, skipped: 0 }]);
        const store = new RestVectorStore({
            dialect: 'evermind', baseUrl: 'https://gw', collection: 'kb', batchSize: 2, fetchImpl: impl,
        });

        const records = Array.from({ length: 5 }, (_, i) => record(`r${i}`, `t${i}`, { tenantId: 'a' }, vec(i)));
        const result = await store.upsert(records);

        expect(calls).toHaveLength(3);          // 2 + 2 + 1
        expect(result.upserted).toBe(6);        // the stub reports 2 per batch
    });

    it('emulates count when a dialect cannot express it', async () => {
        const { impl, calls } = stubFetch([{ matches: [{ id: 'a', text: '', score: 1, metadata: {} }] }]);
        const store = new RestVectorStore({
            dialect: 'pinecone', baseUrl: 'https://p', collection: 'ns', fetchImpl: impl,
        });
        expect(await store.count()).toBe(1);
        expect(calls[0]?.url).toContain('/query');
    });

    it('emulates a filtered delete rather than deleting a superset', async () => {
        const { impl, calls } = stubFetch((url) =>
            url.includes('/points/search')
                ? { result: [{ id: 'a', score: 1, payload: { __text: 't', archived: true } }] }
                : {});
        const store = new RestVectorStore({ dialect: 'qdrant', baseUrl: 'https://q', collection: 'kb', fetchImpl: impl });

        // `ne` is residual for qdrant, so the bulk path must NOT be used.
        const deleted = await store.deleteByFilter({ op: 'ne', field: 'archived', value: false });
        expect(deleted).toBe(1);
        expect(calls.map((c) => c.url).some((u) => u.includes('/points/search'))).toBe(true);
    });

    it('uses the vendor bulk delete when the whole filter pushes down', async () => {
        const { impl, calls } = stubFetch([{}]);
        const store = new RestVectorStore({ dialect: 'qdrant', baseUrl: 'https://q', collection: 'kb', fetchImpl: impl });

        const deleted = await store.deleteByFilter({ op: 'eq', field: 'sourceId', value: 's1' });
        expect(deleted).toBe(-1);
        expect(calls).toHaveLength(1);
        expect(calls[0]?.url).toContain('/points/delete');
    });

    it('retries a 429 with backoff and fails fast on a 400', async () => {
        let attempt = 0;
        const impl = (async () => {
            attempt += 1;
            return attempt < 3
                ? new Response('slow down', { status: 429 })
                : new Response(JSON.stringify({ matches: [] }), { status: 200 });
        }) as unknown as typeof fetch;

        const sleep = jest.fn(async () => {});
        const store = new RestVectorStore({
            dialect: 'evermind', baseUrl: 'https://gw', collection: 'kb',
            fetchImpl: impl, sleep: sleep as unknown as (ms: number) => Promise<void>,
        });
        await store.query({ vector: vec(1) });
        expect(attempt).toBe(3);
        expect(sleep).toHaveBeenCalledTimes(2);

        const badRequest = (async () => new Response('bad filter', { status: 400 })) as unknown as typeof fetch;
        const strict = new RestVectorStore({
            dialect: 'evermind', baseUrl: 'https://gw', collection: 'kb', fetchImpl: badRequest,
        });
        await expect(strict.query({ vector: vec(1) })).rejects.toThrow(/400/);
    });

    it('calls the authorize hook on every request', async () => {
        const authorize = jest.fn(async () => ({ Authorization: 'Bearer fresh' }));
        const { impl, calls } = stubFetch([{ records: [] }]);
        const store = new RestVectorStore({
            dialect: 'evermind', baseUrl: 'https://gw', collection: 'kb',
            fetchImpl: impl, authorize: authorize as unknown as () => Promise<Record<string, string>>,
        });

        await store.fetch(['a']);
        expect(authorize).toHaveBeenCalledTimes(1);
        expect(calls).toHaveLength(1);
        expect(await store.fetch([])).toEqual([]);
        expect(await store.delete([])).toBe(0);
    });

    it('returns nothing from keywordSearch when the dialect has no lexical index', async () => {
        const { impl } = stubFetch([{}]);
        const store = new RestVectorStore({ dialect: 'qdrant', baseUrl: 'https://q', collection: 'kb', fetchImpl: impl });
        expect(await store.keywordSearch({ text: 'anything' })).toEqual([]);
    });

    it('runs keyword search on a dialect that has one', async () => {
        const { impl, calls } = stubFetch([{ matches: [{ id: 'a', text: 'lexical', score: 0.4, metadata: {} }] }]);
        const store = new RestVectorStore({ dialect: 'evermind', baseUrl: 'https://gw', collection: 'kb', fetchImpl: impl });

        const hits = await store.keywordSearch({ text: 'ERR-4021', topK: 3 });
        expect(hits[0]?.text).toBe('lexical');
        expect(calls[0]?.url).toContain('/keyword');
        expect(await store.keywordSearch({ text: '  ' })).toEqual([]);
    });
});
