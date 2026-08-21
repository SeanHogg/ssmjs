import { describe, expect, it, jest } from '@jest/globals';

import {
    IngestionPipeline,
    InMemoryIngestManifest,
    hashText,
    mapWithConcurrency,
} from '../src/ingest/IngestionPipeline.js';
import {
    ParserRegistry,
    csvParser,
    htmlParser,
    jsonParser,
    markdownParser,
    parseCsv,
    plainTextParser,
    rowsParser,
    serializeRow,
} from '../src/ingest/parsers.js';
import { CONTENT_HASH_FIELD, SOURCE_ID_FIELD, type SourceDocument } from '../src/ingest/types.js';
import { MemoryVectorStore } from '../src/vectorstore/MemoryVectorStore.js';
import { ACL_FIELD, TENANT_FIELD } from '../src/vectorstore/types.js';
import { Tracer } from '../src/telemetry/Tracer.js';

/** A deterministic embedder: length + first-char code. Enough to be a vector. */
const embedder = async (texts: string[]): Promise<Float32Array[]> =>
    texts.map((t) => Float32Array.from([t.length, t.charCodeAt(0) || 0]));

function textSource(id: string, text: string, over: Partial<SourceDocument> = {}): SourceDocument {
    return { id, tenantId: 'acme', content: { kind: 'text', text }, ...over };
}

describe('parsers', () => {
    it('serialises a row with its field names so the embedding has something to match', () => {
        expect(serializeRow({ status: 'open', priority: 1, tags: ['billing', 'p1'], empty: '' }))
            .toBe('status: open\npriority: 1\ntags: billing, p1');
    });

    it('turns rows into sections that keep every typed column as metadata', () => {
        const sections = rowsParser.parse({
            id: 't', tenantId: 'acme',
            content: { kind: 'rows', rows: [{ id: 'T-1', status: 'open', priority: 1 }], rowIdField: 'id' },
        });
        expect(sections[0]?.text).toContain('status: open');
        expect(sections[0]?.metadata).toMatchObject({ status: 'open', priority: 1, rowId: 'T-1', rowIndex: 0 });
    });

    it('parses RFC4180 CSV including quotes, embedded commas and newlines', () => {
        const rows = parseCsv('id,note,open\n1,"hello, world",true\n2,"line\none",false\n');
        expect(rows).toEqual([
            { id: 1, note: 'hello, world', open: true },
            { id: 2, note: 'line\none', open: false },
        ]);
    });

    it('coerces numbers and booleans but preserves identifier-shaped strings', () => {
        const rows = parseCsv('account,amount,flag\n007,12.5,true\n');
        expect(rows[0]).toEqual({ account: '007', amount: 12.5, flag: true });
    });

    it('handles doubled quotes and an empty file', () => {
        expect(parseCsv('a\n"say ""hi"""\n')[0]).toEqual({ a: 'say "hi"' });
        expect(parseCsv('')).toEqual([]);
        expect(parseCsv('a,b\n\n')).toEqual([]);
    });

    it('routes tab-separated content through the csv parser', () => {
        const sections = csvParser.parse(
            textSource('t', 'a\tb\n1\t2\n', { content: { kind: 'text', text: 'a\tb\n1\t2\n', mediaType: 'text/tab-separated-values' } }),
        );
        expect(sections[0]?.metadata).toMatchObject({ a: 1, b: 2 });
    });

    it('treats a JSON array of objects as rows, and other JSON as leaf paths', () => {
        const asRows = jsonParser.parse(textSource('j', '[{"a":1},{"a":2}]', {
            content: { kind: 'text', text: '[{"a":1},{"a":2}]', mediaType: 'application/json' },
        }));
        expect(asRows).toHaveLength(2);

        const nested = jsonParser.parse(textSource('j', '{"a":{"b":[1,2]}}', {
            content: { kind: 'text', text: '{"a":{"b":[1,2]}}', mediaType: 'application/json' },
        }));
        expect(nested[0]?.text).toContain('/a/b/0: 1');
        expect(nested[0]?.metadata).toMatchObject({ leafCount: 2 });
    });

    it('degrades malformed JSON to plain text instead of dropping the document', () => {
        const sections = jsonParser.parse(textSource('j', 'not json at all', {
            content: { kind: 'text', text: 'not json at all', mediaType: 'application/json' },
        }));
        expect(sections[0]?.text).toBe('not json at all');
    });

    it('keeps the full markdown heading breadcrumb and ignores headings inside fences', () => {
        const md = [
            '# Security',
            'intro',
            '## Data retention',
            'we keep records for 7 years',
            '```',
            '# not a heading',
            '```',
            '# Billing',
            'invoices monthly',
        ].join('\n');

        const sections = markdownParser.parse(textSource('m', md, {
            content: { kind: 'text', text: md, mediaType: 'text/markdown' },
        }));

        const retention = sections.find((s) => s.text.includes('7 years'));
        expect(retention?.metadata?.['headingPath']).toBe('Security › Data retention');
        expect(retention?.metadata?.['headingDepth']).toBe(2);
        expect(retention?.text).toContain('# not a heading');

        const billing = sections.find((s) => s.text.includes('invoices'));
        expect(billing?.metadata?.['headingPath']).toBe('Billing');
    });

    it('strips scripts, styles and entities from HTML and keeps the title', () => {
        const html = '<html><head><title>Policy</title></head><body>' +
            '<script>evil()</script><style>.a{}</style>' +
            '<p>Retention &amp; deletion</p><p>Second&nbsp;line</p></body></html>';

        const sections = htmlParser.parse(textSource('h', html, {
            content: { kind: 'text', text: html, mediaType: 'text/html' },
        }));

        expect(sections[0]?.text).toContain('Retention & deletion');
        expect(sections[0]?.text).not.toContain('evil()');
        expect(sections[0]?.metadata?.['htmlTitle']).toBe('Policy');
        expect(htmlParser.parse(textSource('h', '<p></p>', {
            content: { kind: 'text', text: '<p></p>', mediaType: 'text/html' },
        }))).toEqual([]);
    });

    it('returns nothing for empty plain text', () => {
        expect(plainTextParser.parse(textSource('t', '   '))).toEqual([]);
    });

    it('resolves parsers in registry order and lets a custom one override a builtin', () => {
        const registry = new ParserRegistry();
        expect(registry.resolve(textSource('t', 'plain'))?.id).toBe('text');
        expect(registry.list()).toContain('markdown');

        registry.register({ id: 'custom', accepts: () => true, parse: () => [{ text: 'overridden' }] });
        expect(registry.resolve(textSource('t', 'plain'))?.id).toBe('custom');
    });
});

describe('hashText and mapWithConcurrency', () => {
    it('hashes deterministically and differently for different content', () => {
        expect(hashText('abc')).toBe(hashText('abc'));
        expect(hashText('abc')).not.toBe(hashText('abd'));
        expect(hashText('')).toHaveLength(8);
    });

    it('never exceeds the concurrency limit', async () => {
        let active = 0;
        let peak = 0;
        await mapWithConcurrency(Array.from({ length: 20 }, (_, i) => i), 3, async () => {
            active += 1;
            peak = Math.max(peak, active);
            await Promise.resolve();
            active -= 1;
        });
        expect(peak).toBeLessThanOrEqual(3);
    });

    it('handles an empty input list', async () => {
        const fn = jest.fn(async () => {});
        await mapWithConcurrency([], 4, fn as unknown as (item: never, i: number) => Promise<void>);
        expect(fn).not.toHaveBeenCalled();
    });
});

describe('IngestionPipeline', () => {
    it('writes tenant, a non-empty ACL, source id and content hash onto every chunk', async () => {
        const store = new MemoryVectorStore();
        const pipeline = new IngestionPipeline({ store, embed: embedder });

        const report = await pipeline.ingest([
            textSource('doc-1', 'the retention policy is seven years', { title: 'Policy', uri: 'https://x/p' }),
        ]);

        expect(report.documents).toBe(1);
        expect(report.chunks).toBe(1);
        expect(report.embedded).toBe(1);

        const [record] = await store.fetch(['doc-1#0']);
        expect(record?.metadata[TENANT_FIELD]).toBe('acme');
        expect(record?.metadata[ACL_FIELD]).toEqual(['*']);   // fails closed by default
        expect(record?.metadata[SOURCE_ID_FIELD]).toBe('doc-1');
        expect(record?.metadata[CONTENT_HASH_FIELD]).toBeDefined();
        expect(record?.metadata['title']).toBe('Policy');
        expect(record?.vector).toBeInstanceOf(Float32Array);
    });

    it('carries an explicit ACL and sensitivity through to the chunks', async () => {
        const store = new MemoryVectorStore();
        await new IngestionPipeline({ store, embed: embedder }).ingest([
            textSource('secret', 'incident detail', { acl: ['sec-team'], sensitivity: 5 }),
        ]);

        const visible = await store.query({ topK: 10, scope: { tenantId: 'acme' } });
        expect(visible).toHaveLength(0);

        const asSecurity = await store.query({
            topK: 10, scope: { tenantId: 'acme', principals: ['sec-team'] },
        });
        expect(asSecurity).toHaveLength(1);
    });

    it('ingests structured rows with their columns filterable', async () => {
        const store = new MemoryVectorStore();
        await new IngestionPipeline({ store, embed: embedder }).ingest([{
            id: 'tickets', tenantId: 'acme',
            content: {
                kind: 'rows',
                rows: [
                    { id: 'T-1', status: 'open', priority: 1, subject: 'billing double charge' },
                    { id: 'T-2', status: 'closed', priority: 3, subject: 'password reset' },
                ],
                rowIdField: 'id',
            },
        }]);

        const openP1 = await store.query({
            topK: 10,
            scope: { tenantId: 'acme' },
            filter: {
                op: 'and',
                filters: [
                    { op: 'eq', field: 'status', value: 'open' },
                    { op: 'lte', field: 'priority', value: 2 },
                ],
            },
        });
        expect(openP1).toHaveLength(1);
        expect(openP1[0]?.metadata['rowId']).toBe('T-1');
    });

    it('re-embeds only the changed chunk on a diff sync', async () => {
        const store = new MemoryVectorStore();
        const manifest = new InMemoryIngestManifest();
        const embed = jest.fn(embedder) as unknown as typeof embedder;
        const pipeline = new IngestionPipeline({
            store, embed, manifest, chunk: { chunkSize: 40, chunkOverlap: 0 },
        });

        const first = 'alpha section text here.\n\nbeta section text here.';
        const firstReport = await pipeline.ingest([textSource('doc', first)]);
        expect(firstReport.embedded).toBe(firstReport.chunks);
        const embeddedFirst = firstReport.embedded;

        // Same document, second section reworded: only that chunk is re-embedded.
        const second = 'alpha section text here.\n\nbeta section CHANGED text.';
        const secondReport = await pipeline.ingest([textSource('doc', second)]);

        expect(secondReport.skippedUnchanged).toBeGreaterThan(0);
        expect(secondReport.embedded).toBeLessThan(embeddedFirst);
        expect(secondReport.deletedStale).toBe(0);
    });

    it('deletes chunks that vanished from a shrinking source', async () => {
        const store = new MemoryVectorStore();
        const manifest = new InMemoryIngestManifest();
        const pipeline = new IngestionPipeline({
            store, embed: embedder, manifest, chunk: { chunkSize: 30, chunkOverlap: 0 },
        });

        await pipeline.ingest([textSource('doc', 'one paragraph here.\n\ntwo paragraph here.\n\nthree here.')]);
        const before = await store.count();

        const report = await pipeline.ingest([textSource('doc', 'one paragraph here.')]);
        expect(report.deletedStale).toBeGreaterThan(0);
        expect(await store.count()).toBeLessThan(before);
    });

    it('replaces a source wholesale when no manifest is supplied', async () => {
        const store = new MemoryVectorStore();
        const pipeline = new IngestionPipeline({ store, embed: embedder, chunk: { chunkSize: 30, chunkOverlap: 0 } });

        await pipeline.ingest([textSource('doc', 'first version here.\n\nsecond part here.')]);
        await pipeline.ingest([textSource('doc', 'only one now.')]);

        const all = await store.query({ topK: 50, scope: { tenantId: 'acme' } });
        expect(all).toHaveLength(1);
        expect(all[0]?.text).toContain('only one now');
    });

    it('appends without deleting under the append strategy', async () => {
        const store = new MemoryVectorStore();
        const pipeline = new IngestionPipeline({ store, embed: embedder, syncStrategy: 'append' });
        await pipeline.ingest([textSource('doc', 'v1')]);
        await pipeline.ingest([textSource('doc2', 'v2')]);
        expect(await store.count()).toBe(2);
    });

    it('isolates a failing document and reports it by id', async () => {
        const store = new MemoryVectorStore();
        const parsers = new ParserRegistry([{
            id: 'picky',
            accepts: (s) => s.id !== 'unsupported',
            parse: () => [{ text: 'ok' }],
        }]);
        const pipeline = new IngestionPipeline({ store, embed: embedder, parsers });

        const report = await pipeline.ingest([
            textSource('good', 'fine'),
            textSource('unsupported', 'nope'),
        ]);

        expect(report.documents).toBe(1);
        expect(report.failures).toEqual([
            { sourceId: 'unsupported', error: 'No parser accepts source "unsupported".' },
        ]);
    });

    it('surfaces an embedder failure as a per-document failure', async () => {
        const store = new MemoryVectorStore();
        const pipeline = new IngestionPipeline({
            store,
            embed: async () => { throw new Error('embedding quota exceeded'); },
        });
        const report = await pipeline.ingest([textSource('doc', 'text')]);
        expect(report.failures[0]?.error).toBe('embedding quota exceeded');
        expect(report.documents).toBe(0);
    });

    it('forgets every chunk of a source — the erasure primitive', async () => {
        const store = new MemoryVectorStore();
        const manifest = new InMemoryIngestManifest();
        const pipeline = new IngestionPipeline({ store, embed: embedder, manifest });

        await pipeline.ingest([textSource('doc', 'personal data'), textSource('keep', 'other data')]);
        const removed = await pipeline.forget('doc');

        expect(removed).toBe(1);
        expect(await store.count()).toBe(1);
        expect(await manifest.get('doc')).toBeUndefined();
    });

    it('traces the run with embedding spans that carry token usage', async () => {
        const tracer = new Tracer();
        const store = new MemoryVectorStore();
        const pipeline = new IngestionPipeline({ store, embed: embedder, tracer, embeddingModel: 'text-embedding-005' });

        const report = await pipeline.ingest([textSource('doc', 'traced content here')]);
        const spans = tracer.trace(report.traceId as string);

        expect(spans.some((s) => s.name === 'ingest.run')).toBe(true);
        expect(spans.some((s) => s.name === 'ingest.document')).toBe(true);
        const embedSpan = spans.find((s) => s.name === 'embed.batch');
        expect(embedSpan?.usage?.model).toBe('text-embedding-005');
        expect(embedSpan?.usage?.estimated).toBe(true);
    });

    it('marks the run span failed when a traced document throws', async () => {
        const tracer = new Tracer();
        const store = new MemoryVectorStore();
        const pipeline = new IngestionPipeline({
            store, embed: async () => { throw new Error('down'); }, tracer,
        });

        const report = await pipeline.ingest([textSource('doc', 'text')]);
        const spans = tracer.trace(report.traceId as string);
        expect(spans.find((s) => s.name === 'ingest.document')?.status).toBe('error');
        expect(spans.find((s) => s.name === 'ingest.run')?.status).toBe('error');
    });

    it('batches embeddings and upserts', async () => {
        const store = new MemoryVectorStore();
        const embed = jest.fn(embedder) as unknown as typeof embedder;
        const pipeline = new IngestionPipeline({
            store, embed, embedBatchSize: 2, upsertBatchSize: 2,
            chunk: { chunkSize: 20, chunkOverlap: 0 },
        });

        const text = Array.from({ length: 6 }, (_, i) => `paragraph number ${i}`).join('\n\n');
        const report = await pipeline.ingest([textSource('doc', text)]);

        expect(report.chunks).toBeGreaterThan(2);
        expect((embed as unknown as jest.Mock).mock.calls.length).toBe(Math.ceil(report.chunks / 2));
    });
});
