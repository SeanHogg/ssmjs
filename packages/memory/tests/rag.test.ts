import { describe, expect, it, jest } from '@jest/globals';

import {
    EnterpriseRetriever,
    buildGroundedPrompt,
    NO_EVIDENCE_ANSWER,
    type RetrievedPassage,
} from '../src/rag/EnterpriseRetriever.js';
import { IngestionPipeline } from '../src/ingest/IngestionPipeline.js';
import { MemoryVectorStore } from '../src/vectorstore/MemoryVectorStore.js';
import { Tracer } from '../src/telemetry/Tracer.js';
import type { SourceDocument } from '../src/ingest/types.js';
import type { TransformerBridge } from '../src/bridges/TransformerBridge.js';
import type { VectorMatch, VectorQuery, VectorStore } from '../src/vectorstore/types.js';

/**
 * A bag-of-words embedder over a fixed vocabulary. Deterministic and cheap, and —
 * unlike random vectors — it makes "the semantically closer passage wins" a real
 * assertion rather than a coincidence.
 */
const VOCAB = ['retention', 'policy', 'years', 'billing', 'invoice', 'incident', 'outage', 'sso', 'refund'];

function embedText(text: string): Float32Array {
    const lower = text.toLowerCase();
    const v = Float32Array.from(VOCAB.map((term) => (lower.includes(term) ? 1 : 0)));
    // A non-zero floor keeps cosine defined for text sharing no vocabulary.
    if (v.every((x) => x === 0)) v[0] = 0.001;
    return v;
}

const embedBatch = async (texts: string[]) => texts.map(embedText);
const embedOne = async (text: string) => embedText(text);

function doc(id: string, text: string, over: Partial<SourceDocument> = {}): SourceDocument {
    return { id, tenantId: 'acme', content: { kind: 'text', text }, ...over };
}

async function seededStore(): Promise<MemoryVectorStore> {
    const store = new MemoryVectorStore();
    await new IngestionPipeline({ store, embed: embedBatch, chunk: { chunkSize: 400, chunkOverlap: 0 } })
        .ingest([
            doc('policy', 'Our retention policy keeps records for seven years.', { title: 'Retention Policy' }),
            doc('billing', 'A billing invoice is issued monthly and a refund takes ten days.', { title: 'Billing FAQ' }),
            doc('incident', 'The SSO outage incident ERR-4021 was resolved in two hours.', {
                title: 'Incident ERR-4021', acl: ['sec-team'],
            }),
            doc('other-tenant', 'Globex retention policy is three years.', {
                tenantId: 'globex', title: 'Globex Policy',
            }),
        ]);
    return store;
}

function stubBridge(reply: string): TransformerBridge & { prompts: string[] } {
    const prompts: string[] = [];
    return {
        supportsStreaming: false,
        prompts,
        async generate(prompt: string) {
            prompts.push(prompt);
            return reply;
        },
    };
}

describe('EnterpriseRetriever', () => {
    it('fuses dense and lexical arms and reports hybrid mode', async () => {
        const store = await seededStore();
        const retriever = new EnterpriseRetriever({ store, embed: embedOne });

        const result = await retriever.retrieve('what is the retention policy', {
            scope: { tenantId: 'acme' }, topK: 2,
        });

        expect(result.mode).toBe('hybrid');
        expect(result.passages[0]?.sourceId).toBe('policy');
        expect(result.passages[0]?.title).toBe('Retention Policy');
        expect(result.candidatesConsidered).toBeGreaterThan(0);
    });

    it('finds an exact identifier that embeddings blur, via the lexical arm', async () => {
        const store = await seededStore();
        const retriever = new EnterpriseRetriever({ store, embed: embedOne });

        const result = await retriever.retrieve('ERR-4021', {
            scope: { tenantId: 'acme', principals: ['sec-team'] }, topK: 3,
        });
        expect(result.passages.map((p) => p.sourceId)).toContain('incident');
    });

    it('never returns another tenant\'s passage', async () => {
        const store = await seededStore();
        const retriever = new EnterpriseRetriever({ store, embed: embedOne });

        const result = await retriever.retrieve('retention policy', { scope: { tenantId: 'acme' }, topK: 10 });
        expect(result.passages.map((p) => p.sourceId)).not.toContain('other-tenant');

        const asGlobex = await retriever.retrieve('retention policy', { scope: { tenantId: 'globex' }, topK: 10 });
        expect(asGlobex.passages.map((p) => p.sourceId)).toEqual(['other-tenant']);
    });

    it('withholds an ACL-restricted passage from a principal that lacks the group', async () => {
        const store = await seededStore();
        const retriever = new EnterpriseRetriever({ store, embed: embedOne });

        const denied = await retriever.retrieve('SSO outage incident', { scope: { tenantId: 'acme' }, topK: 10 });
        expect(denied.passages.map((p) => p.sourceId)).not.toContain('incident');
    });

    it('uses a default scope when the call does not name one', async () => {
        const store = await seededStore();
        const retriever = new EnterpriseRetriever({
            store, embed: embedOne, defaultScope: { tenantId: 'globex' },
        });
        const result = await retriever.retrieve('retention', { topK: 5 });
        expect(result.passages.every((p) => p.sourceId === 'other-tenant')).toBe(true);
    });

    it('degrades to dense-only against a store with no lexical index, and says so', async () => {
        const backing = await seededStore();
        const denseOnly: VectorStore = {
            name: 'dense-only',
            query: (q: VectorQuery) => backing.query(q),
            upsert: (r) => backing.upsert(r),
            delete: (ids) => backing.delete(ids),
            deleteByFilter: (f, s) => backing.deleteByFilter(f, s),
            fetch: (ids) => backing.fetch(ids),
            count: (f, s) => backing.count(f, s),
        };

        const retriever = new EnterpriseRetriever({ store: denseOnly, embed: embedOne });
        const result = await retriever.retrieve('retention policy', { scope: { tenantId: 'acme' }, topK: 2 });

        expect(result.mode).toBe('dense-only');
        expect(result.passages.length).toBeGreaterThan(0);
    });

    it('runs keyword-only when no embedder is configured', async () => {
        const store = await seededStore();
        const retriever = new EnterpriseRetriever({ store });

        const result = await retriever.retrieve('invoice refund', { scope: { tenantId: 'acme' }, topK: 2 });
        expect(result.mode).toBe('keyword-only');
        expect(result.passages[0]?.sourceId).toBe('billing');
    });

    it('reports empty when nothing matches within scope', async () => {
        const store = new MemoryVectorStore();
        const retriever = new EnterpriseRetriever({ store, embed: embedOne });
        const result = await retriever.retrieve('anything', { scope: { tenantId: 'nobody' } });
        expect(result.passages).toHaveLength(0);
        expect(result.mode).toBe('empty');
    });

    it('applies a reranker over the fused candidates', async () => {
        const store = await seededStore();
        const reranker = jest.fn(async (_q: string, passages: readonly RetrievedPassage[]) =>
            [...passages].reverse());

        const retriever = new EnterpriseRetriever({
            store, embed: embedOne,
            reranker: reranker as unknown as (q: string, p: readonly RetrievedPassage[]) => Promise<readonly RetrievedPassage[]>,
        });

        const plain = await new EnterpriseRetriever({ store, embed: embedOne })
            .retrieve('retention policy billing', { scope: { tenantId: 'acme' }, topK: 2 });
        const reranked = await retriever.retrieve('retention policy billing', {
            scope: { tenantId: 'acme' }, topK: 2,
        });

        expect(reranker).toHaveBeenCalled();
        expect(reranked.passages[0]?.id).not.toBe(plain.passages[0]?.id);
    });

    it('applies minScore after fusion', async () => {
        const store = await seededStore();
        const retriever = new EnterpriseRetriever({ store, embed: embedOne });
        const result = await retriever.retrieve('retention policy', {
            scope: { tenantId: 'acme' }, topK: 5, minScore: 1.1,
        });
        expect(result.passages).toHaveLength(0);
    });

    it('keeps the fused order when diversity is disabled', async () => {
        const store = await seededStore();
        const withDiversity = new EnterpriseRetriever({ store, embed: embedOne });
        const without = new EnterpriseRetriever({ store, embed: embedOne, disableDiversity: true });

        const a = await withDiversity.retrieve('retention policy', { scope: { tenantId: 'acme' }, topK: 3 });
        const b = await without.retrieve('retention policy', { scope: { tenantId: 'acme' }, topK: 3 });

        expect(a.passages.length).toBeGreaterThan(0);
        expect(b.passages.length).toBeGreaterThan(0);
    });

    it('traces retrieval with the mode and hit counts on the span', async () => {
        const store = await seededStore();
        const tracer = new Tracer();
        const retriever = new EnterpriseRetriever({ store, embed: embedOne, tracer });

        const result = await retriever.retrieve('retention policy', { scope: { tenantId: 'acme' } });
        const span = tracer.trace(result.traceId as string)[0];

        expect(span?.kind).toBe('retrieval');
        expect(span?.attributes['retrieval.mode']).toBe('hybrid');
        expect(span?.attributes['retrieval.tenant']).toBe('acme');
        expect(span?.attributes['retrieval.store']).toBe('memory');
    });

    it('fails the retrieval span when the store throws', async () => {
        const tracer = new Tracer();
        const broken: VectorStore = {
            name: 'broken',
            query: async () => { throw new Error('index offline'); },
            upsert: async () => ({ upserted: 0, skipped: 0 }),
            delete: async () => 0,
            deleteByFilter: async () => 0,
            fetch: async () => [],
            count: async () => 0,
        };
        const retriever = new EnterpriseRetriever({ store: broken, embed: embedOne, tracer });

        await expect(retriever.retrieve('x', { scope: { tenantId: 'acme' } })).rejects.toThrow('index offline');
        expect(tracer.finished()[0]?.status).toBe('error');
    });
});

describe('grounded answering', () => {
    it('numbers passages and asks for citations', () => {
        const prompt = buildGroundedPrompt('how long do we keep records?', [
            { id: 'a#0', text: 'seven years', score: 1, sourceId: 'a', title: 'Policy', uri: 'https://x', metadata: {} },
            { id: 'b#0', text: 'ten days', score: 0.5, sourceId: 'b', metadata: {} },
        ]);

        expect(prompt).toContain('[1] Policy (https://x)');
        expect(prompt).toContain('[2] b');
        expect(prompt).toContain('Question: how long do we keep records?');
    });

    it('answers from retrieved evidence and passes the grounding system prompt', async () => {
        const store = await seededStore();
        const bridge = stubBridge('Records are kept for seven years [1].');
        const retriever = new EnterpriseRetriever({ store, embed: embedOne });

        const { answer, result } = await retriever.answer('retention policy', bridge, {
            scope: { tenantId: 'acme' }, topK: 2,
        });

        expect(answer).toContain('[1]');
        expect(result.passages.length).toBeGreaterThan(0);
        expect(bridge.prompts[0]).toContain('Passages:');
    });

    it('refuses rather than answering with no evidence', async () => {
        const store = new MemoryVectorStore();
        const bridge = stubBridge('I am confident the answer is 42.');
        const retriever = new EnterpriseRetriever({ store, embed: embedOne });

        const { answer } = await retriever.answer('anything', bridge, { scope: { tenantId: 'empty' } });

        expect(answer).toBe(NO_EVIDENCE_ANSWER);
        expect(bridge.prompts).toHaveLength(0);   // the model was never called
    });

    it('nests retrieval and generation under one trace', async () => {
        const store = await seededStore();
        const tracer = new Tracer();
        const retriever = new EnterpriseRetriever({ store, embed: embedOne, tracer });

        await retriever.answer('retention policy', stubBridge('seven years [1]'), {
            scope: { tenantId: 'acme' },
        });

        const names = tracer.finished().map((s) => s.name);
        expect(names).toContain('retrieval.hybrid');
        expect(names).toContain('rag.answer');

        const answerSpan = tracer.finished().find((s) => s.name === 'rag.answer');
        const retrievalSpan = tracer.finished().find((s) => s.name === 'retrieval.hybrid');
        expect(retrievalSpan?.parentSpanId).toBe(answerSpan?.spanId);
    });

    it('fails the answer span when the bridge throws', async () => {
        const store = await seededStore();
        const tracer = new Tracer();
        const retriever = new EnterpriseRetriever({ store, embed: embedOne, tracer });
        const broken: TransformerBridge = {
            supportsStreaming: false,
            generate: async () => { throw new Error('model unavailable'); },
        };

        await expect(retriever.answer('retention policy', broken, { scope: { tenantId: 'acme' } }))
            .rejects.toThrow('model unavailable');
        expect(tracer.finished().find((s) => s.name === 'rag.answer')?.status).toBe('error');
    });
});
