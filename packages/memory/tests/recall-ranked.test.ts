/**
 * tests/recall-ranked.test.ts — MemoryStore.recallRanked: the ONE hybrid recall
 * path, its fused scores, and the honest name of the ranker that produced them.
 *
 * Two things are pinned here that nothing else pins:
 *
 *  1. `method` is EARNED. It says `'embedding'` only when the query really did
 *     embed and a dense ranking really did take part in the fusion — an ordering
 *     that quietly fell back to word overlap must not claim to be semantic.
 *  2. The blend is the SHARED primitive (`hybridRetrieve` → `reciprocalRankFusion`
 *     from src/retrieval), not a second hand-rolled mix of cosine and overlap.
 *
 * The semantic case is driven by a REAL, pure-CPU `EvermindLM` — no GPU, no
 * WebGPU device — because that is the whole point: embedding-ranked recall was
 * assumed to need a GPU runtime, and it does not.
 */

import 'fake-indexeddb/auto';
import { jest } from '@jest/globals';
import { EvermindLM, EvermindLMTrainer, EvermindTextEmbedder, BPETokenizer } from '@seanhogg/builderforce-memory-engine';
import { MemoryStore } from '../src/memory/MemoryStore.js';
import { hybridRetrieve, type RetrievalCandidate } from '../src/retrieval/index.js';

let _db = 0;
const freshStore = (opts: Record<string, unknown> = {}) =>
    new MemoryStore({ dbName: `ranked-${_db++}`, ...opts });

// ── method reporting ──────────────────────────────────────────────────────────

test('an empty store recalls nothing, lexically', async () => {
    expect(await freshStore().recallRanked('q', 3)).toEqual({ hits: [], method: 'lexical' });
});

test('no runtime means the ordering is lexical, and says so', async () => {
    const store = freshStore();
    await store.remember('k1', 'the quick brown fox');
    await store.remember('k2', 'completely unrelated content');

    const ranked = await store.recallRanked('quick brown fox', 2);
    expect(ranked.method).toBe('lexical');
    expect(ranked.hits[0]!.entry.key).toBe('k1');
    expect(ranked.hits[0]!.score).toBeGreaterThan(0);
});

test('an embedding runtime makes the ordering semantic, and says so', async () => {
    const store = freshStore();
    await store.remember('match', 'aligned quick fox');
    await store.remember('miss', 'orthogonal turtle');

    const vectors: Record<string, Float32Array> = {
        'query quick fox': new Float32Array([1, 0]),
        'aligned quick fox': new Float32Array([1, 0]),
        'orthogonal turtle': new Float32Array([0, 1]),
    };
    const runtime = { embed: jest.fn<any>(async (t: string) => vectors[t] ?? new Float32Array([0.1, 0.1])) };

    const ranked = await store.recallRanked('query quick fox', 2, runtime);
    expect(ranked.method).toBe('embedding');
    expect(ranked.hits[0]!.entry.key).toBe('match');
});

test('an embed() that yields no vector is reported as lexical, not as embedding', async () => {
    const store = freshStore();
    await store.remember('k1', 'alpha beta gamma');
    await store.remember('k2', 'delta epsilon');

    const runtime = { embed: jest.fn<any>(async () => new Float32Array(0)) };
    const ranked = await store.recallRanked('alpha beta', 2, runtime);
    expect(ranked.method).toBe('lexical');
    expect(ranked.hits[0]!.entry.key).toBe('k1');
    // A failed query embedding must not cost N pointless candidate embeddings.
    expect(runtime.embed).toHaveBeenCalledTimes(1);
});

test('recallHybrid is recallRanked with the evidence dropped (one implementation)', async () => {
    const store = freshStore();
    await store.remember('k1', 'the quick brown fox');
    await store.remember('k2', 'a slow green turtle');

    const ranked = await store.recallRanked('quick fox', 2);
    const plain = await store.recallHybrid('quick fox', 2);
    expect(plain.map((e) => e.key)).toEqual(ranked.hits.map((h) => h.entry.key));
});

// ── the blend is the shared fusion primitive ──────────────────────────────────

test('the ordering IS hybridRetrieve/RRF over the same candidates, not a second blend', async () => {
    const store = freshStore();
    // Dense and sparse deliberately DISAGREE, so any other blend would reorder.
    await store.remember('dense-favourite', 'orthogonal turtle');
    await store.remember('sparse-favourite', 'quick fox lexical');
    await store.remember('filler', 'nothing in common at all');

    const vectors: Record<string, Float32Array> = {
        'quick fox query': new Float32Array([1, 0]),
        'orthogonal turtle': new Float32Array([1, 0]),
        'quick fox lexical': new Float32Array([0, 1]),
        'nothing in common at all': new Float32Array([0.3, 0.3]),
    };
    const embed = async (t: string) => vectors[t] ?? new Float32Array([0, 0]);
    const runtime = { embed: jest.fn<any>(embed) };

    const ranked = await store.recallRanked('quick fox query', 3, runtime);

    // The same inputs pushed straight through the shared primitive.
    const all = await store.recallAll();
    const candidates: RetrievalCandidate[] = [];
    for (const e of all) candidates.push({ id: e.key, text: e.content, vector: await embed(e.content) });
    const expected = hybridRetrieve(
        { text: 'quick fox query', vector: await embed('quick fox query') },
        candidates,
        { topK: 3 },
    );

    expect(ranked.hits.map((h) => h.entry.key)).toEqual(expected.map((h) => h.id));
    expect(ranked.hits.map((h) => h.score)).toEqual(expected.map((h) => h.score));
});

// ── the real CPU SSM embedding, end to end ────────────────────────────────────

/**
 * A corpus in which {puppy, canine}, {chased, pursued} and {toy, ball} are used
 * interchangeably, so the model learns them as substitutable — and separately a
 * finance vocabulary that shares nothing with either.
 */
const CORPUS = [
    'puppy chased toy.', 'canine pursued ball.', 'puppy pursued ball.', 'canine chased toy.',
    'puppy chased ball.', 'canine pursued toy.', 'puppy pursued toy.', 'canine chased ball.',
    'invoices reconciled ledger.', 'quarterly ledger balances.', 'invoices filed quarterly.',
    'ledger reconciled quarterly.', 'quarterly invoices balances.', 'ledger filed invoices.',
].join(' ');

/** A trained CPU EvermindLM wrapped as the text embedder the store consumes. */
function cpuEmbedder(): EvermindTextEmbedder {
    const tok = new BPETokenizer();
    tok.train(CORPUS, { numMerges: 60 });
    const model = new EvermindLM({ vocabSize: tok.vocabSize, dModel: 16, numLayers: 2, hiddenDim: 24, seed: 7 });
    const seqs = CORPUS.split(/(?<=\.)\s+/).map((s) => tok.encode(s.trim())).filter((ids) => ids.length >= 2);
    new EvermindLMTrainer(model, { lr: 0.05, epochs: 40 }).fit(seqs);
    return new EvermindTextEmbedder(model, tok);
}

describe('CPU SSM embedding recall (no GPU)', () => {
    const QUERY = 'puppy chased toy';
    const RELATED = 'canine pursued ball';   // shares NO token with the query
    const UNRELATED = 'invoices reconciled ledger';

    // Trained ONCE: the point is what the embedding ranks, not how many times a
    // 40-epoch CPU fit can be repeated inside a test file.
    let embedder: EvermindTextEmbedder;
    beforeAll(() => { embedder = cpuEmbedder(); }, 180_000);

    async function seeded() {
        const store = freshStore();
        await store.remember('related', RELATED);
        await store.remember('unrelated', UNRELATED);
        return store;
    }

    test('a lexically-disjoint but related memory outranks an unrelated one', async () => {
        const ranked = await (await seeded()).recallRanked(QUERY, 2, embedder);
        expect(ranked.method).toBe('embedding');
        expect(ranked.hits[0]!.entry.key).toBe('related');
    }, 60_000);

    test('word overlap cannot see it at all — which is the gap being closed', async () => {
        const lexical = await (await seeded()).recallRanked(QUERY, 2);
        expect(lexical.method).toBe('lexical');
        // No query term occurs in either memory, so BM25 has nothing to rank.
        expect(lexical.hits.map((h) => h.entry.key)).toEqual([]);
    });

    test('the embedding is deterministic, so a cached vector stays valid', async () => {
        const a = await embedder.embed(RELATED);
        const b = await embedder.embed(RELATED);
        expect(Array.from(b)).toEqual(Array.from(a));
        // ...and a REBUILT model from the same seed agrees, so a vector cached by an
        // earlier process is still the vector this one would have computed.
        expect(Array.from(await cpuEmbedder().embed(RELATED))).toEqual(Array.from(a));
    }, 180_000);
});
