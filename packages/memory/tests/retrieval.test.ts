/**
 * tests/retrieval.test.ts
 * Chunking, BM25, rank fusion, and the HybridRetriever — the RAG retrieval layer.
 */

import { chunkText } from '../src/retrieval/chunk.js';
import { bm25Search, bm25Idf, bm25LengthNorm, bm25TermScore } from '../src/retrieval/bm25.js';
import { reciprocalRankFusion, maximalMarginalRelevance } from '../src/retrieval/fusion.js';
import { hybridRetrieve } from '../src/retrieval/HybridRetriever.js';

// ── chunkText ──────────────────────────────────────────────────────────────────

test('chunkText returns [] for empty / whitespace input', () => {
    expect(chunkText('')).toEqual([]);
    expect(chunkText('   \n  ')).toEqual([]);
});

test('chunkText returns a single chunk when text fits chunkSize', () => {
    const chunks = chunkText('short text', { chunkSize: 100 });
    expect(chunks).toEqual([{ text: 'short text', index: 0, start: 0 }]);
});

test('chunkText splits on paragraph boundaries with overlap', () => {
    const para = 'A'.repeat(40);
    const text = `${para}\n\n${para}\n\n${para}`;
    const chunks = chunkText(text, { chunkSize: 50, chunkOverlap: 10 });
    expect(chunks.length).toBeGreaterThan(1);
    // Indices are contiguous from 0.
    chunks.forEach((c, i) => expect(c.index).toBe(i));
    // Every chunk respects (roughly) the size budget.
    for (const c of chunks) expect(c.text.length).toBeLessThanOrEqual(50);
});

test('chunkText hard-splits a single oversized piece with no separators', () => {
    const text = 'x'.repeat(120);
    const chunks = chunkText(text, { chunkSize: 50, chunkOverlap: 0 });
    expect(chunks.length).toBeGreaterThanOrEqual(3);
    for (const c of chunks) expect(c.text.length).toBeLessThanOrEqual(50);
});

test('chunkText exhausts custom separators without a final "" splitter', () => {
    // No newline in text + separators lacking "" → splitRecursive hits sep===undefined.
    const text = 'y'.repeat(120);
    const chunks = chunkText(text, { chunkSize: 50, chunkOverlap: 5, separators: ['\n'] });
    expect(chunks.length).toBeGreaterThanOrEqual(2);
});

test('chunkText hard-splits an oversized piece with zero overlap', () => {
    // separators lacking "" → an oversized atomic piece reaches the hard-split while
    // loop; overlap 0 exercises the no-carry branch inside it.
    const chunks = chunkText('z'.repeat(120), { chunkSize: 50, chunkOverlap: 0, separators: ['\n'] });
    expect(chunks.length).toBeGreaterThanOrEqual(3);
    for (const c of chunks) expect(c.text.length).toBeLessThanOrEqual(50);
});

test('chunkText with zero overlap does not carry context', () => {
    const text = `${'a'.repeat(30)}\n\n${'b'.repeat(30)}`;
    const chunks = chunkText(text, { chunkSize: 35, chunkOverlap: 0 });
    expect(chunks.length).toBeGreaterThan(1);
});

// ── bm25Search ─────────────────────────────────────────────────────────────────

test('bm25Search returns [] for an empty corpus', () => {
    expect(bm25Search('query', [])).toEqual([]);
});

test('bm25Search returns [] when the query has no usable terms', () => {
    expect(bm25Search('   !!! ', [{ id: '1', text: 'hello world' }])).toEqual([]);
});

test('bm25Search ranks documents by term relevance and drops non-matches', () => {
    const docs = [
        { id: 'a', text: 'the quick brown fox' },
        { id: 'b', text: 'lazy dogs sleep all day' },
        { id: 'c', text: 'a quick fox is a clever fox' },
    ];
    const hits = bm25Search('quick fox', docs);
    const ids = hits.map((h) => h.id);
    expect(ids.sort()).toEqual(['a', 'c']); // both match; 'b' has no overlap → dropped
});

test('bm25Search handles a corpus of empty-tokenizing docs (avgdl guard)', () => {
    // '!!!' tokenizes to [] → totalLen 0 → avgdl falls back to 1; query term absent
    // from every doc → no match → [].
    expect(bm25Search('alpha', [{ id: 'a', text: '!!!' }])).toEqual([]);
});

test('bm25Search honours custom k1/b options', () => {
    const docs = [{ id: 'a', text: 'alpha beta gamma' }];
    const hits = bm25Search('alpha', docs, { k1: 1.2, b: 0.5 });
    expect(hits[0]!.id).toBe('a');
    expect(hits[0]!.score).toBeGreaterThan(0);
});

test('bm25Search accepts an injected tokenizer for both query and documents', () => {
    // A stemming tokenizer: "run" in the query must reach "runs" in the doc, which
    // the plain word-split tokenizer treats as two unrelated terms.
    const stem = (text: string): string[] =>
        (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).map((t) =>
            t.endsWith('s') && !t.endsWith('ss') && t.length > 3 ? t.slice(0, -1) : t,
        );

    const docs = [
        { id: 'match', text: 'the process runs nightly' },
        { id: 'other', text: 'unrelated prose about weather' },
    ];
    expect(bm25Search('run', docs)).toEqual([]);
    expect(bm25Search('run', docs, { tokenize: stem })[0]!.id).toBe('match');
});

test('an injected tokenizer that drops everything yields no hits, not a throw', () => {
    const docs = [{ id: 'a', text: 'alpha beta' }];
    expect(bm25Search('alpha', docs, { tokenize: () => [] })).toEqual([]);
});

// ── the exported scoring core (shared with the api's BM25F ranker) ─────────────

test('bm25Idf is non-negative even for a term present in every document', () => {
    expect(bm25Idf(10, 10)).toBeGreaterThanOrEqual(0);
    expect(bm25Idf(10, 1)).toBeGreaterThan(bm25Idf(10, 9));
});

test('bm25Idf clamps a nonsense corpus rather than returning NaN', () => {
    expect(Number.isFinite(bm25Idf(0, 0))).toBe(true);
    expect(Number.isFinite(bm25Idf(5, 99))).toBe(true);
    expect(Number.isFinite(bm25Idf(5, -1))).toBe(true);
});

test('bm25LengthNorm is 1 at average length and grows with document length', () => {
    expect(bm25LengthNorm(100, 100, 0.75)).toBeCloseTo(1, 10);
    expect(bm25LengthNorm(200, 100, 0.75)).toBeGreaterThan(1);
    expect(bm25LengthNorm(50, 100, 0.75)).toBeLessThan(1);
    // A zero-length corpus must not divide by zero.
    expect(Number.isFinite(bm25LengthNorm(10, 0, 0.75))).toBe(true);
});

test('bm25TermScore saturates: doubling frequency does not double the score', () => {
    const idf = 2;
    const one = bm25TermScore(1, idf, 1, 1.5);
    const two = bm25TermScore(2, idf, 1, 1.5);
    expect(two).toBeGreaterThan(one);
    expect(two).toBeLessThan(one * 2);
    expect(bm25TermScore(0, idf, 1, 1.5)).toBe(0);
});

test('the core reproduces bm25Search exactly — one formula, two index shapes', () => {
    // What the api's BM25F ranker does: read precomputed per-term stats and score
    // them with the shared core. Over a single-term single-doc corpus it must match
    // the inline index bm25Search builds, or the two rankers have drifted.
    const docs = [{ id: 'a', text: 'alpha beta gamma' }];
    const inline = bm25Search('alpha', docs)[0]!.score;
    const fromStats = bm25TermScore(1, bm25Idf(1, 1), bm25LengthNorm(3, 3, 0.75), 1.5);
    expect(fromStats).toBeCloseTo(inline, 12);
});

// ── reciprocalRankFusion ─────────────────────────────────────────────────────

test('reciprocalRankFusion merges rankings and rewards top positions', () => {
    const fused = reciprocalRankFusion([
        { ids: ['a', 'b', 'c'] },
        { ids: ['b', 'a', 'd'] },
    ]);
    // 'a' (ranks 1,2) and 'b' (ranks 2,1) appear in both → top two.
    const top2 = fused.slice(0, 2).map((f) => f.id).sort();
    expect(top2).toEqual(['a', 'b']);
});

test('reciprocalRankFusion respects per-list weight', () => {
    const fused = reciprocalRankFusion([
        { ids: ['a'], weight: 0.1 },
        { ids: ['b'], weight: 10 },
    ]);
    expect(fused[0]!.id).toBe('b');
});

// ── maximalMarginalRelevance ─────────────────────────────────────────────────

test('mmr selects the most relevant first, then diversifies', () => {
    const q = new Float32Array([1, 0]);
    const cands = [
        { id: 'near1', vector: new Float32Array([1, 0]) },        // exact match (rel 1)
        { id: 'near2', vector: new Float32Array([0.99, 0.141]) }, // near-duplicate (rel ~0.99)
        { id: 'div', vector: new Float32Array([0, 1]) },          // orthogonal / diverse
    ];
    // λ=0.3 weights diversity heavily: after the top match, the near-duplicate is
    // penalised for redundancy and the diverse item is promoted.
    const out = maximalMarginalRelevance(q, cands, 2, 0.3);
    expect(out[0]).toBe('near1');     // most relevant picked first
    expect(out[1]).toBe('div');       // diversity beats the near-duplicate
});

test('mmr stops when candidates run out before topK', () => {
    const q = new Float32Array([1, 0]);
    const cands = [{ id: 'a', vector: new Float32Array([1, 0]) }];
    expect(maximalMarginalRelevance(q, cands, 5)).toEqual(['a']);
});

test('mmr keeps the max similarity across multiple already-selected items', () => {
    // topK=3 forces a third selection while two are already selected, exercising the
    // inner "max similarity to selected" loop (the no-update branch included).
    const q = new Float32Array([1, 0]);
    const cands = [
        { id: 'a', vector: new Float32Array([1, 0]) },
        { id: 'b', vector: new Float32Array([0, 1]) },
        { id: 'c', vector: new Float32Array([0.7, 0.7]) },
    ];
    const out = maximalMarginalRelevance(q, cands, 3, 0.5);
    expect(out.sort()).toEqual(['a', 'b', 'c']);
});

// ── hybridRetrieve ───────────────────────────────────────────────────────────

const vec = (x: number, y: number) => Float32Array.from([x, y]);

test('hybridRetrieve returns [] for no candidates', () => {
    expect(hybridRetrieve({ text: 'q' }, [])).toEqual([]);
});

test('hybridRetrieve fuses dense + sparse and reranks', () => {
    const candidates = [
        { id: 'a', text: 'quick brown fox', vector: vec(1, 0) },
        { id: 'b', text: 'slow green turtle', vector: vec(0, 1) },
        { id: 'c', text: 'quick clever fox', vector: vec(0.9, 0.1) },
    ];
    const hits = hybridRetrieve({ text: 'quick fox', vector: vec(1, 0) }, candidates, { topK: 2 });
    expect(hits.length).toBe(2);
    expect(hits.map((h) => h.id)).toContain('a');
});

test('hybridRetrieve works BM25-only when no query vector is supplied', () => {
    const candidates = [
        { id: 'a', text: 'alpha beta' },
        { id: 'b', text: 'gamma delta' },
    ];
    const hits = hybridRetrieve({ text: 'alpha' }, candidates, { topK: 5 });
    expect(hits[0]!.id).toBe('a');
});

test('hybridRetrieve returns [] when nothing matches either signal', () => {
    // No query vector AND no lexical overlap → both rankings empty → fused empty.
    const candidates = [{ id: 'a', text: 'zzz' }];
    expect(hybridRetrieve({ text: 'nomatch' }, candidates)).toEqual([]);
});

test('hybridRetrieve skips MMR when candidates have no vectors', () => {
    const candidates = [
        { id: 'a', text: 'alpha beta' },
        { id: 'b', text: 'alpha gamma' },
    ];
    // query has a vector but candidates do not → mmrCands empty → fused order kept.
    const hits = hybridRetrieve({ text: 'alpha', vector: vec(1, 0) }, candidates, { topK: 2 });
    expect(hits.length).toBe(2);
});

test('hybridRetrieve with rerank disabled keeps fused order', () => {
    const candidates = [
        { id: 'a', text: 'quick fox', vector: vec(1, 0) },
        { id: 'b', text: 'quick fox', vector: vec(0.99, 0.01) },
    ];
    const hits = hybridRetrieve({ text: 'quick fox', vector: vec(1, 0) }, candidates, {
        topK: 2,
        rerank: false,
    });
    expect(hits.length).toBe(2);
});

test('hybridRetrieve appends fused tail when MMR ranks only the vectored subset', () => {
    const candidates = [
        { id: 'withvec', text: 'quick fox', vector: vec(1, 0) },
        { id: 'novec', text: 'quick fox lexical' }, // matches BM25 but has no vector
    ];
    const hits = hybridRetrieve({ text: 'quick fox', vector: vec(1, 0) }, candidates, { topK: 5 });
    const ids = hits.map((h) => h.id);
    expect(ids).toContain('withvec');
    expect(ids).toContain('novec'); // appended after the MMR-ranked vectored subset
});
