/**
 * BM25 (Okapi) lexical ranking.
 *
 * The keyword half of hybrid retrieval. Dense vector search matches meaning but
 * misses exact tokens (identifiers, error codes, rare names); BM25 catches those.
 * Fusing the two (see {@link ./fusion}) is what lifts the memory layer from
 * "cosine only" to a hybrid retriever on par with Weaviate-style search.
 *
 * Two things are separable here and are kept separate on purpose:
 *
 *   • the SCORING CORE ({@link bm25Idf}, {@link bm25LengthNorm}, {@link bm25TermScore})
 *     — the Okapi formula itself, which is the same wherever lexical relevance is
 *     scored, whether the index is built inline from raw text or read out of a
 *     precomputed per-field term table (BM25F). Exported so a consumer with its own
 *     index shape reuses the maths instead of restating it and drifting from it;
 *   • the TOKENIZER — which is genuinely corpus-specific. Prose wants stopword
 *     removal and stemming; identifiers and error codes want neither. So the
 *     tokenizer is an injectable {@link Bm25Options.tokenize} hook defaulting to the
 *     package's plain `tokenize`, rather than a second copy of BM25 growing around
 *     each specialisation.
 *
 * Pure and zero-dependency.
 */

import { tokenize } from '../similarity/index.js';

/** Splits text into the terms BM25 scores over. */
export type Bm25Tokenizer = (text: string) => string[];

export interface Bm25Options {
    /** Term-frequency saturation. Higher = TF matters more. Default 1.5. */
    k1?: number;
    /** Length normalisation, 0..1. Higher = penalise long docs more. Default 0.75. */
    b?: number;
    /**
     * Tokenizer for both the query and the documents. Default is the package's plain
     * `tokenize` (lowercase word split), which preserves identifiers verbatim. Supply
     * a stemming/stopword tokenizer when the corpus is prose — the SAME function must
     * be used for query and documents or nothing will match.
     */
    tokenize?: Bm25Tokenizer;
}

export interface Bm25Doc {
    id: string;
    text: string;
}

export interface Bm25Hit {
    id: string;
    score: number;
}

/** Term-frequency saturation used when {@link Bm25Options.k1} is omitted. */
export const BM25_DEFAULT_K1 = 1.5;
/** Length normalisation used when {@link Bm25Options.b} is omitted. */
export const BM25_DEFAULT_B = 0.75;

/**
 * Inverse document frequency with the +1 smoothing variant, which is always
 * non-negative — the unsmoothed form goes negative for a term present in more than
 * half the corpus, which subtracts score for a MATCH.
 */
export function bm25Idf(documentCount: number, documentFrequency: number): number {
    const n = Math.max(documentCount, 1);
    const df = Math.max(0, Math.min(documentFrequency, n));
    return Math.log(1 + (n - df + 0.5) / (df + 0.5));
}

/**
 * The document-length penalty `1 - b + b·(len/avgdl)`, factored out because it is
 * per-DOCUMENT: computing it once and reusing it across the query's terms is what
 * keeps scoring O(terms) rather than O(terms) divisions of the same quotient.
 */
export function bm25LengthNorm(length: number, averageLength: number, b: number = BM25_DEFAULT_B): number {
    const avg = averageLength > 0 ? averageLength : 1;
    return 1 - b + b * (Math.max(0, length) / avg);
}

/**
 * One term's contribution: `idf · f(k1+1) / (f + k1·lengthNorm)`.
 *
 * `frequency` is whatever the caller counts as the term's occurrence weight — a raw
 * count for plain BM25, or a field-weighted sum for BM25F. Combining fields BEFORE
 * saturation (rather than saturating each field and summing) is what makes BM25F
 * behave like BM25 over a weighted document.
 */
export function bm25TermScore(frequency: number, idf: number, lengthNorm: number, k1: number = BM25_DEFAULT_K1): number {
    if (frequency <= 0) return 0;
    return (idf * (frequency * (k1 + 1))) / (frequency + k1 * lengthNorm);
}

/**
 * Scores every document against `query` with Okapi BM25, returning hits sorted by
 * descending score (documents with no query-term overlap score 0 and are dropped).
 * Builds the index inline — for a recall over a bounded candidate set (the memory
 * store / a vector pre-filter) this is O(N·terms) and needs no persistence.
 */
export function bm25Search(query: string, docs: Bm25Doc[], opts: Bm25Options = {}): Bm25Hit[] {
    const k1 = opts.k1 ?? BM25_DEFAULT_K1;
    const b = opts.b ?? BM25_DEFAULT_B;
    const split = opts.tokenize ?? tokenize;
    const N = docs.length;
    if (N === 0) return [];

    const queryTerms = new Set(split(query));
    if (queryTerms.size === 0) return [];

    // Per-doc term frequencies + document lengths.
    const docTerms: { id: string; tf: Map<string, number>; len: number }[] = [];
    const df = new Map<string, number>();
    let totalLen = 0;

    for (const doc of docs) {
        const tokens = split(doc.text);
        const tf = new Map<string, number>();
        for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
        for (const t of tf.keys()) if (queryTerms.has(t)) df.set(t, (df.get(t) ?? 0) + 1);
        docTerms.push({ id: doc.id, tf, len: tokens.length });
        totalLen += tokens.length;
    }
    const avgdl = totalLen / N || 1;

    const idf = new Map<string, number>();
    for (const term of queryTerms) idf.set(term, bm25Idf(N, df.get(term) ?? 0));

    const hits: Bm25Hit[] = [];
    for (const d of docTerms) {
        const lengthNorm = bm25LengthNorm(d.len, avgdl, b);
        let score = 0;
        for (const term of queryTerms) {
            const f = d.tf.get(term);
            if (!f) continue;
            score += bm25TermScore(f, idf.get(term)!, lengthNorm, k1);
        }
        if (score > 0) hits.push({ id: d.id, score });
    }

    hits.sort((a, b) => b.score - a.score);
    return hits;
}
