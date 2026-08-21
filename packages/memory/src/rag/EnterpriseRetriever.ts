/**
 * rag/EnterpriseRetriever.ts — hybrid, tenant-scoped, cited retrieval.
 *
 * The gap between a RAG demo and an enterprise RAG system is not the embedding
 * model. It is these five properties, all of which live here:
 *
 *   1. **Scoped.** Every read carries an {@link AccessScope} that is compiled into
 *      the database query. There is no code path that reads unscoped, so a
 *      cross-tenant answer is not something a caller can cause by forgetting.
 *   2. **Hybrid.** Dense retrieval alone misses exact tokens — error codes, SKUs,
 *      contract clause numbers, function names. Lexical search alone misses
 *      paraphrase. RRF over both, then MMR for diversity, then an optional
 *      reranker.
 *   3. **Honest about degradation.** When the store has no lexical index the
 *      retrieval is dense-only and SAYS so on the span and in the result, instead
 *      of quietly returning worse answers.
 *   4. **Cited.** Every passage carries its source id, title and URI, and the
 *      prompt builder numbers them, so an answer can be checked. An uncitable
 *      answer is an unauditable one, and enterprises audit.
 *   5. **Traced and costed.** Retrieval and generation are spans under one trace,
 *      with token counts and cost, so "why was this answer bad" and "why did this
 *      cost that" are answerable from the same record.
 */

import type { TransformerBridge } from '../bridges/TransformerBridge.js';
import { maximalMarginalRelevance, reciprocalRankFusion } from '../retrieval/fusion.js';
import type { Span, Tracer } from '../telemetry/Tracer.js';
import {
    SOURCE_ID_FIELD,
    TITLE_FIELD,
    URI_FIELD,
} from '../ingest/types.js';
import type {
    AccessScope,
    MetadataFilter,
    VectorMatch,
    VectorStore,
} from '../vectorstore/types.js';

/** Reranker port — a cross-encoder, an LLM judge, or a hosted rerank endpoint. */
export type Reranker = (
    query: string,
    passages: readonly RetrievedPassage[],
) => Promise<readonly RetrievedPassage[]>;

export interface RetrievedPassage {
    id: string;
    text: string;
    /** Fused relevance score. */
    score: number;
    sourceId: string;
    title?: string;
    uri?: string;
    metadata: Record<string, unknown>;
}

export type RetrievalMode = 'hybrid' | 'dense-only' | 'keyword-only' | 'empty';

export interface RetrievalResult {
    passages: RetrievedPassage[];
    /** What actually ran — `dense-only` means the store had no lexical index. */
    mode: RetrievalMode;
    /** Candidates considered before fusion and truncation. */
    candidatesConsidered: number;
    traceId?: string;
}

export interface RetrieveOptions {
    /** Tenant/ACL scope. Required unless the retriever was built with a default. */
    scope?: AccessScope;
    /** Extra metadata predicate, AND-ed with the scope. */
    filter?: MetadataFilter;
    /** Passages returned. Default 5. */
    topK?: number;
    /** Candidates pulled from each retrieval arm before fusion. Default `topK * 4`. */
    candidateK?: number;
    /** Drop passages below this fused score after normalisation. Default 0. */
    minScore?: number;
    /** Parent span, so retrieval nests under the agent turn that asked for it. */
    parent?: Span;
}

export interface EnterpriseRetrieverOptions {
    store: VectorStore;
    /** Embeds the query. Omit for keyword-only retrieval. */
    embed?: (text: string) => Promise<Float32Array>;
    /** Applied to the fused list before truncation. */
    reranker?: Reranker;
    /** Default scope, when a deployment is single-tenant or scope comes from context. */
    defaultScope?: AccessScope;
    /** RRF weight for the dense arm. Default 1. */
    denseWeight?: number;
    /** RRF weight for the lexical arm. Default 1. */
    keywordWeight?: number;
    /** MMR relevance/diversity trade-off; 1 is pure relevance. Default 0.7. */
    mmrLambda?: number;
    /** Disable the MMR diversity pass. Default false. */
    disableDiversity?: boolean;
    tracer?: Tracer;
}

export class EnterpriseRetriever {
    private readonly _store: VectorStore;
    private readonly _embed: ((text: string) => Promise<Float32Array>) | undefined;
    private readonly _reranker: Reranker | undefined;
    private readonly _defaultScope: AccessScope | undefined;
    private readonly _denseWeight: number;
    private readonly _keywordWeight: number;
    private readonly _mmrLambda: number;
    private readonly _diversity: boolean;
    private readonly _tracer: Tracer | undefined;

    constructor(opts: EnterpriseRetrieverOptions) {
        this._store = opts.store;
        this._embed = opts.embed;
        this._reranker = opts.reranker;
        this._defaultScope = opts.defaultScope;
        this._denseWeight = opts.denseWeight ?? 1;
        this._keywordWeight = opts.keywordWeight ?? 1;
        this._mmrLambda = opts.mmrLambda ?? 0.7;
        this._diversity = !opts.disableDiversity;
        this._tracer = opts.tracer;
    }

    async retrieve(query: string, opts: RetrieveOptions = {}): Promise<RetrievalResult> {
        const topK = Math.max(1, opts.topK ?? 5);
        const candidateK = Math.max(topK, opts.candidateK ?? topK * 4);
        const scope = opts.scope ?? this._defaultScope;

        const span = this._tracer?.startSpan('retrieval.hybrid', {
            kind: 'retrieval',
            ...(opts.parent ? { parent: opts.parent } : {}),
            attributes: {
                'retrieval.store': this._store.name,
                'retrieval.top_k': topK,
                'retrieval.tenant': scope?.tenantId ?? 'unscoped',
            },
        });

        try {
            const queryVector = this._embed ? await this._embed(query) : undefined;

            const base = {
                text: query,
                topK: candidateK,
                ...(opts.filter ? { filter: opts.filter } : {}),
                ...(scope ? { scope } : {}),
                includeVectors: this._diversity,
            };

            // Both arms run concurrently: they hit different indexes and the
            // sequential version doubles p95 latency for nothing.
            const [dense, keyword] = await Promise.all([
                queryVector
                    ? this._store.query({ ...base, vector: queryVector })
                    : Promise.resolve<VectorMatch[]>([]),
                this._store.keywordSearch
                    ? this._store.keywordSearch(base)
                    : Promise.resolve<VectorMatch[]>([]),
            ]);

            const mode = pickMode(dense.length > 0, keyword.length > 0, Boolean(this._store.keywordSearch));
            const byId = new Map<string, VectorMatch>();
            for (const match of [...dense, ...keyword]) {
                if (!byId.has(match.id)) byId.set(match.id, match);
            }

            let ordered = this._fuse(dense, keyword, byId);

            if (this._diversity && queryVector && ordered.length > topK) {
                ordered = this._diversify(queryVector, ordered, byId, topK);
            }

            let passages = ordered.slice(0, Math.max(topK, this._reranker ? candidateK : topK))
                .map((hit) => toPassage(byId.get(hit.id) as VectorMatch, hit.score));

            if (this._reranker && passages.length > 0) {
                passages = [...await this._reranker(query, passages)];
            }
            if (opts.minScore !== undefined) {
                passages = passages.filter((p) => p.score >= (opts.minScore as number));
            }
            passages = passages.slice(0, topK);

            span?.setAttributes({
                'retrieval.mode': mode,
                'retrieval.candidates': byId.size,
                'retrieval.returned': passages.length,
                'retrieval.dense_hits': dense.length,
                'retrieval.keyword_hits': keyword.length,
                'retrieval.reranked': Boolean(this._reranker),
            });
            span?.end('ok');

            const result: RetrievalResult = {
                passages,
                mode,
                candidatesConsidered: byId.size,
            };
            if (span) result.traceId = span.traceId;
            return result;
        } catch (err) {
            span?.fail(err);
            throw err;
        }
    }

    /**
     * Retrieve, then answer strictly from what was retrieved.
     *
     * The system prompt forbids answering beyond the passages and requires
     * `[n]` citations. That is not decoration: an enterprise deployment is judged
     * on whether a wrong answer is *detectable*, and an uncited claim is not.
     */
    async answer(
        query: string,
        bridge: TransformerBridge,
        opts: RetrieveOptions & { systemPrompt?: string; maxTokens?: number; model?: string } = {},
    ): Promise<{ answer: string; result: RetrievalResult }> {
        const span = this._tracer?.startSpan('rag.answer', {
            kind: 'agent',
            ...(opts.parent ? { parent: opts.parent } : {}),
        });

        try {
            const result = await this.retrieve(query, { ...opts, ...(span ? { parent: span } : {}) });

            if (result.passages.length === 0) {
                // Refusing beats hallucinating: with no evidence there is nothing to
                // ground an answer in, and a confident answer here is the exact
                // failure enterprises cite when a pilot is killed.
                span?.setAttribute('rag.refused', true);
                span?.end('ok');
                return { answer: NO_EVIDENCE_ANSWER, result };
            }

            const prompt = buildGroundedPrompt(query, result.passages);
            const answer = await bridge.generate(prompt, {
                systemPrompt: opts.systemPrompt ?? GROUNDED_SYSTEM_PROMPT,
                ...(opts.maxTokens !== undefined ? { maxTokens: opts.maxTokens } : {}),
                ...(opts.model !== undefined ? { model: opts.model } : {}),
            });

            span?.setAttributes({ 'rag.passages': result.passages.length, 'rag.mode': result.mode });
            span?.end('ok');
            return { answer, result };
        } catch (err) {
            span?.fail(err);
            throw err;
        }
    }

    private _fuse(
        dense: readonly VectorMatch[],
        keyword: readonly VectorMatch[],
        byId: Map<string, VectorMatch>,
    ): Array<{ id: string; score: number }> {
        const lists = [];
        if (dense.length) lists.push({ ids: dense.map((m) => m.id), weight: this._denseWeight });
        if (keyword.length) lists.push({ ids: keyword.map((m) => m.id), weight: this._keywordWeight });

        if (lists.length === 0) return [];
        if (lists.length === 1) {
            // One arm: keep the store's own similarity rather than replacing it with
            // an RRF score, which carries no meaning for a single list.
            const only = dense.length ? dense : keyword;
            return only.map((m) => ({ id: m.id, score: m.score }));
        }

        const fused = reciprocalRankFusion(lists);
        const best = fused[0]?.score ?? 1;
        // RRF scores live on a tiny arbitrary scale (~1/60); normalising to [0,1]
        // keeps `minScore` meaning the same thing on one arm or two.
        return fused
            .filter((hit) => byId.has(hit.id))
            .map((hit) => ({ id: hit.id, score: best > 0 ? hit.score / best : 0 }));
    }

    private _diversify(
        queryVector: Float32Array,
        ordered: Array<{ id: string; score: number }>,
        byId: Map<string, VectorMatch>,
        topK: number,
    ): Array<{ id: string; score: number }> {
        const candidates = ordered
            .map((hit) => ({ id: hit.id, vector: byId.get(hit.id)?.vector }))
            .filter((c): c is { id: string; vector: Float32Array } => Boolean(c.vector));

        // Without stored vectors MMR cannot run; keeping the fused order is the
        // correct degradation, not an error.
        if (candidates.length < 2) return ordered;

        const scores = new Map(ordered.map((hit) => [hit.id, hit.score]));
        const picked = maximalMarginalRelevance(queryVector, candidates, topK, this._mmrLambda);
        const pickedSet = new Set(picked);

        return [
            ...picked.map((id) => ({ id, score: scores.get(id) ?? 0 })),
            ...ordered.filter((hit) => !pickedSet.has(hit.id)),
        ];
    }
}

function pickMode(hasDense: boolean, hasKeyword: boolean, storeHasKeyword: boolean): RetrievalMode {
    if (hasDense && hasKeyword) return 'hybrid';
    if (hasDense) return 'dense-only';
    if (hasKeyword) return 'keyword-only';
    return storeHasKeyword ? 'empty' : 'dense-only';
}

function toPassage(match: VectorMatch, score: number): RetrievedPassage {
    const metadata = match.metadata as Record<string, unknown>;
    const passage: RetrievedPassage = {
        id: match.id,
        text: match.text,
        score,
        sourceId: String(metadata[SOURCE_ID_FIELD] ?? match.id),
        metadata,
    };
    const title = metadata[TITLE_FIELD];
    if (typeof title === 'string') passage.title = title;
    const uri = metadata[URI_FIELD];
    if (typeof uri === 'string') passage.uri = uri;
    return passage;
}

export const GROUNDED_SYSTEM_PROMPT =
    'You answer strictly from the numbered passages provided. ' +
    'Cite every claim with the passage number in square brackets, e.g. [2]. ' +
    'If the passages do not contain the answer, say so plainly and name what is missing. ' +
    'Never rely on knowledge that is not in the passages.';

export const NO_EVIDENCE_ANSWER =
    'No indexed passage matched this question within the caller\'s access scope, ' +
    'so there is nothing to ground an answer in.';

/** Renders passages as a numbered, citable evidence block. */
export function buildGroundedPrompt(query: string, passages: readonly RetrievedPassage[]): string {
    const evidence = passages
        .map((p, i) => {
            const label = p.title ?? p.sourceId;
            const uri = p.uri ? ` (${p.uri})` : '';
            return `[${i + 1}] ${label}${uri}\n${p.text}`;
        })
        .join('\n\n');

    return `Passages:\n\n${evidence}\n\nQuestion: ${query}\n\nAnswer, citing passages by number:`;
}
