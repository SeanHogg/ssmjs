/**
 * vectorstore/MemoryVectorStore.ts — the in-process reference adapter.
 *
 * It is the default store, the test double for every layer above it, and the
 * conformance benchmark a remote adapter is checked against. It is also the
 * honest answer for edge deployments (a Worker, a browser tab, an on-prem box)
 * where standing up a vector database is not on the table.
 *
 * Performance notes, because "in-memory" is not an excuse for O(N) per query:
 *   • The HNSW graph is built ONCE and extended incrementally on upsert. The
 *     shared `denseSearch` helper rebuilds its index per call, which is right for
 *     a one-shot rerank over a candidate set and wrong for a store that is queried
 *     repeatedly — so this adapter owns a persistent index instead.
 *   • Deletes tombstone rather than rebuild; the graph is rebuilt only once
 *     tombstones pass a share of the index, amortising the cost.
 *   • Filtered queries over-fetch from the graph and post-filter, falling back to
 *     an exact scan only when the filter is selective enough that the graph could
 *     not fill `topK`. Correct in both regimes, fast in the common one.
 */

import { bm25Search } from '../retrieval/bm25.js';
import { HnswIndex, type HnswOptions } from '../retrieval/hnsw.js';
import { cosineSimilarity } from '../similarity/index.js';
import { combineFilters, matchesFilter } from './filter.js';
import type {
    AccessScope,
    MetadataFilter,
    UpsertResult,
    VectorMatch,
    VectorQuery,
    VectorRecord,
    VectorStore,
} from './types.js';

export interface MemoryVectorStoreOptions {
    /** Adapter name for telemetry. Default `'memory'`. */
    name?: string;
    /**
     * Record count at/above which dense search uses the HNSW graph instead of an
     * exact cosine scan. Below it, a scan is both simpler and faster. Default 256.
     */
    annThreshold?: number;
    /** HNSW construction parameters. */
    hnsw?: HnswOptions;
    /**
     * Over-fetch multiplier for filtered ANN queries — how many graph candidates
     * to pull per requested result before post-filtering. Default 4.
     */
    filterOverfetch?: number;
    /**
     * Rebuild the graph once tombstoned nodes exceed this share of it. Default 0.2.
     */
    rebuildTombstoneRatio?: number;
}

export class MemoryVectorStore implements VectorStore {
    readonly name: string;

    private readonly _records = new Map<string, VectorRecord>();
    private readonly _annThreshold: number;
    private readonly _hnswOpts: HnswOptions;
    private readonly _overfetch: number;
    private readonly _rebuildRatio: number;

    private _index: HnswIndex | undefined;
    /** Ids present in the graph but deleted from the store. */
    private readonly _tombstones = new Set<string>();

    constructor(opts: MemoryVectorStoreOptions = {}) {
        this.name = opts.name ?? 'memory';
        this._annThreshold = Math.max(1, opts.annThreshold ?? 256);
        this._hnswOpts = opts.hnsw ?? {};
        this._overfetch = Math.max(1, opts.filterOverfetch ?? 4);
        this._rebuildRatio = opts.rebuildTombstoneRatio ?? 0.2;
    }

    get size(): number { return this._records.size; }

    async upsert(records: readonly VectorRecord[]): Promise<UpsertResult> {
        let upserted = 0;
        for (const record of records) {
            const previous = this._records.get(record.id);
            this._records.set(record.id, record);
            upserted += 1;

            if (!record.vector) continue;
            // A changed vector under an existing id invalidates its graph node;
            // tombstone the old one so search cannot return the stale neighbour.
            if (previous?.vector && this._index?.has(record.id)) {
                this._tombstones.add(record.id);
                this._maybeRebuild();
                continue;
            }
            if (this._index && !this._index.has(record.id)) {
                this._index.add(record.id, record.vector);
                this._tombstones.delete(record.id);
            }
        }
        return { upserted, skipped: 0 };
    }

    async fetch(ids: readonly string[]): Promise<VectorRecord[]> {
        const out: VectorRecord[] = [];
        for (const id of ids) {
            const record = this._records.get(id);
            if (record) out.push(record);
        }
        return out;
    }

    async delete(ids: readonly string[]): Promise<number> {
        let removed = 0;
        for (const id of ids) {
            if (!this._records.delete(id)) continue;
            removed += 1;
            if (this._index?.has(id)) this._tombstones.add(id);
        }
        this._maybeRebuild();
        return removed;
    }

    async deleteByFilter(filter: MetadataFilter, scope?: AccessScope): Promise<number> {
        const effective = combineFilters(filter, scope);
        const doomed = [...this._records.values()]
            .filter((r) => matchesFilter(r.metadata, effective))
            .map((r) => r.id);
        return this.delete(doomed);
    }

    async count(filter?: MetadataFilter, scope?: AccessScope): Promise<number> {
        const effective = combineFilters(filter, scope);
        if (!effective) return this._records.size;
        let n = 0;
        for (const record of this._records.values()) {
            if (matchesFilter(record.metadata, effective)) n += 1;
        }
        return n;
    }

    async query(query: VectorQuery): Promise<VectorMatch[]> {
        const topK = Math.max(1, query.topK ?? 10);
        const minScore = query.minScore ?? 0;
        const filter = combineFilters(query.filter, query.scope);

        if (!query.vector) {
            // No embedding: degrade to a filter-only listing rather than returning
            // nothing, so metadata-only lookups do not need a second code path.
            return this._visible(filter)
                .slice(0, topK)
                .map((r) => this._toMatch(r, 1, query.includeVectors));
        }

        const scored = this._denseScore(query.vector, topK, filter);
        return scored
            .filter((hit) => hit.score >= minScore)
            .slice(0, topK)
            .map(({ record, score }) => this._toMatch(record, score, query.includeVectors));
    }

    /**
     * BM25 over the visible corpus. Present because lexical recall catches the
     * exact identifiers (error codes, SKUs, function names) that embeddings blur —
     * the failure mode enterprise RAG evaluations reliably surface.
     */
    async keywordSearch(query: VectorQuery): Promise<VectorMatch[]> {
        const text = query.text?.trim();
        if (!text) return [];

        const topK = Math.max(1, query.topK ?? 10);
        const filter = combineFilters(query.filter, query.scope);
        const visible = this._visible(filter);
        if (visible.length === 0) return [];

        const hits = bm25Search(text, visible.map((r) => ({ id: r.id, text: r.text })));
        const best = hits[0]?.score ?? 0;

        return hits.slice(0, topK).map((hit) => {
            const record = this._records.get(hit.id) as VectorRecord;
            // BM25 scores are unbounded; normalising against the top hit puts them
            // on the same [0,1] footing as cosine so fusion weights mean something.
            return this._toMatch(record, best > 0 ? hit.score / best : 0, query.includeVectors);
        });
    }

    private _denseScore(
        vector: Float32Array,
        topK: number,
        filter?: MetadataFilter,
    ): Array<{ record: VectorRecord; score: number }> {
        const embedded = [...this._records.values()].filter((r) => r.vector);

        const exactScan = () => embedded
            .filter((r) => matchesFilter(r.metadata, filter))
            .map((r) => ({ record: r, score: cosineSimilarity(vector, r.vector as Float32Array) }))
            .sort((a, b) => b.score - a.score);

        if (embedded.length < this._annThreshold) return exactScan();

        this._ensureIndex(embedded);
        const index = this._index as HnswIndex;
        const wanted = filter ? topK * this._overfetch : topK;
        const hits = index.search(vector, wanted + this._tombstones.size);

        const matched: Array<{ record: VectorRecord; score: number }> = [];
        for (const hit of hits) {
            if (this._tombstones.has(hit.id)) continue;
            const record = this._records.get(hit.id);
            if (!record || !matchesFilter(record.metadata, filter)) continue;
            matched.push({ record, score: hit.score });
        }

        // The graph could not fill the page under this filter — the filter is more
        // selective than the over-fetch anticipated, so pay for an exact scan
        // rather than return a short, silently-truncated result set.
        if (matched.length < topK && filter) return exactScan();
        return matched;
    }

    private _visible(filter?: MetadataFilter): VectorRecord[] {
        const all = [...this._records.values()];
        return filter ? all.filter((r) => matchesFilter(r.metadata, filter)) : all;
    }

    private _toMatch(record: VectorRecord, score: number, includeVectors?: boolean): VectorMatch {
        const match: VectorMatch = {
            id: record.id,
            text: record.text,
            score,
            metadata: record.metadata,
        };
        if (includeVectors && record.vector) match.vector = record.vector;
        return match;
    }

    private _ensureIndex(embedded: readonly VectorRecord[]): void {
        if (this._index) return;
        const index = new HnswIndex(this._hnswOpts);
        for (const record of embedded) index.add(record.id, record.vector as Float32Array);
        this._index = index;
        this._tombstones.clear();
    }

    private _maybeRebuild(): void {
        const index = this._index;
        if (!index || index.size === 0) return;
        if (this._tombstones.size / index.size < this._rebuildRatio) return;
        this._index = undefined;
        this._tombstones.clear();
    }
}
