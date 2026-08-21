/**
 * MemoryStore – persistent key-value fact store and weight checkpoint helper.
 *
 * Uses IndexedDB with a dedicated 'ssmjs' database containing two object stores:
 *   - 'facts'   : MemoryEntry records keyed by the fact key string
 *   - 'weights' : a single ArrayBuffer keyed by `weightsKey`
 *
 * Weight save/load is delegated to the SSMRuntime passed to saveWeights/loadWeights.
 */

import { SSMError } from '../errors/SSMError.js';
import { tokenize, jaccardSimilarity } from '../similarity/index.js';
import { hybridRetrieve, denseSearch, HnswIndex, type RetrievalCandidate, type HybridRetrieveOptions } from '../retrieval/index.js';

export type FactType = 'text' | 'json' | 'number' | 'boolean';

export interface MemoryEntry {
    key        : string;
    content    : string;
    timestamp  : number;
    /**
     * Monotonic write sequence, used only to break `timestamp` ties so that
     * ordering stays deterministic when several entries are written within the
     * same millisecond (Date.now() resolution). Higher = written later.
     * Optional for backward compatibility with entries persisted/imported
     * before this field existed (those sort as seq 0).
     */
    seq?       : number;
    /** Optional time-to-live in milliseconds. */
    ttlMs?     : number;
    /** Semantic type of the stored value. Default: 'text'. */
    type?      : FactType;
    /** Tags for grouping and filtering. */
    tags?      : string[];
    /** Importance weight in the range 0–1. Default: 0.5. */
    importance?: number;
}

/**
 * Which ranker produced a recall ordering. Matches the vocabulary the gateway's
 * Evermind recall/validate surfaces already report, so a host can render one chip
 * ("Semantic recall" / "Lexical recall (fallback)") regardless of which layer
 * answered.
 */
export type RecallMethod = 'embedding' | 'lexical';

/** A recalled entry with its relevance score (higher = closer). */
export interface ScoredMemory {
    entry: MemoryEntry;
    score: number;
}

/** A ranked recall: the hits plus the ranker that ordered them. */
export interface RankedRecall {
    hits: ScoredMemory[];
    method: RecallMethod;
}

export interface RememberOptions {
    /** Override the store-level defaultTtlMs for this entry. */
    ttlMs?     : number;
    type?      : FactType;
    tags?      : string[];
    importance?: number;
}

export interface MemoryStoreOptions {
    /** IndexedDB database name. Default: 'ssmjs'. */
    dbName?       : string;
    /** Key used for weight storage within the 'weights' object store. Default: 'ssmjs-weights'. */
    weightsKey?   : string;
    /**
     * IDBFactory to use instead of the global `indexedDB`.
     * Use this in Node.js environments with fake-indexeddb:
     *   import { IDBFactory } from 'fake-indexeddb';
     *   const idbFactory = new IDBFactory();
     */
    idbFactory?   : IDBFactory;
    /**
     * Default TTL applied to new entries when no per-entry ttlMs is provided.
     * Entries with an expired TTL are filtered from recallAll() and related methods.
     */
    defaultTtlMs? : number;
    /**
     * Fact count at/above which semantic recall switches from an exact O(N)
     * cosine scan to an O(log N) HNSW ANN index. Default 256 — small stores stay
     * exact (simpler + faster), large stores stay fast. (EVM-1)
     */
    annThreshold? : number;
    /**
     * Hard cap on stored facts. When set, each write evicts the lowest-value
     * entries (expired first, then lowest importance, then oldest) so the store
     * stays bounded. Unset = unbounded (legacy). (EVM-7)
     */
    maxEntries?   : number;
    /**
     * Max content→embedding vectors held in the LRU recall cache. Default 2000.
     * (EVM-8 — bounded LRU instead of clear-on-overflow.)
     */
    embedCacheMax?: number;
}

const FACTS_STORE   = 'facts';
const WEIGHTS_STORE = 'weights';
const DB_VERSION    = 1;

/**
 * Process-wide monotonic counter stamped onto each remembered entry. Strictly
 * increasing per write regardless of store instance, so it reliably breaks
 * `timestamp` ties (same-millisecond writes) in newest-first ordering.
 */
let _writeSeq = 0;

// Minimal interface to avoid importing SSMRuntime (circular dep)
interface SaveLoadRuntime {
    save(opts?: { storage: 'indexedDB'; key: string }): Promise<void>;
    load(opts?: { key: string }): Promise<boolean>;
}

// Forward-declared to avoid circular dependency at import time
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type SSMRuntimeRef = any;

/** Max content→embedding vectors retained (LRU-evicted past this). (EVM-8) */
const EMBED_CACHE_MAX = 2000;

export class MemoryStore {
    private readonly _dbName     : string;
    private readonly _weightsKey : string;
    private readonly _idb        : IDBFactory | undefined;
    private readonly _defaultTtl : number | undefined;
    private readonly _annThreshold : number;
    private readonly _maxEntries : number | undefined;
    private readonly _embedCacheMax : number;
    private _db: IDBDatabase | null = null;
    private _sweepTimer: ReturnType<typeof setInterval> | null = null;
    /** Content → L2-normalised embedding cache, used by recallSimilar. */
    private readonly _embedCache = new Map<string, Float32Array>();
    /** Runtime whose embeddings populate `_embedCache`; a change invalidates it. */
    private _embedRuntime: unknown = null;

    // ── Persistent dense index (EVM-1b) ────────────────────────────────────────
    // Fact embeddings keyed by `${key}@${seq}` so an EDIT (seq bump) is a new key
    // and an unchanged fact is never re-embedded. The HNSW index is derived from
    // these vectors and reused across recalls until the fact set changes — so
    // steady-state recall is O(1) query-embed + O(log N) search, not O(N) embed.
    private readonly _denseVectors = new Map<string, Float32Array>();
    private _denseIndex: HnswIndex | null = null;
    /** Signature of the fact set the current index was built from. */
    private _denseSig = '';
    /** Identity of the runtime whose embeddings populated the vectors. */
    private _denseRuntime: unknown = null;

    constructor(opts: MemoryStoreOptions = {}) {
        this._dbName     = opts.dbName     ?? 'ssmjs';
        this._weightsKey = opts.weightsKey ?? 'ssmjs-weights';
        this._idb        = opts.idbFactory;
        this._defaultTtl = opts.defaultTtlMs;
        this._annThreshold = opts.annThreshold ?? 256;
        this._maxEntries = opts.maxEntries;
        this._embedCacheMax = opts.embedCacheMax ?? EMBED_CACHE_MAX;
    }

    // ── Internal DB open ──────────────────────────────────────────────────────

    private _open(): Promise<IDBDatabase> {
        if (this._db) return Promise.resolve(this._db);

        return new Promise((resolve, reject) => {
            const factory = this._idb ?? (typeof indexedDB !== 'undefined' ? indexedDB : undefined);
            if (!factory) {
                reject(new SSMError(
                    'MEMORY_UNAVAILABLE',
                    'IndexedDB is not available in this environment. Pass an idbFactory option (e.g. from fake-indexeddb) for Node.js support.',
                ));
                return;
            }

            const req = factory.open(this._dbName, DB_VERSION);

            req.onupgradeneeded = (e) => {
                const db = (e.target as IDBOpenDBRequest).result;
                if (!db.objectStoreNames.contains(FACTS_STORE)) {
                    db.createObjectStore(FACTS_STORE, { keyPath: 'key' });
                }
                if (!db.objectStoreNames.contains(WEIGHTS_STORE)) {
                    db.createObjectStore(WEIGHTS_STORE);
                }
            };

            req.onsuccess = () => {
                this._db = req.result;
                resolve(req.result);
            };
            /* istanbul ignore next -- IDB open onerror fires only on storage faults; not reproducible with fake-indexeddb */
            req.onerror = () => reject(new SSMError(
                'MEMORY_UNAVAILABLE',
                `Failed to open IndexedDB "${this._dbName}": ${req.error?.message ?? 'unknown'}`,
                req.error,
            ));
        });
    }

    // ── TTL helpers ───────────────────────────────────────────────────────────

    private _isExpired(entry: MemoryEntry): boolean {
        if (entry.ttlMs == null) return false;
        return Date.now() > entry.timestamp + entry.ttlMs;
    }

    // ── Semantic facts ────────────────────────────────────────────────────────

    /** Stores or overwrites a fact. */
    async remember(key: string, content: string, opts?: RememberOptions): Promise<void> {
        const db = await this._open();
        const entry: MemoryEntry = {
            key,
            content,
            timestamp  : Date.now(),
            seq        : ++_writeSeq,
            ttlMs      : opts?.ttlMs ?? this._defaultTtl,
            type       : opts?.type,
            tags       : opts?.tags,
            importance : opts?.importance,
        };

        const tx = db.transaction(FACTS_STORE, 'readwrite');
        await requestToPromise(tx.objectStore(FACTS_STORE).put(entry), `Failed to store fact "${key}"`, () => undefined);
        // EVM-7: keep the store bounded — evict lowest-value entries past the cap.
        if (this._maxEntries != null) await this._enforceCap();
    }

    /**
     * Evicts the lowest-value entries until at most `maxEntries` remain. Eviction
     * priority (removed first): expired → lowest importance → oldest. No-op when
     * no cap is set or the store is under it. (EVM-7)
     */
    private async _enforceCap(): Promise<void> {
        if (this._maxEntries == null) return;
        const db = await this._open();
        const tx = db.transaction(FACTS_STORE, 'readonly');
        const all = await requestToPromise(
            tx.objectStore(FACTS_STORE).getAll() as IDBRequest<MemoryEntry[]>,
            'Failed to scan facts for cap',
            (r) => r,
        );
        if (all.length <= this._maxEntries) return;

        // Ascending sort = worst first: expired (0) before live (1), then lower
        // importance, then older timestamp, then lower write sequence.
        const rank = (e: MemoryEntry): [number, number, number, number] =>
            [this._isExpired(e) ? 0 : 1, e.importance ?? 0.5, e.timestamp, e.seq ?? 0];
        const sorted = [...all].sort((a, b) => {
            const ra = rank(a), rb = rank(b);
            for (let i = 0; i < ra.length; i++) {
                if (ra[i] !== rb[i]) return ra[i]! - rb[i]!;
            }
            return 0;
        });
        const victims = sorted.slice(0, all.length - this._maxEntries);
        await Promise.all(victims.map((e) => this.forget(e.key)));
    }

    /**
     * Starts a background timer that periodically hard-deletes expired entries —
     * TTL is otherwise only evaluated lazily on read, so an idle store would keep
     * dead entries forever. Idempotent; pair with {@link stopTtlSweeper}. (EVM-7)
     */
    startTtlSweeper(intervalMs: number): void {
        if (this._sweepTimer) return;
        const timer = setInterval(() => { void this.purgeExpired(); }, intervalMs);
        // Don't keep a Node process alive just for the sweeper.
        const maybeUnref = timer as unknown as { unref?: () => void };
        /* istanbul ignore else -- unref exists in Node; absent in the browser */
        if (typeof maybeUnref.unref === 'function') {
            maybeUnref.unref();
        }
        this._sweepTimer = timer;
    }

    /** Stops the background TTL sweeper started by {@link startTtlSweeper}. */
    stopTtlSweeper(): void {
        if (this._sweepTimer) {
            clearInterval(this._sweepTimer);
            this._sweepTimer = null;
        }
    }

    /**
     * Retrieves a fact by key.
     * Returns `undefined` if the key does not exist or the entry has expired.
     */
    async recall(key: string): Promise<MemoryEntry | undefined> {
        const db = await this._open();

        const tx = db.transaction(FACTS_STORE, 'readonly');
        return requestToPromise(
            tx.objectStore(FACTS_STORE).get(key) as IDBRequest<MemoryEntry | undefined>,
            `Failed to recall fact "${key}"`,
            (entry) => (entry && this._isExpired(entry) ? undefined : entry),
        );
    }

    /** Returns all non-expired stored facts, newest first. */
    async recallAll(): Promise<MemoryEntry[]> {
        const db = await this._open();

        const tx = db.transaction(FACTS_STORE, 'readonly');
        return requestToPromise(
            tx.objectStore(FACTS_STORE).getAll() as IDBRequest<MemoryEntry[]>,
            'Failed to recall all facts',
            (entries) => entries
                .filter(e => !this._isExpired(e))
                // Newest first; break same-millisecond ties by write sequence so
                // ordering is deterministic (IndexedDB getAll() returns key order,
                // which would otherwise surface same-ms writes oldest-first).
                .sort((a, b) => b.timestamp - a.timestamp || (b.seq ?? 0) - (a.seq ?? 0)),
        );
    }

    /**
     * Returns the N most recently updated non-expired facts.
     * Equivalent to `recallAll()` truncated to `n` entries.
     */
    async recallRecent(n: number): Promise<MemoryEntry[]> {
        const all = await this.recallAll();
        return all.slice(0, n);
    }

    /**
     * Returns all non-expired entries that contain the given tag.
     */
    async recallByTag(tag: string): Promise<MemoryEntry[]> {
        const all = await this.recallAll();
        return all.filter(e => e.tags?.includes(tag) ?? false);
    }

    /**
     * Finds the top-K semantically similar entries to `query`.
     *
     * When `runtime` exposes an `embed()` method (the SSMRuntime does), similarity
     * is computed as cosine distance between SSM hidden-state embeddings — i.e. the
     * memory layer uses the very model it is attached to, and recall quality
     * improves automatically as that model is adapted/distilled. Embeddings are
     * cached per content string to avoid recomputing across calls.
     *
     * If no embedding-capable runtime is provided, or embedding fails for any
     * reason, it transparently falls back to Jaccard word-overlap similarity.
     */
    async recallSimilar(query: string, topK: number, runtime?: SSMRuntimeRef): Promise<MemoryEntry[]> {
        return (await this.recallSimilarScored(query, topK, runtime)).map(s => s.entry);
    }

    /**
     * Like {@link recallSimilar} but returns each hit's real 0..1 relevance score
     * (SSM-embedding cosine, or Jaccard on the fallback path) alongside the entry —
     * so callers can rank/threshold by TRUE similarity instead of list position.
     * {@link recallSimilar} is the score-dropping convenience wrapper over this.
     */
    async recallSimilarScored(query: string, topK: number, runtime?: SSMRuntimeRef): Promise<Array<{ entry: MemoryEntry; score: number }>> {
        const all = await this.recallAll();
        if (all.length === 0) return [];

        // ── Preferred path: SSM-embedding cosine similarity ───────────────────
        if (runtime != null && typeof runtime.embed === 'function') {
            const queryVec = await this._embedWithCache(runtime, query);
            if (queryVec) {
                // EVM-1b: sync the persistent vector store + index (embeds only the
                // delta), then query it — no per-call re-embed of every fact.
                const synced = await this._syncDense(runtime, all);
                if (synced) {
                    const hits = synced.index
                        // EVM-1: above the threshold, query the persistent HNSW (O(log N)).
                        ? synced.index.search(queryVec, topK)
                        // Below it, an exact scan over the (already-embedded) vectors.
                        : denseSearch(queryVec, synced.items, topK, this._annThreshold);
                    return hits
                        .map(h => { const entry = synced.byKey.get(h.id); return entry ? { entry, score: h.score } : null; })
                        .filter((e): e is { entry: MemoryEntry; score: number } => !!e);
                }
            }
            // Any failure falls through to the Jaccard path below.
        }

        // ── Fallback: Jaccard word-overlap similarity ─────────────────────────
        const queryTokens = new Set(tokenize(query));
        const scored = all.map(entry => {
            const entryTokens = new Set(tokenize(entry.content));
            const score       = jaccardSimilarity(queryTokens, entryTokens);
            return { entry, score };
        });

        scored.sort((a, b) => b.score - a.score);
        return scored.slice(0, topK);
    }

    /**
     * Hybrid recall: fuses dense (SSM-embedding cosine) and sparse (BM25 lexical)
     * rankings via Reciprocal Rank Fusion, then applies an MMR diversity rerank.
     *
     * This is the production RAG retrieval path — it catches both semantic matches
     * (embeddings) and exact-token matches (BM25 — identifiers, codes, rare names)
     * that cosine-only `recallSimilar` misses, and avoids returning near-duplicate
     * facts. Degrades to BM25-only when no embedding-capable runtime is available,
     * so it is always strictly at least as good as the lexical fallback.
     */
    async recallHybrid(
        query: string,
        topK: number,
        runtime?: SSMRuntimeRef,
        opts?: HybridRetrieveOptions,
    ): Promise<MemoryEntry[]> {
        return (await this.recallRanked(query, topK, runtime, opts)).hits.map(h => h.entry);
    }

    /**
     * The hybrid retrieval path with its evidence attached: each hit's fused score,
     * plus WHICH ranker produced the ordering.
     *
     * {@link recallHybrid} is the convenience wrapper that throws both away, so
     * there is exactly one implementation of the blend — and it is the shared
     * `hybridRetrieve` → `reciprocalRankFusion` primitive, never a second hand-
     * rolled mix of cosine and word overlap.
     *
     * `method` is honest rather than aspirational: `'embedding'` means the query
     * really did embed and the dense ranking really did participate in the fusion;
     * `'lexical'` means no model was available (or embedding failed) and the
     * ordering is BM25 alone. Callers surface it so a user can tell semantic
     * recall from the fallback instead of guessing.
     */
    async recallRanked(
        query: string,
        topK: number,
        runtime?: SSMRuntimeRef,
        opts?: HybridRetrieveOptions,
    ): Promise<RankedRecall> {
        const all = await this.recallAll();
        if (all.length === 0) return { hits: [], method: 'lexical' };

        // Embed candidates + query where a runtime is available; null vectors are
        // fine — hybridRetrieve degrades that candidate to BM25-only.
        const canEmbed = runtime != null && typeof runtime.embed === 'function';
        const queryVec = canEmbed ? (await this._embedWithCache(runtime, query)) ?? undefined : undefined;

        const candidates: RetrievalCandidate[] = [];
        for (const entry of all) {
            const vector = queryVec ? (await this._embedWithCache(runtime, entry.content)) ?? undefined : undefined;
            candidates.push({ id: entry.key, text: entry.content, vector });
        }

        // The dense ranking only exists when the query embedded AND at least one
        // candidate did — anything less is a lexical ordering wearing a label.
        const dense = !!queryVec && candidates.some(c => !!c.vector);
        const hits = hybridRetrieve({ text: query, vector: queryVec }, candidates, { topK, annThreshold: this._annThreshold, ...opts });
        const byKey = new Map(all.map(e => [e.key, e]));
        return {
            hits: hits
                .map(h => { const entry = byKey.get(h.id); return entry ? { entry, score: h.score } : null; })
                .filter((h): h is ScoredMemory => !!h),
            method: dense ? 'embedding' : 'lexical',
        };
    }

    /**
     * Returns a cached embedding for `text`, computing it via `runtime.embed()`
     * on a cache miss. Returns `null` (never throws) when embedding is
     * unavailable so callers can fall back to lexical similarity.
     */
    private async _embedWithCache(runtime: SSMRuntimeRef, text: string): Promise<Float32Array | null> {
        // Embeddings are model-specific: if the runtime instance changed, the cached
        // vectors are from a different model and must be dropped. (EVM-1b)
        if (runtime !== this._embedRuntime) {
            this._embedCache.clear();
            this._embedRuntime = runtime;
        }
        const cached = this._embedCache.get(text);
        if (cached) {
            // LRU touch: re-insert so this entry becomes most-recently-used. (EVM-8)
            this._embedCache.delete(text);
            this._embedCache.set(text, cached);
            return cached;
        }
        try {
            const vec = await runtime.embed(text);
            if (vec instanceof Float32Array && vec.length > 0) {
                // EVM-8: bound cache growth by evicting the LEAST-recently-used entry
                // (Map keeps insertion order; the first key is the LRU), instead of
                // dumping the whole cache on every overflow.
                if (this._embedCache.size >= this._embedCacheMax) {
                    const lru = this._embedCache.keys().next().value;
                    if (lru !== undefined) this._embedCache.delete(lru);
                }
                this._embedCache.set(text, vec);
                return vec;
            }
        } catch {
            // Embedding unavailable (no GPU, destroyed runtime, etc.) — signal fallback.
        }
        return null;
    }

    /**
     * Synchronise the persistent dense vector store + HNSW index to `all` (EVM-1b).
     * Embeds ONLY new/changed facts (keyed by `key@seq`), prunes removed ones, and
     * rebuilds the index only when the fact set actually changed — so a read-heavy
     * workload reuses both the embeddings and the index across recalls. Returns
     * `null` (caller falls back to lexical) if any embedding is unavailable, or
     * resets everything when the runtime identity changes (embeddings are
     * model-specific).
     */
    private async _syncDense(
        runtime: SSMRuntimeRef,
        all: MemoryEntry[],
    ): Promise<{ byKey: Map<string, MemoryEntry>; items: Array<{ id: string; vector: Float32Array }>; index: HnswIndex | null } | null> {
        if (runtime !== this._denseRuntime) {
            this._denseVectors.clear();
            this._denseIndex = null;
            this._denseSig = '';
            this._denseRuntime = runtime;
        }

        const vkeyOf = (e: MemoryEntry): string => `${e.key}@${e.seq ?? 0}`;
        const currentVkeys = new Set<string>();
        let sig = `${all.length}`;
        for (const e of all) { const k = vkeyOf(e); currentVkeys.add(k); sig += '|' + k; }

        const byKey = new Map<string, MemoryEntry>();
        const items: Array<{ id: string; vector: Float32Array }> = [];
        for (const entry of all) {
            const vk = vkeyOf(entry);
            let vec = this._denseVectors.get(vk);
            if (!vec) {
                const v = await this._embedWithCache(runtime, entry.content);
                if (!v) return null; // embedding failed → lexical fallback
                vec = v;
                this._denseVectors.set(vk, vec);
            }
            items.push({ id: entry.key, vector: vec });
            byKey.set(entry.key, entry);
        }

        // Drop vectors for facts that are gone or were edited (stale vkeys).
        if (this._denseVectors.size > currentVkeys.size) {
            for (const k of this._denseVectors.keys()) {
                if (!currentVkeys.has(k)) this._denseVectors.delete(k);
            }
        }

        // Maintain the ANN index only above the threshold; (re)build it only when
        // the fact set changed (no embedding involved — purely in-memory).
        if (all.length >= this._annThreshold) {
            if (sig !== this._denseSig || this._denseIndex === null) {
                const index = new HnswIndex();
                for (const it of items) index.add(it.id, it.vector);
                this._denseIndex = index;
            }
        } else {
            this._denseIndex = null;
        }
        this._denseSig = sig;
        return { byKey, items, index: this._denseIndex };
    }

    /**
     * Hard-deletes all entries whose TTL has expired.
     * @returns The number of entries deleted.
     */
    async purgeExpired(): Promise<number> {
        const db = await this._open();

        // Load all raw entries (including expired ones) to check each
        const tx = db.transaction(FACTS_STORE, 'readonly');
        const all = await requestToPromise(
            tx.objectStore(FACTS_STORE).getAll() as IDBRequest<MemoryEntry[]>,
            'Failed to scan facts for purge',
            (r) => r,
        );

        const expired = all.filter(e => this._isExpired(e));
        if (expired.length === 0) return 0;

        await Promise.all(expired.map(e => this.forget(e.key)));
        return expired.length;
    }

    /** Deletes a single fact. No-op if key does not exist. */
    async forget(key: string): Promise<void> {
        const db = await this._open();

        const tx = db.transaction(FACTS_STORE, 'readwrite');
        return requestToPromise(tx.objectStore(FACTS_STORE).delete(key), `Failed to forget fact "${key}"`, () => undefined);
    }

    /** Deletes all facts. Does not affect saved weights. */
    async clear(): Promise<void> {
        const db = await this._open();
        // Drop the persistent dense index/vectors too (EVM-1b) — they describe a
        // fact set that no longer exists.
        this._denseVectors.clear();
        this._denseIndex = null;
        this._denseSig = '';

        const tx = db.transaction(FACTS_STORE, 'readwrite');
        return requestToPromise(tx.objectStore(FACTS_STORE).clear(), 'Failed to clear facts', () => undefined);
    }

    // ── Cross-session memory merge ────────────────────────────────────────────

    /**
     * Returns all non-expired facts as a plain array for export.
     * Suitable for serialisation and import into another MemoryStore instance.
     */
    async exportAll(): Promise<MemoryEntry[]> {
        return this.recallAll();
    }

    /**
     * Imports entries from an external array.
     *
     * - `'merge'`     : only writes an entry if no existing entry with the same
     *                   key exists, or if the incoming entry has a newer timestamp.
     * - `'overwrite'` : writes all entries unconditionally.
     */
    async importAll(entries: MemoryEntry[], strategy: 'merge' | 'overwrite'): Promise<void> {
        for (const entry of entries) {
            if (strategy === 'overwrite') {
                await this._putRaw(entry);
            } else {
                const existing = await this.recall(entry.key);
                if (existing == null || entry.timestamp > existing.timestamp) {
                    await this._putRaw(entry);
                }
            }
        }
    }

    /** Writes a raw MemoryEntry directly (preserves original timestamp / metadata). */
    private async _putRaw(entry: MemoryEntry): Promise<void> {
        const db = await this._open();
        const tx = db.transaction(FACTS_STORE, 'readwrite');
        return requestToPromise(tx.objectStore(FACTS_STORE).put(entry), `Failed to import fact "${entry.key}"`, () => undefined);
    }

    // ── Weight persistence ────────────────────────────────────────────────────

    /**
     * Saves SSM weights via `runtime.save()`.
     * The weights are stored under `weightsKey` in this store's IndexedDB,
     * separate from MambaSession's own key.
     */
    async saveWeights(runtime: SaveLoadRuntime): Promise<void> {
        await runtime.save({ storage: 'indexedDB', key: this._weightsKey });
    }

    /**
     * Loads SSM weights via `runtime.load()`.
     * Returns `false` when no saved weights exist under `weightsKey`.
     */
    async loadWeights(runtime: SaveLoadRuntime): Promise<boolean> {
        return runtime.load({ key: this._weightsKey });
    }
}

// ── IndexedDB helper ──────────────────────────────────────────────────────────

/**
 * Wires an IDBRequest to a promise: resolves with `map(result)` on success,
 * rejects with a normalised MEMORY_UNAVAILABLE error on failure. Extracted so
 * the success/error wiring lives in one place instead of being duplicated across
 * every store method.
 *
 * The onerror branch fires only on IndexedDB storage faults (quota exceeded,
 * disk failure, corruption), which the in-memory fake-indexeddb used in tests
 * cannot reproduce — hence the istanbul ignore on that single line.
 */
function requestToPromise<R, T>(req: IDBRequest<R>, failMsg: string, map: (result: R) => T): Promise<T> {
    return new Promise<T>((resolve, reject) => {
        req.onsuccess = () => resolve(map(req.result));
        /* istanbul ignore next -- IDB storage-fault path; not reproducible with fake-indexeddb */
        req.onerror = () => reject(new SSMError(
            'MEMORY_UNAVAILABLE',
            `${failMsg}: ${req.error?.message ?? 'unknown'}`,
            req.error ?? undefined,
        ));
    });
}

