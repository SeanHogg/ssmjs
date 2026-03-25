/**
 * MemoryStore – persistent key-value fact store and weight checkpoint helper.
 *
 * Uses IndexedDB with a dedicated 'ssmjs' database (separate from mambakit's
 * 'mambakit' DB) containing two object stores:
 *   - 'facts'   : MemoryEntry records keyed by the fact key string
 *   - 'weights' : a single ArrayBuffer keyed by `weightsKey`
 *
 * MemoryStore does not import mambakit directly — weight save/load is
 * delegated to the SSMRuntime passed to saveWeights/loadWeights.
 */

import { SSMError } from '../errors/SSMError.js';

export type FactType = 'text' | 'json' | 'number' | 'boolean';

export interface MemoryEntry {
    key        : string;
    content    : string;
    timestamp  : number;
    /** Optional time-to-live in milliseconds. */
    ttlMs?     : number;
    /** Semantic type of the stored value. Default: 'text'. */
    type?      : FactType;
    /** Tags for grouping and filtering. */
    tags?      : string[];
    /** Importance weight in the range 0–1. Default: 0.5. */
    importance?: number;
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
}

const FACTS_STORE   = 'facts';
const WEIGHTS_STORE = 'weights';
const DB_VERSION    = 1;

// Minimal interface to avoid importing SSMRuntime (circular dep)
interface SaveLoadRuntime {
    save(opts?: { storage: 'indexedDB'; key: string }): Promise<void>;
    load(opts?: { key: string }): Promise<boolean>;
}

// Forward-declared to avoid circular dependency at import time
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type SSMRuntimeRef = any;

export class MemoryStore {
    private readonly _dbName     : string;
    private readonly _weightsKey : string;
    private readonly _idb        : IDBFactory | undefined;
    private readonly _defaultTtl : number | undefined;
    private _db: IDBDatabase | null = null;

    constructor(opts: MemoryStoreOptions = {}) {
        this._dbName     = opts.dbName     ?? 'ssmjs';
        this._weightsKey = opts.weightsKey ?? 'ssmjs-weights';
        this._idb        = opts.idbFactory;
        this._defaultTtl = opts.defaultTtlMs;
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
            ttlMs      : opts?.ttlMs ?? this._defaultTtl,
            type       : opts?.type,
            tags       : opts?.tags,
            importance : opts?.importance,
        };

        return new Promise((resolve, reject) => {
            const tx  = db.transaction(FACTS_STORE, 'readwrite');
            const req = tx.objectStore(FACTS_STORE).put(entry);
            req.onsuccess = () => resolve();
            req.onerror   = () => reject(new SSMError(
                'MEMORY_UNAVAILABLE',
                `Failed to store fact "${key}": ${req.error?.message ?? 'unknown'}`,
                req.error,
            ));
        });
    }

    /**
     * Retrieves a fact by key.
     * Returns `undefined` if the key does not exist or the entry has expired.
     */
    async recall(key: string): Promise<MemoryEntry | undefined> {
        const db = await this._open();

        return new Promise((resolve, reject) => {
            const tx  = db.transaction(FACTS_STORE, 'readonly');
            const req = tx.objectStore(FACTS_STORE).get(key);
            req.onsuccess = () => {
                const entry = req.result as MemoryEntry | undefined;
                if (entry && this._isExpired(entry)) {
                    resolve(undefined);
                } else {
                    resolve(entry);
                }
            };
            req.onerror   = () => reject(new SSMError(
                'MEMORY_UNAVAILABLE',
                `Failed to recall fact "${key}": ${req.error?.message ?? 'unknown'}`,
                req.error,
            ));
        });
    }

    /** Returns all non-expired stored facts, newest first. */
    async recallAll(): Promise<MemoryEntry[]> {
        const db = await this._open();

        return new Promise((resolve, reject) => {
            const tx  = db.transaction(FACTS_STORE, 'readonly');
            const req = tx.objectStore(FACTS_STORE).getAll();
            req.onsuccess = () => {
                const entries = (req.result as MemoryEntry[])
                    .filter(e => !this._isExpired(e))
                    .sort((a, b) => b.timestamp - a.timestamp);
                resolve(entries);
            };
            req.onerror = () => reject(new SSMError(
                'MEMORY_UNAVAILABLE',
                `Failed to recall all facts: ${req.error?.message ?? 'unknown'}`,
            ));
        });
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
     * Finds the top-K semantically similar entries to `query` using Jaccard
     * word-overlap similarity.  The `runtime` parameter is reserved for future
     * SSM-hidden-state embeddings — the current implementation is CPU-only.
     */
    async recallSimilar(query: string, topK: number, _runtime: SSMRuntimeRef): Promise<MemoryEntry[]> {
        const all = await this.recallAll();
        if (all.length === 0) return [];

        const queryTokens = new Set(tokenize(query));

        const scored = all.map(entry => {
            const entryTokens = new Set(tokenize(entry.content));
            const score       = jaccardSimilarity(queryTokens, entryTokens);
            return { entry, score };
        });

        scored.sort((a, b) => b.score - a.score);
        return scored.slice(0, topK).map(s => s.entry);
    }

    /**
     * Hard-deletes all entries whose TTL has expired.
     * @returns The number of entries deleted.
     */
    async purgeExpired(): Promise<number> {
        const db = await this._open();

        // Load all raw entries (including expired ones) to check each
        const all: MemoryEntry[] = await new Promise((resolve, reject) => {
            const tx  = db.transaction(FACTS_STORE, 'readonly');
            const req = tx.objectStore(FACTS_STORE).getAll();
            req.onsuccess = () => resolve(req.result as MemoryEntry[]);
            req.onerror   = () => reject(new SSMError(
                'MEMORY_UNAVAILABLE',
                `Failed to scan facts for purge: ${req.error?.message ?? 'unknown'}`,
                req.error,
            ));
        });

        const expired = all.filter(e => this._isExpired(e));
        if (expired.length === 0) return 0;

        await Promise.all(expired.map(e => this.forget(e.key)));
        return expired.length;
    }

    /** Deletes a single fact. No-op if key does not exist. */
    async forget(key: string): Promise<void> {
        const db = await this._open();

        return new Promise((resolve, reject) => {
            const tx  = db.transaction(FACTS_STORE, 'readwrite');
            const req = tx.objectStore(FACTS_STORE).delete(key);
            req.onsuccess = () => resolve();
            req.onerror   = () => reject(new SSMError(
                'MEMORY_UNAVAILABLE',
                `Failed to forget fact "${key}": ${req.error?.message ?? 'unknown'}`,
                req.error,
            ));
        });
    }

    /** Deletes all facts. Does not affect saved weights. */
    async clear(): Promise<void> {
        const db = await this._open();

        return new Promise((resolve, reject) => {
            const tx  = db.transaction(FACTS_STORE, 'readwrite');
            const req = tx.objectStore(FACTS_STORE).clear();
            req.onsuccess = () => resolve();
            req.onerror   = () => reject(new SSMError(
                'MEMORY_UNAVAILABLE',
                `Failed to clear facts: ${req.error?.message ?? 'unknown'}`,
                req.error,
            ));
        });
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
        return new Promise((resolve, reject) => {
            const tx  = db.transaction(FACTS_STORE, 'readwrite');
            const req = tx.objectStore(FACTS_STORE).put(entry);
            req.onsuccess = () => resolve();
            req.onerror   = () => reject(new SSMError(
                'MEMORY_UNAVAILABLE',
                `Failed to import fact "${entry.key}": ${req.error?.message ?? 'unknown'}`,
                req.error,
            ));
        });
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

// ── Text similarity helpers ───────────────────────────────────────────────────

/** Splits text into lowercase word tokens, removing punctuation. */
function tokenize(text: string): string[] {
    return text.toLowerCase().split(/[\s\W]+/).filter(Boolean);
}

/** Jaccard similarity between two token sets: |A ∩ B| / |A ∪ B|. */
function jaccardSimilarity(a: Set<string>, b: Set<string>): number {
    if (a.size === 0 && b.size === 0) return 1;
    let intersection = 0;
    for (const token of a) {
        if (b.has(token)) intersection++;
    }
    const union = a.size + b.size - intersection;
    return union === 0 ? 0 : intersection / union;
}
