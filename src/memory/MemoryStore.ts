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

export interface MemoryEntry {
    key      : string;
    content  : string;
    timestamp: number;
}

export interface MemoryStoreOptions {
    /** IndexedDB database name. Default: 'ssmjs'. */
    dbName?     : string;
    /** Key used for weight storage within the 'weights' object store. Default: 'ssmjs-weights'. */
    weightsKey? : string;
}

const FACTS_STORE   = 'facts';
const WEIGHTS_STORE = 'weights';
const DB_VERSION    = 1;

// Minimal interface to avoid importing SSMRuntime (circular dep)
interface SaveLoadRuntime {
    save(opts?: { storage: 'indexedDB'; key: string }): Promise<void>;
    load(opts?: { key: string }): Promise<boolean>;
}

export class MemoryStore {
    private readonly _dbName    : string;
    private readonly _weightsKey: string;
    private _db: IDBDatabase | null = null;

    constructor(opts: MemoryStoreOptions = {}) {
        this._dbName     = opts.dbName     ?? 'ssmjs';
        this._weightsKey = opts.weightsKey ?? 'ssmjs-weights';
    }

    // ── Internal DB open ──────────────────────────────────────────────────────

    private _open(): Promise<IDBDatabase> {
        if (this._db) return Promise.resolve(this._db);

        return new Promise((resolve, reject) => {
            if (typeof indexedDB === 'undefined') {
                reject(new SSMError(
                    'MEMORY_UNAVAILABLE',
                    'IndexedDB is not available in this environment.',
                ));
                return;
            }

            const req = indexedDB.open(this._dbName, DB_VERSION);

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

    // ── Semantic facts ────────────────────────────────────────────────────────

    /** Stores or overwrites a fact. */
    async remember(key: string, content: string): Promise<void> {
        const db    = await this._open();
        const entry : MemoryEntry = { key, content, timestamp: Date.now() };

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
     * Returns `undefined` if the key does not exist.
     */
    async recall(key: string): Promise<MemoryEntry | undefined> {
        const db = await this._open();

        return new Promise((resolve, reject) => {
            const tx  = db.transaction(FACTS_STORE, 'readonly');
            const req = tx.objectStore(FACTS_STORE).get(key);
            req.onsuccess = () => resolve(req.result as MemoryEntry | undefined);
            req.onerror   = () => reject(new SSMError(
                'MEMORY_UNAVAILABLE',
                `Failed to recall fact "${key}": ${req.error?.message ?? 'unknown'}`,
                req.error,
            ));
        });
    }

    /** Returns all stored facts, newest first. */
    async recallAll(): Promise<MemoryEntry[]> {
        const db = await this._open();

        return new Promise((resolve, reject) => {
            const tx  = db.transaction(FACTS_STORE, 'readonly');
            const req = tx.objectStore(FACTS_STORE).getAll();
            req.onsuccess = () => {
                const entries = (req.result as MemoryEntry[])
                    .sort((a, b) => b.timestamp - a.timestamp);
                resolve(entries);
            };
            req.onerror = () => reject(new SSMError(
                'MEMORY_UNAVAILABLE',
                `Failed to recall all facts: ${req.error?.message ?? 'unknown'}`,
                req.error,
            ));
        });
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
