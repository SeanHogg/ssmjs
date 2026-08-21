/**
 * MemoryStoreBackend — local adapter mapping @seanhogg/builderforce-memory's MemoryStore
 * onto the MemoryBackend seam.
 *
 * Recall quality: when a text embedder is supplied it is forwarded to the store's
 * `recallRanked`, so the ordering is the shared hybrid retrieval — the SSM
 * embedding's cosine ranking FUSED with BM25 by Reciprocal Rank Fusion — and it
 * sharpens as the model is adapted/distilled. With no embedder the same call
 * degrades to the lexical ranking alone. Either way the backend reports which of
 * the two produced the ordering (`method`), so a silent degrade is visible.
 *
 * The embedder does NOT have to be a GPU runtime: `EvermindTextEmbedder` from the
 * engine is a pure-CPU `EvermindLM`, which is what lets the headless stdio server
 * rank recall semantically (see ./evermind-embedder.ts).
 */

import type { MemoryBackend, RankedRecall, RecallHit, RememberInput } from "../backend.js";

// Structural views of the @seanhogg/builderforce-memory surface we use, so this package
// type-checks without a hard dependency on the runtime package.
interface MemoryEntryLike {
    key: string;
    content: string;
    timestamp?: number;
    tags?: string[];
    importance?: number;
}

interface MemoryStoreLike {
    remember(key: string, content: string, opts?: { ttlMs?: number; tags?: string[]; importance?: number }): Promise<void>;
    recall(key: string): Promise<MemoryEntryLike | undefined>;
    recallAll(): Promise<MemoryEntryLike[]>;
    recallByTag(tag: string): Promise<MemoryEntryLike[]>;
    /** Hybrid (dense⊕BM25) recall with each hit's fused score and the ranker used. */
    recallRanked(
        query: string,
        topK: number,
        runtime?: unknown,
    ): Promise<{ hits: Array<{ entry: MemoryEntryLike; score: number }>; method: "embedding" | "lexical" }>;
    forget(key: string): Promise<void>;
}

function toHit(e: MemoryEntryLike, score?: number): RecallHit {
    return {
        key: e.key,
        content: e.content,
        ...(score === undefined ? {} : { score }),
        tags: e.tags,
        importance: e.importance,
        timestamp: e.timestamp,
    };
}

export class MemoryStoreBackend implements MemoryBackend {
    constructor(
        private readonly store: MemoryStoreLike,
        /** Optional SSMRuntime; enables embedding-based recall when present. */
        private readonly runtime?: unknown,
    ) {}

    /** Plain recall — {@link recallRanked} with the ranker's name dropped. ONE path. */
    async recall(query: string, topK: number): Promise<RecallHit[]> {
        return (await this.recallRanked(query, topK)).hits;
    }

    async recallRanked(query: string, topK: number): Promise<RankedRecall> {
        const ranked = await this.store.recallRanked(query, topK, this.runtime);
        return { hits: ranked.hits.map((h) => toHit(h.entry, h.score)), method: ranked.method };
    }

    async get(key: string): Promise<RecallHit | undefined> {
        const e = await this.store.recall(key);
        return e ? toHit(e) : undefined;
    }

    async recallByTag(tag: string, limit: number): Promise<RecallHit[]> {
        const entries = await this.store.recallByTag(tag);
        return entries.slice(0, limit).map(toHit);
    }

    async remember(input: RememberInput): Promise<void> {
        await this.store.remember(input.key, input.content, {
            ttlMs: input.ttlMs,
            tags: input.tags,
            importance: input.importance,
        });
    }

    /** The store has no batch write; sequencing here keeps ONE definition of a write. */
    async rememberMany(inputs: RememberInput[]): Promise<void> {
        for (const input of inputs) await this.remember(input);
    }

    async forget(key: string): Promise<void> {
        await this.store.forget(key);
    }
}

/** Options for {@link createLocalMemoryStoreBackend}. */
export interface LocalBackendOptions {
    /** IndexedDB database name. Defaults to MemoryStore's own default ('ssmjs'). */
    dbName?: string;
    /**
     * Optional text embedder — anything with `embed(text): Promise<Float32Array>`.
     * Omit for lexical-only recall. Pass the agent-runtime's
     * `ssmMemoryService.runtime` to reuse an already-loaded GPU Evermind, or an
     * `EvermindTextEmbedder` (pure CPU) via {@link createEvermindEmbedder} on a
     * headless host.
     */
    runtime?: unknown;
    /**
     * Absolute path to a JSON file that mirrors the store to disk. Without it the
     * store is purely in-memory (fake-indexeddb) and evaporates when the process
     * exits — fine for a long-lived server, fatal for a per-session subprocess
     * (e.g. an MCP stdio client that respawns the server each launch).
     *
     * When set, the store is hydrated from the file on creation and re-snapshotted
     * after every remember/forget, giving durable cross-process memory. TTLs are
     * dropped on persist: the snapshot is the durable long-term tier.
     *
     * The file is also WATCHED: it is a shared artifact (other processes compact it
     * in place), so the store re-hydrates from disk whenever it changed underneath —
     * see {@link DiskPersistedBackend} for the loop guard.
     */
    persistFile?: string;
}

/** The on-disk snapshot shape — a flat array of durable entries. */
interface SnapshotEntry {
    key: string;
    content: string;
    tags?: string[];
    importance?: number;
}

type FsLike = {
    readFileSync(path: string, enc: "utf8"): string;
    writeFileSync(path: string, data: string): void;
    mkdirSync(path: string, opts: { recursive: boolean }): void;
    existsSync(path: string): boolean;
    statSync(path: string): { mtimeMs: number; size: number };
};
type PathLike = { dirname(p: string): string };

/** Identity of the snapshot file as this process last left it. */
interface FileStamp {
    mtimeMs: number;
    size: number;
    /** Digest of the exact bytes — settles the case where a file is touched but unchanged. */
    hash: string;
}

/**
 * FNV-1a over the snapshot text. Non-cryptographic on purpose: this only has to
 * separate "someone rewrote the file" from "the mtime moved but the bytes are
 * ours", and node:crypto is deliberately not imported here (this module must stay
 * bundleable for the browser, where the disk path is never taken).
 */
function contentHash(text: string): string {
    let h = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) {
        h ^= text.charCodeAt(i);
        h = Math.imul(h, 0x01000193);
    }
    return (h >>> 0).toString(16);
}

/** Parse a snapshot's text into its durable entries, or null when unusable. */
function parseSnapshot(text: string): SnapshotEntry[] | null {
    let parsed: unknown;
    try {
        parsed = JSON.parse(text);
    } catch {
        return null;
    }
    if (!Array.isArray(parsed)) return null;
    return (parsed as SnapshotEntry[]).filter(
        (raw) => !!raw && typeof raw.key === "string" && typeof raw.content === "string",
    );
}

/**
 * Wraps a MemoryStoreBackend so every write is mirrored to a JSON file, and
 * hydrates that file back into the store on boot. This is what turns a respawned
 * stdio subprocess into a persistent memory: the store itself is in-memory, the
 * file is the source of truth across process lifetimes.
 *
 * It also WATCHES the file for external edits. The snapshot is a shared artifact —
 * the BuilderForce VS Code extension compacts absorbed entries by rewriting it
 * directly, and a second server instance may be mirroring the same path. Without a
 * watch this server, still holding the pre-edit bodies, would silently re-snapshot
 * over that work on its very next remember/forget; the edit would only stick once
 * the client respawned the subprocess. So every operation first calls
 * {@link DiskPersistedBackend.ensureFresh}, which re-hydrates the store from disk
 * when — and only when — the file changed underneath us.
 *
 * The loop guard is the {@link FileStamp} recorded immediately AFTER each of our own
 * writes: the next check stats the file, sees the same mtime+size, and returns
 * without reading a byte. A stat per call is the entire steady-state cost.
 */
class DiskPersistedBackend implements MemoryBackend {
    /** How the file looked when this process last wrote or read it. */
    private stamp: FileStamp | null = null;

    constructor(
        private readonly inner: MemoryStoreBackend,
        private readonly store: MemoryStoreLike,
        private readonly file: string,
        private readonly fs: FsLike,
    ) {}

    async recall(query: string, topK: number): Promise<RecallHit[]> {
        await this.ensureFresh();
        return this.inner.recall(query, topK);
    }
    async recallRanked(query: string, topK: number): Promise<RankedRecall> {
        await this.ensureFresh();
        return this.inner.recallRanked(query, topK);
    }
    async get(key: string): Promise<RecallHit | undefined> {
        await this.ensureFresh();
        return this.inner.get(key);
    }
    async recallByTag(tag: string, limit: number): Promise<RecallHit[]> {
        await this.ensureFresh();
        return this.inner.recallByTag(tag, limit);
    }

    async remember(input: RememberInput): Promise<void> {
        await this.ensureFresh();
        await this.inner.remember(input);
        await this.snapshot();
    }

    /** Batch write — ONE snapshot for the whole set, so compacting N keys is not N full rewrites. */
    async rememberMany(inputs: RememberInput[]): Promise<void> {
        await this.ensureFresh();
        await this.inner.rememberMany(inputs);
        await this.snapshot();
    }

    async forget(key: string): Promise<void> {
        await this.ensureFresh();
        await this.inner.forget(key);
        await this.snapshot();
    }

    /**
     * Re-read the snapshot into the store when the file changed since we last
     * touched it. Also the BOOT hydration: with no stamp yet, an existing file is
     * always read in — one routine, so disk and memory can never diverge by taking
     * two different paths.
     */
    async ensureFresh(): Promise<void> {
        if (!this.fs.existsSync(this.file)) return;

        let st: { mtimeMs: number; size: number };
        try {
            st = this.fs.statSync(this.file);
        } catch {
            return;
        }
        // Fast path: byte-for-byte what we last wrote. No read, no parse, no writes.
        if (this.stamp && st.mtimeMs === this.stamp.mtimeMs && st.size === this.stamp.size) return;

        let text: string;
        try {
            text = this.fs.readFileSync(this.file, "utf8");
        } catch {
            return;
        }
        const hash = contentHash(text);
        // Touched (mtime moved) but identical content — re-stamp, don't re-hydrate.
        if (this.stamp && hash === this.stamp.hash) {
            this.stamp = { mtimeMs: st.mtimeMs, size: st.size, hash };
            return;
        }

        const entries = parseSnapshot(text);
        // Corrupt/partial snapshot (a half-written file, say): keep what we have
        // rather than wiping the live store, and leave the stamp alone so the next
        // call re-checks.
        if (!entries) return;

        await this.replaceStore(entries);
        this.stamp = { mtimeMs: st.mtimeMs, size: st.size, hash };
    }

    /**
     * Make the in-memory store equal the snapshot: drop keys the file no longer has,
     * then replay every entry (a re-remember overwrites, so a body shortened to a
     * stub on disk becomes the stub in memory). Replayed as durable — TTLs are
     * dropped on persist; the snapshot is the long-term tier.
     */
    private async replaceStore(entries: SnapshotEntry[]): Promise<void> {
        const wanted = new Set(entries.map((e) => e.key));
        for (const existing of await this.store.recallAll()) {
            if (!wanted.has(existing.key)) await this.store.forget(existing.key);
        }
        for (const raw of entries) {
            await this.store.remember(raw.key, raw.content, { tags: raw.tags, importance: raw.importance });
        }
    }

    private async snapshot(): Promise<void> {
        const entries = await this.store.recallAll();
        const out: SnapshotEntry[] = entries.map((e) => ({
            key: e.key,
            content: e.content,
            tags: e.tags,
            importance: e.importance,
        }));
        const json = JSON.stringify(out, null, 2);
        this.fs.writeFileSync(this.file, json);
        // Stamp OUR write immediately — this is what stops the watch above from
        // treating our own output as an external edit (and looping).
        try {
            const st = this.fs.statSync(this.file);
            this.stamp = { mtimeMs: st.mtimeMs, size: st.size, hash: contentHash(json) };
        } catch {
            this.stamp = null;
        }
    }
}

/**
 * Builds a MemoryStoreBackend over a fresh MemoryStore, wiring fake-indexeddb in
 * Node exactly as SsmMemoryService does. @seanhogg/builderforce-memory and fake-indexeddb
 * are imported indirectly so they remain optional peers — a consumer that only
 * wants a custom backend (or the HTTP thin-client) never has to install them.
 */
export async function createLocalMemoryStoreBackend(opts: LocalBackendOptions = {}): Promise<MemoryBackend> {
    // Indirect import prevents the bundler/tsc from resolving optional peers.
    const _import = (m: string): Promise<unknown> =>
        // eslint-disable-next-line @typescript-eslint/no-implied-eval, no-new-func
        new Function("m", "return import(m)")(m) as Promise<unknown>;

    const memoryMod = (await _import("@seanhogg/builderforce-memory")) as { MemoryStore: new (o: unknown) => MemoryStoreLike };
    const { MemoryStore } = memoryMod;

    // IndexedDB shim for Node. In the browser the global is used automatically.
    let idbFactory: unknown;
    try {
        const fake = (await _import("fake-indexeddb")) as { IDBFactory: new () => unknown };
        idbFactory = new fake.IDBFactory();
    } catch {
        // Browser or a host that provides global indexedDB — MemoryStore handles it.
    }

    const store = new MemoryStore({ idbFactory, dbName: opts.dbName });
    const backend = new MemoryStoreBackend(store, opts.runtime);

    if (!opts.persistFile) return backend;

    // Disk-mirror requested. node:fs/path are loaded indirectly so a browser
    // bundle of this module never statically pulls in Node builtins.
    const fs = (await _import("node:fs")) as FsLike;
    const path = (await _import("node:path")) as PathLike;
    const dir = path.dirname(opts.persistFile);
    if (dir && !fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

    const persisted = new DiskPersistedBackend(backend, store, opts.persistFile, fs);
    // Boot hydration IS the freshness check with no stamp yet — see ensureFresh().
    await persisted.ensureFresh();
    return persisted;
}
