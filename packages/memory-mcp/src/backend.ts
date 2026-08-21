/**
 * MemoryBackend — the storage seam every transport is written against.
 *
 * The MCP tools (src/tools.ts) and all three transports (SDK / stdio / HTTP)
 * depend ONLY on this interface, never on a concrete store. Ship the local
 * `MemoryStoreBackend` (IndexedDB via @seanhogg/builderforce-memory) today; drop in a
 * networked builderforce.ai adapter later with zero changes to the tools or
 * transports.
 */

/** A single recalled memory, normalised across backends. */
export interface RecallHit {
    /** Stable identifier for the memory. */
    key: string;
    /** The stored value. */
    content: string;
    /**
     * Optional relevance score (higher = closer). Semantic backends that expose
     * ranking can populate this; the local MemoryStore ranks but does not surface
     * a score, so it is left undefined and recall order carries the signal.
     */
    score?: number;
    /** Tags for grouping/filtering. */
    tags?: string[];
    /** Importance weight 0–1. */
    importance?: number;
    /** Unix-ms write time. */
    timestamp?: number;
}

/**
 * Which ranker produced a recall ordering.
 *
 * Same vocabulary the gateway's Evermind recall/validate result uses, so a host
 * renders one chip ("Semantic recall" / "Lexical recall (fallback)") no matter
 * which layer answered.
 */
export type RecallMethod = "embedding" | "lexical";

/** A recall ordering plus the ranker that produced it. */
export interface RankedRecall {
    hits: RecallHit[];
    method: RecallMethod;
}

/** Arguments for writing a memory. */
export interface RememberInput {
    key: string;
    content: string;
    tags?: string[];
    /** Importance weight 0–1. */
    importance?: number;
    /** Time-to-live in milliseconds. */
    ttlMs?: number;
}

/**
 * The minimal capability surface the MCP layer needs. Deliberately small —
 * the token-saving design exposes recall-on-demand, not a "dump everything"
 * call, so this interface has no `recallAll`.
 */
export interface MemoryBackend {
    /**
     * Semantic top-K recall. Backends with an embedding model (the SSM
     * runtime) should use it; lexical fallback is acceptable. `topK` is already
     * clamped by the caller — the backend may return fewer, never more.
     */
    recall(query: string, topK: number): Promise<RecallHit[]>;

    /**
     * {@link recall} with the ranker named. Optional; callers MUST fall back to
     * plain `recall` (and report nothing) when it is absent.
     *
     * It exists because "did this ordering come from the SSM embedding or from
     * word overlap?" is the difference between semantic recall and a fallback, and
     * an answer that silently degrades to lexical is indistinguishable from one
     * that did not. A backend that knows which path it took says so here.
     */
    recallRanked?(query: string, topK: number): Promise<RankedRecall>;

    /** Exact lookup by key. Returns undefined when absent or expired. */
    get(key: string): Promise<RecallHit | undefined>;

    /** All non-expired entries carrying `tag`, capped to `limit`. */
    recallByTag(tag: string, limit: number): Promise<RecallHit[]>;

    /** Store or overwrite a memory. Optional — read-only backends omit it. */
    remember?(input: RememberInput): Promise<void>;

    /**
     * Store several memories as ONE durable unit. Optional; callers MUST fall
     * back to a `remember` loop when it is absent.
     *
     * It exists for the batch writer (`memory_compact`): a durable backend that
     * flushes after every write would otherwise re-serialise the entire store
     * once per compacted key, which is quadratic in the size of the store.
     */
    rememberMany?(inputs: RememberInput[]): Promise<void>;

    /** Delete a memory by key. Optional — read-only backends omit it. */
    forget?(key: string): Promise<void>;
}
