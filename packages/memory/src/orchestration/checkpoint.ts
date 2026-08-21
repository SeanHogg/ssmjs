/**
 * orchestration/checkpoint.ts — durable state between supersteps.
 *
 * Checkpointing is what turns three separate features into one mechanism:
 * crash recovery, human-in-the-loop approval, and time-travel debugging. All three
 * are "load the state as of superstep N and continue", so they share an
 * implementation rather than each growing their own.
 */

import type { Checkpoint, Checkpointer, GraphState } from './types.js';

export interface InMemoryCheckpointerOptions {
    /**
     * Checkpoints retained per thread, newest kept. Default 50 — bounded, because
     * an unbounded history in a long-lived process is a memory leak with a nice name.
     */
    maxPerThread?: number;
}

export class InMemoryCheckpointer<S extends GraphState = GraphState> implements Checkpointer<S> {
    private readonly _threads = new Map<string, Checkpoint<S>[]>();
    private readonly _max: number;

    constructor(opts: InMemoryCheckpointerOptions = {}) {
        this._max = Math.max(1, opts.maxPerThread ?? 50);
    }

    async save(checkpoint: Checkpoint<S>): Promise<void> {
        const history = this._threads.get(checkpoint.threadId) ?? [];
        // Same id means the same superstep was re-entered (a resumed run): replace
        // rather than append, so history stays a line and not a tangle.
        const existing = history.findIndex((c) => c.id === checkpoint.id);
        if (existing >= 0) history[existing] = checkpoint;
        else history.push(checkpoint);

        history.sort((a, b) => a.step - b.step);
        if (history.length > this._max) history.splice(0, history.length - this._max);
        this._threads.set(checkpoint.threadId, history);
    }

    async load(threadId: string, checkpointId?: string): Promise<Checkpoint<S> | undefined> {
        const history = this._threads.get(threadId);
        if (!history || history.length === 0) return undefined;
        if (checkpointId) return history.find((c) => c.id === checkpointId);
        return history[history.length - 1];
    }

    async list(threadId: string): Promise<Checkpoint<S>[]> {
        return [...(this._threads.get(threadId) ?? [])].reverse();
    }

    async clear(threadId: string): Promise<void> {
        this._threads.delete(threadId);
    }

    get threadCount(): number { return this._threads.size; }
}

/**
 * Persists checkpoints through an injected key-value store.
 *
 * The `KeyValueLike` shape is deliberately the smallest common denominator of
 * Cloudflare KV, Redis, Firestore-as-KV and a plain `Map`, so a deployment binds
 * its own storage without this package taking a dependency on any of them.
 */
export interface KeyValueLike {
    get(key: string): Promise<string | null | undefined>;
    put(key: string, value: string): Promise<void>;
    delete(key: string): Promise<void>;
}

export interface KeyValueCheckpointerOptions {
    kv: KeyValueLike;
    /** Key prefix. Default `'evermind:ckpt:'`. */
    prefix?: string;
    /** Checkpoints retained per thread. Default 50. */
    maxPerThread?: number;
}

export class KeyValueCheckpointer<S extends GraphState = GraphState> implements Checkpointer<S> {
    private readonly _kv: KeyValueLike;
    private readonly _prefix: string;
    private readonly _max: number;

    constructor(opts: KeyValueCheckpointerOptions) {
        this._kv = opts.kv;
        this._prefix = opts.prefix ?? 'evermind:ckpt:';
        this._max = Math.max(1, opts.maxPerThread ?? 50);
    }

    async save(checkpoint: Checkpoint<S>): Promise<void> {
        const history = await this.list(checkpoint.threadId);
        const without = history.filter((c) => c.id !== checkpoint.id);
        const next = [...without, checkpoint]
            .sort((a, b) => a.step - b.step)
            .slice(-this._max);
        // One key per thread rather than one per checkpoint: a thread's history is
        // always read whole, and a single round trip beats a list-then-fan-out.
        await this._kv.put(this._key(checkpoint.threadId), JSON.stringify(next));
    }

    async load(threadId: string, checkpointId?: string): Promise<Checkpoint<S> | undefined> {
        const history = await this.list(threadId);
        if (history.length === 0) return undefined;
        return checkpointId ? history.find((c) => c.id === checkpointId) : history[0];
    }

    /** Newest first. */
    async list(threadId: string): Promise<Checkpoint<S>[]> {
        const raw = await this._kv.get(this._key(threadId));
        if (!raw) return [];
        try {
            const parsed = JSON.parse(raw) as Checkpoint<S>[];
            return Array.isArray(parsed) ? [...parsed].sort((a, b) => b.step - a.step) : [];
        } catch {
            // A corrupt entry must not wedge the thread forever; treat it as empty
            // so the next run rebuilds rather than throwing on every resume.
            return [];
        }
    }

    async clear(threadId: string): Promise<void> {
        await this._kv.delete(this._key(threadId));
    }

    private _key(threadId: string): string {
        return `${this._prefix}${threadId}`;
    }
}
