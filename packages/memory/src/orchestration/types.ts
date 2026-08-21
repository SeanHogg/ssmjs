/**
 * orchestration/types.ts — the multi-agent vocabulary.
 *
 * Multi-agent systems fail in production for reasons that are all state problems,
 * not prompting problems: two agents write the same key and one silently wins; a
 * loop never terminates; a run dies at minute nine of a twelve-minute job with no
 * way to resume; nobody can say which agent produced the claim in the final
 * answer. So state is modelled explicitly here — as CHANNELS with declared merge
 * semantics — rather than left as an object everyone mutates.
 *
 * The execution model is supersteps (Pregel): a frontier of nodes runs, their
 * updates are merged through the channel reducers, and the next frontier is
 * computed from the edges. Concurrent fan-out is therefore the default rather than
 * a special case, and "two agents wrote the same channel" has a defined answer
 * instead of a race.
 */

import type { Span } from '../telemetry/Tracer.js';

/** Reserved node names: the graph's entry and terminal sentinels. */
export const START = '__start__';
export const END = '__end__';

export type GraphState = Record<string, unknown>;

/**
 * Merges a channel's existing value with an update produced this superstep.
 * Called once per writing node, in node-name order, so a fan-out merge is
 * deterministic and reproducible from a checkpoint.
 */
export type ChannelReducer<V = unknown> = (current: V | undefined, update: V) => V;

export interface ChannelSpec<V = unknown> {
    /** Omit for last-write-wins. */
    reducer?: ChannelReducer<V>;
    /** Initial value factory, used when the channel is absent from the input. */
    default?: () => V;
}

export type ChannelSpecs<S extends GraphState> = {
    [K in keyof S]?: ChannelSpec<S[K]>;
};

export interface NodeContext<S extends GraphState> {
    /** Superstep number, from 0. */
    step: number;
    /** The node's own span — pass it as `parent` so model calls nest correctly. */
    span?: Span;
    /** Emits a custom event into the run's stream (progress, intermediate output). */
    emit(event: { type: string; data?: unknown }): void;
    /** Cooperative cancellation — long nodes should check it between calls. */
    signal?: AbortSignal;
    /** Read-only view of the merged state at the start of this superstep. */
    state: Readonly<S>;
}

/**
 * A node returns a PARTIAL state update, never a whole state. Returning the whole
 * object is how a fan-out silently discards a sibling's work.
 */
export type NodeFn<S extends GraphState> = (
    state: Readonly<S>,
    ctx: NodeContext<S>,
) => Promise<Partial<S> | void> | Partial<S> | void;

/** Chooses the next node(s). Return {@link END} to finish this branch. */
export type EdgeCondition<S extends GraphState> = (
    state: Readonly<S>,
    ctx: { step: number },
) => string | string[] | Promise<string | string[]>;

export type GraphEvent<S extends GraphState> =
    | { type: 'step-start'; step: number; frontier: string[] }
    | { type: 'node-start'; step: number; node: string }
    | { type: 'node-end'; step: number; node: string; update: Partial<S> }
    | { type: 'node-error'; step: number; node: string; error: string }
    | { type: 'custom'; step: number; node: string; event: { type: string; data?: unknown } }
    | { type: 'step-end'; step: number; state: S }
    | { type: 'interrupt'; step: number; before: string[]; state: S }
    | { type: 'end'; step: number; state: S; reason: GraphEndReason };

export type GraphEndReason =
    /** Every branch reached END. */
    | 'complete'
    /** The recursion limit tripped — a cycle that did not converge. */
    | 'recursion-limit'
    /** Paused at an `interruptBefore` node, awaiting a human. */
    | 'interrupted'
    /** A node threw and the graph is configured to stop on error. */
    | 'error';

export interface GraphRunResult<S extends GraphState> {
    state: S;
    reason: GraphEndReason;
    steps: number;
    /** Node names executed, in order — the audit trail of a run. */
    path: string[];
    traceId?: string;
    /** Present when `reason` is `'interrupted'`; resume by passing it back. */
    checkpointId?: string;
    error?: string;
}

/** One durable snapshot of a run between supersteps. */
export interface Checkpoint<S extends GraphState> {
    id: string;
    threadId: string;
    step: number;
    state: S;
    /** Nodes queued to run next — without this, resume cannot know where it was. */
    frontier: string[];
    path: string[];
    createdAt: number;
}

/**
 * Durable state between supersteps. An in-process implementation ships here; a
 * production deployment points this at Durable Objects, Firestore or Postgres, at
 * which point a twelve-minute multi-agent run survives a deploy.
 */
export interface Checkpointer<S extends GraphState = GraphState> {
    save(checkpoint: Checkpoint<S>): Promise<void>;
    /** Latest checkpoint for a thread, or a specific one by id. */
    load(threadId: string, checkpointId?: string): Promise<Checkpoint<S> | undefined>;
    /** History, newest first — the basis for time-travel debugging. */
    list(threadId: string): Promise<Checkpoint<S>[]>;
    clear(threadId: string): Promise<void>;
}
