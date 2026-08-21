/**
 * orchestration/AgentGraph.ts — a stateful multi-agent graph.
 *
 * Execution is a superstep loop: run the current frontier of nodes concurrently,
 * merge their partial updates through the channel reducers, then compute the next
 * frontier from the edges. This is the model LangGraph popularised, and it is the
 * right one for agents because the three hard parts of multi-agent orchestration —
 * concurrent fan-out, deterministic merge, and resumability — all fall out of it:
 *
 *   • Fan-out is just a frontier with more than one node.
 *   • A merge conflict has a declared answer (the channel's reducer), not a race.
 *   • The state between two supersteps is a complete description of the run, so a
 *     checkpoint written there can resume it exactly — which is what makes
 *     human-in-the-loop approval and crash recovery the same mechanism.
 *
 * Everything is traced: one span per run, one per node, so a model call inside a
 * node nests under the node that made it and cost rolls up per agent.
 */

import type { Span, Tracer } from '../telemetry/Tracer.js';
import {
    END,
    START,
    type ChannelSpecs,
    type Checkpointer,
    type EdgeCondition,
    type GraphEndReason,
    type GraphEvent,
    type GraphRunResult,
    type GraphState,
    type NodeContext,
    type NodeFn,
} from './types.js';

interface NodeEntry<S extends GraphState> {
    name: string;
    fn: NodeFn<S>;
    /** Retries on throw before the node is treated as failed. Default 0. */
    retries: number;
}

interface EdgeEntry<S extends GraphState> {
    from: string;
    /** Static target, or a condition that computes targets from state. */
    to?: string;
    condition?: EdgeCondition<S>;
    /** Maps a condition's return value onto node names. */
    mapping?: Record<string, string>;
}

export interface AgentGraphOptions<S extends GraphState> {
    /** Merge semantics per state key. Keys without a spec are last-write-wins. */
    channels?: ChannelSpecs<S>;
    /**
     * Maximum supersteps before the run is stopped. Default 25. A cyclic graph
     * (which is the point of ReAct and reflection) needs a hard stop, or a model
     * that will not converge becomes an unbounded bill.
     */
    recursionLimit?: number;
    /** Node names to pause BEFORE — the human-in-the-loop approval gate. */
    interruptBefore?: string[];
    /**
     * When true (default), a node that throws after its retries fails the run.
     * Set false for best-effort fan-out where one dead branch is acceptable; the
     * error is recorded on the span and emitted either way.
     */
    failFast?: boolean;
    checkpointer?: Checkpointer<S>;
    tracer?: Tracer;
    /** Name used on the run span. Default `'agent-graph'`. */
    name?: string;
}

export interface RunOptions {
    /** Correlates checkpoints for one conversation/job. Required to checkpoint. */
    threadId?: string;
    /** Resume from this checkpoint instead of starting fresh. */
    checkpointId?: string;
    signal?: AbortSignal;
    /** Parent span, so a graph nested inside another agent nests in the trace. */
    parent?: Span;
    /** Overrides the graph-level limit for this run. */
    recursionLimit?: number;
}

export class AgentGraph<S extends GraphState = GraphState> {
    private readonly _nodes = new Map<string, NodeEntry<S>>();
    private readonly _edges: EdgeEntry<S>[] = [];
    private readonly _channels: ChannelSpecs<S>;
    private readonly _recursionLimit: number;
    private readonly _interruptBefore: Set<string>;
    private readonly _failFast: boolean;
    private readonly _checkpointer: Checkpointer<S> | undefined;
    private readonly _tracer: Tracer | undefined;
    private readonly _name: string;

    private _entry: string | undefined;
    private _compiled = false;

    constructor(opts: AgentGraphOptions<S> = {}) {
        this._channels = opts.channels ?? {};
        this._recursionLimit = Math.max(1, opts.recursionLimit ?? 25);
        this._interruptBefore = new Set(opts.interruptBefore ?? []);
        this._failFast = opts.failFast ?? true;
        this._checkpointer = opts.checkpointer;
        this._tracer = opts.tracer;
        this._name = opts.name ?? 'agent-graph';
    }

    addNode(name: string, fn: NodeFn<S>, opts: { retries?: number } = {}): this {
        if (name === START || name === END) {
            throw new Error(`"${name}" is a reserved node name.`);
        }
        if (this._nodes.has(name)) throw new Error(`Node "${name}" is already defined.`);
        this._nodes.set(name, { name, fn, retries: Math.max(0, opts.retries ?? 0) });
        this._compiled = false;
        return this;
    }

    addEdge(from: string, to: string): this {
        this._edges.push({ from, to });
        if (from === START) this._entry = to;
        this._compiled = false;
        return this;
    }

    /**
     * A conditional edge is how delegation, retry-on-failure and "am I done yet"
     * are expressed. `mapping` translates a condition's domain language
     * (`'approve' | 'revise'`) into node names, keeping routing readable.
     */
    addConditionalEdges(
        from: string,
        condition: EdgeCondition<S>,
        mapping?: Record<string, string>,
    ): this {
        const edge: EdgeEntry<S> = { from, condition };
        if (mapping) edge.mapping = mapping;
        this._edges.push(edge);
        this._compiled = false;
        return this;
    }

    setEntryPoint(name: string): this {
        this._entry = name;
        this._compiled = false;
        return this;
    }

    /**
     * Validates the graph. Catching a dangling edge here rather than at superstep
     * seven — after six model calls have been billed — is the entire point.
     */
    compile(): this {
        if (!this._entry) throw new Error('Graph has no entry point: call setEntryPoint() or addEdge(START, …).');
        if (!this._nodes.has(this._entry)) throw new Error(`Entry point "${this._entry}" is not a node.`);

        for (const edge of this._edges) {
            if (edge.from !== START && !this._nodes.has(edge.from)) {
                throw new Error(`Edge source "${edge.from}" is not a node.`);
            }
            if (edge.to && edge.to !== END && !this._nodes.has(edge.to)) {
                throw new Error(`Edge target "${edge.to}" is not a node.`);
            }
            for (const target of Object.values(edge.mapping ?? {})) {
                if (target !== END && !this._nodes.has(target)) {
                    throw new Error(`Conditional edge from "${edge.from}" maps to unknown node "${target}".`);
                }
            }
        }

        for (const name of this._nodes.keys()) {
            const hasOutgoing = this._edges.some((e) => e.from === name);
            if (!hasOutgoing) {
                throw new Error(`Node "${name}" has no outgoing edge; add one to END if it is terminal.`);
            }
        }

        for (const name of this._interruptBefore) {
            if (!this._nodes.has(name)) throw new Error(`interruptBefore names unknown node "${name}".`);
        }

        this._compiled = true;
        return this;
    }

    async invoke(input: Partial<S>, opts: RunOptions = {}): Promise<GraphRunResult<S>> {
        let last: GraphEvent<S> | undefined;
        let result: GraphRunResult<S> | undefined;
        for await (const event of this.stream(input, opts)) {
            last = event;
            if (event.type === 'end') result = this._lastResult;
        }
        if (!result) {
            throw new Error(`Graph run produced no terminal event (last: ${last?.type ?? 'none'}).`);
        }
        return result;
    }

    private _lastResult: GraphRunResult<S> | undefined;

    /** Streams execution events — what a live orchestration view renders. */
    async *stream(input: Partial<S>, opts: RunOptions = {}): AsyncIterable<GraphEvent<S>> {
        if (!this._compiled) this.compile();

        const limit = Math.max(1, opts.recursionLimit ?? this._recursionLimit);
        const runSpan = this._tracer?.startSpan(this._name, {
            kind: 'graph',
            ...(opts.parent ? { parent: opts.parent } : {}),
            attributes: { 'graph.nodes': this._nodes.size, 'graph.recursion_limit': limit },
        });

        const resumed = opts.threadId && this._checkpointer
            ? await this._checkpointer.load(opts.threadId, opts.checkpointId)
            : undefined;

        let state: S = resumed ? { ...resumed.state } : this._initialState(input);
        let frontier: string[] = resumed ? [...resumed.frontier] : [this._entry as string];
        const path: string[] = resumed ? [...resumed.path] : [];
        let step = resumed ? resumed.step : 0;
        let reason: GraphEndReason = 'complete';
        let error: string | undefined;
        let checkpointId: string | undefined;

        try {
            while (frontier.length > 0) {
                if (step >= limit) {
                    reason = 'recursion-limit';
                    break;
                }

                const pausing = frontier.filter((n) => this._interruptBefore.has(n));
                if (pausing.length > 0) {
                    reason = 'interrupted';
                    checkpointId = await this._checkpoint(opts.threadId, step, state, frontier, path);
                    yield { type: 'interrupt', step, before: pausing, state };
                    break;
                }

                yield { type: 'step-start', step, frontier: [...frontier] };

                const pending: Array<{ node: string; update: Partial<S> }> = [];
                const customEvents: Array<{ node: string; event: { type: string; data?: unknown } }> = [];
                const failures: Array<{ node: string; error: string }> = [];

                // Deterministic order so a fan-out merge reproduces from a checkpoint.
                const running = [...frontier].sort();
                for (const name of running) yield { type: 'node-start', step, node: name };

                await Promise.all(running.map(async (name) => {
                    const entry = this._nodes.get(name) as NodeEntry<S>;
                    const nodeSpan = this._tracer?.startSpan(`node.${name}`, {
                        kind: 'agent',
                        ...(runSpan ? { parent: runSpan } : {}),
                        attributes: { 'graph.node': name, 'graph.step': step },
                    });

                    const ctx: NodeContext<S> = {
                        step,
                        ...(nodeSpan ? { span: nodeSpan } : {}),
                        ...(opts.signal ? { signal: opts.signal } : {}),
                        state,
                        emit: (event) => customEvents.push({ node: name, event }),
                    };

                    for (let attempt = 0; attempt <= entry.retries; attempt++) {
                        try {
                            const update = await entry.fn(state, ctx);
                            pending.push({ node: name, update: (update ?? {}) as Partial<S> });
                            nodeSpan?.setAttribute('graph.attempts', attempt + 1);
                            nodeSpan?.end('ok');
                            return;
                        } catch (err) {
                            if (attempt === entry.retries) {
                                const message = err instanceof Error ? err.message : String(err);
                                failures.push({ node: name, error: message });
                                nodeSpan?.fail(err);
                            }
                        }
                    }
                }));

                for (const { node, event } of customEvents) {
                    yield { type: 'custom', step, node, event };
                }
                for (const failure of failures) {
                    yield { type: 'node-error', step, node: failure.node, error: failure.error };
                }

                // Merge in the same sorted order the nodes ran in.
                pending.sort((a, b) => a.node.localeCompare(b.node));
                for (const { node, update } of pending) {
                    state = this._merge(state, update);
                    path.push(node);
                    yield { type: 'node-end', step, node, update };
                }

                if (failures.length > 0 && this._failFast) {
                    reason = 'error';
                    error = failures.map((f) => `${f.node}: ${f.error}`).join('; ');
                    break;
                }

                yield { type: 'step-end', step, state };
                checkpointId = await this._checkpoint(opts.threadId, step + 1, state, [], path);

                const succeeded = new Set(pending.map((p) => p.node));
                frontier = await this._nextFrontier([...succeeded], state, step);
                step += 1;

                if (opts.signal?.aborted) {
                    reason = 'error';
                    error = 'Run aborted.';
                    break;
                }
            }

            runSpan?.setAttributes({
                'graph.steps': step,
                'graph.reason': reason,
                'graph.path_length': path.length,
            });
            runSpan?.end(reason === 'error' ? 'error' : 'ok');

            const result: GraphRunResult<S> = { state, reason, steps: step, path };
            if (runSpan) result.traceId = runSpan.traceId;
            if (checkpointId) result.checkpointId = checkpointId;
            if (error) result.error = error;
            this._lastResult = result;

            yield { type: 'end', step, state, reason };
        } catch (err) {
            runSpan?.fail(err);
            throw err;
        } finally {
            await this._tracer?.flush();
        }
    }

    /** Resumes an interrupted run, optionally applying a human's edits to state. */
    async resume(threadId: string, update: Partial<S> = {}, opts: RunOptions = {}): Promise<GraphRunResult<S>> {
        if (!this._checkpointer) throw new Error('resume() requires a checkpointer.');
        const checkpoint = await this._checkpointer.load(threadId, opts.checkpointId);
        if (!checkpoint) throw new Error(`No checkpoint for thread "${threadId}".`);

        // The human's edit is merged through the SAME reducers a node's update goes
        // through, so approving-with-changes cannot corrupt a channel's invariants.
        const merged = this._merge(checkpoint.state, update);
        await this._checkpointer.save({ ...checkpoint, state: merged });

        // Clearing interruptBefore for the resumed run is what prevents the graph
        // from pausing again on the very node the human just approved.
        const resumedGraph = this._withoutInterrupts();
        return resumedGraph.invoke({}, { ...opts, threadId, checkpointId: checkpoint.id });
    }

    private _withoutInterrupts(): AgentGraph<S> {
        const clone = new AgentGraph<S>({
            channels: this._channels,
            recursionLimit: this._recursionLimit,
            failFast: this._failFast,
            ...(this._checkpointer ? { checkpointer: this._checkpointer } : {}),
            ...(this._tracer ? { tracer: this._tracer } : {}),
            name: this._name,
        });
        for (const [name, entry] of this._nodes) clone.addNode(name, entry.fn, { retries: entry.retries });
        for (const edge of this._edges) clone._edges.push(edge);
        clone._entry = this._entry as string;
        return clone.compile();
    }

    private async _nextFrontier(from: string[], state: S, step: number): Promise<string[]> {
        const next = new Set<string>();
        for (const node of from) {
            for (const edge of this._edges.filter((e) => e.from === node)) {
                if (edge.to) {
                    if (edge.to !== END) next.add(edge.to);
                    continue;
                }
                const decision = await (edge.condition as EdgeCondition<S>)(state, { step });
                for (const raw of Array.isArray(decision) ? decision : [decision]) {
                    const target = edge.mapping?.[raw] ?? raw;
                    if (target === END) continue;
                    if (!this._nodes.has(target)) {
                        throw new Error(`Conditional edge from "${node}" returned unknown target "${target}".`);
                    }
                    next.add(target);
                }
            }
        }
        return [...next];
    }

    private _initialState(input: Partial<S>): S {
        const state = {} as S;
        for (const [key, spec] of Object.entries(this._channels) as Array<[keyof S, { default?: () => unknown }]>) {
            if (spec?.default) state[key] = spec.default() as S[keyof S];
        }
        return this._merge(state, input);
    }

    private _merge(state: S, update: Partial<S>): S {
        const next = { ...state };
        for (const [key, value] of Object.entries(update) as Array<[keyof S, S[keyof S]]>) {
            if (value === undefined) continue;
            const reducer = (this._channels[key] as { reducer?: (c: unknown, u: unknown) => unknown } | undefined)?.reducer;
            next[key] = reducer ? reducer(state[key], value) as S[keyof S] : value;
        }
        return next;
    }

    private async _checkpoint(
        threadId: string | undefined,
        step: number,
        state: S,
        frontier: string[],
        path: string[],
    ): Promise<string | undefined> {
        if (!threadId || !this._checkpointer) return undefined;
        const id = `${threadId}:${step}`;
        await this._checkpointer.save({
            id, threadId, step, state,
            frontier: [...frontier],
            path: [...path],
            createdAt: Date.now(),
        });
        return id;
    }
}

// ── channel reducers ──────────────────────────────────────────────────────────

/** Appends to an array channel — the merge every message log wants. */
export function appendReducer<T>(current: T[] | undefined, update: T[] | T): T[] {
    const additions = Array.isArray(update) ? update : [update];
    return [...(current ?? []), ...additions];
}

/** Shallow-merges object channels instead of replacing them. */
export function mergeReducer<T extends Record<string, unknown>>(current: T | undefined, update: T): T {
    return { ...(current ?? {} as T), ...update };
}

/** Sums numeric channels — how a run counts tokens or tool calls across agents. */
export function sumReducer(current: number | undefined, update: number): number {
    return (current ?? 0) + update;
}

/** Set-union for array channels, preserving first-seen order. */
export function unionReducer<T>(current: T[] | undefined, update: T[] | T): T[] {
    const additions = Array.isArray(update) ? update : [update];
    const seen = new Set(current ?? []);
    const out = [...(current ?? [])];
    for (const item of additions) {
        if (seen.has(item)) continue;
        seen.add(item);
        out.push(item);
    }
    return out;
}

export { START, END };
