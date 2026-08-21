/**
 * orchestration/patterns.ts — ReAct, self-reflection, hierarchical delegation.
 *
 * These are the three patterns enterprise agent work actually reduces to, and each
 * is a small graph rather than a bespoke loop, so they compose: a supervisor's
 * worker can be a ReAct agent, and a ReAct agent's answer can be routed through a
 * reflection pass. That composability is why they are built on {@link AgentGraph}
 * instead of as three independent classes.
 *
 * The bridge port is plain text in, text out, so tool use is expressed as a strict
 * textual protocol the model is instructed to follow, and violations are handled
 * as data (an observation telling the model what it got wrong) rather than as
 * exceptions. A model that emits a malformed action must be able to recover on the
 * next turn; throwing would end the run over a formatting slip.
 */

import type { TransformerBridge } from '../bridges/TransformerBridge.js';
import type { Tracer } from '../telemetry/Tracer.js';
import { AgentGraph, appendReducer, sumReducer } from './AgentGraph.js';
import type { Checkpointer, NodeContext } from './types.js';
import { END, START } from './types.js';

// ── shared message channel ────────────────────────────────────────────────────

export type AgentTurnRole = 'user' | 'assistant' | 'system' | 'tool';

export interface AgentTurn {
    role: AgentTurnRole;
    content: string;
    /** Set on `tool` turns — which tool produced this observation. */
    name?: string;
}

/** Renders a transcript for a text-completion bridge. */
export function renderTranscript(turns: readonly AgentTurn[]): string {
    return turns
        .map((turn) => {
            if (turn.role === 'tool') return `Observation (${turn.name ?? 'tool'}): ${turn.content}`;
            if (turn.role === 'assistant') return turn.content;
            if (turn.role === 'system') return turn.content;
            return `Question: ${turn.content}`;
        })
        .join('\n');
}

// ── tools ─────────────────────────────────────────────────────────────────────

export interface AgentTool {
    name: string;
    description: string;
    /** Parameter documentation, rendered into the prompt. */
    parameters?: Record<string, string>;
    run(input: Record<string, unknown>): Promise<string> | string;
}

export function renderToolCatalog(tools: readonly AgentTool[]): string {
    return tools
        .map((tool) => {
            const params = tool.parameters
                ? ` Input fields: ${Object.entries(tool.parameters).map(([k, v]) => `${k} (${v})`).join(', ')}.`
                : '';
            return `- ${tool.name}: ${tool.description}${params}`;
        })
        .join('\n');
}

export interface ParsedAction {
    kind: 'action' | 'final' | 'malformed';
    thought?: string;
    tool?: string;
    input?: Record<string, unknown>;
    answer?: string;
    problem?: string;
}

/**
 * Parses one ReAct turn.
 *
 * Tolerant on purpose: models drift on whitespace, casing and code fences far more
 * often than they drift on intent, and a parser that rejects `**Action:**` burns a
 * turn for no reason. Genuinely unparseable output returns `malformed` with a
 * message that is fed back as an observation.
 */
export function parseReactTurn(text: string): ParsedAction {
    const clean = text.replace(/\*\*/g, '').trim();

    const finalMatch = /final\s*answer\s*:\s*([\s\S]*)$/i.exec(clean);
    const actionMatch = /action\s*:\s*([^\n]+)/i.exec(clean);
    const thoughtMatch = /thought\s*:\s*([^\n]+)/i.exec(clean);
    const thought = thoughtMatch?.[1]?.trim();

    // A turn containing both is a model hedging; the final answer wins only when it
    // appears AFTER the action, matching the order the protocol asks for.
    if (finalMatch && (!actionMatch || (finalMatch.index > actionMatch.index))) {
        return {
            kind: 'final',
            ...(thought ? { thought } : {}),
            answer: (finalMatch[1] ?? '').trim(),
        };
    }

    if (!actionMatch) {
        return {
            kind: 'malformed',
            ...(thought ? { thought } : {}),
            problem: 'No "Action:" or "Final Answer:" line was present.',
        };
    }

    const tool = (actionMatch[1] as string).trim().replace(/[.`]+$/, '');
    const inputMatch = /action\s*input\s*:\s*([\s\S]*?)(?:\n\s*(?:thought|action|observation|final answer)\s*:|$)/i
        .exec(clean);

    const rawInput = (inputMatch?.[1] ?? '').trim().replace(/^```(?:json)?/i, '').replace(/```$/, '').trim();
    let input: Record<string, unknown> = {};
    if (rawInput) {
        try {
            const parsed = JSON.parse(rawInput) as unknown;
            input = parsed && typeof parsed === 'object' && !Array.isArray(parsed)
                ? parsed as Record<string, unknown>
                // A bare scalar is a common and reasonable shape for single-argument
                // tools; wrapping it beats rejecting the turn.
                : { input: parsed };
        } catch {
            input = { input: rawInput };
        }
    }

    return { kind: 'action', ...(thought ? { thought } : {}), tool, input };
}

// ── ReAct ─────────────────────────────────────────────────────────────────────

export type ReactState = {
    task: string;
    turns: AgentTurn[];
    /** Tool invocations so far — the loop's own budget meter. */
    toolCalls: number;
    answer: string;
    done: boolean;
};

export interface ReactAgentOptions {
    bridge: TransformerBridge;
    tools: readonly AgentTool[];
    /** Prepended to the generated protocol instructions. */
    systemPrompt?: string;
    /** Reason/act cycles before the agent must answer. Default 6. */
    maxIterations?: number;
    maxTokens?: number;
    model?: string;
    tracer?: Tracer;
    checkpointer?: Checkpointer<ReactState>;
}

export const REACT_PROTOCOL =
    'Work in cycles. Each reply must be EITHER:\n' +
    'Thought: <your reasoning>\nAction: <tool name>\nAction Input: <JSON object>\n' +
    'OR, when you can answer:\nThought: <your reasoning>\nFinal Answer: <the answer>\n' +
    'Use one action per reply and wait for the Observation before continuing.';

/**
 * ReAct: interleaved reasoning and tool use.
 *
 * Two nodes and a conditional edge — `reason` produces a thought plus either an
 * action or a final answer; `act` executes the tool and appends an observation;
 * the edge routes back to `reason` or to END. The iteration cap is enforced in the
 * routing rather than trusted to the model, because "decide when to stop" is the
 * one judgement an unconverged agent is worst at.
 */
export function createReactAgent(opts: ReactAgentOptions): AgentGraph<ReactState> {
    const maxIterations = Math.max(1, opts.maxIterations ?? 6);
    const toolsByName = new Map(opts.tools.map((t) => [t.name, t]));
    const catalog = renderToolCatalog(opts.tools);

    const systemPrompt = [
        opts.systemPrompt ?? 'You are a careful analyst that uses tools to gather evidence before answering.',
        `Available tools:\n${catalog}`,
        REACT_PROTOCOL,
    ].join('\n\n');

    const graph = new AgentGraph<ReactState>({
        name: 'react-agent',
        channels: {
            turns: { reducer: appendReducer, default: () => [] },
            toolCalls: { reducer: sumReducer, default: () => 0 },
            done: { default: () => false },
            answer: { default: () => '' },
            task: { default: () => '' },
        },
        recursionLimit: maxIterations * 2 + 2,
        ...(opts.tracer ? { tracer: opts.tracer } : {}),
        ...(opts.checkpointer ? { checkpointer: opts.checkpointer } : {}),
    });

    graph.addNode('reason', async (state, ctx: NodeContext<ReactState>) => {
        const transcript = renderTranscript([
            { role: 'user', content: state.task },
            ...state.turns,
        ]);
        const remaining = maxIterations - state.toolCalls;
        const pressure = remaining <= 1
            ? '\n\nYou have no tool calls left. Reply with a Final Answer using what you already know.'
            : '';

        const reply = await opts.bridge.generate(`${transcript}${pressure}`, {
            systemPrompt,
            ...(opts.maxTokens !== undefined ? { maxTokens: opts.maxTokens } : {}),
            ...(opts.model !== undefined ? { model: opts.model } : {}),
        });

        const parsed = parseReactTurn(reply);
        ctx.span?.setAttributes({ 'react.turn_kind': parsed.kind, 'react.tool': parsed.tool ?? '' });

        if (parsed.kind === 'final') {
            return {
                turns: [{ role: 'assistant', content: reply }] as AgentTurn[],
                answer: parsed.answer ?? '',
                done: true,
            };
        }
        return { turns: [{ role: 'assistant', content: reply }] as AgentTurn[], done: false };
    });

    graph.addNode('act', async (state, ctx: NodeContext<ReactState>) => {
        const lastAssistant = [...state.turns].reverse().find((t) => t.role === 'assistant');
        const parsed = parseReactTurn(lastAssistant?.content ?? '');

        if (parsed.kind === 'malformed') {
            // Fed back as an observation, not thrown: the model gets one clear
            // chance to correct its formatting rather than killing the run.
            return {
                turns: [{
                    role: 'tool',
                    name: 'protocol',
                    content: `Your reply could not be parsed (${parsed.problem}). Reply using the exact format.`,
                }] as AgentTurn[],
                toolCalls: 1,
            };
        }

        const tool = toolsByName.get(parsed.tool as string);
        if (!tool) {
            return {
                turns: [{
                    role: 'tool',
                    name: 'protocol',
                    content: `Unknown tool "${parsed.tool}". Choose one of: ${[...toolsByName.keys()].join(', ')}.`,
                }] as AgentTurn[],
                toolCalls: 1,
            };
        }

        const span = ctx.span?.child(`tool.${tool.name}`, { kind: 'tool' });
        try {
            const observation = await tool.run(parsed.input ?? {});
            span?.setAttribute('tool.result_chars', observation.length);
            span?.end('ok');
            return {
                turns: [{ role: 'tool', name: tool.name, content: observation }] as AgentTurn[],
                toolCalls: 1,
            };
        } catch (err) {
            // A failing tool is an observation too. An agent that can read "the API
            // returned 503" can retry or route around it; one that crashes cannot.
            span?.fail(err);
            const message = err instanceof Error ? err.message : String(err);
            return {
                turns: [{ role: 'tool', name: tool.name, content: `Tool failed: ${message}` }] as AgentTurn[],
                toolCalls: 1,
            };
        }
    });

    graph.addEdge(START, 'reason');
    graph.addConditionalEdges('reason', (state) => {
        if (state.done) return END;
        return state.toolCalls >= maxIterations ? END : 'act';
    });
    graph.addEdge('act', 'reason');

    return graph.compile();
}

// ── self-reflection ───────────────────────────────────────────────────────────

export type ReflectionState = {
    task: string;
    draft: string;
    critique: string;
    revisions: number;
    accepted: boolean;
    history: string[];
};

export interface ReflectionAgentOptions {
    bridge: TransformerBridge;
    /** Revision rounds allowed after the first draft. Default 2. */
    maxRevisions?: number;
    /** Instructions for the drafting pass. */
    generatorPrompt?: string;
    /** Instructions for the critic. Must ask for an explicit verdict token. */
    criticPrompt?: string;
    /** Decides acceptance from the critique. Default: looks for `APPROVED`. */
    isAcceptable?: (critique: string) => boolean;
    maxTokens?: number;
    model?: string;
    tracer?: Tracer;
    checkpointer?: Checkpointer<ReflectionState>;
}

export const DEFAULT_CRITIC_PROMPT =
    'You are a demanding reviewer. Judge the draft against the task on correctness, ' +
    'completeness and evidence. List concrete, actionable defects. ' +
    'End your reply with exactly one verdict line: "VERDICT: APPROVED" or "VERDICT: REVISE".';

/**
 * Generate → critique → revise, bounded.
 *
 * Reflection earns its cost only when the critic is allowed to be harsh and the
 * loop is allowed to stop: an unbounded critic converges on nitpicking, and a
 * critic that cannot say APPROVED doubles spend for no quality gain. Both limits
 * are enforced here rather than requested in the prompt.
 */
export function createReflectionAgent(opts: ReflectionAgentOptions): AgentGraph<ReflectionState> {
    const maxRevisions = Math.max(0, opts.maxRevisions ?? 2);
    const isAcceptable = opts.isAcceptable ?? ((critique) => /verdict\s*:\s*approved/i.test(critique));

    const graph = new AgentGraph<ReflectionState>({
        name: 'reflection-agent',
        channels: {
            history: { reducer: appendReducer, default: () => [] },
            revisions: { default: () => 0 },
            accepted: { default: () => false },
            draft: { default: () => '' },
            critique: { default: () => '' },
            task: { default: () => '' },
        },
        recursionLimit: (maxRevisions + 1) * 2 + 2,
        ...(opts.tracer ? { tracer: opts.tracer } : {}),
        ...(opts.checkpointer ? { checkpointer: opts.checkpointer } : {}),
    });

    const generate = async (state: ReflectionState): Promise<Partial<ReflectionState>> => {
        const prompt = state.draft
            ? `Task: ${state.task}\n\nPrevious draft:\n${state.draft}\n\nReviewer feedback:\n${state.critique}\n\nProduce an improved draft that resolves every point.`
            : `Task: ${state.task}\n\nProduce your best draft.`;

        const draft = await opts.bridge.generate(prompt, {
            systemPrompt: opts.generatorPrompt ?? 'You produce precise, well-evidenced work.',
            ...(opts.maxTokens !== undefined ? { maxTokens: opts.maxTokens } : {}),
            ...(opts.model !== undefined ? { model: opts.model } : {}),
        });

        return {
            draft,
            history: [draft],
            ...(state.draft ? { revisions: state.revisions + 1 } : {}),
        };
    };

    graph.addNode('generate', generate);

    graph.addNode('critique', async (state, ctx) => {
        const critique = await opts.bridge.generate(
            `Task: ${state.task}\n\nDraft:\n${state.draft}\n\nReview it.`,
            {
                systemPrompt: opts.criticPrompt ?? DEFAULT_CRITIC_PROMPT,
                ...(opts.maxTokens !== undefined ? { maxTokens: opts.maxTokens } : {}),
                ...(opts.model !== undefined ? { model: opts.model } : {}),
            },
        );
        const accepted = isAcceptable(critique);
        ctx.span?.setAttribute('reflection.accepted', accepted);
        return { critique, accepted };
    });

    graph.addEdge(START, 'generate');
    graph.addEdge('generate', 'critique');
    graph.addConditionalEdges('critique', (state) => {
        if (state.accepted) return END;
        return state.revisions >= maxRevisions ? END : 'generate';
    });

    return graph.compile();
}

// ── hierarchical delegation ───────────────────────────────────────────────────

export interface Worker {
    name: string;
    /** What this worker is for — the supervisor routes on this text. */
    description: string;
    /** Runs the sub-task and returns its result. May itself be a graph. */
    run(task: string, context: { transcript: readonly AgentTurn[] }): Promise<string>;
}

export type SupervisorState = {
    task: string;
    turns: AgentTurn[];
    /** Worker chosen for the next hop, or `FINISH`. */
    next: string;
    handoffs: number;
    answer: string;
};

export interface SupervisorOptions {
    bridge: TransformerBridge;
    workers: readonly Worker[];
    /** Delegations before the supervisor must answer. Default 5. */
    maxHandoffs?: number;
    systemPrompt?: string;
    maxTokens?: number;
    model?: string;
    tracer?: Tracer;
    checkpointer?: Checkpointer<SupervisorState>;
}

export const FINISH = 'FINISH';

/**
 * A supervisor that delegates to named workers and then composes the answer.
 *
 * Hierarchical delegation is worth its coordination overhead exactly when workers
 * hold context the supervisor should not — a different tool set, a different
 * corpus, a different tenant's data. Each worker returns a RESULT rather than its
 * transcript, which is what keeps the supervisor's context from growing into the
 * sum of everything its workers read.
 */
export function createSupervisor(opts: SupervisorOptions): AgentGraph<SupervisorState> {
    const maxHandoffs = Math.max(1, opts.maxHandoffs ?? 5);
    const workersByName = new Map(opts.workers.map((w) => [w.name, w]));
    const roster = opts.workers.map((w) => `- ${w.name}: ${w.description}`).join('\n');

    const systemPrompt = [
        opts.systemPrompt ?? 'You are a supervisor coordinating specialist workers.',
        `Workers:\n${roster}`,
        `Reply with exactly one line: "NEXT: <worker name>" to delegate, or "NEXT: ${FINISH}" ` +
        'followed by a line "ANSWER: <the final answer>" when you have enough to answer.',
    ].join('\n\n');

    const graph = new AgentGraph<SupervisorState>({
        name: 'supervisor',
        channels: {
            turns: { reducer: appendReducer, default: () => [] },
            handoffs: { reducer: sumReducer, default: () => 0 },
            next: { default: () => '' },
            answer: { default: () => '' },
            task: { default: () => '' },
        },
        recursionLimit: maxHandoffs * 2 + 2,
        ...(opts.tracer ? { tracer: opts.tracer } : {}),
        ...(opts.checkpointer ? { checkpointer: opts.checkpointer } : {}),
    });

    graph.addNode('supervisor', async (state, ctx) => {
        const transcript = renderTranscript([{ role: 'user', content: state.task }, ...state.turns]);
        const pressure = state.handoffs >= maxHandoffs
            ? `\n\nYou have used all delegations. Reply with "NEXT: ${FINISH}" and your ANSWER.`
            : '';

        const reply = await opts.bridge.generate(`${transcript}${pressure}`, {
            systemPrompt,
            ...(opts.maxTokens !== undefined ? { maxTokens: opts.maxTokens } : {}),
            ...(opts.model !== undefined ? { model: opts.model } : {}),
        });

        const nextMatch = /next\s*:\s*([^\n]+)/i.exec(reply.replace(/\*\*/g, ''));
        const answerMatch = /answer\s*:\s*([\s\S]*)$/i.exec(reply.replace(/\*\*/g, ''));
        const requested = nextMatch?.[1]?.trim().replace(/[.`]+$/, '') ?? FINISH;

        // An unknown worker name resolves to FINISH rather than throwing: a
        // supervisor that hallucinates a worker should degrade to answering, not
        // crash a run that may already have useful worker results in state.
        const next = workersByName.has(requested) ? requested : FINISH;
        ctx.span?.setAttributes({ 'supervisor.requested': requested, 'supervisor.next': next });

        return {
            turns: [{ role: 'assistant', content: reply }] as AgentTurn[],
            next,
            ...(next === FINISH ? { answer: (answerMatch?.[1] ?? reply).trim() } : {}),
        };
    });

    graph.addNode('delegate', async (state, ctx) => {
        const worker = workersByName.get(state.next);
        if (!worker) return { next: FINISH };

        const span = ctx.span?.child(`worker.${worker.name}`, { kind: 'agent' });
        try {
            const result = await worker.run(state.task, { transcript: state.turns });
            span?.setAttribute('worker.result_chars', result.length);
            span?.end('ok');
            return {
                turns: [{ role: 'tool', name: worker.name, content: result }] as AgentTurn[],
                handoffs: 1,
            };
        } catch (err) {
            span?.fail(err);
            const message = err instanceof Error ? err.message : String(err);
            return {
                turns: [{ role: 'tool', name: worker.name, content: `Worker failed: ${message}` }] as AgentTurn[],
                handoffs: 1,
            };
        }
    });

    graph.addEdge(START, 'supervisor');
    graph.addConditionalEdges('supervisor', (state) => {
        if (state.next === FINISH) return END;
        return state.handoffs >= maxHandoffs ? END : 'delegate';
    });
    graph.addEdge('delegate', 'supervisor');

    return graph.compile();
}

/** Adapts a ReAct agent into a {@link Worker}, so hierarchies nest. */
export function asWorker(
    name: string,
    description: string,
    graph: AgentGraph<ReactState>,
): Worker {
    return {
        name,
        description,
        run: async (task) => {
            const result = await graph.invoke({ task } as Partial<ReactState>);
            return result.state.answer || 'The worker produced no answer.';
        },
    };
}
