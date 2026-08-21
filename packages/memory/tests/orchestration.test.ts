import { describe, expect, it, jest } from '@jest/globals';

import {
    AgentGraph,
    appendReducer,
    mergeReducer,
    sumReducer,
    unionReducer,
} from '../src/orchestration/AgentGraph.js';
import { InMemoryCheckpointer, KeyValueCheckpointer, type KeyValueLike } from '../src/orchestration/checkpoint.js';
import {
    FINISH,
    asWorker,
    createReactAgent,
    createReflectionAgent,
    createSupervisor,
    parseReactTurn,
    renderToolCatalog,
    renderTranscript,
    type AgentTool,
} from '../src/orchestration/patterns.js';
import { END, START } from '../src/orchestration/types.js';
import { Tracer } from '../src/telemetry/Tracer.js';
import type { TransformerBridge } from '../src/bridges/TransformerBridge.js';

type Counter = { count: number; log: string[] };

/** A bridge that replays a scripted sequence of replies, recording its prompts. */
function scriptedBridge(replies: string[]): TransformerBridge & { prompts: string[]; calls: number } {
    let i = 0;
    const prompts: string[] = [];
    return {
        supportsStreaming: false,
        prompts,
        get calls() { return i; },
        async generate(prompt: string) {
            prompts.push(prompt);
            return replies[Math.min(i++, replies.length - 1)] as string;
        },
    };
}

describe('channel reducers', () => {
    it('append, union, sum and merge behave as declared', () => {
        expect(appendReducer([1], 2)).toEqual([1, 2]);
        expect(appendReducer(undefined, [1, 2])).toEqual([1, 2]);
        expect(unionReducer(['a'], ['a', 'b'])).toEqual(['a', 'b']);
        expect(unionReducer(undefined, 'a')).toEqual(['a']);
        expect(sumReducer(undefined, 3)).toBe(3);
        expect(sumReducer(2, 3)).toBe(5);
        expect(mergeReducer({ a: 1 }, { b: 2 })).toEqual({ a: 1, b: 2 });
        expect(mergeReducer(undefined, { b: 2 })).toEqual({ b: 2 });
    });
});

describe('AgentGraph', () => {
    it('runs a linear graph and returns the merged state', async () => {
        const graph = new AgentGraph<Counter>({
            channels: { count: { reducer: sumReducer, default: () => 0 }, log: { reducer: appendReducer, default: () => [] } },
        });
        graph.addNode('a', () => ({ count: 1, log: ['a'] }));
        graph.addNode('b', () => ({ count: 2, log: ['b'] }));
        graph.addEdge(START, 'a');
        graph.addEdge('a', 'b');
        graph.addEdge('b', END);

        const result = await graph.invoke({});
        expect(result.state.count).toBe(3);
        expect(result.state.log).toEqual(['a', 'b']);
        expect(result.reason).toBe('complete');
        expect(result.path).toEqual(['a', 'b']);
    });

    it('fans out concurrently and merges both writers through the reducer', async () => {
        const graph = new AgentGraph<Counter>({
            channels: { count: { reducer: sumReducer, default: () => 0 }, log: { reducer: appendReducer, default: () => [] } },
        });
        graph.addNode('start', () => ({ log: ['start'] }));
        graph.addNode('left', () => ({ count: 1, log: ['left'] }));
        graph.addNode('right', () => ({ count: 10, log: ['right'] }));
        graph.addNode('join', () => ({ log: ['join'] }));

        graph.addEdge(START, 'start');
        graph.addConditionalEdges('start', () => ['left', 'right']);
        graph.addEdge('left', 'join');
        graph.addEdge('right', 'join');
        graph.addEdge('join', END);

        const result = await graph.invoke({});
        // Both siblings' writes survive — the merge is declared, not a race.
        expect(result.state.count).toBe(11);
        expect(result.state.log).toEqual(['start', 'left', 'right', 'join']);
        // `join` ran ONCE despite two inbound edges: the frontier is a set.
        expect(result.path.filter((n) => n === 'join')).toHaveLength(1);
    });

    it('last-write-wins on a channel with no reducer', async () => {
        const graph = new AgentGraph<{ value: string }>();
        graph.addNode('a', () => ({ value: 'first' }));
        graph.addNode('b', () => ({ value: 'second' }));
        graph.addEdge(START, 'a');
        graph.addEdge('a', 'b');
        graph.addEdge('b', END);

        expect((await graph.invoke({ value: 'seed' })).state.value).toBe('second');
    });

    it('ignores an undefined update value rather than clobbering the channel', async () => {
        const graph = new AgentGraph<{ value: string }>();
        graph.addNode('a', () => ({ value: undefined }));
        graph.addEdge(START, 'a');
        graph.addEdge('a', END);
        expect((await graph.invoke({ value: 'kept' })).state.value).toBe('kept');
    });

    it('routes through a conditional mapping', async () => {
        const graph = new AgentGraph<{ verdict: string; log: string[] }>({
            channels: { log: { reducer: appendReducer, default: () => [] } },
        });
        graph.addNode('check', (s) => ({ log: [`check:${s.verdict}`] }));
        graph.addNode('approve', () => ({ log: ['approved'] }));
        graph.addNode('revise', () => ({ log: ['revised'] }));
        graph.addEdge(START, 'check');
        graph.addConditionalEdges('check', (s) => s.verdict, { ok: 'approve', bad: 'revise' });
        graph.addEdge('approve', END);
        graph.addEdge('revise', END);

        expect((await graph.invoke({ verdict: 'bad' })).state.log).toEqual(['check:bad', 'revised']);
        expect((await graph.invoke({ verdict: 'ok' })).state.log).toEqual(['check:ok', 'approved']);
    });

    it('stops a non-converging cycle at the recursion limit', async () => {
        const graph = new AgentGraph<Counter>({
            channels: { count: { reducer: sumReducer, default: () => 0 }, log: { default: () => [] } },
            recursionLimit: 4,
        });
        graph.addNode('loop', () => ({ count: 1 }));
        graph.addEdge(START, 'loop');
        graph.addConditionalEdges('loop', () => 'loop');

        const result = await graph.invoke({});
        expect(result.reason).toBe('recursion-limit');
        expect(result.state.count).toBe(4);
    });

    it('retries a flaky node before failing the run', async () => {
        let attempts = 0;
        const graph = new AgentGraph<{ ok: boolean }>();
        graph.addNode('flaky', () => {
            attempts += 1;
            if (attempts < 3) throw new Error('transient');
            return { ok: true };
        }, { retries: 3 });
        graph.addEdge(START, 'flaky');
        graph.addEdge('flaky', END);

        const result = await graph.invoke({ ok: false });
        expect(attempts).toBe(3);
        expect(result.state.ok).toBe(true);
    });

    it('fails fast by default and continues when failFast is off', async () => {
        const build = (failFast: boolean) => {
            const graph = new AgentGraph<Counter>({
                channels: { count: { reducer: sumReducer, default: () => 0 }, log: { reducer: appendReducer, default: () => [] } },
                failFast,
            });
            graph.addNode('bad', () => { throw new Error('exploded'); });
            graph.addNode('good', () => ({ count: 1, log: ['good'] }));
            graph.addNode('after', () => ({ log: ['after'] }));
            graph.addEdge(START, 'good');
            graph.addConditionalEdges('good', () => ['bad', 'after']);
            graph.addEdge('bad', END);
            graph.addEdge('after', END);
            return graph;
        };

        const strict = await build(true).invoke({});
        expect(strict.reason).toBe('error');
        expect(strict.error).toContain('bad: exploded');

        const lenient = await build(false).invoke({});
        expect(lenient.reason).toBe('complete');
        expect(lenient.state.log).toEqual(['good', 'after']);
    });

    it('streams node and step events, including custom node emissions', async () => {
        const graph = new AgentGraph<{ done: boolean }>();
        graph.addNode('work', (_s, ctx) => {
            ctx.emit({ type: 'progress', data: 50 });
            return { done: true };
        });
        graph.addEdge(START, 'work');
        graph.addEdge('work', END);

        const types: string[] = [];
        for await (const event of graph.stream({ done: false })) types.push(event.type);

        expect(types).toEqual(['step-start', 'node-start', 'custom', 'node-end', 'step-end', 'end']);
    });

    it('emits node-error for a failed node', async () => {
        const graph = new AgentGraph<{ x: number }>({ failFast: true });
        graph.addNode('bad', () => { throw new Error('nope'); });
        graph.addEdge(START, 'bad');
        graph.addEdge('bad', END);

        const errors: string[] = [];
        for await (const event of graph.stream({ x: 0 })) {
            if (event.type === 'node-error') errors.push(event.error);
        }
        expect(errors).toEqual(['nope']);
    });

    it('validates the graph at compile time, not at superstep seven', () => {
        expect(() => new AgentGraph().compile()).toThrow(/no entry point/);

        const missingTarget = new AgentGraph();
        missingTarget.addNode('a', () => ({}));
        missingTarget.addEdge(START, 'a');
        missingTarget.addEdge('a', 'ghost');
        expect(() => missingTarget.compile()).toThrow(/Edge target "ghost"/);

        const dangling = new AgentGraph();
        dangling.addNode('a', () => ({}));
        dangling.addEdge(START, 'a');
        expect(() => dangling.compile()).toThrow(/no outgoing edge/);

        const badMapping = new AgentGraph();
        badMapping.addNode('a', () => ({}));
        badMapping.addEdge(START, 'a');
        badMapping.addConditionalEdges('a', () => 'x', { x: 'ghost' });
        expect(() => badMapping.compile()).toThrow(/unknown node "ghost"/);

        const badInterrupt = new AgentGraph({ interruptBefore: ['ghost'] });
        badInterrupt.addNode('a', () => ({}));
        badInterrupt.addEdge(START, 'a');
        badInterrupt.addEdge('a', END);
        expect(() => badInterrupt.compile()).toThrow(/interruptBefore names unknown node/);

        const badSource = new AgentGraph();
        badSource.addNode('a', () => ({}));
        badSource.addEdge(START, 'a');
        badSource.addEdge('a', END);
        badSource.addEdge('ghost', END);
        expect(() => badSource.compile()).toThrow(/Edge source "ghost"/);

        const reserved = new AgentGraph();
        expect(() => reserved.addNode(START, () => ({}))).toThrow(/reserved node name/);

        const duplicate = new AgentGraph();
        duplicate.addNode('a', () => ({}));
        expect(() => duplicate.addNode('a', () => ({}))).toThrow(/already defined/);
    });

    it('throws when a conditional edge invents a target at run time', async () => {
        const graph = new AgentGraph<{ x: number }>();
        graph.addNode('a', () => ({ x: 1 }));
        graph.addEdge(START, 'a');
        graph.addConditionalEdges('a', () => 'nowhere');
        graph.compile();

        await expect(graph.invoke({ x: 0 })).rejects.toThrow(/unknown target "nowhere"/);
    });

    it('pauses before an interrupt node and resumes with a human edit', async () => {
        const checkpointer = new InMemoryCheckpointer<{ draft: string; approved: boolean; log: string[] }>();
        const build = () => {
            const graph = new AgentGraph<{ draft: string; approved: boolean; log: string[] }>({
                channels: { log: { reducer: appendReducer, default: () => [] } },
                interruptBefore: ['publish'],
                checkpointer,
            });
            graph.addNode('draft', () => ({ draft: 'machine draft', log: ['draft'] }));
            graph.addNode('publish', (s) => ({ log: [`published:${s.draft}`], approved: true }));
            graph.addEdge(START, 'draft');
            graph.addEdge('draft', 'publish');
            graph.addEdge('publish', END);
            return graph.compile();
        };

        const graph = build();
        const paused = await graph.invoke({ draft: '', approved: false }, { threadId: 'thread-1' });

        expect(paused.reason).toBe('interrupted');
        expect(paused.state.approved).toBe(false);
        expect(paused.checkpointId).toBeDefined();

        // The human corrects the draft; the edit goes through the same reducers.
        const resumed = await graph.resume('thread-1', { draft: 'human-edited draft' });
        expect(resumed.reason).toBe('complete');
        expect(resumed.state.log).toEqual(['draft', 'published:human-edited draft']);
    });

    it('rejects resume without a checkpointer or a known thread', async () => {
        const graph = new AgentGraph<{ x: number }>();
        graph.addNode('a', () => ({ x: 1 }));
        graph.addEdge(START, 'a');
        graph.addEdge('a', END);
        graph.compile();
        await expect(graph.resume('t')).rejects.toThrow(/requires a checkpointer/);

        const withCkpt = new AgentGraph<{ x: number }>({ checkpointer: new InMemoryCheckpointer() });
        withCkpt.addNode('a', () => ({ x: 1 }));
        withCkpt.addEdge(START, 'a');
        withCkpt.addEdge('a', END);
        withCkpt.compile();
        await expect(withCkpt.resume('missing')).rejects.toThrow(/No checkpoint for thread/);
    });

    it('aborts cooperatively when the signal fires', async () => {
        const controller = new AbortController();
        const graph = new AgentGraph<Counter>({
            channels: { count: { reducer: sumReducer, default: () => 0 }, log: { default: () => [] } },
            recursionLimit: 50,
        });
        graph.addNode('loop', () => {
            controller.abort();
            return { count: 1 };
        });
        graph.addEdge(START, 'loop');
        graph.addConditionalEdges('loop', () => 'loop');

        const result = await graph.invoke({}, { signal: controller.signal });
        expect(result.reason).toBe('error');
        expect(result.error).toBe('Run aborted.');
    });

    it('nests node spans under the run span', async () => {
        const tracer = new Tracer();
        const graph = new AgentGraph<{ x: number }>({ tracer, name: 'traced-graph' });
        graph.addNode('work', () => ({ x: 1 }));
        graph.addEdge(START, 'work');
        graph.addEdge('work', END);

        const result = await graph.invoke({ x: 0 });
        const spans = tracer.trace(result.traceId as string);
        const run = spans.find((s) => s.name === 'traced-graph');
        const node = spans.find((s) => s.name === 'node.work');

        expect(node?.parentSpanId).toBe(run?.spanId);
        expect(run?.attributes['graph.reason']).toBe('complete');
    });

    it('compiles lazily on first run', async () => {
        const graph = new AgentGraph<{ x: number }>();
        graph.addNode('a', () => ({ x: 1 }));
        graph.addEdge(START, 'a');
        graph.addEdge('a', END);
        expect((await graph.invoke({ x: 0 })).state.x).toBe(1);
    });
});

describe('checkpointers', () => {
    const sample = (step: number) => ({
        id: `t:${step}`, threadId: 't', step, state: { x: step },
        frontier: [], path: [], createdAt: step,
    });

    it('keeps a bounded, ordered history and loads the newest by default', async () => {
        const ckpt = new InMemoryCheckpointer<{ x: number }>({ maxPerThread: 2 });
        await ckpt.save(sample(0));
        await ckpt.save(sample(1));
        await ckpt.save(sample(2));

        expect((await ckpt.load('t'))?.step).toBe(2);
        expect((await ckpt.load('t', 't:1'))?.step).toBe(1);
        expect(await ckpt.list('t')).toHaveLength(2);      // step 0 evicted
        expect(await ckpt.load('missing')).toBeUndefined();

        await ckpt.clear('t');
        expect(await ckpt.list('t')).toHaveLength(0);
    });

    it('replaces rather than duplicates a re-entered superstep', async () => {
        const ckpt = new InMemoryCheckpointer<{ x: number }>();
        await ckpt.save(sample(1));
        await ckpt.save({ ...sample(1), state: { x: 99 } });
        const history = await ckpt.list('t');
        expect(history).toHaveLength(1);
        expect(history[0]?.state.x).toBe(99);
    });

    it('persists through a key-value binding', async () => {
        const map = new Map<string, string>();
        const kv: KeyValueLike = {
            get: async (k) => map.get(k) ?? null,
            put: async (k, v) => { map.set(k, v); },
            delete: async (k) => { map.delete(k); },
        };
        const ckpt = new KeyValueCheckpointer<{ x: number }>({ kv, prefix: 'p:', maxPerThread: 2 });

        await ckpt.save(sample(0));
        await ckpt.save(sample(1));
        await ckpt.save(sample(2));

        expect((await ckpt.load('t'))?.step).toBe(2);
        expect((await ckpt.load('t', 't:1'))?.step).toBe(1);
        expect(await ckpt.list('t')).toHaveLength(2);
        expect(map.has('p:t')).toBe(true);

        await ckpt.clear('t');
        expect(await ckpt.list('t')).toHaveLength(0);
        expect(await ckpt.load('t')).toBeUndefined();
    });

    it('treats a corrupt entry as empty instead of wedging the thread', async () => {
        const kv: KeyValueLike = {
            get: async () => 'not json',
            put: async () => {},
            delete: async () => {},
        };
        const ckpt = new KeyValueCheckpointer({ kv });
        expect(await ckpt.list('t')).toEqual([]);
    });
});

describe('parseReactTurn', () => {
    it('reads a well-formed action with JSON input', () => {
        const parsed = parseReactTurn('Thought: I should search\nAction: search\nAction Input: {"q": "retention"}');
        expect(parsed).toMatchObject({ kind: 'action', tool: 'search', thought: 'I should search' });
        expect(parsed.input).toEqual({ q: 'retention' });
    });

    it('tolerates markdown bold, code fences and a bare scalar input', () => {
        const fenced = parseReactTurn('**Action:** search\n**Action Input:**\n```json\n{"q":"x"}\n```');
        expect(fenced.tool).toBe('search');
        expect(fenced.input).toEqual({ q: 'x' });

        const scalar = parseReactTurn('Action: lookup\nAction Input: ERR-4021');
        expect(scalar.input).toEqual({ input: 'ERR-4021' });

        const jsonScalar = parseReactTurn('Action: lookup\nAction Input: 42');
        expect(jsonScalar.input).toEqual({ input: 42 });
    });

    it('recognises a final answer, and prefers it when it follows an action', () => {
        expect(parseReactTurn('Thought: done\nFinal Answer: seven years'))
            .toMatchObject({ kind: 'final', answer: 'seven years' });

        const hedged = parseReactTurn('Action: search\nAction Input: {}\nFinal Answer: actually I know it');
        expect(hedged.kind).toBe('final');

        const actionFirst = parseReactTurn('Final Answer mentioned earlier\nAction: search\nAction Input: {}');
        expect(actionFirst.kind).toBe('action');
    });

    it('reports malformed output as data, not an exception', () => {
        const parsed = parseReactTurn('I will just ramble without the protocol.');
        expect(parsed.kind).toBe('malformed');
        expect(parsed.problem).toContain('No "Action:"');
    });

    it('renders a tool catalog and a transcript', () => {
        const catalog = renderToolCatalog([
            { name: 'search', description: 'Searches the corpus.', parameters: { q: 'query text' }, run: () => '' },
        ]);
        expect(catalog).toContain('- search: Searches the corpus. Input fields: q (query text).');

        expect(renderTranscript([
            { role: 'user', content: 'why?' },
            { role: 'assistant', content: 'thinking' },
            { role: 'tool', name: 'search', content: 'found' },
            { role: 'system', content: 'note' },
        ])).toBe('Question: why?\nthinking\nObservation (search): found\nnote');
    });
});

describe('createReactAgent', () => {
    const searchTool: AgentTool = {
        name: 'search',
        description: 'Searches the knowledge base.',
        parameters: { q: 'query' },
        run: (input) => `result for ${String(input['q'])}`,
    };

    it('reasons, acts, observes, and answers', async () => {
        const bridge = scriptedBridge([
            'Thought: I need data\nAction: search\nAction Input: {"q":"retention"}',
            'Thought: I have it\nFinal Answer: seven years',
        ]);
        const agent = createReactAgent({ bridge, tools: [searchTool] });

        const result = await agent.invoke({ task: 'How long do we keep records?' });

        expect(result.state.answer).toBe('seven years');
        expect(result.state.toolCalls).toBe(1);
        expect(result.state.turns.some((t) => t.role === 'tool' && t.content.includes('result for retention'))).toBe(true);
        expect(bridge.prompts[1]).toContain('Observation (search): result for retention');
    });

    it('feeds an unknown tool back as an observation instead of crashing', async () => {
        const bridge = scriptedBridge([
            'Action: teleport\nAction Input: {}',
            'Final Answer: I used what I had',
        ]);
        const agent = createReactAgent({ bridge, tools: [searchTool] });
        const result = await agent.invoke({ task: 'go' });

        expect(result.state.turns.some((t) => t.content.includes('Unknown tool "teleport"'))).toBe(true);
        expect(result.state.answer).toBe('I used what I had');
    });

    it('feeds a protocol violation back as an observation', async () => {
        const bridge = scriptedBridge([
            'I refuse to follow the format.',
            'Final Answer: fine, here it is',
        ]);
        const agent = createReactAgent({ bridge, tools: [searchTool] });
        const result = await agent.invoke({ task: 'go' });

        expect(result.state.turns.some((t) => t.content.includes('could not be parsed'))).toBe(true);
        expect(result.state.answer).toBe('fine, here it is');
    });

    it('turns a failing tool into an observation the agent can route around', async () => {
        const broken: AgentTool = {
            name: 'search',
            description: 'Searches.',
            run: () => { throw new Error('index offline'); },
        };
        const bridge = scriptedBridge([
            'Action: search\nAction Input: {"q":"x"}',
            'Final Answer: the index is down',
        ]);
        const agent = createReactAgent({ bridge, tools: [broken] });
        const result = await agent.invoke({ task: 'go' });

        expect(result.state.turns.some((t) => t.content === 'Tool failed: index offline')).toBe(true);
        expect(result.state.answer).toBe('the index is down');
    });

    it('enforces the iteration cap in routing rather than trusting the model', async () => {
        const bridge = scriptedBridge(['Action: search\nAction Input: {"q":"again"}']);
        const agent = createReactAgent({ bridge, tools: [searchTool], maxIterations: 2 });

        const result = await agent.invoke({ task: 'loop forever' });
        expect(result.state.toolCalls).toBe(2);
        expect(result.state.done).toBe(false);
        expect(bridge.prompts.some((p) => p.includes('no tool calls left'))).toBe(true);
    });

    it('traces tool execution under the node span', async () => {
        const tracer = new Tracer();
        const bridge = scriptedBridge([
            'Action: search\nAction Input: {"q":"x"}',
            'Final Answer: done',
        ]);
        const agent = createReactAgent({ bridge, tools: [searchTool], tracer });

        const result = await agent.invoke({ task: 'go' });
        const names = tracer.trace(result.traceId as string).map((s) => s.name);
        expect(names).toContain('node.reason');
        expect(names).toContain('tool.search');
    });
});

describe('createReflectionAgent', () => {
    it('revises until the critic approves', async () => {
        const bridge = scriptedBridge([
            'draft one',
            'Missing the number.\nVERDICT: REVISE',
            'draft two with seven years',
            'Good.\nVERDICT: APPROVED',
        ]);
        const agent = createReflectionAgent({ bridge, maxRevisions: 3 });

        const result = await agent.invoke({ task: 'state the retention period' });

        expect(result.state.accepted).toBe(true);
        expect(result.state.revisions).toBe(1);
        expect(result.state.draft).toBe('draft two with seven years');
        expect(result.state.history).toEqual(['draft one', 'draft two with seven years']);
        expect(bridge.prompts[2]).toContain('Reviewer feedback');
    });

    it('stops at the revision cap when the critic never approves', async () => {
        const bridge = scriptedBridge(['a draft', 'Still wrong.\nVERDICT: REVISE']);
        const agent = createReflectionAgent({ bridge, maxRevisions: 1 });

        const result = await agent.invoke({ task: 'impossible task' });
        expect(result.state.accepted).toBe(false);
        expect(result.state.revisions).toBe(1);
    });

    it('accepts a custom acceptance predicate', async () => {
        const bridge = scriptedBridge(['draft', 'looks fine to me']);
        const agent = createReflectionAgent({
            bridge,
            isAcceptable: (critique) => critique.includes('fine'),
        });
        expect((await agent.invoke({ task: 't' })).state.accepted).toBe(true);
    });
});

describe('createSupervisor', () => {
    const researcher = {
        name: 'researcher',
        description: 'Looks facts up.',
        run: async () => 'retention is seven years',
    };
    const writer = {
        name: 'writer',
        description: 'Writes prose.',
        run: async () => 'a polished paragraph',
    };

    it('delegates, collects worker results, then answers', async () => {
        const bridge = scriptedBridge([
            'NEXT: researcher',
            'NEXT: writer',
            `NEXT: ${FINISH}\nANSWER: records are kept seven years`,
        ]);
        const supervisor = createSupervisor({ bridge, workers: [researcher, writer] });

        const result = await supervisor.invoke({ task: 'explain retention' });

        expect(result.state.handoffs).toBe(2);
        expect(result.state.answer).toBe('records are kept seven years');
        expect(result.state.turns.filter((t) => t.role === 'tool').map((t) => t.name))
            .toEqual(['researcher', 'writer']);
    });

    it('degrades a hallucinated worker to FINISH instead of crashing the run', async () => {
        const bridge = scriptedBridge(['NEXT: oracle\nANSWER: I made that worker up']);
        const supervisor = createSupervisor({ bridge, workers: [researcher] });

        const result = await supervisor.invoke({ task: 'x' });
        expect(result.state.next).toBe(FINISH);
        expect(result.state.answer).toBe('I made that worker up');
        expect(result.state.handoffs).toBe(0);
    });

    it('records a worker failure as an observation', async () => {
        const broken = {
            name: 'researcher',
            description: 'Breaks.',
            run: async () => { throw new Error('rate limited'); },
        };
        const bridge = scriptedBridge([
            'NEXT: researcher',
            `NEXT: ${FINISH}\nANSWER: could not research`,
        ]);
        const supervisor = createSupervisor({ bridge, workers: [broken] });

        const result = await supervisor.invoke({ task: 'x' });
        expect(result.state.turns.some((t) => t.content === 'Worker failed: rate limited')).toBe(true);
        expect(result.state.answer).toBe('could not research');
    });

    it('caps delegations and pressures the supervisor to answer', async () => {
        const bridge = scriptedBridge(['NEXT: researcher']);
        const supervisor = createSupervisor({ bridge, workers: [researcher], maxHandoffs: 2 });

        const result = await supervisor.invoke({ task: 'x' });
        expect(result.state.handoffs).toBe(2);
        expect(bridge.prompts.some((p) => p.includes('used all delegations'))).toBe(true);
    });

    it('nests a ReAct agent as a worker — hierarchies compose', async () => {
        const workerBridge = scriptedBridge(['Final Answer: the sub-answer']);
        const reactWorker = createReactAgent({
            bridge: workerBridge,
            tools: [{ name: 'noop', description: 'Does nothing.', run: () => 'nothing' }],
        });

        const supervisorBridge = scriptedBridge([
            'NEXT: analyst',
            `NEXT: ${FINISH}\nANSWER: composed from the sub-answer`,
        ]);
        const supervisor = createSupervisor({
            bridge: supervisorBridge,
            workers: [asWorker('analyst', 'Runs a ReAct loop.', reactWorker)],
        });

        const result = await supervisor.invoke({ task: 'delegate down' });
        expect(result.state.turns.some((t) => t.content === 'the sub-answer')).toBe(true);
        expect(result.state.answer).toBe('composed from the sub-answer');
    });

    it('reports a worker that produced no answer', async () => {
        const emptyBridge = scriptedBridge(['Action: noop\nAction Input: {}']);
        const reactWorker = createReactAgent({
            bridge: emptyBridge,
            tools: [{ name: 'noop', description: 'Does nothing.', run: () => 'nothing' }],
            maxIterations: 1,
        });
        const worker = asWorker('analyst', 'Runs a ReAct loop.', reactWorker);
        expect(await worker.run('t', { transcript: [] })).toBe('The worker produced no answer.');
    });
});
