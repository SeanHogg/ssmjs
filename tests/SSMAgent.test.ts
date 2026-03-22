/**
 * tests/SSMAgent.test.ts
 * Unit tests for SSMAgent — prompt building, history, memory, routing.
 */

import { jest } from '@jest/globals';
import { SSMAgent }     from '../src/agent/SSMAgent.js';
import type { SSMRuntime } from '../src/runtime/SSMRuntime.js';
import type { MemoryStore, MemoryEntry } from '../src/memory/MemoryStore.js';

// ── Mock factories ────────────────────────────────────────────────────────────

function makeRuntime(overrides: Partial<SSMRuntime> = {}): SSMRuntime {
    return {
        generate    : jest.fn<any>().mockResolvedValue('assistant reply'),
        stream      : jest.fn<any>().mockImplementation(async function*() { yield 'tok'; }),
        adapt       : jest.fn<any>().mockResolvedValue({ losses: [], epochCount: 0 }),
        evaluate    : jest.fn<any>().mockResolvedValue(50),
        save        : jest.fn<any>().mockResolvedValue(undefined),
        load        : jest.fn<any>().mockResolvedValue(false),
        destroy     : jest.fn<any>(),
        streamHybrid: jest.fn<any>(),
        get bridge()    { return undefined; },
        get destroyed() { return false; },
        get internals() { return {} as never; },
        ...overrides,
    } as unknown as SSMRuntime;
}

type MockMemory = {
    remember  : jest.Mock;
    recall    : jest.Mock;
    recallAll : jest.Mock;
    forget    : jest.Mock;
    clear     : jest.Mock;
    saveWeights : jest.Mock;
    loadWeights : jest.Mock;
};

function makeMemory(facts: MemoryEntry[] = []): MockMemory & MemoryStore {
    return {
        remember    : jest.fn<any>().mockResolvedValue(undefined),
        recall      : jest.fn<any>().mockResolvedValue(undefined),
        recallAll   : jest.fn<any>().mockResolvedValue(facts),
        forget      : jest.fn<any>().mockResolvedValue(undefined),
        clear       : jest.fn<any>().mockResolvedValue(undefined),
        saveWeights : jest.fn<any>().mockResolvedValue(undefined),
        loadWeights : jest.fn<any>().mockResolvedValue(false),
    } as unknown as MockMemory & MemoryStore;
}

// ── think() — basic ───────────────────────────────────────────────────────────

test('think calls runtime.generate and returns the response', async () => {
    const runtime = makeRuntime();
    const agent   = new SSMAgent({ runtime });

    const reply = await agent.think('hello');
    expect(reply).toBe('assistant reply');
    expect(runtime.generate).toHaveBeenCalledTimes(1);
});

test('think trims hallucinated User turns from model output', async () => {
    const runtime = makeRuntime({
        generate: jest.fn<any>().mockResolvedValue('answer\nUser: next question\nAssistant: ...'),
    });
    const agent = new SSMAgent({ runtime });

    const reply = await agent.think('hello');
    expect(reply).toBe('answer');
});

test('think appends user + assistant messages to history', async () => {
    const agent = new SSMAgent({ runtime: makeRuntime() });

    await agent.think('first');
    expect(agent.history).toHaveLength(2);
    expect(agent.history[0]).toEqual({ role: 'user',      content: 'first' });
    expect(agent.history[1]).toEqual({ role: 'assistant', content: 'assistant reply' });
});

test('turnCount increments after each think()', async () => {
    const agent = new SSMAgent({ runtime: makeRuntime() });
    expect(agent.turnCount).toBe(0);
    await agent.think('one');
    expect(agent.turnCount).toBe(1);
    await agent.think('two');
    expect(agent.turnCount).toBe(2);
});

// ── think() — prompt structure ────────────────────────────────────────────────

test('prompt includes System: line', async () => {
    const runtime = makeRuntime();
    const agent   = new SSMAgent({ runtime, systemPrompt: 'You are a pirate.' });

    await agent.think('arr');
    const prompt = (runtime.generate as jest.Mock<any>).mock.calls[0][0] as string;
    expect(prompt).toContain('System: You are a pirate.');
});

test('prompt ends with "User: <input>\\nAssistant:"', async () => {
    const runtime = makeRuntime();
    const agent   = new SSMAgent({ runtime });

    await agent.think('my question');
    const prompt = (runtime.generate as jest.Mock<any>).mock.calls[0][0] as string;
    expect(prompt).toMatch(/User: my question\nAssistant:$/);
});

test('prompt includes history from prior turns', async () => {
    const runtime = makeRuntime();
    const agent   = new SSMAgent({ runtime });

    await agent.think('first');
    await agent.think('second');

    const secondPrompt = (runtime.generate as jest.Mock<any>).mock.calls[1][0] as string;
    expect(secondPrompt).toContain('User: first');
    expect(secondPrompt).toContain('Assistant: assistant reply');
    expect(secondPrompt).toContain('User: second');
});

test('per-turn systemPrompt override is used', async () => {
    const runtime = makeRuntime();
    const agent   = new SSMAgent({ runtime, systemPrompt: 'default' });

    await agent.think('hi', { systemPrompt: 'override' });
    const prompt = (runtime.generate as jest.Mock<any>).mock.calls[0][0] as string;
    expect(prompt).toContain('System: override');
    expect(prompt).not.toContain('System: default');
});

// ── history trimming ──────────────────────────────────────────────────────────

test('history is trimmed to maxHistoryTurns pairs', async () => {
    const runtime = makeRuntime();
    const agent   = new SSMAgent({ runtime, maxHistoryTurns: 1 });

    await agent.think('turn1');
    await agent.think('turn2');

    // After turn2, the third prompt should only contain turn2 history (not turn1)
    await agent.think('turn3');
    const thirdPrompt = (runtime.generate as jest.Mock<any>).mock.calls[2][0] as string;
    expect(thirdPrompt).toContain('User: turn2');
    expect(thirdPrompt).not.toContain('User: turn1');
});

// ── clearHistory() ────────────────────────────────────────────────────────────

test('clearHistory resets history to empty', async () => {
    const agent = new SSMAgent({ runtime: makeRuntime() });
    await agent.think('one');
    agent.clearHistory();
    expect(agent.history).toHaveLength(0);
    expect(agent.turnCount).toBe(0);
});

// ── thinkStream() ─────────────────────────────────────────────────────────────

test('thinkStream yields tokens from runtime.stream', async () => {
    const runtime = makeRuntime({
        stream: jest.fn<any>().mockImplementation(async function*() {
            yield 'hello';
            yield ' world';
        }),
    });
    const agent  = new SSMAgent({ runtime });
    const tokens: string[] = [];

    for await (const t of agent.thinkStream('hi')) {
        tokens.push(t);
    }

    expect(tokens).toEqual(['hello', ' world']);
});

test('thinkStream appends history after stream completes', async () => {
    const runtime = makeRuntime({
        stream: jest.fn<any>().mockImplementation(async function*() {
            yield 'response';
        }),
    });
    const agent = new SSMAgent({ runtime });

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    for await (const _t of agent.thinkStream('prompt')) { /* drain */ }

    expect(agent.turnCount).toBe(1);
    expect(agent.history[1].content).toBe('response');
});

// ── memory ────────────────────────────────────────────────────────────────────

test('remember throws SSMError(MEMORY_UNAVAILABLE) without a MemoryStore', async () => {
    const agent = new SSMAgent({ runtime: makeRuntime() });
    await expect(agent.remember('key', 'val'))
        .rejects.toMatchObject({ code: 'MEMORY_UNAVAILABLE' });
});

test('remember delegates to memory.remember', async () => {
    const memory = makeMemory();
    const agent  = new SSMAgent({ runtime: makeRuntime(), memory });

    await agent.remember('lang', 'TypeScript');
    expect(memory.remember).toHaveBeenCalledWith('lang', 'TypeScript');
});

test('recall returns undefined without a MemoryStore', async () => {
    const agent = new SSMAgent({ runtime: makeRuntime() });
    expect(await agent.recall('anything')).toBeUndefined();
});

test('recall delegates to memory.recall', async () => {
    const memory = makeMemory();
    (memory.recall as jest.Mock<any>).mockResolvedValue({ key: 'lang', content: 'TypeScript', timestamp: 0 });
    const agent = new SSMAgent({ runtime: makeRuntime(), memory });

    const val = await agent.recall('lang');
    expect(val).toBe('TypeScript');
});

test('facts matching input keys are injected into the prompt', async () => {
    const runtime = makeRuntime();
    const memory  = makeMemory([
        { key: 'stack', content: 'React + TypeScript', timestamp: 1 },
        { key: 'goal',  content: 'Build a chat app',   timestamp: 0 },
    ]);
    const agent = new SSMAgent({ runtime, memory });

    await agent.think('What stack should I use?');
    const prompt = (runtime.generate as jest.Mock<any>).mock.calls[0][0] as string;

    // "stack" appears in the input
    expect(prompt).toContain('Fact (stack): React + TypeScript');
    // "goal" does NOT appear in the input
    expect(prompt).not.toContain('Fact (goal):');
});

test('injectAllFacts injects all facts regardless of key match', async () => {
    const runtime = makeRuntime();
    const memory  = makeMemory([
        { key: 'stack', content: 'React',          timestamp: 1 },
        { key: 'goal',  content: 'Build chat app', timestamp: 0 },
    ]);
    const agent = new SSMAgent({ runtime, memory });

    await agent.think('unrelated question', { injectAllFacts: true });
    const prompt = (runtime.generate as jest.Mock<any>).mock.calls[0][0] as string;

    expect(prompt).toContain('Fact (stack):');
    expect(prompt).toContain('Fact (goal):');
});

// ── learn() ───────────────────────────────────────────────────────────────────

test('learn delegates to runtime.adapt', async () => {
    const runtime = makeRuntime();
    const agent   = new SSMAgent({ runtime });

    await agent.learn('training text', { wsla: true, epochs: 5 });
    expect(runtime.adapt).toHaveBeenCalledWith('training text', { wsla: true, epochs: 5 });
});
