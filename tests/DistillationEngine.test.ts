/**
 * tests/DistillationEngine.test.ts
 * Unit tests for DistillationEngine using mock runtime + bridge.
 */

import { DistillationEngine } from '../src/distillation/DistillationEngine.js';
import { SSMError }           from '../src/errors/SSMError.js';
import type { SSMRuntime }    from '../src/runtime/SSMRuntime.js';
import type { TransformerBridge } from '../src/bridges/TransformerBridge.js';

// ── Mock factories ────────────────────────────────────────────────────────────

function makeRuntime(overrides: Partial<SSMRuntime> = {}): SSMRuntime {
    return {
        generate  : jest.fn().mockResolvedValue('ssm response'),
        stream    : jest.fn(),
        adapt     : jest.fn().mockResolvedValue({ losses: [0.5, 0.3], epochCount: 3 }),
        evaluate  : jest.fn().mockResolvedValue(50),
        save      : jest.fn().mockResolvedValue(undefined),
        load      : jest.fn().mockResolvedValue(false),
        destroy   : jest.fn(),
        get bridge()    { return undefined; },
        get destroyed() { return false; },
        get internals() { return {} as never; },
        streamHybrid: jest.fn(),
        ...overrides,
    } as unknown as SSMRuntime;
}

function makeBridge(overrides: Partial<TransformerBridge> = {}): TransformerBridge {
    return {
        generate        : jest.fn().mockResolvedValue('teacher response'),
        stream          : jest.fn(),
        supportsStreaming: true,
        ...overrides,
    };
}

// ── distill() ────────────────────────────────────────────────────────────────

test('distill calls bridge.generate with the input', async () => {
    const bridge  = makeBridge();
    const runtime = makeRuntime();
    const engine  = new DistillationEngine(runtime, bridge);

    await engine.distill('What is Mamba?');

    expect(bridge.generate).toHaveBeenCalledWith('What is Mamba?', undefined);
});

test('distill calls runtime.adapt with input + teacher output', async () => {
    const bridge  = makeBridge();
    const runtime = makeRuntime();
    const engine  = new DistillationEngine(runtime, bridge);

    await engine.distill('What is Mamba?');

    expect(runtime.adapt).toHaveBeenCalledWith(
        'What is Mamba?\nteacher response',
        expect.objectContaining({ wsla: true, epochs: 3 }),
    );
});

test('distill returns correct DistillResult shape', async () => {
    const bridge  = makeBridge();
    const runtime = makeRuntime();
    const engine  = new DistillationEngine(runtime, bridge);

    const result = await engine.distill('prompt');

    expect(result).toMatchObject({
        input        : 'prompt',
        teacherOutput: 'teacher response',
        adaptResult  : { losses: [0.5, 0.3], epochCount: 3 },
    });
});

test('distill forwards opts.generate to bridge.generate', async () => {
    const bridge  = makeBridge();
    const runtime = makeRuntime();
    const engine  = new DistillationEngine(runtime, bridge);

    await engine.distill('prompt', { generate: { maxTokens: 256, model: 'gpt-4o' } });

    expect(bridge.generate).toHaveBeenCalledWith('prompt', { maxTokens: 256, model: 'gpt-4o' });
});

test('distill forwards opts.adapt to runtime.adapt', async () => {
    const bridge  = makeBridge();
    const runtime = makeRuntime();
    const engine  = new DistillationEngine(runtime, bridge);

    await engine.distill('prompt', { adapt: { wsla: false, epochs: 10 } });

    expect(runtime.adapt).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ wsla: false, epochs: 10 }),
    );
});

test('distill throws SSMError(DISTILL_FAILED) when bridge.generate throws', async () => {
    const bridge  = makeBridge({ generate: jest.fn().mockRejectedValue(new Error('network error')) });
    const runtime = makeRuntime();
    const engine  = new DistillationEngine(runtime, bridge);

    await expect(engine.distill('prompt')).rejects.toMatchObject({ code: 'DISTILL_FAILED' });
});

test('distill throws SSMError(DISTILL_FAILED) when runtime.adapt throws', async () => {
    const bridge  = makeBridge();
    const runtime = makeRuntime({ adapt: jest.fn().mockRejectedValue(new Error('GPU error')) });
    const engine  = new DistillationEngine(runtime, bridge);

    await expect(engine.distill('prompt')).rejects.toMatchObject({ code: 'DISTILL_FAILED' });
});

test('distill thrown SSMError has correct code type', async () => {
    const bridge  = makeBridge({ generate: jest.fn().mockRejectedValue(new Error('fail')) });
    const runtime = makeRuntime();

    let caught: unknown;
    try {
        await new DistillationEngine(runtime, bridge).distill('x');
    } catch (e) {
        caught = e;
    }

    expect(caught).toBeInstanceOf(SSMError);
});

// ── distillBatch() ───────────────────────────────────────────────────────────

test('distillBatch processes all inputs in order', async () => {
    const bridge  = makeBridge();
    const runtime = makeRuntime();
    const engine  = new DistillationEngine(runtime, bridge);

    const result = await engine.distillBatch(['a', 'b', 'c']);

    expect(result.results).toHaveLength(3);
    expect(result.results.map(r => r.input)).toEqual(['a', 'b', 'c']);
});

test('distillBatch aggregates totalEpochs', async () => {
    const bridge  = makeBridge();
    const runtime = makeRuntime();
    const engine  = new DistillationEngine(runtime, bridge);

    const result = await engine.distillBatch(['a', 'b']);

    // Each distill returns epochCount: 3, two inputs → 6 total
    expect(result.totalEpochs).toBe(6);
});

test('distillBatch returns totalMs >= 0', async () => {
    const bridge  = makeBridge();
    const runtime = makeRuntime();

    const result = await new DistillationEngine(runtime, bridge).distillBatch(['x']);
    expect(result.totalMs).toBeGreaterThanOrEqual(0);
});

test('distillBatch with empty array returns zero results', async () => {
    const result = await new DistillationEngine(makeRuntime(), makeBridge()).distillBatch([]);
    expect(result.results).toHaveLength(0);
    expect(result.totalEpochs).toBe(0);
});
