/**
 * tests/InferenceRouter.test.ts
 * Unit tests for the InferenceRouter heuristics.
 * No external dependencies — all pure logic.
 */

import { InferenceRouter } from '../src/router/InferenceRouter.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

function router(opts: ConstructorParameters<typeof InferenceRouter>[0] = {}) {
    return new InferenceRouter({ hasBridge: true, ...opts });
}

// ── No bridge ─────────────────────────────────────────────────────────────────

test('returns ssm when no bridge is attached', async () => {
    const r = new InferenceRouter({ hasBridge: false, strategy: 'auto' });
    expect(await r.route('analyze everything step by step')).toBe('ssm');
});

// ── Fixed strategies ──────────────────────────────────────────────────────────

test('strategy=ssm always returns ssm', async () => {
    const r = router({ strategy: 'ssm' });
    expect(await r.route('analyze everything step by step')).toBe('ssm');
});

test('strategy=transformer always returns transformer', async () => {
    const r = router({ strategy: 'transformer' });
    expect(await r.route('hi')).toBe('transformer');
});

// ── Complexity patterns ───────────────────────────────────────────────────────

test('detects "step by step"', async () => {
    const r = router({ strategy: 'auto' });
    expect(await r.route('explain step by step')).toBe('transformer');
});

test('detects "step-by-step"', async () => {
    const r = router({ strategy: 'auto' });
    expect(await r.route('step-by-step guide')).toBe('transformer');
});

test('detects "analyze"', async () => {
    const r = router({ strategy: 'auto' });
    expect(await r.route('analyze this code')).toBe('transformer');
});

test('detects "analyse" (British spelling)', async () => {
    const r = router({ strategy: 'auto' });
    expect(await r.route('analyse the data')).toBe('transformer');
});

test('detects "explain why"', async () => {
    const r = router({ strategy: 'auto' });
    expect(await r.route('explain why this fails')).toBe('transformer');
});

test('detects "explain how"', async () => {
    const r = router({ strategy: 'auto' });
    expect(await r.route('explain how recursion works')).toBe('transformer');
});

test('detects "compare X and Y"', async () => {
    const r = router({ strategy: 'auto' });
    expect(await r.route('compare Mamba and Transformer and explain')).toBe('transformer');
});

test('detects "contrast"', async () => {
    const r = router({ strategy: 'auto' });
    expect(await r.route('contrast the two approaches')).toBe('transformer');
});

test('detects "pros and cons"', async () => {
    const r = router({ strategy: 'auto' });
    expect(await r.route('what are the pros and cons')).toBe('transformer');
});

test('detects "summarize"', async () => {
    const r = router({ strategy: 'auto' });
    expect(await r.route('summarize this article')).toBe('transformer');
});

test('detects "write a detailed"', async () => {
    const r = router({ strategy: 'auto' });
    expect(await r.route('write a detailed report')).toBe('transformer');
});

test('detects "what are the key differences"', async () => {
    const r = router({ strategy: 'auto' });
    expect(await r.route('what are the key differences between A and B')).toBe('transformer');
});

// ── Input length ──────────────────────────────────────────────────────────────

test('short input with no patterns routes to ssm', async () => {
    const r = router({ strategy: 'auto', longInputThreshold: 1200 });
    expect(await r.route('hello world')).toBe('ssm');
});

test('input over length threshold routes to transformer', async () => {
    const r = router({ strategy: 'auto', longInputThreshold: 50 });
    const longInput = 'x'.repeat(51);
    expect(await r.route(longInput)).toBe('transformer');
});

test('input exactly at threshold routes to ssm', async () => {
    const r = router({ strategy: 'auto', longInputThreshold: 50 });
    const input = 'x'.repeat(50);
    expect(await r.route(input)).toBe('ssm');
});

// ── Perplexity probe ──────────────────────────────────────────────────────────

test('high perplexity routes to transformer', async () => {
    const probe = jest.fn().mockResolvedValue(90);
    const r = router({ strategy: 'auto', perplexityThreshold: 80, perplexityProbe: probe });
    expect(await r.route('what is a quark')).toBe('transformer');
    expect(probe).toHaveBeenCalledWith('what is a quark');
});

test('low perplexity routes to ssm', async () => {
    const probe = jest.fn().mockResolvedValue(40);
    const r = router({ strategy: 'auto', perplexityThreshold: 80, perplexityProbe: probe });
    expect(await r.route('what is a quark')).toBe('ssm');
});

test('perplexity equal to threshold routes to ssm', async () => {
    const probe = jest.fn().mockResolvedValue(80);
    const r = router({ strategy: 'auto', perplexityThreshold: 80, perplexityProbe: probe });
    expect(await r.route('simple question')).toBe('ssm');
});

test('ctx.perplexity skips the probe call', async () => {
    const probe = jest.fn().mockResolvedValue(10);
    const r = router({ strategy: 'auto', perplexityThreshold: 80, perplexityProbe: probe });
    // Pass high perplexity via context — should route to transformer without calling probe
    const result = await r.route('simple', { perplexity: 90 });
    expect(result).toBe('transformer');
    expect(probe).not.toHaveBeenCalled();
});

test('no probe provided, short simple input → ssm', async () => {
    const r = router({ strategy: 'auto' });
    expect(await r.route('hello')).toBe('ssm');
});

// ── Custom thresholds ─────────────────────────────────────────────────────────

test('custom longInputThreshold is respected', async () => {
    const r = router({ strategy: 'auto', longInputThreshold: 10 });
    expect(await r.route('x'.repeat(11))).toBe('transformer');
    expect(await r.route('x'.repeat(10))).toBe('ssm');
});

test('custom perplexityThreshold is respected', async () => {
    const probe = jest.fn().mockResolvedValue(55);
    const r = router({ strategy: 'auto', perplexityThreshold: 50, perplexityProbe: probe });
    expect(await r.route('simple question')).toBe('transformer');
});
