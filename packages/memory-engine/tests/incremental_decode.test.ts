/**
 * tests/incremental_decode.test.ts — the prefix cache.
 *
 * A tool-calling turn used to cost ONE FULL FORWARD PASS PER CANDIDATE: scoring
 * ~20 tool names replayed the whole (capped) prompt ~20 times, plus another pass
 * per scored enum/boolean/optional argument. The only mitigation was truncating
 * the prompt.
 *
 * `EvermindLM`'s single cross-position dependency is the depthwise causal conv,
 * so the exact decode state is the trailing `convKernel - 1` normalised conv
 * inputs per layer. These tests prove the incremental path is numerically
 * identical to the batch path, and measure the saving in real evaluated
 * positions.
 */

import { EvermindLM } from '../src/lm/evermind_lm';

const CFG = { vocabSize: 32, dModel: 16, numLayers: 2, convKernel: 3, hiddenDim: 24, numExperts: 3, topK: 2, seed: 4242 };

function lm(): EvermindLM {
    return new EvermindLM(CFG);
}

const PROMPT = [3, 9, 1, 14, 7, 2, 22, 5, 11, 8, 19, 4];
const CANDIDATES = [
    [12, 6], [30, 1, 5], [7], [18, 18, 2], [25, 3],
];

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
    let m = 0;
    for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]! - b[i]!));
    return m;
}

test('stepDecode over a whole sequence equals forward() position for position', () => {
    const m = lm();
    const batch = m.forward(PROMPT).logits;
    const inc = m.stepDecode(m.newDecodeState(), PROMPT).logits;
    expect(inc).toHaveLength(batch.length);
    for (let t = 0; t < batch.length; t++) {
        expect(maxAbsDiff(batch[t]!, inc[t]!)).toBeLessThan(1e-5);
    }
});

test('a cached prefix plus a continuation equals one batch pass over the concatenation', () => {
    const m = lm();
    for (const cand of CANDIDATES) {
        const whole = m.forward([...PROMPT, ...cand]).logits;
        const prefix = m.prefixState(PROMPT);
        const { logits } = m.stepDecode(prefix, cand);
        for (let i = 0; i < cand.length; i++) {
            expect(maxAbsDiff(whole[PROMPT.length + i]!, logits[i]!)).toBeLessThan(1e-5);
        }
    }
});

test('a prefix state is reusable — extending it does not mutate it', () => {
    const m = lm();
    const prefix = m.prefixState(PROMPT);
    const lengthBefore = prefix.length;
    const first = m.stepDecode(prefix, CANDIDATES[0]!).logits;
    m.stepDecode(prefix, CANDIDATES[1]!);
    const again = m.stepDecode(prefix, CANDIDATES[0]!).logits;
    expect(prefix.length).toBe(lengthBefore);
    for (let i = 0; i < first.length; i++) expect(maxAbsDiff(first[i]!, again[i]!)).toBe(0);
});

test('scoreContinuation matches a teacher-forced score computed the batch way', () => {
    const m = lm();
    const prefix = m.prefixState(PROMPT);
    for (const cand of CANDIDATES) {
        const cached = m.scoreContinuation(prefix, cand);

        // The batch equivalent: one full pass over prompt+candidate, reading the
        // log-prob of each candidate token from the position BEFORE it.
        const { logits } = m.forward([...PROMPT, ...cand]);
        let total = 0;
        for (let i = 0; i < cand.length; i++) {
            const row = logits[PROMPT.length + i - 1]!;
            let max = -Infinity;
            for (let v = 0; v < row.length; v++) if (row[v]! > max) max = row[v]!;
            let sum = 0;
            for (let v = 0; v < row.length; v++) sum += Math.exp(row[v]! - max);
            total += row[cand[i]!]! - (max + Math.log(sum));
        }
        expect(Math.abs(cached - total / cand.length)).toBeLessThan(1e-4);
    }
});

test('scoring N candidates costs the prompt ONCE, not N times', () => {
    const m = lm();
    const layers = CFG.numLayers;

    // Before: one full pass per candidate.
    m.resetStats();
    for (const cand of CANDIDATES) m.forward([...PROMPT, ...cand]);
    const replayed = m.positionsEvaluated;

    // After: one prefix pass, then only the candidate's own positions.
    m.resetStats();
    const prefix = m.prefixState(PROMPT);
    for (const cand of CANDIDATES) m.scoreContinuation(prefix, cand);
    const cached = m.positionsEvaluated;

    const candidateTokens = CANDIDATES.reduce((n, c) => n + c.length, 0);
    // Exact accounting, not a vague "it is faster".
    expect(replayed).toBe((CANDIDATES.length * PROMPT.length + candidateTokens) * layers);
    // scoreContinuation walks candidate.length - 1 positions (the last token is
    // predicted, never fed), on top of the single prompt pass.
    expect(cached).toBe((PROMPT.length + candidateTokens - CANDIDATES.length) * layers);
    expect(cached).toBeLessThan(replayed / 3);
});

test('generate() no longer replays the prompt for every new token', () => {
    const m = lm();
    const layers = CFG.numLayers;
    m.resetStats();
    const produced = m.generate(PROMPT, { maxNewTokens: 6, temperature: 0 });
    expect(produced.length).toBeGreaterThan(0);
    // Linear: the prompt once, plus one position per token fed back in. The old
    // implementation was quadratic — sum over n of (prompt + n).
    const quadratic = Array.from({ length: 6 }, (_, n) => PROMPT.length + n).reduce((a, b) => a + b, 0) * layers;
    expect(m.positionsEvaluated).toBeLessThanOrEqual((PROMPT.length + produced.length) * layers);
    expect(m.positionsEvaluated).toBeLessThan(quadratic);
});

test('greedy generation is unchanged by the incremental path', () => {
    const a = lm();
    const b = lm();
    // Reference: the old algorithm — a full forward per produced token.
    const tokens = [...PROMPT];
    const reference: number[] = [];
    for (let n = 0; n < 6; n++) {
        const { logits } = b.forward(tokens);
        const row = logits[logits.length - 1]!;
        let best = 0;
        for (let v = 1; v < row.length; v++) if (row[v]! > row[best]!) best = v;
        reference.push(best);
        tokens.push(best);
    }
    expect(a.generate(PROMPT, { maxNewTokens: 6, temperature: 0 })).toEqual(reference);
});
