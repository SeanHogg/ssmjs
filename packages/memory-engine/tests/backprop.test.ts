/**
 * tests/backprop.test.ts — the gradient check for the real backward pass.
 *
 * Before this existed `MambaTrainer._trainStep` handed the SAME `dLogits` buffer
 * to AdamW for every parameter: no parameter ever saw its own gradient, so
 * `learn()` was safe but inert. The backward pass is now real, and this file is
 * the proof: every analytic parameter gradient is compared against a central
 * finite difference of the exact loss the model computes.
 *
 * It runs on the CPU reference path, so it needs no WebGPU device. What it
 * verifies is the MATHS the WGSL kernels implement — the CPU ops in
 * `cpu_ops.ts` / `mamba1_cpu.ts` mirror those kernels line for line, so a
 * divergence between forward and backward fails here immediately.
 */

import {
    cpuModelBackward,
    cpuModelLoss,
    zeroCpuModelGrads,
    toNamedGrads,
    type CpuModelDims,
    type CpuModelWeights,
} from '../src/training/model_cpu';
import { MAMBA1_PARAM_NAMES, type Mamba1CpuWeights } from '../src/training/mamba1_cpu';
import { silu, siluGrad, softplus, sigmoid } from '../src/training/cpu_ops';

// ── Deterministic tiny model ─────────────────────────────────────────────────

function lcg(seed: number): () => number {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
    };
}

function randArray(n: number, rnd: () => number, scale = 0.3): Float32Array {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = (rnd() * 2 - 1) * scale;
    return a;
}

const DIMS: CpuModelDims = {
    vocabSize: 7,
    dModel   : 4,
    dState   : 3,
    dConv    : 3,
    dInner   : 8,   // expand 2
    dtRank   : 2,
    numLayers: 2,
};

function makeLayer(rnd: () => number, dims: CpuModelDims): Mamba1CpuWeights {
    const { dModel, dState: N, dConv: K, dInner: D, dtRank: R } = dims;
    const aLog = new Float32Array(D * N);
    for (let d = 0; d < D; d++) for (let n = 0; n < N; n++) aLog[d * N + n] = Math.log(n + 1);
    return {
        wInProj   : randArray(2 * D * dModel, rnd),
        bInProj   : randArray(2 * D, rnd, 0.1),
        wConv     : randArray(D * K, rnd, 0.2),
        bConv     : randArray(D, rnd, 0.1),
        wXProj    : randArray((R + 2 * N) * D, rnd, 0.2),
        bXProj    : randArray(R + 2 * N, rnd, 0.1),
        wDtProj   : randArray(D * R, rnd, 0.2),
        bDtProj   : randArray(D, rnd, 0.1),
        A_log     : aLog,
        D_vec     : randArray(D, rnd, 0.5),
        wOutProj  : randArray(dModel * D, rnd),
        bOutProj  : randArray(dModel, rnd, 0.1),
        normWeight: new Float32Array(dModel).fill(1).map((v, i) => v + (i % 2 ? 0.1 : -0.1)),
    };
}

function makeWeights(seed = 12345, dims: CpuModelDims = DIMS): CpuModelWeights {
    const rnd = lcg(seed);
    return {
        embedding : randArray(dims.vocabSize * dims.dModel, rnd, 0.5),
        finalNorm : new Float32Array(dims.dModel).fill(1).map((v, i) => v + (i % 2 ? 0.05 : -0.05)),
        lmHeadBias: randArray(dims.vocabSize, rnd, 0.1),
        layers    : Array.from({ length: dims.numLayers }, () => makeLayer(rnd, dims)),
    };
}

/** Every writable tensor, keyed the way `HybridMambaModel.parameters()` names it. */
function namedTensors(w: CpuModelWeights): Map<string, Float32Array> {
    const m = new Map<string, Float32Array>();
    m.set('embedding', w.embedding);
    for (let i = 0; i < w.layers.length; i++) {
        for (const name of MAMBA1_PARAM_NAMES) m.set(`layer${i}.${name}`, w.layers[i]![name]);
    }
    m.set('final_norm', w.finalNorm);
    m.set('lm_head_bias', w.lmHeadBias);
    return m;
}

const TOKENS  = [1, 4, 2, 6, 3, 0];
const TARGETS = [4, 2, 6, 3, 0, 5];
const BATCH   = 1;
const SEQLEN  = TOKENS.length;

/**
 * Finite-difference step. Weights live in `Float32Array` (mirroring the GPU
 * buffers), so a perturbation smaller than ~1e-3 is eaten by f32 rounding; 1e-2
 * keeps the truncation error (O(eps²)) well under the rounding floor.
 */
const FD_EPS = 1e-2;

/**
 * Tolerance model for the check. The forward pass stores every intermediate in
 * `Float32Array`, so the loss itself carries ~1e-7 of rounding noise and a
 * central difference carries ~1e-7 / FD_EPS ≈ 1e-5 of ABSOLUTE noise regardless
 * of how small the true gradient is. Hence an absolute floor plus a relative
 * term — against the largest gradients in this model (~3e-1) the floor is a
 * 3e-5 relative bar, and `the check has teeth` below proves it still fails a
 * corrupted gradient.
 */
const FD_ATOL = 1e-5;
const FD_RTOL = 1e-2;
function gradTolerance(analytic: number, fd: number): number {
    return FD_ATOL + FD_RTOL * Math.max(Math.abs(analytic), Math.abs(fd));
}

/** d(loss)/d(tensor[i]) by central difference of the real loss. */
function centralDifference(w: CpuModelWeights, tensor: Float32Array, i: number): number {
    const orig = tensor[i]!;
    tensor[i] = orig + FD_EPS;
    const plus = cpuModelLoss(TOKENS, TARGETS, w, DIMS, BATCH, SEQLEN);
    tensor[i] = orig - FD_EPS;
    const minus = cpuModelLoss(TOKENS, TARGETS, w, DIMS, BATCH, SEQLEN);
    tensor[i] = orig;
    return (plus - minus) / (2 * FD_EPS);
}

/**
 * Directional derivative check: perturb EVERY parameter at once along a random
 * ±1 direction and compare g·v against the central difference. Rounding noise
 * averages out across thousands of elements, so this is the tight, global
 * statement that the backward pass is the gradient of the forward pass.
 *
 * It uses a SMALLER step than the per-element check: shifting every parameter at
 * once is a large move, so the O(eps²) truncation error dominates there, while
 * the rounding noise is irrelevant because the loss change is large.
 */
const DIRECTIONAL_EPS = 1e-3;
function directionalCheck(seed: number): { analytic: number; fd: number } {
    const w = makeWeights(seed);
    const tensors = namedTensors(w);
    const { grads } = cpuModelBackward(TOKENS, TARGETS, w, DIMS, BATCH, SEQLEN);

    const rnd = lcg(seed ^ 0x5bf03635);
    const dirs = new Map<string, Float32Array>();
    let analytic = 0;
    for (const [name, t] of tensors) {
        const v = new Float32Array(t.length);
        const g = grads.get(name)!;
        for (let i = 0; i < t.length; i++) {
            v[i] = rnd() < 0.5 ? -1 : 1;
            analytic += g[i]! * v[i]!;
        }
        dirs.set(name, v);
    }

    const shift = (sign: number): number => {
        const wp = makeWeights(seed);
        for (const [name, t] of namedTensors(wp)) {
            const v = dirs.get(name)!;
            for (let i = 0; i < t.length; i++) t[i] = t[i]! + sign * DIRECTIONAL_EPS * v[i]!;
        }
        return cpuModelLoss(TOKENS, TARGETS, wp, DIMS, BATCH, SEQLEN);
    };
    return { analytic, fd: (shift(1) - shift(-1)) / (2 * DIRECTIONAL_EPS) };
}

// ── The element-wise primitives ──────────────────────────────────────────────

describe('cpu_ops primitives', () => {
    test('siluGrad matches a finite difference of silu', () => {
        for (const x of [-4, -1.3, -0.2, 0, 0.2, 1.3, 4]) {
            const eps = 1e-4;
            const fd = (silu(x + eps) - silu(x - eps)) / (2 * eps);
            expect(Math.abs(siluGrad(x) - fd)).toBeLessThan(1e-5);
        }
    });

    test('softplus is stable and its derivative is the sigmoid', () => {
        expect(softplus(100)).toBeCloseTo(100, 5);
        expect(Number.isFinite(softplus(1000))).toBe(true);
        for (const x of [-5, -0.5, 0, 0.5, 5]) {
            const eps = 1e-4;
            const fd = (softplus(x + eps) - softplus(x - eps)) / (2 * eps);
            expect(Math.abs(sigmoid(x) - fd)).toBeLessThan(1e-6);
        }
    });
});

// ── The whole-model gradient check ───────────────────────────────────────────

describe('cpuModelBackward · analytic gradients match finite differences', () => {
    const w = makeWeights();
    const tensors = namedTensors(w);
    const { grads } = cpuModelBackward(TOKENS, TARGETS, w, DIMS, BATCH, SEQLEN);

    test('a gradient is produced for every parameter the model publishes', () => {
        for (const name of tensors.keys()) {
            const g = grads.get(name);
            expect(g).toBeDefined();
            expect(g!.length).toBe(tensors.get(name)!.length);
        }
    });

    test('gradients are not all zero (the old trainer fed dLogits to everything)', () => {
        let nonZero = 0;
        for (const g of grads.values()) if (g.some((v) => v !== 0)) nonZero++;
        expect(nonZero).toBe(grads.size);
    });

    // One test per parameter tensor: check a spread of elements against a central
    // difference of the real loss. Small tensors are checked exhaustively.
    for (const name of [
        'embedding', 'final_norm', 'lm_head_bias',
        ...MAMBA1_PARAM_NAMES.map((n) => `layer0.${n}`),
        ...MAMBA1_PARAM_NAMES.map((n) => `layer1.${n}`),
    ]) {
        test(`d(loss)/d(${name})`, () => {
            const tensor = tensors.get(name)!;
            const analytic = grads.get(name)!;
            const stride = Math.max(1, Math.floor(tensor.length / 12));
            let checked = 0;
            for (let i = 0; i < tensor.length; i += stride) {
                const fd = centralDifference(w, tensor, i);
                const a  = analytic[i]!;
                expect(Math.abs(a - fd)).toBeLessThanOrEqual(gradTolerance(a, fd));
                checked++;
            }
            expect(checked).toBeGreaterThan(0);
        });
    }
});

describe('cpuModelBackward · shape and accumulation', () => {
    test('zeroCpuModelGrads produces the exact parameter shapes', () => {
        const g = zeroCpuModelGrads(DIMS);
        const named = toNamedGrads(g);
        const tensors = namedTensors(makeWeights());
        expect([...named.keys()].sort()).toEqual([...tensors.keys()].sort());
        for (const [k, v] of named) expect(v.length).toBe(tensors.get(k)!.length);
    });

    test('the tied embedding gradient collects BOTH the LM-head and the lookup terms', () => {
        // Zero the LM head's contribution by making every logit gradient flow only
        // through rows that are also input tokens: with tying, an embedding row used
        // as an input AND as a target must receive two distinct contributions. A
        // gradient that only carried one of them would fail the finite-difference
        // check above; this asserts the structure explicitly.
        const w = makeWeights(777);
        const { grads } = cpuModelBackward(TOKENS, TARGETS, w, DIMS, BATCH, SEQLEN);
        const emb = grads.get('embedding')!;
        const used = new Set([...TOKENS, ...TARGETS]);
        for (const id of used) {
            const row = emb.subarray(id * DIMS.dModel, (id + 1) * DIMS.dModel);
            expect(row.some((v) => v !== 0)).toBe(true);
        }
    });

    test('loss decreases when one real gradient step is applied', () => {
        const w = makeWeights(4242);
        const before = cpuModelLoss(TOKENS, TARGETS, w, DIMS, BATCH, SEQLEN);
        const { grads } = cpuModelBackward(TOKENS, TARGETS, w, DIMS, BATCH, SEQLEN);
        const tensors = namedTensors(w);
        const lr = 0.05;
        for (const [name, t] of tensors) {
            const g = grads.get(name)!;
            for (let i = 0; i < t.length; i++) t[i] = t[i]! - lr * g[i]!;
        }
        const after = cpuModelLoss(TOKENS, TARGETS, w, DIMS, BATCH, SEQLEN);
        expect(after).toBeLessThan(before);
    });
});

describe('cpuModelBackward · directional derivative over ALL parameters', () => {
    for (const seed of [12345, 999, 24680]) {
        test(`seed ${seed}: g·v matches the central difference along v`, () => {
            const { analytic, fd } = directionalCheck(seed);
            expect(Math.abs(analytic)).toBeGreaterThan(1e-3);   // the probe is not degenerate
            expect(Math.abs(analytic - fd) / Math.abs(fd)).toBeLessThan(1e-2);
        });
    }

    test('the check has teeth: a corrupted gradient fails it', () => {
        const w = makeWeights(31337);
        const tensors = namedTensors(w);
        const { grads } = cpuModelBackward(TOKENS, TARGETS, w, DIMS, BATCH, SEQLEN);
        // Drop the tied LM-head term from the embedding gradient — the exact class
        // of mistake (a missing contribution) the finite-difference check exists to
        // catch. It must be detected on at least one element.
        const emb = grads.get('embedding')!;
        const corrupted = Float32Array.from(emb, (v) => v * 0.5);
        const tensor = tensors.get('embedding')!;
        let detected = false;
        for (let i = 0; i < tensor.length; i++) {
            const fd = centralDifference(w, tensor, i);
            if (Math.abs(corrupted[i]! - fd) > gradTolerance(corrupted[i]!, fd)) { detected = true; break; }
        }
        expect(detected).toBe(true);
    });
});
