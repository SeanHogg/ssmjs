/**
 * tests/peft_trainer.test.ts — MambaTrainer end to end, plus the efficient-training
 * toolkit it now carries: LoRA, gradient accumulation, optimizer-state sharding and
 * activation checkpointing.
 *
 * These were previously logged as blocked on "WebGPU + GPU numerical validation".
 * They are not: the trainer's gradients and optimiser run on the CPU reference, and
 * the GPU is only a byte store it reads parameters from and writes them back to —
 * which `tests/fakeGpu.ts` models exactly. What genuinely still needs a device is an
 * all-GPU (WGSL) backward pass, which is a different piece of work.
 */

import { HybridMambaModel } from '../src/model/mamba_model';
import { MambaTrainer } from '../src/training/trainer';
import { cpuModelBackward, paramMatrixShape, type CpuModelDims } from '../src/training/model_cpu';
import { fakeGpu } from './fakeGpu';

const CONFIG = {
    vocabSize: 24,
    dModel: 8,
    numLayers: 2,
    dState: 4,
    dConv: 3,
    expand: 2,
    seed: 99,
};

const DIMS: CpuModelDims = {
    vocabSize: CONFIG.vocabSize, dModel: CONFIG.dModel, dState: CONFIG.dState,
    dConv: CONFIG.dConv, dInner: CONFIG.expand * CONFIG.dModel,
    dtRank: Math.ceil(CONFIG.dModel / 16), numLayers: CONFIG.numLayers,
};

/** A repeating token pattern the model can actually overfit. */
const CORPUS = Array.from({ length: 64 }, (_, i) => (i * 5 + 3) % CONFIG.vocabSize);

function newModel(): { model: HybridMambaModel; gpu: ReturnType<typeof fakeGpu> } {
    const gpu = fakeGpu();
    return { model: new HybridMambaModel(gpu.device, CONFIG), gpu };
}

/** Read a parameter's bytes back out of the fake device's real backing store. */
function readParam(buf: GPUBuffer): Float32Array {
    const bytes = (buf as unknown as { bytes: Uint8Array }).bytes;
    return new Float32Array(bytes.slice().buffer);
}

function sameBytes(a: Float32Array, b: Float32Array): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
}

describe('paramMatrixShape', () => {
    test('names every trainable matrix and rejects the vectors', () => {
        const D = DIMS.dInner, N = DIMS.dState, R = DIMS.dtRank, K = DIMS.dConv, dM = DIMS.dModel;
        expect(paramMatrixShape('embedding', DIMS)).toEqual([DIMS.vocabSize, dM]);
        expect(paramMatrixShape('layer0.wInProj', DIMS)).toEqual([2 * D, dM]);
        expect(paramMatrixShape('layer1.wConv', DIMS)).toEqual([D, K]);
        expect(paramMatrixShape('layer0.wXProj', DIMS)).toEqual([R + 2 * N, D]);
        expect(paramMatrixShape('layer0.wDtProj', DIMS)).toEqual([D, R]);
        expect(paramMatrixShape('layer0.A_log', DIMS)).toEqual([D, N]);
        expect(paramMatrixShape('layer0.wOutProj', DIMS)).toEqual([dM, D]);
        for (const v of ['layer0.bInProj', 'layer0.bConv', 'layer0.D_vec', 'layer0.normWeight', 'final_norm', 'lm_head_bias']) {
            expect(paramMatrixShape(v, DIMS)).toBeNull();
        }
    });

    test('each shape multiplies out to the element count the model publishes', () => {
        const { model } = newModel();
        for (const p of model.parameters()) {
            const shape = paramMatrixShape(p.name, DIMS);
            if (!shape) continue;
            expect(shape[0] * shape[1]).toBe(p.numel);
        }
    });
});

describe('activation checkpointing', () => {
    test('produces bit-identical gradients to the retained-cache path', () => {
        const { model } = newModel();
        const trainer = new MambaTrainer(model);
        // Reach the CPU mirror through a train() call, then compare both backward paths.
        return trainer.train(CORPUS.slice(0, 32), { epochs: 1, seqLen: 8, learningRate: 0 }).then(() => {
            const mirror = (trainer as unknown as { _mirror: Parameters<typeof cpuModelBackward>[2] })._mirror;
            const inputs = CORPUS.slice(0, 8);
            const targets = CORPUS.slice(1, 9);
            const plain = cpuModelBackward(inputs, targets, mirror, DIMS, 1, 8);
            const ckpt = cpuModelBackward(inputs, targets, mirror, DIMS, 1, 8, { activationCheckpointing: true });
            expect(ckpt.loss).toBe(plain.loss);
            for (const [name, g] of plain.grads) {
                expect(Array.from(ckpt.grads.get(name)!)).toEqual(Array.from(g));
            }
        });
    });
});

describe('gradient accumulation', () => {
    test('accumulating two micro-batches equals one pass at the doubled batch', () => {
        const { model } = newModel();
        const trainer = new MambaTrainer(model);
        return trainer.train(CORPUS.slice(0, 32), { epochs: 1, seqLen: 8, learningRate: 0 }).then(() => {
            const mirror = (trainer as unknown as { _mirror: Parameters<typeof cpuModelBackward>[2] })._mirror;
            const a = { in: CORPUS.slice(0, 8), out: CORPUS.slice(1, 9) };
            const b = { in: CORPUS.slice(8, 16), out: CORPUS.slice(9, 17) };

            const ga = cpuModelBackward(a.in, a.out, mirror, DIMS, 1, 8);
            const gb = cpuModelBackward(b.in, b.out, mirror, DIMS, 1, 8);
            const both = cpuModelBackward([...a.in, ...b.in], [...a.out, ...b.out], mirror, DIMS, 2, 8);

            expect(both.loss).toBeCloseTo((ga.loss + gb.loss) / 2, 5);
            for (const [name, g] of both.grads) {
                const accumulated = ga.grads.get(name)!;
                const other = gb.grads.get(name)!;
                for (let i = 0; i < g.length; i++) {
                    expect(Math.abs((accumulated[i]! + other[i]!) / 2 - g[i]!)).toBeLessThan(1e-6);
                }
            }
        });
    });
});

describe('MambaTrainer on a fake device', () => {
    test('a real training run reduces the loss', async () => {
        const { model } = newModel();
        const trainer = new MambaTrainer(model);
        const losses = await trainer.train(CORPUS, { epochs: 6, seqLen: 8, learningRate: 0.02, maxDelta: 0 });
        expect(losses).toHaveLength(6);
        expect(losses.every((l) => Number.isFinite(l))).toBe(true);
        expect(losses[losses.length - 1]!).toBeLessThan(losses[0]!);
    });

    test('the reported gradient norm is real and finite', async () => {
        const { model } = newModel();
        const trainer = new MambaTrainer(model);
        const norms: number[] = [];
        await trainer.train(CORPUS, {
            epochs: 2, seqLen: 8, learningRate: 0.01,
            onEpochEnd: (_e, _l, gradNorm) => { norms.push(gradNorm!); },
        });
        expect(norms).toHaveLength(2);
        for (const n of norms) {
            expect(Number.isFinite(n)).toBe(true);
            expect(n).toBeGreaterThan(0);
        }
    });

    test('WSLA trains only the narrow projection and leaves the backbone frozen', async () => {
        const { model } = newModel();
        const trainer = new MambaTrainer(model);
        const before = new Map(model.parameters().map((p) => [p.name, readParam(p.buf)]));
        await trainer.train(CORPUS, { epochs: 2, seqLen: 8, learningRate: 0.05, wsla: true });
        const after = new Map(model.parameters().map((p) => [p.name, readParam(p.buf)]));

        const changed = [...before.keys()].filter((n) => !sameBytes(before.get(n)!, after.get(n)!));
        // Only the selective (delta, B, C) projection may move; A_log, the conv, the
        // output projection, the embedding and the final norm must all be untouched.
        expect(changed.sort()).toEqual(['layer0.bXProj', 'layer0.wXProj', 'layer1.bXProj', 'layer1.wXProj']);
    });

    test('gradient accumulation changes the step count, not the outcome shape', async () => {
        const { model } = newModel();
        const trainer = new MambaTrainer(model);
        const losses = await trainer.train(CORPUS, {
            epochs: 3, seqLen: 8, batchSize: 1, gradientAccumulation: 2, learningRate: 0.02, maxDelta: 0,
        });
        expect(losses).toHaveLength(3);
        expect(losses[2]!).toBeLessThan(losses[0]!);
    });
});

describe('LoRA on MambaTrainer', () => {
    test('trains adapters, not weights — and the base is provably frozen', async () => {
        const { model } = newModel();
        const trainer = new MambaTrainer(model);
        await trainer.train(CORPUS, { epochs: 3, seqLen: 8, learningRate: 0.05, maxDelta: 0, lora: { rank: 2, seed: 5 } });

        expect(trainer.adapters.size).toBeGreaterThan(0);
        for (const name of trainer.adapters.keys()) {
            expect(name).toMatch(/^layer\d+\.(wInProj|wXProj|wDtProj|wOutProj)$/);
        }
        // The base snapshot is untouched; the merged weight differs from it by
        // exactly the adapter delta.
        const bases = (trainer as unknown as { _loraBase: Map<string, Float32Array> })._loraBase;
        const mirror = (trainer as unknown as { _mirrorByName: Map<string, Float32Array> })._mirrorByName;
        let moved = 0;
        for (const [name, adapter] of trainer.adapters) {
            const base = bases.get(name)!;
            const merged = mirror.get(name)!;
            const delta = adapter.delta();
            for (let i = 0; i < base.length; i++) {
                expect(Math.abs(merged[i]! - (base[i]! + delta[i]!))).toBeLessThan(1e-6);
                if (delta[i] !== 0) moved++;
            }
        }
        expect(moved).toBeGreaterThan(0);   // B started at zero; training moved it
    });

    test('trains far fewer scalars than a full fine-tune', async () => {
        const { model } = newModel();
        const full = new MambaTrainer(model);
        const fullCount = full.trainableParamCount();

        const { model: m2 } = newModel();
        const lora = new MambaTrainer(m2);
        await lora.train(CORPUS.slice(0, 24), { epochs: 1, seqLen: 8, learningRate: 0.01, lora: { rank: 1 } });
        expect(lora.trainableParamCount()).toBeLessThan(fullCount / 4);
    });

    test('a rank larger than the smallest target dimension is clamped, not rejected', async () => {
        const { model } = newModel();
        const trainer = new MambaTrainer(model);
        // dtRank is 1 for dModel=8, so wDtProj is dInner x 1 — rank 64 must clamp.
        await trainer.train(CORPUS.slice(0, 24), { epochs: 1, seqLen: 8, learningRate: 0.01, lora: { rank: 64 } });
        for (const [name, adapter] of trainer.adapters) {
            const shape = paramMatrixShape(name, DIMS)!;
            expect(adapter.rank).toBeLessThanOrEqual(Math.min(shape[0], shape[1]));
        }
    });

    test('an unmatched target list fails loudly instead of silently training nothing', async () => {
        const { model } = newModel();
        const trainer = new MambaTrainer(model);
        await expect(
            trainer.train(CORPUS.slice(0, 24), { epochs: 1, seqLen: 8, lora: { targets: ['bConv'] } }),
        ).rejects.toThrow(/no trainable parameter matched targets/);
    });
});

describe('optimizer-state sharding', () => {
    test('two shards together hold what one unsharded trainer holds', async () => {
        const opts = { epochs: 1, seqLen: 8, learningRate: 0.01, maxDelta: 0 };
        const { model: mAll } = newModel();
        const all = new MambaTrainer(mAll);
        await all.train(CORPUS, opts);

        const { model: m0 } = newModel();
        const s0 = new MambaTrainer(m0);
        await s0.train(CORPUS, { ...opts, shard: { index: 0, count: 2 } });

        const { model: m1 } = newModel();
        const s1 = new MambaTrainer(m1);
        await s1.train(CORPUS, { ...opts, shard: { index: 1, count: 2 } });

        expect(s0.optimizerStateBytes()).toBeGreaterThan(0);
        expect(s1.optimizerStateBytes()).toBeGreaterThan(0);
        expect(s0.optimizerStateBytes() + s1.optimizerStateBytes()).toBe(all.optimizerStateBytes());
        // Each shard holds strictly less than the whole.
        expect(s0.optimizerStateBytes()).toBeLessThan(all.optimizerStateBytes());
    });

    test('an invalid shard is rejected', async () => {
        const { model } = newModel();
        const trainer = new MambaTrainer(model);
        await expect(trainer.train(CORPUS, { epochs: 1, seqLen: 8, shard: { index: 2, count: 2 } }))
            .rejects.toThrow(/invalid shard/);
    });
});

describe('the gradient engine refuses what it cannot differentiate', () => {
    test('a hybrid schedule fails loudly instead of training on absent gradients', () => {
        const gpu = fakeGpu();
        expect(() => new MambaTrainer(new HybridMambaModel(gpu.device, {
            ...CONFIG,
            nHeads: 2,
            layers: [{ type: 'mamba1' }, { type: 'attention' }],
        }))).toThrow(/differentiates mamba1 layers only/);
    });
});
