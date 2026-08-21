/**
 * model_cpu.ts — the exact CPU forward + backward for a whole
 * {@link HybridMambaModel}: token embedding → layer stack → final RMSNorm →
 * tied LM head → cross-entropy.
 *
 * This is the gradient engine {@link MambaTrainer} uses. Before it existed the
 * trainer handed the SAME `dLogits` buffer to AdamW for every parameter, so no
 * parameter ever received its own gradient — `learn()` was safe but inert. Now
 * the backward pass is real, and `tests/backprop.test.ts` proves it against
 * finite differences without needing a GPU.
 *
 * Parameter names match {@link HybridMambaModel.parameters} exactly
 * (`embedding`, `layer{i}.{name}`, `final_norm`), which is how the trainer
 * pairs each parameter with its OWN gradient.
 */

import {
    linearForward,
    rmsNormForward,
    rmsNormBackward,
    RMSNORM_EPS,
} from './cpu_ops.js';
import {
    mamba1CpuForward,
    mamba1CpuBackward,
    zeroMamba1Grads,
    MAMBA1_PARAM_NAMES,
    type Mamba1CpuWeights,
    type Mamba1CpuCache,
    type Mamba1CpuDims,
    type Mamba1CpuGrads,
} from './mamba1_cpu.js';
import { crossEntropyLoss, crossEntropyGrad } from './autograd.js';

/** Layer types the CPU gradient engine can differentiate today. */
export const CPU_GRADIENT_LAYER_TYPES = ['mamba1'] as const;
export type CpuGradientLayerType = (typeof CPU_GRADIENT_LAYER_TYPES)[number];

export interface CpuModelDims {
    vocabSize: number;
    dModel   : number;
    dState   : number;
    dConv    : number;
    dInner   : number;
    dtRank   : number;
    numLayers: number;
}

/** Every tensor the CPU model reads, mirrored out of the GPU buffers. */
export interface CpuModelWeights {
    embedding  : Float32Array;            // (vocabSize, dModel) — tied with the LM head
    finalNorm  : Float32Array;            // (dModel,)
    lmHeadBias : Float32Array;            // (vocabSize,)
    layers     : Mamba1CpuWeights[];
}

export interface CpuModelGrads {
    embedding  : Float32Array;
    finalNorm  : Float32Array;
    lmHeadBias : Float32Array;
    layers     : Mamba1CpuGrads[];
}

/** Knobs on one backward pass. */
export interface CpuBackwardOptions {
    /**
     * Recompute each layer's activations during the backward pass instead of
     * retaining every layer's cache from the forward. Trades one extra forward per
     * layer for O(1) instead of O(layers) retained activations; the gradients are
     * numerically identical (`tests/peft_trainer.test.ts` asserts bit equality).
     */
    activationCheckpointing?: boolean;
}

/** Loss + the gradient of every parameter, keyed by `parameters()` name. */
export interface CpuBackwardResult {
    loss : number;
    grads: Map<string, Float32Array>;
}

export function zeroCpuModelGrads(dims: CpuModelDims): CpuModelGrads {
    const blockDims: Mamba1CpuDims = {
        dModel: dims.dModel, dState: dims.dState, dConv: dims.dConv,
        dInner: dims.dInner, dtRank: dims.dtRank, batch: 1, seqLen: 1,
    };
    return {
        embedding : new Float32Array(dims.vocabSize * dims.dModel),
        finalNorm : new Float32Array(dims.dModel),
        lmHeadBias: new Float32Array(dims.vocabSize),
        layers    : Array.from({ length: dims.numLayers }, () => zeroMamba1Grads(blockDims)),
    };
}

/** Flatten a {@link CpuModelGrads} into the `parameters()` naming the optimizer uses. */
export function toNamedGrads(g: CpuModelGrads): Map<string, Float32Array> {
    const out = new Map<string, Float32Array>();
    out.set('embedding', g.embedding);
    for (let i = 0; i < g.layers.length; i++) {
        const lg = g.layers[i]!;
        for (const name of MAMBA1_PARAM_NAMES) out.set(`layer${i}.${name}`, lg[name]);
    }
    out.set('final_norm', g.finalNorm);
    out.set('lm_head_bias', g.lmHeadBias);
    return out;
}

/** The model's tensors keyed the same way — the inverse of {@link cpuWeightsFromNamed}. */
export function namedWeights(w: CpuModelWeights): Map<string, Float32Array> {
    const out = new Map<string, Float32Array>();
    out.set('embedding', w.embedding);
    for (let i = 0; i < w.layers.length; i++) {
        const lw = w.layers[i]!;
        for (const name of MAMBA1_PARAM_NAMES) out.set(`layer${i}.${name}`, lw[name]);
    }
    out.set('final_norm', w.finalNorm);
    out.set('lm_head_bias', w.lmHeadBias);
    return out;
}

/**
 * Assemble a {@link CpuModelWeights} from tensors keyed by `parameters()` name —
 * how {@link MambaTrainer} mirrors the GPU buffers into the gradient engine.
 * Throws on a missing or wrongly-sized tensor rather than training on garbage.
 */
export function cpuWeightsFromNamed(named: Map<string, Float32Array>, dims: CpuModelDims): CpuModelWeights {
    const take = (key: string, expected: number): Float32Array => {
        const t = named.get(key);
        if (!t) throw new Error(`cpuWeightsFromNamed: missing tensor "${key}"`);
        if (t.length !== expected) {
            throw new Error(`cpuWeightsFromNamed: "${key}" has ${t.length} elements, expected ${expected}`);
        }
        return t;
    };
    const { dModel, dState: N, dConv: K, dInner: D, dtRank: R } = dims;
    const sizes: Record<keyof Mamba1CpuWeights, number> = {
        wInProj: 2 * D * dModel, bInProj: 2 * D, wConv: D * K, bConv: D,
        wXProj: (R + 2 * N) * D, bXProj: R + 2 * N, wDtProj: D * R, bDtProj: D,
        A_log: D * N, D_vec: D, wOutProj: dModel * D, bOutProj: dModel, normWeight: dModel,
    };
    const layers: Mamba1CpuWeights[] = [];
    for (let i = 0; i < dims.numLayers; i++) {
        const lw = {} as Mamba1CpuWeights;
        for (const name of MAMBA1_PARAM_NAMES) lw[name] = take(`layer${i}.${name}`, sizes[name]);
        layers.push(lw);
    }
    return {
        embedding : take('embedding', dims.vocabSize * dModel),
        finalNorm : take('final_norm', dModel),
        lmHeadBias: take('lm_head_bias', dims.vocabSize),
        layers,
    };
}

interface ModelCache {
    hiddenIn : Float32Array[];   // per layer: the input that layer saw
    layer    : Mamba1CpuCache[];
    lastHidden: Float32Array;    // (M, dModel) output of the final layer
    normOut  : Float32Array;     // (M, dModel) after the final RMSNorm
    normInv  : Float32Array;     // (M,)
    logits   : Float32Array;     // (M, vocabSize)
}

function forwardInternal(
    tokenIds: ArrayLike<number>,
    w: CpuModelWeights,
    dims: CpuModelDims,
    batch: number,
    seqLen: number,
    /** Drop each layer's activation cache after use — see {@link CpuBackwardOptions}. */
    checkpointed = false,
): ModelCache {
    const { vocabSize, dModel } = dims;
    const M = batch * seqLen;

    // Token embedding lookup.
    let hidden: Float32Array = new Float32Array(M * dModel);
    for (let r = 0; r < M; r++) {
        const id = tokenIds[r]!;
        if (id < 0 || id >= vocabSize) throw new Error(`token id ${id} out of range [0, ${vocabSize})`);
        hidden.set(w.embedding.subarray(id * dModel, (id + 1) * dModel), r * dModel);
    }

    const blockDims: Mamba1CpuDims = {
        dModel, dState: dims.dState, dConv: dims.dConv,
        dInner: dims.dInner, dtRank: dims.dtRank, batch, seqLen,
    };

    const hiddenIn: Float32Array[] = [];
    const layerCaches: Mamba1CpuCache[] = [];
    for (let i = 0; i < dims.numLayers; i++) {
        hiddenIn.push(hidden);
        const { output, cache } = mamba1CpuForward(hidden, w.layers[i]!, blockDims);
        // Under activation checkpointing only the layer INPUTS are retained; each
        // layer's cache is recomputed from its input when its backward runs. Memory
        // drops from O(layers) caches to one, at the cost of a second forward per
        // layer — gradients are numerically identical either way.
        if (!checkpointed) layerCaches.push(cache);
        hidden = output;
    }

    // Final RMSNorm.
    const normOut = new Float32Array(M * dModel);
    const normInv = new Float32Array(M);
    rmsNormForward(hidden, w.finalNorm, M, dModel, normOut, normInv, RMSNORM_EPS);

    // Tied LM head: logits = normOut @ embedding^T + lmHeadBias.
    const logits = new Float32Array(M * vocabSize);
    linearForward(normOut, w.embedding, w.lmHeadBias, M, dModel, vocabSize, logits);

    return { hiddenIn, layer: layerCaches, lastHidden: hidden, normOut, normInv, logits };
}

/** Logits only — the CPU mirror of `HybridMambaModel.forward`. */
export function cpuModelForward(
    tokenIds: ArrayLike<number>,
    w: CpuModelWeights,
    dims: CpuModelDims,
    batch: number,
    seqLen: number,
): Float32Array {
    return forwardInternal(tokenIds, w, dims, batch, seqLen).logits;
}

/** Mean next-token cross-entropy over the sequence — the trainer's loss. */
export function cpuModelLoss(
    tokenIds: ArrayLike<number>,
    targets: ArrayLike<number>,
    w: CpuModelWeights,
    dims: CpuModelDims,
    batch: number,
    seqLen: number,
): number {
    const logits = cpuModelForward(tokenIds, w, dims, batch, seqLen);
    const M = batch * seqLen;
    const V = dims.vocabSize;
    let total = 0;
    for (let r = 0; r < M; r++) {
        total += crossEntropyLoss(logits.subarray(r * V, (r + 1) * V), targets[r]!);
    }
    return total / M;
}

/**
 * Forward + backward. Returns the mean cross-entropy loss and the gradient of
 * EVERY parameter, keyed by its `parameters()` name.
 *
 * The loss is averaged over the `batch * seqLen` predicted positions, so the
 * gradients are the gradients of that mean — the quantity the trainer reports.
 */
export function cpuModelBackward(
    tokenIds: ArrayLike<number>,
    targets: ArrayLike<number>,
    w: CpuModelWeights,
    dims: CpuModelDims,
    batch: number,
    seqLen: number,
    opts: CpuBackwardOptions = {},
): CpuBackwardResult {
    const { vocabSize: V, dModel } = dims;
    const M = batch * seqLen;
    const checkpointed = opts.activationCheckpointing ?? false;
    const cache = forwardInternal(tokenIds, w, dims, batch, seqLen, checkpointed);
    const g = zeroCpuModelGrads(dims);

    // Cross-entropy over every position, averaged.
    let total = 0;
    const dLogits = new Float32Array(M * V);
    for (let r = 0; r < M; r++) {
        const slice = cache.logits.subarray(r * V, (r + 1) * V);
        const target = targets[r]!;
        total += crossEntropyLoss(slice, target);
        const gr = crossEntropyGrad(slice, target);
        for (let v = 0; v < V; v++) dLogits[r * V + v] = gr[v]! / M;
    }
    const loss = total / M;

    // LM head (tied): logits = normOut @ embedding^T + bias.
    // dNormOut = dLogits @ embedding ; dEmbedding += dLogits^T @ normOut.
    const dNormOut = new Float32Array(M * dModel);
    for (let r = 0; r < M; r++) {
        const lo = r * V;
        const no = r * dModel;
        for (let v = 0; v < V; v++) {
            const gl = dLogits[lo + v]!;
            if (gl === 0) continue;
            g.lmHeadBias[v] = g.lmHeadBias[v]! + gl;
            const eo = v * dModel;
            for (let i = 0; i < dModel; i++) {
                dNormOut[no + i] = dNormOut[no + i]! + gl * w.embedding[eo + i]!;
                g.embedding[eo + i] = g.embedding[eo + i]! + gl * cache.normOut[no + i]!;
            }
        }
    }

    // Final RMSNorm.
    const dLastHidden = new Float32Array(M * dModel);
    rmsNormBackward(dNormOut, cache.lastHidden, w.finalNorm, cache.normInv, M, dModel,
        dLastHidden, g.finalNorm, RMSNORM_EPS);

    // Layer stack, in reverse.
    const blockDims: Mamba1CpuDims = {
        dModel, dState: dims.dState, dConv: dims.dConv,
        dInner: dims.dInner, dtRank: dims.dtRank, batch, seqLen,
    };
    let d: Float32Array = dLastHidden;
    for (let i = dims.numLayers - 1; i >= 0; i--) {
        const layerCache = checkpointed
            ? mamba1CpuForward(cache.hiddenIn[i]!, w.layers[i]!, blockDims).cache
            : cache.layer[i]!;
        d = mamba1CpuBackward(d, w.layers[i]!, layerCache, blockDims, g.layers[i]!);
    }

    // Token embedding — the SAME table the LM head reads, so this accumulates
    // on top of the head's contribution (the tie is why both terms are needed).
    for (let r = 0; r < M; r++) {
        const id = tokenIds[r]!;
        const eo = id * dModel;
        const ho = r * dModel;
        for (let i = 0; i < dModel; i++) g.embedding[eo + i] = g.embedding[eo + i]! + d[ho + i]!;
    }

    return { loss, grads: toNamedGrads(g) };
}

// ── Parameter shapes (for PEFT) ──────────────────────────────────────────────

/**
 * The 2-D shape of a parameter, or `null` when it is a vector (bias / norm gain
 * / `D_vec`).
 *
 * Low-rank adaptation only means anything over a matrix, so this is the single
 * place that says which parameters an adapter can cover — and it derives the
 * shapes from `dims` rather than restating the block's layout, so it cannot
 * drift from {@link cpuWeightsFromNamed}.
 */
export function paramMatrixShape(name: string, dims: CpuModelDims): [rows: number, cols: number] | null {
    const { dModel, dState: N, dConv: K, dInner: D, dtRank: R } = dims;
    if (name === 'embedding') return [dims.vocabSize, dModel];
    const local = name.startsWith('layer') ? name.slice(name.indexOf('.') + 1) : name;
    switch (local) {
        case 'wInProj':  return [2 * D, dModel];
        case 'wConv':    return [D, K];
        case 'wXProj':   return [R + 2 * N, D];
        case 'wDtProj':  return [D, R];
        case 'A_log':    return [D, N];
        case 'wOutProj': return [dModel, D];
        default:         return null;   // bInProj, bConv, bXProj, bDtProj, D_vec, bOutProj, normWeight, final_norm, lm_head_bias
    }
}

/**
 * Parameters a low-rank adapter covers by default: the four projections inside
 * each block.
 *
 * Deliberately NOT the token embedding (by far the largest tensor, and tied to
 * the output head — adapting it low-rank changes the head too) and NOT `A_log`
 * (the state-transition decay, which WSLA freezes precisely because drifting it
 * destabilises the SSM).
 */
export const DEFAULT_LORA_TARGETS = ['wInProj', 'wXProj', 'wDtProj', 'wOutProj'] as const;
