/**
 * trainer.ts – MambaTrainer class
 *
 * REAL backpropagation. Until 2026-08 `_trainStep` handed the SAME `dLogits`
 * buffer to AdamW for every parameter, so no parameter ever received its own
 * gradient: `learn()` was SAFE (WSLA freezes the backbone, the trust region
 * bounds every step, the NaN guard blocks a poisoned weight, and the session
 * rolls back a regression) but INERT. It now runs a genuine backward pass —
 * `cpuModelBackward` in `model_cpu.ts` — and each parameter is stepped with its
 * OWN gradient.
 *
 * Where the maths runs, and why:
 *   • the backward pass is the CPU reference (`cpu_ops.ts` + `mamba1_cpu.ts`),
 *     a line-for-line mirror of the WGSL forward kernels. It is exact and it is
 *     PROVEN — `tests/backprop.test.ts` checks every parameter gradient against
 *     finite differences of the real loss, with no GPU involved;
 *   • the optimiser is {@link adamwUpdateInPlace}, the CPU mirror of
 *     `WEIGHT_UPDATE_WGSL`'s `adamw_update`, keeping the trust region and the
 *     NaN guard the safety story depends on;
 *   • the model's GPU buffers are written back after every optimiser step, so
 *     inference (which IS GPU) always serves the trained weights.
 *
 * The residual is a WGSL backward pass so training itself runs on the GPU. The
 * backward kernels exist (`SELECTIVE_SCAN_BACKWARD_WGSL`, `SSD_BACKWARD_WGSL`,
 * `CONV1D_BACKWARD_WGSL`, `LINEAR_BACKWARD_WGSL`, `ACTIVATIONS_BACKWARD_WGSL`)
 * but wiring them cannot be validated numerically without a WebGPU device, and
 * an unvalidated GPU backward is worse than a slower exact one.
 */

import {
    readBuffer,
    uploadBuffer,
} from '../utils/gpu_utils.js';

import { crossEntropyLoss } from './autograd.js';
import { adamwUpdateInPlace, type ShardSpec } from '../optim/adamw.js';
import { LoRAAdapter, type LoRAConfig } from './lora.js';
import {
    cpuModelBackward,
    cpuWeightsFromNamed,
    namedWeights,
    paramMatrixShape,
    DEFAULT_LORA_TARGETS,
    CPU_GRADIENT_LAYER_TYPES,
    type CpuModelDims,
    type CpuModelWeights,
} from './model_cpu.js';
import { HybridMambaModel, MambaModel } from '../model/mamba_model.js';
import { BPETokenizer } from '../tokenizer/bpe.js';
import type { LayerParam as BlockParam } from '../model/sequence_layer.js';

export interface TrainOptions {
  learningRate?: number;
  epochs?: number;
  /** Sequences trained together per optimiser step. Chunks of equal length are
   *  grouped into batches of this size; a ragged tail trains on its own. */
  batchSize?: number;
  seqLen?: number;
  maxGradNorm?: number;
  weightDecay?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  wsla?: boolean;
  /** Trust region: max |Δθ| per optimizer step. 0 disables. Defaults to
   *  {@link WSLA_MAX_DELTA} in WSLA (write-through) mode, else 0 (full training). */
  maxDelta?: number;
  /**
   * @deprecated The pre-clip gradient L2 norm is now computed exactly on the way
   * through the optimiser and is ALWAYS reported — it no longer costs a GPU
   * readback, so there is nothing to opt into. Accepted for API compatibility.
   */
  trackGradNorm?: boolean;
  /**
   * Train LOW-RANK ADAPTERS over the block projections instead of the weights
   * themselves (see {@link LoRAAdapter}). The base weights are frozen; only A and
   * B train, so an adapt costs a fraction of the optimiser state and the result is
   * a small, mergeable delta rather than a whole checkpoint.
   *
   * `targets` selects which parameters get an adapter, by their unqualified name
   * (`wInProj`, `wXProj`, `wDtProj`, `wOutProj`, `wConv`, `A_log`, `embedding`);
   * omit for {@link DEFAULT_LORA_TARGETS}. A rank larger than a target's smaller
   * dimension is clamped to it rather than throwing — the small projections
   * (`wDtProj` is `dInner x dtRank`) would otherwise reject a sensible global rank.
   */
  lora?: LoRAConfig & { targets?: readonly string[] };
  /**
   * Batches to accumulate gradients over before one optimiser step. Default 1.
   * The effective batch size is `batchSize * gradientAccumulation` at the memory
   * cost of `batchSize` — the standard way to train at a batch size the host
   * cannot hold at once.
   */
  gradientAccumulation?: number;
  /**
   * Shard the optimiser state (the Adam moments — 2x the parameter bytes, the
   * dominant cost of a full fine-tune). This instance then OWNS only parameters
   * where `index % count === index`, and allocates moments for those alone. Run
   * `count` trainers over the same model and the union of their steps equals one
   * unsharded step. Same contract as {@link ShardSpec} on {@link AdamW}.
   */
  shard?: ShardSpec;
  /**
   * Recompute each layer's activations in the backward pass instead of retaining
   * all of them. Gradients are identical; peak memory drops from O(layers)
   * activation caches to one, at the cost of a second forward per layer.
   */
  activationCheckpointing?: boolean;
  onEpochEnd?: ((epoch: number, loss: number, gradNorm?: number) => void) | null;
}

/**
 * Default per-step trust region for WSLA / write-through adaptation. Small
 * enough that a single `adapt()` nudges the narrow params without lurching, so
 * repeated adapts stay stable (and any that regress are cheaply rolled back by
 * the session). Full-training callers pass `maxDelta: 0` (or omit it in
 * non-WSLA mode) for unbounded steps.
 */
export const WSLA_MAX_DELTA = 0.05;

interface AdamMoments {
  m: Float32Array;
  v: Float32Array;
}

interface AdamHyperparams {
  learningRate: number;
  weightDecay: number;
  beta1: number;
  beta2: number;
  eps: number;
  beta1_t: number;
  beta2_t: number;
  maxDelta: number;
  gradScale: number;
}

/** One training example: a token window and its next-token targets. */
interface TrainChunk { inputs: number[]; targets: number[] }

/** A batch of equal-length chunks trained in one optimiser step. */
interface TrainBatch { inputs: number[]; targets: number[]; batch: number; seqLen: number }

export class MambaTrainer {
    model: HybridMambaModel;
    tokenizer: BPETokenizer | null;
    device: GPUDevice;
    /**
     * Adam moments keyed by parameter NAME (not array index). Name-keying is what
     * lets WSLA toggle safely: a narrow write-through step updates only the
     * `layer{i}.wXProj/bXProj` subset, a full fine-tune updates everything, and
     * both reuse the SAME `m`/`v` per parameter. Index-keyed moments (the old
     * scheme) silently misaligned the moment with the wrong parameter the moment
     * the trainable set changed shape — corrupting the update.
     */
    private _moments: Map<string, AdamMoments>;
    private _step: number;
    private readonly _dims: CpuModelDims;
    /** CPU mirror of every model tensor — the gradient engine's working set. */
    private _mirror: CpuModelWeights | null;
    private _mirrorByName: Map<string, Float32Array> | null;
    /** Low-rank adapters by parameter name, when training with `lora`. */
    private _adapters: Map<string, LoRAAdapter>;
    /** Frozen base weights the adapters sit on top of (LoRA only). */
    private _loraBase: Map<string, Float32Array>;
    private _shard: ShardSpec | null;

    constructor(model: HybridMambaModel | MambaModel, tokenizer: BPETokenizer | null = null) {
        this.model     = model;
        this.tokenizer = tokenizer;
        this.device    = model.device;

        this._moments = new Map();
        this._step = 0;
        this._dims = cpuDimsFor(model);
        this._mirror = null;
        this._mirrorByName = null;
        this._adapters = new Map();
        this._loraBase = new Map();
        this._shard = null;
    }

    /**
     * The low-rank adapters this trainer is training, by parameter name — empty
     * unless `train({ lora })` was used. Exposed so a caller can serialise just the
     * adapter (a few hundred KB) instead of the whole checkpoint.
     */
    get adapters(): ReadonlyMap<string, LoRAAdapter> { return this._adapters; }

    /** Trainable scalar count for the current mode — the LoRA saving, measured. */
    trainableParamCount(): number {
        const params = this.model.getTrainableParams();
        if (this._adapters.size === 0) return params.reduce((n, p) => n + p.numel, 0);
        let n = 0;
        for (const p of params) {
            const a = this._adapters.get(p.name);
            if (a) n += a.numParams();
        }
        return n;
    }

    /** Whether this trainer owns parameter `index` under the active shard. */
    private _owns(index: number): boolean {
        return this._shard === null || index % this._shard.count === this._shard.index;
    }

    /**
     * Build (once) an adapter for every LoRA-target parameter, snapshot the frozen
     * base, and return the set. Rank is clamped per parameter to its smaller
     * dimension so one global rank works across projections of very different shape.
     */
    private _installAdapters(cfg: LoRAConfig & { targets?: readonly string[] }): void {
        if (this._adapters.size > 0) return;   // already installed; keep training them
        const targets = new Set<string>(cfg.targets ?? DEFAULT_LORA_TARGETS);
        const weights = this._mirrorByName!;
        for (const p of this.model.getTrainableParams()) {
            const local = p.name.includes('.') ? p.name.slice(p.name.indexOf('.') + 1) : p.name;
            if (!targets.has(local)) continue;
            const shape = paramMatrixShape(p.name, this._dims);
            if (!shape) continue;   // a vector parameter — nothing for a low-rank adapter to do
            const [rows, cols] = shape;
            const rank = Math.max(1, Math.min(cfg.rank ?? 8, rows, cols));
            this._adapters.set(p.name, new LoRAAdapter(rows, cols, { ...cfg, rank }));
            this._loraBase.set(p.name, Float32Array.from(weights.get(p.name)!));
        }
        if (this._adapters.size === 0) {
            throw new Error(
                `MambaTrainer: lora was requested but no trainable parameter matched targets ` +
                `[${[...targets].join(', ')}]. Valid targets: wInProj, wXProj, wDtProj, wOutProj, wConv, A_log, embedding.`,
            );
        }
    }

    /** Write `base + adapter.delta()` into the mirror, so the forward sees the adapted weights. */
    private _materialiseAdapters(): void {
        const weights = this._mirrorByName!;
        for (const [name, adapter] of this._adapters) {
            const w = weights.get(name)!;
            const base = this._loraBase.get(name)!;
            const delta = adapter.delta();
            for (let i = 0; i < w.length; i++) w[i] = base[i]! + delta[i]!;
        }
    }

    /** Get-or-create the Adam first/second moments for a parameter, by name. */
    private _momentFor(p: BlockParam): AdamMoments {
        let mom = this._moments.get(p.name);
        if (!mom) {
            mom = { m: new Float32Array(p.numel), v: new Float32Array(p.numel) };
            this._moments.set(p.name, mom);
        }
        return mom;
    }

    /**
     * Pull every model tensor off the GPU into the CPU mirror. Done once per
     * `train()` call (not per step) so an externally loaded checkpoint or a
     * previous session's weights are picked up, after which the mirror is
     * authoritative and each step writes the updated tensors back.
     */
    private async _syncMirrorFromGpu(): Promise<void> {
        const named = new Map<string, Float32Array>();
        for (const p of this.model.parameters()) {
            named.set(p.name, await readBuffer(this.device, p.buf, p.numel * 4));
        }
        named.set('lm_head_bias',
            await readBuffer(this.device, this.model.gpuLMHeadBias, this._dims.vocabSize * 4));
        this._mirror = cpuWeightsFromNamed(named, this._dims);
        this._mirrorByName = namedWeights(this._mirror);
    }

    async train(input: string | number[], opts: TrainOptions = {}): Promise<number[]> {
        const {
            learningRate = 1e-4,
            epochs       = 5,
            batchSize    = 1,
            seqLen       = 512,
            maxGradNorm  = 1.0,
            weightDecay  = 0.01,
            beta1        = 0.9,
            beta2        = 0.999,
            eps          = 1e-8,
            wsla         = false,
            onEpochEnd   = null,
        } = opts;
        const accumulation = Math.max(1, Math.floor(opts.gradientAccumulation ?? 1));
        this._shard = opts.shard ?? null;
        if (this._shard && (this._shard.count <= 0 || this._shard.index < 0 || this._shard.index >= this._shard.count)) {
            throw new Error(`MambaTrainer: invalid shard ${this._shard.index}/${this._shard.count}`);
        }
        // Trust region defaults ON for write-through (WSLA) adapts, OFF otherwise.
        const maxDelta = opts.maxDelta ?? (wsla ? WSLA_MAX_DELTA : 0);

        if (wsla) this.model.setWSLAMode(true);

        let tokenIds: number[];
        if (typeof input === 'string') {
            if (!this.tokenizer) {
                throw new Error(
                    'MambaTrainer requires a tokenizer when input is a string. ' +
                    'Pass a BPETokenizer instance as the second constructor argument.'
                );
            }
            tokenIds = this.tokenizer.encode(input);
        } else {
            tokenIds = Array.from(input);
        }

        if (tokenIds.length < 2) {
            throw new Error('Input must contain at least 2 tokens to form a training pair.');
        }

        const chunks = buildChunks(tokenIds, seqLen);
        if (chunks.length === 0) {
            throw new Error('Input is too short to form any training chunk.');
        }
        const batches = buildBatches(chunks, Math.max(1, Math.floor(batchSize)));

        // The mirror is the gradient engine's working set; refresh it from the GPU
        // once per train() so a freshly loaded checkpoint is trained, not stale weights.
        await this._syncMirrorFromGpu();
        if (opts.lora) this._installAdapters(opts.lora);

        const epochLosses: number[] = [];

        for (let epoch = 0; epoch < epochs; epoch++) {
            let epochLoss = 0;
            let epochGradNorm = 0;
            let numSteps  = 0;

            // Gradient accumulation: `accumulation` batches contribute to one
            // optimiser step, so the effective batch is batchSize x accumulation at
            // the memory of one batch.
            for (let i = 0; i < batches.length; i += accumulation) {
                const group = batches.slice(i, i + accumulation);
                const { loss, gradNorm } = await this._trainStep(
                    group,
                    { learningRate, maxGradNorm, weightDecay, beta1, beta2, eps, wsla, maxDelta,
                      ...(opts.activationCheckpointing ? { activationCheckpointing: true } : {}) },
                );
                epochLoss += loss;
                epochGradNorm += gradNorm;
                numSteps++;
            }

            const avgLoss = epochLoss / numSteps;
            epochLosses.push(avgLoss);
            if (onEpochEnd) onEpochEnd(epoch + 1, avgLoss, epochGradNorm / numSteps);
        }

        if (wsla) this.model.setWSLAMode(false);
        return epochLosses;
    }

    /**
     * One optimiser step over one batch: real forward + backward, a global
     * gradient-norm clip over the trainable set, then AdamW per parameter with
     * its OWN gradient, written straight back to the GPU buffer.
     */
    private async _trainStep(
        group: TrainBatch[],
        hyperparams: TrainOptions & { learningRate: number; maxGradNorm: number; weightDecay: number; beta1: number; beta2: number; eps: number; maxDelta: number }
    ): Promise<{ loss: number; gradNorm: number }> {
        const { learningRate, maxGradNorm, weightDecay, beta1, beta2, eps, maxDelta } = hyperparams;

        if (!this._mirror || !this._mirrorByName) await this._syncMirrorFromGpu();
        const mirror = this._mirror!;
        const weights = this._mirrorByName!;

        this._step++;
        // Under LoRA the forward must see base + Delta, not the stale merge from the
        // previous step.
        if (this._adapters.size > 0) this._materialiseAdapters();

        const bwOpts = hyperparams.activationCheckpointing ? { activationCheckpointing: true } : {};
        let loss = 0;
        let grads = new Map<string, Float32Array>();
        for (const b of group) {
            const step = cpuModelBackward(b.inputs, b.targets, mirror, this._dims, b.batch, b.seqLen, bwOpts);
            loss += step.loss / group.length;
            if (grads.size === 0) {
                // First micro-batch owns the accumulator; scale it as we go so the
                // accumulated gradient is the MEAN over the group, matching a single
                // pass at the larger batch size.
                for (const v of step.grads.values()) for (let i = 0; i < v.length; i++) v[i] = v[i]! / group.length;
                grads = step.grads;
            } else {
                for (const [k, v] of step.grads) {
                    const acc = grads.get(k)!;
                    for (let i = 0; i < v.length; i++) acc[i] = acc[i]! + v[i]! / group.length;
                }
            }
        }

        // Only the trainable set is updated. Under WSLA (write-through) that is
        // the narrow per-layer subset and the backbone (incl. every A_log) stays
        // frozen — the guarantee that repeated adapts can't destabilise the SSM.
        const params = this.model.getTrainableParams();

        // Under LoRA the base weights are frozen: project each base gradient onto
        // its adapter, and the adapter's A/B are what the optimiser steps.
        if (this._adapters.size > 0) {
            for (const [name, adapter] of this._adapters) {
                const g = grads.get(name);
                if (!g) continue;
                adapter.zeroGrad();
                adapter.accumulateGradient(g);
            }
        }

        // Global pre-clip L2 norm ACROSS the whole trainable set. The old GPU
        // reduce clipped one buffer at a time and raced on `norm_sq[0]` across
        // workgroups; this is the true global norm and needs no readback.
        let sumSq = 0;
        for (const p of params) {
            const g = grads.get(p.name);
            if (!g) continue;
            for (let i = 0; i < g.length; i++) sumSq += g[i]! * g[i]!;
        }
        const gradNorm = Math.sqrt(sumSq);
        const gradScale = maxGradNorm > 0 && gradNorm > maxGradNorm ? maxGradNorm / gradNorm : 1;

        this._adamwStep(params, grads, weights, {
            learningRate, weightDecay, beta1, beta2, eps,
            beta1_t: Math.pow(beta1, this._step),
            beta2_t: Math.pow(beta2, this._step),
            maxDelta, gradScale,
        });

        return { loss, gradNorm };
    }

    /**
     * AdamW over the trainable set. Each parameter is paired with its OWN
     * gradient by NAME — the bug this replaces passed one shared `dLogits`
     * buffer to every parameter — and the updated tensor is uploaded to the GPU
     * so inference serves it.
     */
    private _adamwStep(
        params: BlockParam[],
        grads: Map<string, Float32Array>,
        weights: Map<string, Float32Array>,
        hp: AdamHyperparams,
    ): void {
        const { learningRate, weightDecay, beta1, beta2, eps, beta1_t, beta2_t, maxDelta, gradScale } = hp;
        const step = { lr: learningRate, beta1, beta2, eps, weightDecay, beta1_t, beta2_t, maxDelta, gradScale };

        for (let idx = 0; idx < params.length; idx++) {
            const p = params[idx]!;
            if (!this._owns(idx)) continue;   // another shard holds this one's moments

            // LoRA: step A and B, then re-merge base + Delta into the mirror so the
            // GPU buffer (what inference reads) always holds the adapted weights.
            const adapter = this._adapters.get(p.name);
            if (adapter) {
                const w = weights.get(p.name)!;
                const base = this._loraBase.get(p.name)!;
                const [aP, bP] = [adapter.A, adapter.B];
                const [gA, gB] = [adapter.gradients()[1]!.data, adapter.gradients()[0]!.data];
                const momB = this._momentNamed(`${p.name}::loraB`, bP.length);
                const momA = this._momentNamed(`${p.name}::loraA`, aP.length);
                adamwUpdateInPlace(bP, gB, momB.m, momB.v, step);
                adamwUpdateInPlace(aP, gA, momA.m, momA.v, step);
                const delta = adapter.delta();
                for (let i = 0; i < w.length; i++) w[i] = base[i]! + delta[i]!;
                uploadBuffer(this.device, p.buf, w);
                continue;
            }

            const g = grads.get(p.name);
            const w = weights.get(p.name);
            if (!g || !w) continue;
            if (g.length !== p.numel || w.length !== p.numel) {
                throw new Error(
                    `MambaTrainer: parameter "${p.name}" is ${p.numel} elements but the ` +
                    `gradient engine produced ${g.length} — model and CPU mirror disagree.`
                );
            }

            const mom = this._momentFor(p);
            adamwUpdateInPlace(w, g, mom.m, mom.v, step);
            uploadBuffer(this.device, p.buf, w);
        }
    }

    /** Get-or-create moments for a synthetic parameter (a LoRA factor), by name. */
    private _momentNamed(name: string, numel: number): AdamMoments {
        let mom = this._moments.get(name);
        if (!mom) {
            mom = { m: new Float32Array(numel), v: new Float32Array(numel) };
            this._moments.set(name, mom);
        }
        return mom;
    }

    /**
     * Bytes of optimiser state (Adam m + v) this trainer holds. Under a shard that
     * is 1/count of the full state; under LoRA it is the adapters' state, not the
     * weights'. The measurable point of both features.
     */
    optimizerStateBytes(): number {
        let n = 0;
        for (const mom of this._moments.values()) n += mom.m.byteLength + mom.v.byteLength;
        return n;
    }

    async evaluate(input: string | number[]): Promise<number> {
        let tokenIds: number[];
        if (typeof input === 'string') {
            if (!this.tokenizer) throw new Error('Tokenizer required for string input.');
            tokenIds = Array.from(this.tokenizer.encode(input));
        } else {
            tokenIds = Array.from(input);
        }

        const seqLen    = tokenIds.length;
        const vocabSize = this.model.config.vocabSize;

        const { logits } = await this.model.forward(
            new Uint32Array(tokenIds.slice(0, -1)), 1, seqLen - 1
        );

        let totalLoss = 0;
        for (let i = 0; i < seqLen - 1; i++) {
            const offset = i * vocabSize;
            totalLoss += crossEntropyLoss(
                logits.slice(offset, offset + vocabSize),
                tokenIds[i + 1]!
            );
        }

        const avgLoss = totalLoss / (seqLen - 1);
        return Math.exp(avgLoss);
    }
}

/**
 * Derive the gradient engine's dimensions from a model, rejecting layer types
 * the CPU reference cannot differentiate.
 *
 * A hybrid schedule (`jamba`/`zamba`, or any explicit schedule containing
 * `mamba2`/`mamba3`/`attention`) fails LOUDLY here rather than silently training
 * on gradients that do not exist — which is exactly what the old trainer did.
 */
export function cpuDimsFor(model: HybridMambaModel): CpuModelDims {
    const unsupported = model.layerSpecs
        .map((s, i) => ({ i, type: s.type }))
        .filter(({ type }) => !(CPU_GRADIENT_LAYER_TYPES as readonly string[]).includes(type));
    if (unsupported.length > 0) {
        const detail = unsupported.map(({ i, type }) => `layer${i}=${type}`).join(', ');
        throw new Error(
            `MambaTrainer: the gradient engine differentiates ${CPU_GRADIENT_LAYER_TYPES.join('/')} ` +
            `layers only, but this model has ${detail}. Build the model with the default ` +
            `mamba1 schedule to train it, or extend the CPU reference to cover those blocks.`
        );
    }

    const first = model.layers[0] as unknown as { dInner?: number; dtRank?: number } | undefined;
    const dInner = first?.dInner ?? model.config.expand * model.config.dModel;
    const dtRank = first?.dtRank ?? Math.ceil(model.config.dModel / 16);

    return {
        vocabSize: model.config.vocabSize,
        dModel   : model.config.dModel,
        dState   : model.config.dState,
        dConv    : model.config.dConv,
        dInner,
        dtRank,
        numLayers: model.layers.length,
    };
}

export function buildChunks(ids: number[], seqLen: number): TrainChunk[] {
    const chunks: TrainChunk[] = [];
    for (let start = 0; start + seqLen < ids.length; start += seqLen) {
        chunks.push({
            inputs : ids.slice(start, start + seqLen),
            targets: ids.slice(start + 1, start + seqLen + 1),
        });
    }
    const rem = ids.length % seqLen;
    if (rem > 1) {
        const start = ids.length - rem;
        chunks.push({
            inputs : ids.slice(start, -1),
            targets: ids.slice(start + 1),
        });
    }
    return chunks;
}

/**
 * Group equal-length chunks into batches of at most `batchSize`.
 *
 * `batchSize` used to be accepted and ignored — every chunk trained alone. The
 * batch dimension the model and the gradient engine both take is real, so it is
 * honoured here; a ragged tail chunk simply trains as its own batch of one.
 */
export function buildBatches(chunks: TrainChunk[], batchSize: number): TrainBatch[] {
    const batches: TrainBatch[] = [];
    let i = 0;
    while (i < chunks.length) {
        const seqLen = chunks[i]!.inputs.length;
        const group: TrainChunk[] = [];
        while (i < chunks.length && group.length < batchSize && chunks[i]!.inputs.length === seqLen) {
            group.push(chunks[i]!);
            i++;
        }
        batches.push({
            inputs : group.flatMap((c) => c.inputs),
            targets: group.flatMap((c) => c.targets),
            batch  : group.length,
            seqLen,
        });
    }
    return batches;
}
