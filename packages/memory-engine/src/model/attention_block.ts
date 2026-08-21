/**
 * attention_block.ts – Causal Multi-Head Self-Attention Block.
 *
 * Intentionally simple for WebGPU — naive O(L²) tiled attention,
 * no Flash-Attention dependency. Suitable for hybrid (Jamba/Zamba) schedules
 * where a few attention layers interleave with SSM layers.
 *
 * Data flow:
 *   Input (B, L, D_model)
 *     └─ RMSNorm
 *     └─ wQKV → Q (B,L,H,dh), K (B,L,H,dh), V (B,L,H,dh)
 *     └─ causal attention scores = Q·Kᵀ / √dh  (masked)
 *     └─ softmax
 *     └─ weighted V sum
 *     └─ concat heads → wO → D_model
 *     └─ + residual
 *   [optional FFN sublayer]
 *
 * Implements SequenceLayer.
 */

import {
    createComputePipeline,
    createBindGroup,
    createStorageBuffer,
    createEmptyStorageBuffer,
    createUniformBuffer,
    dispatchKernel,
    cdiv,
} from '../utils/gpu_utils.js';

import {
    ATTENTION_FORWARD_WGSL,
    SOFTMAX_WGSL,
} from '../kernels/attention.js';
import { LINEAR_FORWARD_WGSL } from '../kernels/linear_projection.js';
import { gaussianArray } from '../utils/rng.js';
import { ACTIVATIONS_WGSL }    from '../kernels/activations.js';
import { COL_SLICE_WGSL, COL_SLICE_ENTRY, dispatchColumnSlice } from '../kernels/slice.js';

import type { SequenceLayer, LayerForwardResult, LayerParam } from './sequence_layer.js';

export interface AttentionBlockConfig {
    dModel  : number;
    nHeads  : number;
    dHead?  : number;   // default dModel / nHeads
    hasFfn? : boolean;  // include 4×dModel FFN sublayer
    ffnMult?: number;   // FFN expansion factor (default 4)
}

export interface AttentionCache {
    scores: GPUBuffer;  // post-softmax scores for backward
}

const ADD_SHADER = /* wgsl */`
@group(0) @binding(0) var<storage, read>       a : array<f32>;
@group(0) @binding(1) var<storage, read>       b : array<f32>;
@group(0) @binding(2) var<storage, read_write> c : array<f32>;
@group(0) @binding(3) var<uniform>             n : u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < n) { c[i] = a[i] + b[i]; }
}
`;

// SiLU for FFN
const SILU_SHADER = /* wgsl */`
struct ActParams { num_elements: u32; };
@group(0) @binding(0) var<uniform>             p : ActParams;
@group(0) @binding(1) var<storage, read>       x : array<f32>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@compute @workgroup_size(256, 1, 1)
fn silu_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.num_elements) { return; }
    let v = x[i];
    y[i] = v / (1.0 + exp(-v));
}
`;

export class AttentionBlock implements SequenceLayer {
    readonly layerType = 'attention' as const;

    device : GPUDevice;
    config : Required<AttentionBlockConfig>;
    dHead  : number;

    gpuWeights: Record<string, GPUBuffer>;
    pipelines : Record<string, GPUComputePipeline>;

    constructor(device: GPUDevice, config: AttentionBlockConfig) {
        this.device = device;

        if (config.dModel % config.nHeads !== 0) {
            throw new Error(
                `AttentionBlock: dModel (${config.dModel}) must be divisible by nHeads (${config.nHeads}).`
            );
        }

        this.config = {
            dHead  : config.dModel / config.nHeads,
            hasFfn : false,
            ffnMult: 4,
            ...config,
        } as Required<AttentionBlockConfig>;

        this.dHead = this.config.dHead;

        this.gpuWeights = {};
        this.pipelines  = {};

        this._initWeights();
        this._buildPipelines();
    }

    private _initWeights(): void {
        const { dModel, hasFfn, ffnMult } = this.config;

        const randn = (n: number, std = 0.02): Float32Array => gaussianArray(n, std);

        const zeros = (n: number) => new Float32Array(n);
        const ones  = (n: number) => new Float32Array(n).fill(1.0);
        const mk    = (arr: Float32Array) => createStorageBuffer(this.device, arr, true);

        this.gpuWeights = {
            wQKV      : mk(randn(3 * dModel * dModel)),
            bQKV      : mk(zeros(3 * dModel)),
            wO        : mk(randn(dModel * dModel)),
            bO        : mk(zeros(dModel)),
            normWeight: mk(ones(dModel)),
        };

        if (hasFfn) {
            const ffnDim = dModel * ffnMult;
            this.gpuWeights['wFfn1'] = mk(randn(ffnDim * dModel));
            this.gpuWeights['bFfn1'] = mk(zeros(ffnDim));
            this.gpuWeights['wFfn2'] = mk(randn(dModel * ffnDim));
            this.gpuWeights['bFfn2'] = mk(zeros(dModel));
        }
    }

    private _buildPipelines(): void {
        const d = this.device;
        this.pipelines = {
            linear  : createComputePipeline(d, LINEAR_FORWARD_WGSL,     'linear_forward'),
            rmsnorm : createComputePipeline(d, ACTIVATIONS_WGSL,        'rmsnorm_forward'),
            attn_fwd: createComputePipeline(d, ATTENTION_FORWARD_WGSL,  'attention_forward'),
            attn_val: createComputePipeline(d, ATTENTION_FORWARD_WGSL,  'attention_value'),
            softmax : createComputePipeline(d, SOFTMAX_WGSL,            'softmax_forward'),
            colSlice: createComputePipeline(d, COL_SLICE_WGSL,          COL_SLICE_ENTRY),
            elAdd   : createComputePipeline(d, ADD_SHADER,              'main'),
        };

        if (this.config.hasFfn) {
            this.pipelines['silu'] = createComputePipeline(d, SILU_SHADER, 'silu_forward');
        }
    }

    forward(xBuf: GPUBuffer, batch: number, seqLen: number): LayerForwardResult {
        const d = this.device;
        const { dModel, nHeads, hasFfn } = this.config;
        const dh = this.dHead;
        const B  = batch;
        const L  = seqLen;
        const M  = B * L;
        const H  = nHeads;

        // 1. Pre-block RMSNorm
        const normOut = createEmptyStorageBuffer(d, M * dModel * 4, true);
        const normInv = createEmptyStorageBuffer(d, M * 4, true);
        {
            const params = new ArrayBuffer(16);
            new Uint32Array(params, 0, 2).set([M, dModel]);
            new Float32Array(params, 8, 1).set([1e-6]);
            const pBuf = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['rmsnorm']!,
                [pBuf, xBuf, this.gpuWeights['normWeight']!, normOut, normInv]);
            dispatchKernel(d, this.pipelines['rmsnorm']!, bg, [cdiv(M, 64), 1, 1]);
        }
        normInv.destroy();

        // 2. QKV projection: [B, L, 3*D]
        const qkvOut = createEmptyStorageBuffer(d, M * 3 * dModel * 4, true);
        {
            const params = new Uint32Array([M, dModel, 3 * dModel]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['linear']!,
                [pBuf, normOut, this.gpuWeights['wQKV']!, this.gpuWeights['bQKV']!, qkvOut]);
            dispatchKernel(d, this.pipelines['linear']!, bg, [cdiv(M, 16), cdiv(3 * dModel, 16), 1]);
        }
        normOut.destroy();

        // Split QKV into Q, K, V: each [B, L, H, dh] = [B, L, D]
        const QBuf = createEmptyStorageBuffer(d, M * dModel * 4, true);
        const KBuf = createEmptyStorageBuffer(d, M * dModel * 4, true);
        const VBuf = createEmptyStorageBuffer(d, M * dModel * 4, true);
        {
            // COLUMN split. `linear_forward` writes (M, 3*dModel) ROW-major, so Q, K
            // and V are ranges of columns WITHIN every row — a strided gather, not a
            // contiguous copy (which only coincides for M == 1).
            const slice = this.pipelines['colSlice']!;
            dispatchColumnSlice(d, slice, qkvOut, QBuf, M, 3 * dModel, 0, dModel);
            dispatchColumnSlice(d, slice, qkvOut, KBuf, M, 3 * dModel, dModel, dModel);
            dispatchColumnSlice(d, slice, qkvOut, VBuf, M, 3 * dModel, 2 * dModel, dModel);
        }
        qkvOut.destroy();

        // 3. Attention scores: [B, H, L, L]
        const scores = createEmptyStorageBuffer(d, B * H * L * L * 4, true);
        {
            const attnParams = new Uint32Array([B, L, dModel, H, dh]).buffer;
            const pBuf = createUniformBuffer(d, attnParams);
            const bg = createBindGroup(d, this.pipelines['attn_fwd']!,
                [pBuf, QBuf, KBuf, VBuf, scores,
                 createEmptyStorageBuffer(d, M * dModel * 4, true)]);  // out_buf placeholder
            dispatchKernel(d, this.pipelines['attn_fwd']!, bg, [cdiv(L, 16), H, B]);
        }

        // 4. Softmax (causal) per row: dispatch (L, H, B)
        {
            const smParams = new Uint32Array([L, L, 1]).buffer;
            const pBuf = createUniformBuffer(d, smParams);
            const bg = createBindGroup(d, this.pipelines['softmax']!,
                [pBuf, scores]);
            dispatchKernel(d, this.pipelines['softmax']!, bg, [L, H, B]);
        }

        // 5. Weighted V sum → attn output [B, L, H, dh]
        const attnOut = createEmptyStorageBuffer(d, M * dModel * 4, true);
        {
            const attnParams = new Uint32Array([B, L, dModel, H, dh]).buffer;
            const pBuf = createUniformBuffer(d, attnParams);
            const bg = createBindGroup(d, this.pipelines['attn_val']!,
                [pBuf, QBuf, KBuf, VBuf, scores, attnOut]);
            dispatchKernel(d, this.pipelines['attn_val']!, bg, [cdiv(L, 16), H, B]);
        }
        QBuf.destroy();
        KBuf.destroy();
        VBuf.destroy();

        // 6. Output projection: [B, L, D] → [B, L, D]
        const outProjOut = createEmptyStorageBuffer(d, M * dModel * 4, true);
        {
            const params = new Uint32Array([M, dModel, dModel]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['linear']!,
                [pBuf, attnOut, this.gpuWeights['wO']!, this.gpuWeights['bO']!, outProjOut]);
            dispatchKernel(d, this.pipelines['linear']!, bg, [cdiv(M, 16), cdiv(dModel, 16), 1]);
        }
        attnOut.destroy();

        // 7. Residual add
        let current = createEmptyStorageBuffer(d, M * dModel * 4, true);
        {
            const nBuf = createUniformBuffer(d, new Uint32Array([M * dModel]).buffer);
            const bg   = createBindGroup(d, this.pipelines['elAdd']!,
                [outProjOut, xBuf, current, nBuf]);
            dispatchKernel(d, this.pipelines['elAdd']!, bg, [cdiv(M * dModel, 256), 1, 1]);
        }
        outProjOut.destroy();

        // 8. Optional FFN sublayer
        if (hasFfn) {
            const { ffnMult } = this.config;
            const ffnDim = dModel * ffnMult;

            const ffn1Out = createEmptyStorageBuffer(d, M * ffnDim * 4, true);
            {
                const params = new Uint32Array([M, dModel, ffnDim]).buffer;
                const pBuf   = createUniformBuffer(d, params);
                const bg = createBindGroup(d, this.pipelines['linear']!,
                    [pBuf, current, this.gpuWeights['wFfn1']!, this.gpuWeights['bFfn1']!, ffn1Out]);
                dispatchKernel(d, this.pipelines['linear']!, bg, [cdiv(M, 16), cdiv(ffnDim, 16), 1]);
            }

            const siluOut = createEmptyStorageBuffer(d, M * ffnDim * 4, true);
            {
                const nBuf = createUniformBuffer(d, new Uint32Array([M * ffnDim]).buffer);
                const bg   = createBindGroup(d, this.pipelines['silu']!,
                    [nBuf, ffn1Out, siluOut]);
                dispatchKernel(d, this.pipelines['silu']!, bg, [cdiv(M * ffnDim, 256), 1, 1]);
            }
            ffn1Out.destroy();

            const ffn2Out = createEmptyStorageBuffer(d, M * dModel * 4, true);
            {
                const params = new Uint32Array([M, ffnDim, dModel]).buffer;
                const pBuf   = createUniformBuffer(d, params);
                const bg = createBindGroup(d, this.pipelines['linear']!,
                    [pBuf, siluOut, this.gpuWeights['wFfn2']!, this.gpuWeights['bFfn2']!, ffn2Out]);
                dispatchKernel(d, this.pipelines['linear']!, bg, [cdiv(M, 16), cdiv(dModel, 16), 1]);
            }
            siluOut.destroy();

            const residual2 = createEmptyStorageBuffer(d, M * dModel * 4, true);
            {
                const nBuf = createUniformBuffer(d, new Uint32Array([M * dModel]).buffer);
                const bg   = createBindGroup(d, this.pipelines['elAdd']!,
                    [ffn2Out, current, residual2, nBuf]);
                dispatchKernel(d, this.pipelines['elAdd']!, bg, [cdiv(M * dModel, 256), 1, 1]);
            }
            ffn2Out.destroy();
            current.destroy();
            current = residual2;
        }

        const cache: AttentionCache = { scores };
        return { output: current, cache };
    }

    parameters(): LayerParam[] {
        const { dModel, hasFfn, ffnMult } = this.config;
        const params: LayerParam[] = [
            { buf: this.gpuWeights['wQKV']!,      numel: 3 * dModel * dModel, name: 'wQKV'      },
            { buf: this.gpuWeights['bQKV']!,      numel: 3 * dModel,          name: 'bQKV'      },
            { buf: this.gpuWeights['wO']!,         numel: dModel * dModel,     name: 'wO'        },
            { buf: this.gpuWeights['bO']!,         numel: dModel,              name: 'bO'        },
            { buf: this.gpuWeights['normWeight']!, numel: dModel,              name: 'normWeight'},
        ];

        if (hasFfn) {
            const ffnDim = dModel * ffnMult;
            params.push(
                { buf: this.gpuWeights['wFfn1']!, numel: ffnDim * dModel, name: 'wFfn1' },
                { buf: this.gpuWeights['bFfn1']!, numel: ffnDim,          name: 'bFfn1' },
                { buf: this.gpuWeights['wFfn2']!, numel: dModel * ffnDim, name: 'wFfn2' },
                { buf: this.gpuWeights['bFfn2']!, numel: dModel,          name: 'bFfn2' },
            );
        }

        return params;
    }

    getTrainableParams(): LayerParam[] {
        // Attention layers are always fully trained — no WSLA subset
        return this.parameters();
    }

    setWSLAMode(_enabled: boolean): void {
        // No-op for attention: WSLA does not apply
    }

    destroy(): void {
        for (const buf of Object.values(this.gpuWeights)) buf.destroy();
        this.gpuWeights = {};
    }
}
