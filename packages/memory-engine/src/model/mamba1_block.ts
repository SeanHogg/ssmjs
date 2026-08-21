/**
 * mamba1_block.ts – Mamba-1 Mixer Block (S6 selective scan).
 *
 * Renamed from mamba_block.ts; MambaBlock is kept as a deprecated alias.
 * Implements SequenceLayer so HybridMambaModel can iterate blocks generically.
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

import { SELECTIVE_SCAN_FORWARD_WGSL }  from '../kernels/selective_scan.js';
import { gaussianArray } from '../utils/rng.js';
import { CONV1D_FORWARD_WGSL }          from '../kernels/conv1d.js';
import { LINEAR_FORWARD_WGSL }          from '../kernels/linear_projection.js';
import { ACTIVATIONS_WGSL }             from '../kernels/activations.js';
import { COL_SLICE_WGSL, COL_SLICE_ENTRY, dispatchColumnSlice } from '../kernels/slice.js';

import type { SequenceLayer, LayerForwardResult, LayerParam } from './sequence_layer.js';

export interface Mamba1BlockConfig {
    dModel   : number;
    dState?  : number;
    dConv?   : number;
    expand?  : number;
    dtRank?  : number;
    biasConv?: boolean;
}

/** @deprecated Use LayerParam */
export type BlockParam = LayerParam;

export interface BlockCache {
    normInv   : GPUBuffer;
    normIn    : GPUBuffer;
    normOut   : GPUBuffer;
    zBuf      : GPUBuffer;
    xConvIn   : GPUBuffer;
    convOut   : GPUBuffer;
    siluOut   : GPUBuffer;
    deltaFull : GPUBuffer;
    B_raw     : GPUBuffer;
    C_raw     : GPUBuffer;
    hCache    : GPUBuffer;
}

export interface BlockForwardResult extends LayerForwardResult {
    output : GPUBuffer;
    cache  : BlockCache;
}

// ── Element-wise helper shaders (compiled once per pipeline) ─────────────────

const MUL_SHADER = /* wgsl */`
@group(0) @binding(0) var<storage, read>       a : array<f32>;
@group(0) @binding(1) var<storage, read>       b : array<f32>;
@group(0) @binding(2) var<storage, read_write> c : array<f32>;
@group(0) @binding(3) var<uniform>             n : u32;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < n) { c[i] = a[i] * b[i]; }
}
`;

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

// ── Mamba1Block ───────────────────────────────────────────────────────────────

export class Mamba1Block implements SequenceLayer {
    readonly layerType = 'mamba1' as const;

    device  : GPUDevice;
    config  : Required<Mamba1BlockConfig>;
    dInner  : number;
    dtRank  : number;

    wInProj   : Float32Array;
    bInProj   : Float32Array;
    wConv     : Float32Array;
    bConv     : Float32Array;
    wXProj    : Float32Array;
    bXProj    : Float32Array;
    wDtProj   : Float32Array;
    bDtProj   : Float32Array;
    A_log     : Float32Array;
    D_vec     : Float32Array;
    wOutProj  : Float32Array;
    bOutProj  : Float32Array;
    normWeight: Float32Array;

    gpuWeights : Record<string, GPUBuffer>;
    pipelines  : Record<string, GPUComputePipeline>;

    private _wslaMode = false;

    constructor(device: GPUDevice, config: Mamba1BlockConfig) {
        this.device = device;
        this.config = {
            dState  : 16,
            dConv   : 4,
            expand  : 2,
            biasConv: true,
            dtRank  : Math.ceil(config.dModel / 16),
            ...config,
        } as Required<Mamba1BlockConfig>;

        const { dModel, expand } = this.config;
        this.dInner = expand * dModel;
        this.dtRank = config.dtRank ?? Math.ceil(dModel / 16);

        this.wInProj    = new Float32Array(0);
        this.bInProj    = new Float32Array(0);
        this.wConv      = new Float32Array(0);
        this.bConv      = new Float32Array(0);
        this.wXProj     = new Float32Array(0);
        this.bXProj     = new Float32Array(0);
        this.wDtProj    = new Float32Array(0);
        this.bDtProj    = new Float32Array(0);
        this.A_log      = new Float32Array(0);
        this.D_vec      = new Float32Array(0);
        this.wOutProj   = new Float32Array(0);
        this.bOutProj   = new Float32Array(0);
        this.normWeight = new Float32Array(0);
        this.gpuWeights = {};
        this.pipelines  = {};

        this._initWeights();
        this._buildPipelines();
    }

    private _initWeights(): void {
        const { dModel, dState, dConv } = this.config;
        const D = this.dInner;
        const N = dState;
        const K = dConv;
        const R = this.dtRank;

        const randn = (n: number, std = 0.02): Float32Array => gaussianArray(n, std);

        const zeros = (n: number): Float32Array => new Float32Array(n);
        const ones  = (n: number): Float32Array => new Float32Array(n).fill(1.0);

        this.wInProj  = randn(2 * D * dModel);
        this.bInProj  = zeros(2 * D);
        this.wConv    = randn(D * K, 0.01);
        this.bConv    = zeros(D);
        this.wXProj   = randn((R + 2 * N) * D, 0.01);
        this.bXProj   = zeros(R + 2 * N);
        this.wDtProj  = randn(D * R, 0.02);
        this.bDtProj  = zeros(D);

        this.A_log = new Float32Array(D * N);
        for (let d = 0; d < D; d++) {
            for (let n = 0; n < N; n++) {
                this.A_log[d * N + n] = Math.log(n + 1);
            }
        }

        this.D_vec     = ones(D);
        this.wOutProj  = randn(dModel * D, 0.02);
        this.bOutProj  = zeros(dModel);
        this.normWeight = ones(dModel);

        this._uploadWeightsToGPU();
    }

    private _uploadWeightsToGPU(): void {
        const d  = this.device;
        const mk = (arr: Float32Array): GPUBuffer => createStorageBuffer(d, arr, true);

        this.gpuWeights = {
            wInProj   : mk(this.wInProj),
            bInProj   : mk(this.bInProj),
            wConv     : mk(this.wConv),
            bConv     : mk(this.bConv),
            wXProj    : mk(this.wXProj),
            bXProj    : mk(this.bXProj),
            wDtProj   : mk(this.wDtProj),
            bDtProj   : mk(this.bDtProj),
            A_log     : mk(this.A_log),
            D_vec     : mk(this.D_vec),
            wOutProj  : mk(this.wOutProj),
            bOutProj  : mk(this.bOutProj),
            normWeight: mk(this.normWeight),
        };
    }

    private _buildPipelines(): void {
        const d = this.device;
        this.pipelines = {
            linear      : createComputePipeline(d, LINEAR_FORWARD_WGSL,          'linear_forward'),
            conv1d      : createComputePipeline(d, CONV1D_FORWARD_WGSL,          'conv1d_forward'),
            silu        : createComputePipeline(d, ACTIVATIONS_WGSL,             'silu_forward'),
            rmsnorm     : createComputePipeline(d, ACTIVATIONS_WGSL,             'rmsnorm_forward'),
            scan_fwd    : createComputePipeline(d, SELECTIVE_SCAN_FORWARD_WGSL,  'forward_scan'),
            scan_reduce : createComputePipeline(d, SELECTIVE_SCAN_FORWARD_WGSL,  'forward_reduce'),
            colSlice    : createComputePipeline(d, COL_SLICE_WGSL, COL_SLICE_ENTRY),
            elMul       : createComputePipeline(d, MUL_SHADER, 'main'),
            elAdd       : createComputePipeline(d, ADD_SHADER, 'main'),
        };
    }

    forward(xBuf: GPUBuffer, batch: number, seqLen: number): BlockForwardResult {
        const d = this.device;
        const { dModel, dState, dConv } = this.config;
        const D = this.dInner;
        const N = dState;
        const B = batch;
        const L = seqLen;
        const M = B * L;
        const R = this.dtRank;

        const cache = {} as BlockCache;

        // 1. Pre-block RMSNorm
        const normOut = createEmptyStorageBuffer(d, M * dModel * 4, true);
        const normInv = createEmptyStorageBuffer(d, M * 4, true);
        cache.normInv = normInv;
        cache.normIn  = xBuf;
        {
            const params = new ArrayBuffer(16);
            new Uint32Array(params, 0, 2).set([M, dModel]);
            new Float32Array(params, 8, 1).set([1e-6]);
            const pBuf = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['rmsnorm']!,
                [pBuf, xBuf, this.gpuWeights['normWeight']!, normOut, normInv]);
            dispatchKernel(d, this.pipelines['rmsnorm']!, bg, [cdiv(M, 64), 1, 1]);
        }

        // 2. Input projection → x and z
        const inProjOut = createEmptyStorageBuffer(d, M * 2 * D * 4, true);
        cache.normOut = normOut;
        {
            const params = new Uint32Array([M, dModel, 2 * D]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['linear']!,
                [pBuf, normOut, this.gpuWeights['wInProj']!, this.gpuWeights['bInProj']!, inProjOut]);
            dispatchKernel(d, this.pipelines['linear']!, bg, [cdiv(M, 16), cdiv(2 * D, 16), 1]);
        }

        // 3. Split into x and z.
        //    in_proj writes (M, 2D) ROW-major, so x and z are interleaved COLUMN
        //    slices — one per row — not two contiguous halves of the buffer. A
        //    `copyBufferToBuffer` of the first M*D floats is only the right answer
        //    when M == 1; for any real sequence it hands the block the first half of
        //    the ROWS twice. Split with a strided gather instead.
        const xConvIn = createEmptyStorageBuffer(d, M * D * 4, true);
        const zBuf    = createEmptyStorageBuffer(d, M * D * 4, true);
        {
            const slice = this.pipelines['colSlice']!;
            dispatchColumnSlice(d, slice, inProjOut, xConvIn, M, 2 * D, 0, D);
            dispatchColumnSlice(d, slice, inProjOut, zBuf, M, 2 * D, D, D);
        }
        inProjOut.destroy();
        cache.zBuf    = zBuf;
        cache.xConvIn = xConvIn;

        // 4. Causal conv1d on x
        const convOut = createEmptyStorageBuffer(d, M * D * 4, true);
        cache.convOut = convOut;
        {
            // ConvParams is FIVE u32s (seq_len, d_channels, kernel_size, batch,
            // groups). A four-element uniform is smaller than the struct's minimum
            // binding size and WebGPU rejects the bind group outright. groups = 1
            // is standard depthwise, matching Mamba-2/3's call.
            const params = new Uint32Array([L, D, dConv, B, 1]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['conv1d']!,
                [pBuf, xConvIn, this.gpuWeights['wConv']!, this.gpuWeights['bConv']!, convOut]);
            dispatchKernel(d, this.pipelines['conv1d']!, bg, [cdiv(L, 16), cdiv(D, 16), B]);
        }

        // 5. SiLU activation
        const siluOut = createEmptyStorageBuffer(d, M * D * 4, true);
        cache.siluOut = siluOut;
        {
            const params = new Uint32Array([M * D]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['silu']!,
                [pBuf, convOut, siluOut]);
            dispatchKernel(d, this.pipelines['silu']!, bg, [cdiv(M * D, 256), 1, 1]);
        }

        // 6. x_proj → Δ (dtRaw), B, C
        const xProjOut = createEmptyStorageBuffer(d, M * (R + 2 * N) * 4, true);
        {
            const params = new Uint32Array([M, D, R + 2 * N]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['linear']!,
                [pBuf, siluOut, this.gpuWeights['wXProj']!, this.gpuWeights['bXProj']!, xProjOut]);
            dispatchKernel(d, this.pipelines['linear']!, bg, [cdiv(M, 16), cdiv(R + 2 * N, 16), 1]);
        }

        //    Same story as the (x, z) split: Δ, B and C are COLUMN slices of the
        //    (M, R+2N) projection, so they need a strided gather, not a prefix copy.
        const dtRaw = createEmptyStorageBuffer(d, M * R * 4, true);
        const B_raw = createEmptyStorageBuffer(d, B * L * N * 4, true);
        const C_raw = createEmptyStorageBuffer(d, B * L * N * 4, true);
        {
            const P = R + 2 * N;
            const slice = this.pipelines['colSlice']!;
            dispatchColumnSlice(d, slice, xProjOut, dtRaw, M, P, 0, R);
            dispatchColumnSlice(d, slice, xProjOut, B_raw, M, P, R, N);
            dispatchColumnSlice(d, slice, xProjOut, C_raw, M, P, R + N, N);
        }
        xProjOut.destroy();
        cache.B_raw = B_raw;
        cache.C_raw = C_raw;

        // 7. dt_proj: expand Δ to full dim
        const deltaFull = createEmptyStorageBuffer(d, M * D * 4, true);
        cache.deltaFull = deltaFull;
        {
            const params = new Uint32Array([M, R, D]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['linear']!,
                [pBuf, dtRaw, this.gpuWeights['wDtProj']!, this.gpuWeights['bDtProj']!, deltaFull]);
            dispatchKernel(d, this.pipelines['linear']!, bg, [cdiv(M, 16), cdiv(D, 16), 1]);
        }
        dtRaw.destroy();

        // 8. Selective scan (S6)
        const scanY  = createEmptyStorageBuffer(d, B * L * D * 4, true);
        const hCache = createEmptyStorageBuffer(d, 2 * B * L * D * N * 4, true);
        cache.hCache = hCache;
        {
            const params = new Uint32Array([L, N, D, B]).buffer;
            const pBuf   = createUniformBuffer(d, params);

            const bg1 = createBindGroup(d, this.pipelines['scan_fwd']!,
                [pBuf, siluOut, deltaFull, this.gpuWeights['A_log']!, B_raw, C_raw,
                 this.gpuWeights['D_vec']!, scanY, hCache]);
            // One workgroup per (d, n, batch): the kernel reads `wgid.x` as d and
            // `wgid.y` as n, so the grid must be D x N x B. Dispatching
            // ceil(D/8) x ceil(N/8) left all but a 1/64th corner of the state
            // matrix unscanned (and unwritten in h_cache).
            dispatchKernel(d, this.pipelines['scan_fwd']!, bg1, [D, N, B]);

            const bg2 = createBindGroup(d, this.pipelines['scan_reduce']!,
                [pBuf, siluOut, deltaFull, this.gpuWeights['A_log']!, B_raw, C_raw,
                 this.gpuWeights['D_vec']!, scanY, hCache]);
            dispatchKernel(d, this.pipelines['scan_reduce']!, bg2, [cdiv(L, 64), D, B]);
        }

        // 9. Gate: y ⊗ SiLU(z)
        const siluZ    = createEmptyStorageBuffer(d, M * D * 4, true);
        const gatedOut = createEmptyStorageBuffer(d, M * D * 4, true);
        {
            const nBuf = createUniformBuffer(d, new Uint32Array([M * D]).buffer);
            const bgZ  = createBindGroup(d, this.pipelines['silu']!,
                [nBuf, zBuf, siluZ]);
            dispatchKernel(d, this.pipelines['silu']!, bgZ, [cdiv(M * D, 256), 1, 1]);

            const nBuf2 = createUniformBuffer(d, new Uint32Array([M * D]).buffer);
            const bgMul = createBindGroup(d, this.pipelines['elMul']!,
                [scanY, siluZ, gatedOut, nBuf2]);
            dispatchKernel(d, this.pipelines['elMul']!, bgMul, [cdiv(M * D, 256), 1, 1]);
        }
        siluZ.destroy();
        scanY.destroy();

        // 10. Output projection
        const outProjOut = createEmptyStorageBuffer(d, M * dModel * 4, true);
        {
            const params = new Uint32Array([M, D, dModel]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['linear']!,
                [pBuf, gatedOut, this.gpuWeights['wOutProj']!, this.gpuWeights['bOutProj']!, outProjOut]);
            dispatchKernel(d, this.pipelines['linear']!, bg, [cdiv(M, 16), cdiv(dModel, 16), 1]);
        }
        gatedOut.destroy();

        // 11. Residual add
        const output = createEmptyStorageBuffer(d, M * dModel * 4, true);
        {
            const nBuf = createUniformBuffer(d, new Uint32Array([M * dModel]).buffer);
            const bg   = createBindGroup(d, this.pipelines['elAdd']!,
                [outProjOut, xBuf, output, nBuf]);
            dispatchKernel(d, this.pipelines['elAdd']!, bg, [cdiv(M * dModel, 256), 1, 1]);
        }
        outProjOut.destroy();

        return { output, cache };
    }

    parameters(): LayerParam[] {
        const { dModel, dState, dConv } = this.config;
        const D = this.dInner;
        const N = dState;
        const K = dConv;
        const R = this.dtRank;

        return [
            { buf: this.gpuWeights['wInProj']!,    numel: 2 * D * dModel,   name: 'wInProj'    },
            { buf: this.gpuWeights['bInProj']!,    numel: 2 * D,            name: 'bInProj'    },
            { buf: this.gpuWeights['wConv']!,      numel: D * K,            name: 'wConv'      },
            { buf: this.gpuWeights['bConv']!,      numel: D,                name: 'bConv'      },
            { buf: this.gpuWeights['wXProj']!,     numel: (R + 2 * N) * D, name: 'wXProj'    },
            { buf: this.gpuWeights['bXProj']!,     numel: R + 2 * N,       name: 'bXProj'    },
            { buf: this.gpuWeights['wDtProj']!,    numel: D * R,            name: 'wDtProj'   },
            { buf: this.gpuWeights['bDtProj']!,    numel: D,                name: 'bDtProj'   },
            { buf: this.gpuWeights['A_log']!,      numel: D * N,            name: 'A_log'     },
            { buf: this.gpuWeights['D_vec']!,      numel: D,                name: 'D_vec'     },
            { buf: this.gpuWeights['wOutProj']!,   numel: dModel * D,       name: 'wOutProj'  },
            { buf: this.gpuWeights['bOutProj']!,   numel: dModel,           name: 'bOutProj'  },
            { buf: this.gpuWeights['normWeight']!, numel: dModel,           name: 'normWeight'},
        ];
    }

    getTrainableParams(): LayerParam[] {
        if (this._wslaMode) {
            return [
                { buf: this.gpuWeights['wXProj']!, numel: this.wXProj.length, name: 'wXProj' },
                { buf: this.gpuWeights['bXProj']!, numel: this.bXProj.length, name: 'bXProj' },
            ];
        }
        return this.parameters();
    }

    setWSLAMode(enabled: boolean): void {
        this._wslaMode = enabled;
    }

    destroy(): void {
        for (const buf of Object.values(this.gpuWeights)) {
            buf.destroy();
        }
        this.gpuWeights = {};
    }
}

// Deprecated alias — kept until mambacode.js 3.0.0
export { Mamba1Block as MambaBlock };

/** @deprecated Use Mamba1BlockConfig */
export type MambaBlockConfig = Mamba1BlockConfig;
