/**
 * mamba3_block.ts – Mamba-3 Mixer Block (Complex-valued MIMO SSM, inference-first).
 *
 * Three improvements over Mamba-2:
 *   1. Complex-valued states  — h ∈ ℂ^(N/2), stored as interleaved f32 pairs
 *   2. MIMO recurrence        — G×G block recurrence per head (default G=1 = SISO)
 *   3. ET discretisation      — B_bar = (A_bar − 1)·A⁻¹·B  (exact, not approx)
 *
 * Weight shapes vs Mamba-2 (same 9 tensors, different A_log shape):
 *   wInProj    : (D + 2*G*N_c*2 + H, dModel)   where N_c = dState (complex count)
 *   wConv      : (D + 2*G*N_c*2, K)
 *   bConv      : (D + 2*G*N_c*2,)
 *   A_log      : (H, 2)   ← [log|A|, arg(A)] per head
 *   dt_bias    : (H,)
 *   D_vec      : (H,)
 *   wOutProj   : (dModel, D)
 *   normWeight : (D,)
 *   preNormWeight: (dModel,)
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

import { COMPLEX_SSD_FORWARD_WGSL } from '../kernels/complex_ssd.js';
import { gaussianArray } from '../utils/rng.js';
import { CONV1D_FORWARD_WGSL }      from '../kernels/conv1d.js';
import { COL_SLICE_WGSL, COL_SLICE_ENTRY, dispatchColumnSlice } from '../kernels/slice.js';
import { LINEAR_FORWARD_WGSL }      from '../kernels/linear_projection.js';
import { ACTIVATIONS_WGSL }         from '../kernels/activations.js';

import type { Mamba2BlockConfig }                  from './mamba2_block.js';
import type { SequenceLayer, LayerForwardResult, LayerParam } from './sequence_layer.js';

export interface Mamba3BlockConfig extends Mamba2BlockConfig {
    /** MIMO group size G. Default 1 = SISO (same as Mamba-2). */
    mimoGroup?: number;
    // dState here is the complex state count N_c (real state count = 2*N_c)
}

export interface Mamba3Cache {
    stateCarry: GPUBuffer;
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

export class Mamba3Block implements SequenceLayer {
    readonly layerType = 'mamba3' as const;

    device    : GPUDevice;
    config    : Required<Mamba3BlockConfig>;
    dInner    : number;
    dHead     : number;
    /** Complex state count per head (N_c = dState in config). */
    nComplex  : number;

    gpuWeights: Record<string, GPUBuffer>;
    pipelines : Record<string, GPUComputePipeline>;

    private _wslaMode = false;

    constructor(device: GPUDevice, config: Mamba3BlockConfig) {
        this.device = device;
        this.config = {
            ...{ dState: 16, dConv: 4, expand: 2, nGroups: 1, chunkLen: 256, mimoGroup: 1 },
            ...config,
        } as Required<Mamba3BlockConfig>;

        const { dModel, expand, nHeads } = this.config;
        this.dInner   = expand * dModel;
        this.dHead    = this.dInner / nHeads;
        this.nComplex = this.config.dState; // N_c

        if (this.dInner % nHeads !== 0) {
            throw new Error(
                `Mamba3Block: dInner (${this.dInner}) must be divisible by nHeads (${nHeads}).`
            );
        }

        this.gpuWeights = {};
        this.pipelines  = {};

        this._initWeights();
        this._buildPipelines();
    }

    private _initWeights(): void {
        const { dModel, dConv, nHeads, nGroups } = this.config;
        const D  = this.dInner;
        const Nc = this.nComplex;
        const K  = dConv;
        const H  = nHeads;
        const G  = nGroups;
        // Each complex state = 2 f32 values
        const convD = D + 2 * G * Nc * 2;  // x-channels + complex B/C

        const randn = (n: number, std = 0.02): Float32Array => gaussianArray(n, std);

        const zeros = (n: number) => new Float32Array(n);
        const ones  = (n: number) => new Float32Array(n).fill(1.0);

        // A_log: (H, 2) = [log|A|, arg(A)] per head
        // Initialise to unit magnitude (|A|=1, phase=0) → purely oscillatory
        const A_log = new Float32Array(H * 2);
        for (let h = 0; h < H; h++) {
            A_log[h * 2 + 0] = 0.0;                         // log|A| = 0 → |A| = 1
            A_log[h * 2 + 1] = (2 * Math.PI * h) / H;      // evenly spaced phases
        }

        const mk = (arr: Float32Array) => createStorageBuffer(this.device, arr, true);

        const inProjRows = D + 2 * G * Nc * 2 + H;
        this.gpuWeights = {
            wInProj     : mk(randn(inProjRows * dModel)),
            wConv       : mk(randn(convD * K, 0.01)),
            bConv       : mk(zeros(convD)),
            A_log       : mk(A_log),
            dt_bias     : mk(zeros(H)),
            D_vec       : mk(ones(H)),
            wOutProj    : mk(randn(dModel * D, 0.02)),
            normWeight  : mk(ones(D)),
            preNormWeight: mk(ones(dModel)),
        };
    }

    private _buildPipelines(): void {
        const d = this.device;
        this.pipelines = {
            linear     : createComputePipeline(d, LINEAR_FORWARD_WGSL,       'linear_forward'),
            conv1d     : createComputePipeline(d, CONV1D_FORWARD_WGSL,       'conv1d_forward'),
            colSlice   : createComputePipeline(d, COL_SLICE_WGSL, COL_SLICE_ENTRY),
            rmsnorm    : createComputePipeline(d, ACTIVATIONS_WGSL,          'rmsnorm_forward'),
            cssd_fwd   : createComputePipeline(d, COMPLEX_SSD_FORWARD_WGSL,  'complex_ssd_forward'),
            elAdd      : createComputePipeline(d, ADD_SHADER,                'main'),
        };
    }

    forward(xBuf: GPUBuffer, batch: number, seqLen: number): LayerForwardResult {
        const d = this.device;
        const { dModel, dConv, nHeads, nGroups, chunkLen } = this.config;
        const D  = this.dInner;
        const Nc = this.nComplex;
        const K  = dConv;
        const H  = nHeads;
        const G  = nGroups;
        const dh = this.dHead;
        const B  = batch;
        const L  = seqLen;
        const M  = B * L;
        const convD = D + 2 * G * Nc * 2;
        const numChunks = Math.ceil(L / chunkLen);

        // 1. Pre-block RMSNorm
        const normOut = createEmptyStorageBuffer(d, M * dModel * 4, true);
        const normInv = createEmptyStorageBuffer(d, M * 4, true);
        {
            const params = new ArrayBuffer(16);
            new Uint32Array(params, 0, 2).set([M, dModel]);
            new Float32Array(params, 8, 1).set([1e-6]);
            const pBuf = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['rmsnorm']!,
                [pBuf, xBuf, this.gpuWeights['preNormWeight']!, normOut, normInv]);
            dispatchKernel(d, this.pipelines['rmsnorm']!, bg, [cdiv(M, 64), 1, 1]);
        }
        normInv.destroy();

        // 2. Fused in_proj
        const inProjRows = D + 2 * G * Nc * 2 + H;
        const inProjOut  = createEmptyStorageBuffer(d, M * inProjRows * 4, true);
        {
            const params = new Uint32Array([M, dModel, inProjRows]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const zeroBias = createStorageBuffer(d, new Float32Array(inProjRows), true);
            const bg = createBindGroup(d, this.pipelines['linear']!,
                [pBuf, normOut, this.gpuWeights['wInProj']!, zeroBias, inProjOut]);
            dispatchKernel(d, this.pipelines['linear']!, bg, [cdiv(M, 16), cdiv(inProjRows, 16), 1]);
            zeroBias.destroy();
        }
        normOut.destroy();

        // Split: xConv [convD], dt [H]
        const xConvBuf = createEmptyStorageBuffer(d, M * convD * 4, true);
        const dtBuf    = createEmptyStorageBuffer(d, M * H * 4, true);
        {
            // COLUMN split. `linear_forward` writes (M, convD + H) ROW-major, so each
            // part is a range of columns WITHIN every row — a strided gather, not a
            // contiguous copy (which only coincides for M == 1).
            const slice = this.pipelines['colSlice']!;
            const stride = convD + H;
            dispatchColumnSlice(d, slice, inProjOut, xConvBuf, M, stride, 0, convD);
            dispatchColumnSlice(d, slice, inProjOut, dtBuf, M, stride, convD, H);
        }
        inProjOut.destroy();

        // 3. Causal conv1d (fused convD channels)
        const convOut = createEmptyStorageBuffer(d, M * convD * 4, true);
        {
            const params = new Uint32Array([L, convD, K, B, 1]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['conv1d']!,
                [pBuf, xConvBuf, this.gpuWeights['wConv']!, this.gpuWeights['bConv']!, convOut]);
            dispatchKernel(d, this.pipelines['conv1d']!, bg, [cdiv(L, 16), cdiv(convD, 16), B]);
        }
        xConvBuf.destroy();

        // Split: xSsd [D], B_proj_complex [G*Nc*2], C_proj_complex [G*Nc*2]
        const xSsdBuf  = createEmptyStorageBuffer(d, M * D * 4, true);
        const bProjBuf = createEmptyStorageBuffer(d, M * G * Nc * 2 * 4, true);
        const cProjBuf = createEmptyStorageBuffer(d, M * G * Nc * 2 * 4, true);
        {
            // COLUMN split of the (M, convD) conv output — same reasoning as above.
            const slice = this.pipelines['colSlice']!;
            dispatchColumnSlice(d, slice, convOut, xSsdBuf, M, convD, 0, D);
            dispatchColumnSlice(d, slice, convOut, bProjBuf, M, convD, D, G * Nc * 2);
            dispatchColumnSlice(d, slice, convOut, cProjBuf, M, convD, D + G * Nc * 2, G * Nc * 2);
        }
        convOut.destroy();

        // 4. Complex SSD scan
        // state_carry: [numChunks+1, B, H, Nc*2, dHead]
        const stateCarry = createEmptyStorageBuffer(
            d, (numChunks + 1) * B * H * Nc * 2 * dh * 4, true);
        const cssdOut = createEmptyStorageBuffer(d, M * D * 4, true);

        {
            const params = new Uint32Array([L, D, H, dh, G, Nc, chunkLen, numChunks, B]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['cssd_fwd']!,
                [pBuf, xSsdBuf, bProjBuf, cProjBuf, dtBuf,
                 this.gpuWeights['A_log']!, this.gpuWeights['dt_bias']!,
                 this.gpuWeights['D_vec']!, cssdOut, stateCarry]);
            dispatchKernel(d, this.pipelines['cssd_fwd']!, bg, [numChunks, H, B]);
        }
        xSsdBuf.destroy();
        bProjBuf.destroy();
        cProjBuf.destroy();
        dtBuf.destroy();

        // 5. Inner RMSNorm
        const innerNormOut = createEmptyStorageBuffer(d, M * D * 4, true);
        const innerNormInv = createEmptyStorageBuffer(d, M * 4, true);
        {
            const params = new ArrayBuffer(16);
            new Uint32Array(params, 0, 2).set([M, D]);
            new Float32Array(params, 8, 1).set([1e-6]);
            const pBuf = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['rmsnorm']!,
                [pBuf, cssdOut, this.gpuWeights['normWeight']!, innerNormOut, innerNormInv]);
            dispatchKernel(d, this.pipelines['rmsnorm']!, bg, [cdiv(M, 64), 1, 1]);
        }
        cssdOut.destroy();
        innerNormInv.destroy();

        // 6. Output projection
        const outProjOut = createEmptyStorageBuffer(d, M * dModel * 4, true);
        {
            const params = new Uint32Array([M, D, dModel]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const zeroBias = createStorageBuffer(d, new Float32Array(dModel), true);
            const bg = createBindGroup(d, this.pipelines['linear']!,
                [pBuf, innerNormOut, this.gpuWeights['wOutProj']!, zeroBias, outProjOut]);
            dispatchKernel(d, this.pipelines['linear']!, bg, [cdiv(M, 16), cdiv(dModel, 16), 1]);
            zeroBias.destroy();
        }
        innerNormOut.destroy();

        // 7. Residual add
        const output = createEmptyStorageBuffer(d, M * dModel * 4, true);
        {
            const nBuf = createUniformBuffer(d, new Uint32Array([M * dModel]).buffer);
            const bg   = createBindGroup(d, this.pipelines['elAdd']!,
                [outProjOut, xBuf, output, nBuf]);
            dispatchKernel(d, this.pipelines['elAdd']!, bg, [cdiv(M * dModel, 256), 1, 1]);
        }
        outProjOut.destroy();

        const cache: Mamba3Cache = { stateCarry };
        return { output, cache };
    }

    parameters(): LayerParam[] {
        const { dModel, dConv, nHeads, nGroups } = this.config;
        const D     = this.dInner;
        const Nc    = this.nComplex;
        const K     = dConv;
        const H     = nHeads;
        const G     = nGroups;
        const convD = D + 2 * G * Nc * 2;

        return [
            { buf: this.gpuWeights['wInProj']!,      numel: (D + 2 * G * Nc * 2 + H) * dModel, name: 'wInProj'      },
            { buf: this.gpuWeights['wConv']!,         numel: convD * K,                          name: 'wConv'        },
            { buf: this.gpuWeights['bConv']!,         numel: convD,                              name: 'bConv'        },
            { buf: this.gpuWeights['A_log']!,         numel: H * 2,                              name: 'A_log'        },
            { buf: this.gpuWeights['dt_bias']!,       numel: H,                                  name: 'dt_bias'      },
            { buf: this.gpuWeights['D_vec']!,         numel: H,                                  name: 'D_vec'        },
            { buf: this.gpuWeights['wOutProj']!,      numel: dModel * D,                         name: 'wOutProj'     },
            { buf: this.gpuWeights['normWeight']!,    numel: D,                                  name: 'normWeight'   },
            { buf: this.gpuWeights['preNormWeight']!, numel: dModel,                             name: 'preNormWeight'},
        ];
    }

    getTrainableParams(): LayerParam[] {
        if (this._wslaMode) {
            return [
                { buf: this.gpuWeights['wInProj']!,
                  numel: (this.config.nGroups * this.nComplex * 2 * 2) * this.config.dModel,
                  name: 'wInProj_BC' },
            ];
        }
        return this.parameters();
    }

    setWSLAMode(enabled: boolean): void {
        this._wslaMode = enabled;
    }

    destroy(): void {
        for (const buf of Object.values(this.gpuWeights)) buf.destroy();
        this.gpuWeights = {};
    }
}
