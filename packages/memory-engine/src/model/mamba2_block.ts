/**
 * mamba2_block.ts – Mamba-2 Mixer Block (Structured State Space Duality).
 *
 * Key differences from Mamba-1:
 *   - Multi-head SSM with scalar A per head
 *   - Single fused in_proj (no separate dt_proj expansion)
 *   - SSD (chunked) scan replaces S6 selective scan
 *   - Inner RMSNorm on scan output instead of SiLU gate
 *   - No separate z gate
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

import { SSD_FORWARD_WGSL }    from '../kernels/ssd.js';
import { gaussianArray } from '../utils/rng.js';
import { CONV1D_FORWARD_WGSL } from '../kernels/conv1d.js';
import { COL_SLICE_WGSL, COL_SLICE_ENTRY, dispatchColumnSlice } from '../kernels/slice.js';
import { LINEAR_FORWARD_WGSL } from '../kernels/linear_projection.js';
import { ACTIVATIONS_WGSL }    from '../kernels/activations.js';

import type { SequenceLayer, LayerForwardResult, LayerParam } from './sequence_layer.js';

export interface Mamba2BlockConfig {
    dModel   : number;
    dState   : number;   // N — state dim per group
    dConv    : number;   // K — conv kernel width
    expand   : number;   // dInner = expand * dModel
    nHeads   : number;   // H — number of SSM heads
    nGroups  : number;   // number of B/C groups (default 1)
    chunkLen : number;   // SSD chunk length (default 256)
}

export interface Mamba2Cache {
    stateCarry : GPUBuffer;  // inter-chunk states
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

export class Mamba2Block implements SequenceLayer {
    readonly layerType = 'mamba2' as const;

    device : GPUDevice;
    config : Required<Mamba2BlockConfig>;
    dInner : number;
    dHead  : number;

    gpuWeights : Record<string, GPUBuffer>;
    pipelines  : Record<string, GPUComputePipeline>;

    private _wslaMode = false;

    constructor(device: GPUDevice, config: Mamba2BlockConfig) {
        this.device = device;
        this.config = {
            ...{ dState: 16, dConv: 4, expand: 2, nGroups: 1, chunkLen: 256 },
            ...config,
        } as Required<Mamba2BlockConfig>;

        const { dModel, expand, nHeads } = this.config;
        this.dInner = expand * dModel;
        this.dHead  = this.dInner / nHeads;

        if (this.dInner % nHeads !== 0) {
            throw new Error(
                `Mamba2Block: dInner (${this.dInner}) must be divisible by nHeads (${nHeads}).`
            );
        }

        this.gpuWeights = {};
        this.pipelines  = {};

        this._initWeights();
        this._buildPipelines();
    }

    private _initWeights(): void {
        const { dModel, dState, dConv, nHeads, nGroups } = this.config;
        const D  = this.dInner;
        const N  = dState;
        const K  = dConv;
        const H  = nHeads;
        const G  = nGroups;

        const randn = (n: number, std = 0.02): Float32Array => gaussianArray(n, std);

        const zeros = (n: number) => new Float32Array(n);
        const ones  = (n: number) => new Float32Array(n).fill(1.0);

        // wInProj: (D_inner + 2*n_groups*N + H, D_model) — no bias per Mamba-2 spec
        const inProjRows = D + 2 * G * N + H;
        const mk = (arr: Float32Array) => createStorageBuffer(this.device, arr, true);

        this.gpuWeights = {
            wInProj     : mk(randn(inProjRows * dModel)),
            wConv       : mk(randn((D + 2 * G * N) * K, 0.01)),
            bConv       : mk(zeros(D + 2 * G * N)),
            A_log       : mk(new Float32Array(H).fill(Math.log(1.0))),
            dt_bias     : mk(zeros(H)),
            D_vec       : mk(ones(H)),
            wOutProj    : mk(randn(dModel * D, 0.02)),
            normWeight  : mk(ones(D)),          // inner RMSNorm
            preNormWeight: mk(ones(dModel)),    // pre-block RMSNorm
        };
    }

    private _buildPipelines(): void {
        const d = this.device;
        this.pipelines = {
            linear   : createComputePipeline(d, LINEAR_FORWARD_WGSL,  'linear_forward'),
            conv1d   : createComputePipeline(d, CONV1D_FORWARD_WGSL,  'conv1d_forward'),
            colSlice   : createComputePipeline(d, COL_SLICE_WGSL, COL_SLICE_ENTRY),
            rmsnorm  : createComputePipeline(d, ACTIVATIONS_WGSL,     'rmsnorm_forward'),
            ssd_fwd  : createComputePipeline(d, SSD_FORWARD_WGSL,     'ssd_chunk_forward'),
            elAdd    : createComputePipeline(d, ADD_SHADER,           'main'),
        };
    }

    forward(xBuf: GPUBuffer, batch: number, seqLen: number): LayerForwardResult {
        const d = this.device;
        const { dModel, dState, dConv, nHeads, nGroups, chunkLen } = this.config;
        const D  = this.dInner;
        const N  = dState;
        const K  = dConv;
        const H  = nHeads;
        const G  = nGroups;
        const dh = this.dHead;
        const B  = batch;
        const L  = seqLen;
        const M  = B * L;
        const convD = D + 2 * G * N;  // channels for conv (x, B_proj, C_proj)
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

        // 2. Fused in_proj → [x (D), B_proj (G*N), C_proj (G*N), dt (H)]
        const inProjRows = D + 2 * G * N + H;
        const inProjOut  = createEmptyStorageBuffer(d, M * inProjRows * 4, true);
        {
            const params = new Uint32Array([M, dModel, inProjRows]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            // wInProj has no bias — pass a zero-filled buffer
            const zeroBias = createStorageBuffer(d, new Float32Array(inProjRows), true);
            const bg = createBindGroup(d, this.pipelines['linear']!,
                [pBuf, normOut, this.gpuWeights['wInProj']!, zeroBias, inProjOut]);
            dispatchKernel(d, this.pipelines['linear']!, bg, [cdiv(M, 16), cdiv(inProjRows, 16), 1]);
            zeroBias.destroy();
        }
        normOut.destroy();

        // Split: xConv [D+2GN], dt [H]
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

        // 3. Causal conv1d over x + B_proj + C_proj (fused, convD channels)
        const convOut = createEmptyStorageBuffer(d, M * convD * 4, true);
        {
            const params = new Uint32Array([L, convD, K, B, 1]).buffer;
            const pBuf   = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['conv1d']!,
                [pBuf, xConvBuf, this.gpuWeights['wConv']!, this.gpuWeights['bConv']!, convOut]);
            dispatchKernel(d, this.pipelines['conv1d']!, bg, [cdiv(L, 16), cdiv(convD, 16), B]);
        }
        xConvBuf.destroy();

        // Split conv output: x [D], B_proj [G*N], C_proj [G*N]
        const xSsdBuf = createEmptyStorageBuffer(d, M * D * 4, true);
        const bProjBuf = createEmptyStorageBuffer(d, M * G * N * 4, true);
        const cProjBuf = createEmptyStorageBuffer(d, M * G * N * 4, true);
        {
            // COLUMN split of the (M, convD) conv output — same reasoning as above.
            const slice = this.pipelines['colSlice']!;
            dispatchColumnSlice(d, slice, convOut, xSsdBuf, M, convD, 0, D);
            dispatchColumnSlice(d, slice, convOut, bProjBuf, M, convD, D, G * N);
            dispatchColumnSlice(d, slice, convOut, cProjBuf, M, convD, D + G * N, G * N);
        }
        convOut.destroy();

        // 4. SSD scan
        // state_carry: [numChunks+1, B, H, N, dHead]
        const stateCarry = createEmptyStorageBuffer(
            d, (numChunks + 1) * B * H * N * dh * 4, true);
        const ssdOut = createEmptyStorageBuffer(d, M * D * 4, true);

        {
            const ssdParams = new Uint32Array([L, D, H, dh, G, N, chunkLen, numChunks, B]).buffer;
            const pBuf = createUniformBuffer(d, ssdParams);
            const bg = createBindGroup(d, this.pipelines['ssd_fwd']!,
                [pBuf, xSsdBuf, bProjBuf, cProjBuf, dtBuf,
                 this.gpuWeights['A_log']!, this.gpuWeights['dt_bias']!,
                 this.gpuWeights['D_vec']!, ssdOut, stateCarry]);
            dispatchKernel(d, this.pipelines['ssd_fwd']!, bg, [numChunks, H, B]);
        }
        xSsdBuf.destroy();
        bProjBuf.destroy();
        cProjBuf.destroy();
        dtBuf.destroy();

        // 5. Inner RMSNorm on scan output
        const innerNormOut = createEmptyStorageBuffer(d, M * D * 4, true);
        const innerNormInv = createEmptyStorageBuffer(d, M * 4, true);
        {
            const params = new ArrayBuffer(16);
            new Uint32Array(params, 0, 2).set([M, D]);
            new Float32Array(params, 8, 1).set([1e-6]);
            const pBuf = createUniformBuffer(d, params);
            const bg = createBindGroup(d, this.pipelines['rmsnorm']!,
                [pBuf, ssdOut, this.gpuWeights['normWeight']!, innerNormOut, innerNormInv]);
            dispatchKernel(d, this.pipelines['rmsnorm']!, bg, [cdiv(M, 64), 1, 1]);
        }
        ssdOut.destroy();
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

        const cache: Mamba2Cache = { stateCarry };
        return { output, cache };
    }

    parameters(): LayerParam[] {
        const { dModel, dState, dConv, nHeads, nGroups } = this.config;
        const D     = this.dInner;
        const N     = dState;
        const K     = dConv;
        const H     = nHeads;
        const G     = nGroups;
        const convD = D + 2 * G * N;

        return [
            { buf: this.gpuWeights['wInProj']!,      numel: (D + 2 * G * N + H) * dModel, name: 'wInProj'      },
            { buf: this.gpuWeights['wConv']!,         numel: convD * K,                    name: 'wConv'        },
            { buf: this.gpuWeights['bConv']!,         numel: convD,                        name: 'bConv'        },
            { buf: this.gpuWeights['A_log']!,         numel: H,                            name: 'A_log'        },
            { buf: this.gpuWeights['dt_bias']!,       numel: H,                            name: 'dt_bias'      },
            { buf: this.gpuWeights['D_vec']!,         numel: H,                            name: 'D_vec'        },
            { buf: this.gpuWeights['wOutProj']!,      numel: dModel * D,                   name: 'wOutProj'     },
            { buf: this.gpuWeights['normWeight']!,    numel: D,                            name: 'normWeight'   },
            { buf: this.gpuWeights['preNormWeight']!, numel: dModel,                       name: 'preNormWeight'},
        ];
    }

    getTrainableParams(): LayerParam[] {
        if (this._wslaMode) {
            // WSLA: train only B/C rows of wInProj (the selective projection part)
            return [
                { buf: this.gpuWeights['wInProj']!,
                  numel: (this.config.nGroups * this.config.dState * 2) * this.config.dModel,
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
