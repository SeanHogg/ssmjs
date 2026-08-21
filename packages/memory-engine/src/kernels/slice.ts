/**
 * slice.ts — strided COLUMN slice of a row-major (rows, stride) tensor.
 *
 * Every block in this engine projects into one fused tensor and then splits it:
 * Mamba-1 into (x, z) and (Δ, B, C), Mamba-2/3 into (x, Δ) and (x, B, C),
 * attention into (Q, K, V). `LINEAR_FORWARD_WGSL` writes its output `(M, N)`
 * ROW-major, so each of those parts is a contiguous range of COLUMNS **within
 * every row** — not a contiguous range of the buffer.
 *
 * Splitting them with `copyBufferToBuffer` therefore took whole ROWS instead: it
 * is only the right answer when `M == 1` (batch × seq_len), and for any real
 * sequence it silently handed the block the wrong tensor. This kernel is the one
 * correct implementation; every block uses it.
 */

import { createBindGroup, createUniformBuffer, dispatchKernel, cdiv } from '../utils/gpu_utils.js';

export const COL_SLICE_WGSL: string = /* wgsl */`
struct SliceParams {
    rows   : u32,
    stride : u32,   // source row width
    offset : u32,   // first column to take
    width  : u32,   // number of columns to take
};
@group(0) @binding(0) var<uniform>             p   : SliceParams;
@group(0) @binding(1) var<storage, read>       src : array<f32>;
@group(0) @binding(2) var<storage, read_write> dst : array<f32>;

// Dispatch: (ceil(rows * width / 256), 1, 1)
@compute @workgroup_size(256)
fn col_slice(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.rows * p.width) { return; }
    let r = i / p.width;
    let c = i % p.width;
    dst[i] = src[r * p.stride + p.offset + c];
}
`;

/** Entry point name of {@link COL_SLICE_WGSL}. */
export const COL_SLICE_ENTRY = 'col_slice';

/**
 * `dst[r, 0..width) = src[r, offset..offset+width)` for every row.
 *
 * @param pipeline a pipeline built from {@link COL_SLICE_WGSL} / {@link COL_SLICE_ENTRY}
 * @param rows     batch * seqLen
 * @param stride   source row width (the fused projection's output features)
 */
export function dispatchColumnSlice(
    device: GPUDevice,
    pipeline: GPUComputePipeline,
    src: GPUBuffer,
    dst: GPUBuffer,
    rows: number,
    stride: number,
    offset: number,
    width: number,
): void {
    const pBuf = createUniformBuffer(device, new Uint32Array([rows, stride, offset, width]).buffer);
    const bg = createBindGroup(device, pipeline, [pBuf, src, dst]);
    dispatchKernel(device, pipeline, bg, [cdiv(rows * width, 256), 1, 1]);
    pBuf.destroy();
}
