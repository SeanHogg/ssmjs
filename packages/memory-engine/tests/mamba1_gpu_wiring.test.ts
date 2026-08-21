/**
 * tests/mamba1_gpu_wiring.test.ts — structural checks on the Mamba-1 GPU path.
 *
 * Numerical validation of the WGSL kernels needs a real WebGPU device, which CI
 * does not have. But most of what went wrong in this block was NOT numerical: it
 * was wiring — a dispatch grid that covered a 64th of the state matrix, a uniform
 * buffer smaller than the struct it binds, and two "splits" done with
 * `copyBufferToBuffer` when the tensors they slice are strided columns. Every one
 * of those is observable from a recording fake device, so they are pinned here.
 *
 * A fake device records buffer creation, bind groups and dispatches; the block's
 * forward pass is driven exactly as `HybridMambaModel` drives it.
 */

import { Mamba1Block } from '../src/model/mamba1_block';
import { fakeGpu } from './fakeGpu';

const CFG = { dModel: 8, dState: 4, dConv: 3, expand: 2, dtRank: 2 };
const D = CFG.expand * CFG.dModel;   // 16
const N = CFG.dState;                // 4
const BATCH = 2;
const SEQ = 5;

function runForward() {
    const ctx = fakeGpu();
    const block = new Mamba1Block(ctx.device, CFG);
    const M = BATCH * SEQ;
    const xBuf = ctx.device.createBuffer({ size: M * CFG.dModel * 4, usage: 0x8c }) as GPUBuffer;
    ctx.dispatches.length = 0;   // ignore construction-time work
    ctx.copies.length = 0;
    block.forward(xBuf, BATCH, SEQ);
    return ctx;
}

test('the selective scan is dispatched over EVERY (d, n, batch) triplet', () => {
    const { dispatches } = runForward();
    const scan = dispatches.filter((d) => d.entryPoint === 'forward_scan');
    expect(scan).toHaveLength(1);
    // The kernel reads wgid.x as d and wgid.y as n, one workgroup per pair.
    // ceil(D/8) x ceil(N/8) would leave the rest of the state matrix unscanned.
    expect(scan[0]!.workgroups).toEqual([D, N, BATCH]);
});

test('the scan reduction covers the full (time, channel, batch) grid', () => {
    const { dispatches } = runForward();
    const reduce = dispatches.filter((d) => d.entryPoint === 'forward_reduce');
    expect(reduce).toHaveLength(1);
    expect(reduce[0]!.workgroups[0] * 64).toBeGreaterThanOrEqual(SEQ);
    expect(reduce[0]!.workgroups[1]).toBe(D);
    expect(reduce[0]!.workgroups[2]).toBe(BATCH);
});

test('the conv1d uniform is large enough for all five ConvParams fields', () => {
    const { dispatches } = runForward();
    const conv = dispatches.find((d) => d.entryPoint === 'conv1d_forward');
    expect(conv).toBeDefined();
    // ConvParams = 5 x u32. A 16-byte uniform is below the struct's minimum
    // binding size and WebGPU rejects the bind group.
    expect(conv!.buffers[0]!.size).toBeGreaterThanOrEqual(20);
});

test('the (x, z) and (delta, B, C) splits are strided gathers, not buffer copies', () => {
    const { dispatches, copies } = runForward();
    const slices = dispatches.filter((d) => d.entryPoint === 'col_slice');
    // x, z, delta, B, C — five column slices.
    const sliceOutputs = slices.map((d) => d.buffers[2]!.size);
    const M = BATCH * SEQ;
    expect(sliceOutputs).toEqual(expect.arrayContaining([
        M * D * 4,                 // x
        M * D * 4,                 // z
        M * CFG.dtRank * 4,        // delta_raw
        M * N * 4,                 // B
    ]));
    expect(slices.length).toBeGreaterThanOrEqual(5);
    // No copyBufferToBuffer is used to split a row-major projection any more.
    expect(copies).toHaveLength(0);
});

test('every parameter tensor is published with the right element count', () => {
    const ctx = fakeGpu();
    const block = new Mamba1Block(ctx.device, CFG);
    const byName = new Map(block.parameters().map((p) => [p.name, p.numel]));
    const R = CFG.dtRank;
    expect(byName.get('wInProj')).toBe(2 * D * CFG.dModel);
    expect(byName.get('wConv')).toBe(D * CFG.dConv);
    expect(byName.get('wXProj')).toBe((R + 2 * N) * D);
    expect(byName.get('wDtProj')).toBe(D * R);
    expect(byName.get('A_log')).toBe(D * N);
    expect(byName.get('wOutProj')).toBe(CFG.dModel * D);
    expect(byName.get('normWeight')).toBe(CFG.dModel);
});

test('WSLA narrows the trainable set to the selective (Δ, B, C) projection', () => {
    const ctx = fakeGpu();
    const block = new Mamba1Block(ctx.device, CFG);
    expect(block.getTrainableParams()).toHaveLength(13);
    block.setWSLAMode(true);
    expect(block.getTrainableParams().map((p) => p.name)).toEqual(['wXProj', 'bXProj']);
});
