/**
 * tests/fakeGpu.ts — a recording, byte-accurate stand-in for `GPUDevice`.
 *
 * It does NOT execute WGSL: it records shader modules, bind groups and dispatch
 * grids, and it stores buffer bytes for real so `writeBuffer` / `copyBufferToBuffer`
 * / `mapAsync` behave exactly as WebGPU's do.
 *
 * That is enough to test two things CI could never reach before:
 *   • the WIRING of the GPU path (dispatch grids, uniform sizes, which kernel is
 *     given which buffer) — see `mamba1_gpu_wiring.test.ts`;
 *   • `MambaTrainer` end to end — its gradients and optimiser are CPU, and the GPU
 *     is only a byte store it reads parameters from and writes them back to.
 */

const STORAGE = 0x80;
const COPY_DST = 0x08;
const COPY_SRC = 0x04;
const MAP_READ = 0x01;

export interface FakeBuffer {
    size: number;
    usage: number;
    bytes: Uint8Array;
    destroyed: boolean;
    destroy(): void;
    getMappedRange(): ArrayBuffer;
    unmap(): void;
    mapAsync(mode: number): Promise<void>;
}

export interface Dispatch {
    entryPoint: string;
    workgroups: [number, number, number];
    buffers: FakeBuffer[];
}

export interface FakeGpu {
    device: GPUDevice;
    dispatches: Dispatch[];
    copies: Array<{ size: number }>;
    /** Every buffer ever created, for leak/allocation assertions. */
    buffers: FakeBuffer[];
}

export function fakeGpu(): FakeGpu {
    const dispatches: Dispatch[] = [];
    const copies: Array<{ size: number }> = [];
    const buffers: FakeBuffer[] = [];
    const bindGroups = new WeakMap<object, FakeBuffer[]>();
    const pipelineEntry = new WeakMap<object, string>();

    const mkBuffer = (size: number, usage: number): FakeBuffer => {
        // A real backing store, so a round-trip through the "GPU" preserves bytes.
        const store = new ArrayBuffer(size);
        let mapped: ArrayBuffer | null = null;
        const buf: FakeBuffer = {
            size,
            usage,
            bytes: new Uint8Array(store),
            destroyed: false,
            destroy() { this.destroyed = true; },
            getMappedRange() {
                mapped = store;
                return store;
            },
            unmap() { mapped = null; void mapped; },
            mapAsync: () => Promise.resolve(),
        };
        buffers.push(buf);
        return buf;
    };

    const device = {
        createBuffer: (d: { size: number; usage: number }) => mkBuffer(d.size, d.usage),
        createShaderModule: (d: { code: string }) => ({ code: d.code }),
        createComputePipeline: (d: { compute: { entryPoint: string } }) => {
            const p = { getBindGroupLayout: () => ({}) };
            pipelineEntry.set(p, d.compute.entryPoint);
            return p;
        },
        createBindGroup: (d: { entries: Array<{ binding: number; resource: { buffer: FakeBuffer } }> }) => {
            const bg = {};
            bindGroups.set(bg, d.entries.map((e) => e.resource.buffer));
            return bg;
        },
        createCommandEncoder: () => {
            let pipeline: object | null = null;
            let bindGroup: object | null = null;
            return {
                copyBufferToBuffer: (src: FakeBuffer, srcOff: number, dst: FakeBuffer, dstOff: number, size: number) => {
                    copies.push({ size });
                    dst.bytes.set(src.bytes.subarray(srcOff, srcOff + size), dstOff);
                },
                beginComputePass: () => ({
                    setPipeline: (p: object) => { pipeline = p; },
                    setBindGroup: (_i: number, bg: object) => { bindGroup = bg; },
                    dispatchWorkgroups: (x: number, y: number, z: number) => {
                        dispatches.push({
                            entryPoint: pipelineEntry.get(pipeline!) ?? '?',
                            workgroups: [x, y, z],
                            buffers: bindGroups.get(bindGroup!) ?? [],
                        });
                    },
                    end: () => { /* no-op */ },
                }),
                finish: () => ({}),
            };
        },
        queue: {
            submit: () => { /* no-op */ },
            writeBuffer: (buf: FakeBuffer, offset: number, data: ArrayBuffer, dataOffset = 0, size?: number) => {
                const src = new Uint8Array(data, dataOffset, size ?? data.byteLength - dataOffset);
                buf.bytes.set(src, offset);
            },
        },
    };

    return { device: device as unknown as GPUDevice, dispatches, copies, buffers };
}

/** Usage bits a readable storage buffer carries — for allocation assertions. */
export const READABLE_STORAGE = STORAGE | COPY_DST | COPY_SRC;
export const STAGING = MAP_READ | COPY_DST;
