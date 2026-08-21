/**
 * mamba1_cpu.ts — exact CPU forward + backward for a Mamba-1 (S6) block.
 *
 * This is the REFERENCE implementation of the same maths the WGSL kernels run
 * (`rmsnorm_forward`, `linear_forward`, `conv1d_forward`, `silu_forward`,
 * `forward_scan` + `forward_reduce`). It exists so real per-parameter gradients
 * can be computed — and, crucially, PROVEN correct by finite differences —
 * without a WebGPU device. {@link MambaTrainer} consumes it as its gradient
 * engine: the optimizer (AdamW + trust region + NaN guard) stays on the GPU,
 * the gradients are exact.
 *
 * Every formula below is a line-for-line mirror of the corresponding kernel; if
 * a kernel changes, this file changes with it and `tests/backprop.test.ts`
 * catches the drift (the gradient check fails the moment forward and backward
 * disagree).
 *
 * Layout conventions (identical to the GPU buffers):
 *   x        (M, dModel)        M = batch * seqLen, row r = b * seqLen + t
 *   wInProj  (2*dInner, dModel) row-major, Y = X @ W^T + b
 *   wConv    (dInner, dConv)    depthwise, causal
 *   wXProj   (dtRank + 2*dState, dInner)
 *   wDtProj  (dInner, dtRank)
 *   A_log    (dInner, dState)
 *   wOutProj (dModel, dInner)
 *   hCache   (B, L, dInner, dState)
 */

import {
    linearForward,
    linearBackward,
    rmsNormForward,
    rmsNormBackward,
    silu,
    siluGrad,
    softplus,
    sigmoid,
    RMSNORM_EPS,
} from './cpu_ops.js';

/** The 13 learnable tensors of a Mamba-1 block, keyed by the SAME names
 *  {@link Mamba1Block.parameters} publishes. */
export interface Mamba1CpuWeights {
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
}

/** Parameter names of a Mamba-1 block, in `parameters()` order. */
export const MAMBA1_PARAM_NAMES = [
    'wInProj', 'bInProj', 'wConv', 'bConv', 'wXProj', 'bXProj',
    'wDtProj', 'bDtProj', 'A_log', 'D_vec', 'wOutProj', 'bOutProj', 'normWeight',
] as const satisfies ReadonlyArray<keyof Mamba1CpuWeights>;

export interface Mamba1CpuDims {
    dModel : number;
    dState : number;
    dConv  : number;
    dInner : number;
    dtRank : number;
    batch  : number;
    seqLen : number;
}

/** Everything the backward pass needs from the forward pass. */
export interface Mamba1CpuCache {
    x        : Float32Array;   // (M, dModel) block input (also the residual)
    normInv  : Float32Array;   // (M,)
    normOut  : Float32Array;   // (M, dModel)
    xConvIn  : Float32Array;   // (M, D)
    z        : Float32Array;   // (M, D)
    convOut  : Float32Array;   // (M, D)
    u        : Float32Array;   // (M, D) = SiLU(convOut)
    dtRaw    : Float32Array;   // (M, R)
    bRaw     : Float32Array;   // (M, N)
    cRaw     : Float32Array;   // (M, N)
    deltaFull: Float32Array;   // (M, D)
    dv       : Float32Array;   // (M, D) = softplus(deltaFull)
    aCont    : Float32Array;   // (D, N)
    aBar     : Float32Array;   // (M, D, N)
    bBar     : Float32Array;   // (M, D, N)
    h        : Float32Array;   // (M, D, N)
    scanY    : Float32Array;   // (M, D)
    siluZ    : Float32Array;   // (M, D)
    gated    : Float32Array;   // (M, D)
}

export interface Mamba1CpuForwardResult {
    output: Float32Array;      // (M, dModel)
    cache : Mamba1CpuCache;
}

/** Gradients for every Mamba-1 parameter, same keys as {@link Mamba1CpuWeights}. */
export type Mamba1CpuGrads = Record<keyof Mamba1CpuWeights, Float32Array>;

/**
 * Clamp bounds on `A_log` applied by `discretise_A`/`discretise_B` in
 * `SELECTIVE_SCAN_FORWARD_WGSL`. Outside this range the kernel's output is
 * constant in `A_log`, so the gradient is exactly zero — mirrored here so the
 * analytic gradient matches the function the model actually computes.
 */
export const A_LOG_CLAMP_LO = -10;
export const A_LOG_CLAMP_HI = 5;

/** Allocate a zeroed gradient set matching `dims`. */
export function zeroMamba1Grads(dims: Mamba1CpuDims): Mamba1CpuGrads {
    const { dModel, dState: N, dConv: K, dInner: D, dtRank: R } = dims;
    return {
        wInProj   : new Float32Array(2 * D * dModel),
        bInProj   : new Float32Array(2 * D),
        wConv     : new Float32Array(D * K),
        bConv     : new Float32Array(D),
        wXProj    : new Float32Array((R + 2 * N) * D),
        bXProj    : new Float32Array(R + 2 * N),
        wDtProj   : new Float32Array(D * R),
        bDtProj   : new Float32Array(D),
        A_log     : new Float32Array(D * N),
        D_vec     : new Float32Array(D),
        wOutProj  : new Float32Array(dModel * D),
        bOutProj  : new Float32Array(dModel),
        normWeight: new Float32Array(dModel),
    };
}

/**
 * Exact CPU forward for one Mamba-1 block.
 *
 * Mirrors `Mamba1Block.forward` step for step:
 *   RMSNorm → in_proj → split(x, z) → causal conv1d → SiLU → x_proj(Δ, B, C)
 *   → dt_proj → selective scan → gate by SiLU(z) → out_proj → residual add.
 */
export function mamba1CpuForward(
    x: Float32Array,
    w: Mamba1CpuWeights,
    dims: Mamba1CpuDims,
): Mamba1CpuForwardResult {
    const { dModel, dState: N, dConv: K, dInner: D, dtRank: R, batch: B, seqLen: L } = dims;
    const M = B * L;
    const P = R + 2 * N;

    // 1. Pre-block RMSNorm.
    const normOut = new Float32Array(M * dModel);
    const normInv = new Float32Array(M);
    rmsNormForward(x, w.normWeight, M, dModel, normOut, normInv);

    // 2. Input projection → (x, z).
    const inProj = new Float32Array(M * 2 * D);
    linearForward(normOut, w.wInProj, w.bInProj, M, dModel, 2 * D, inProj);

    // 3. COLUMN split: x = inProj[:, 0:D], z = inProj[:, D:2D].
    const xConvIn = new Float32Array(M * D);
    const z       = new Float32Array(M * D);
    for (let r = 0; r < M; r++) {
        for (let d = 0; d < D; d++) {
            xConvIn[r * D + d] = inProj[r * 2 * D + d]!;
            z[r * D + d]       = inProj[r * 2 * D + D + d]!;
        }
    }

    // 4. Depthwise causal conv1d over the sequence axis.
    const convOut = new Float32Array(M * D);
    for (let b = 0; b < B; b++) {
        for (let t = 0; t < L; t++) {
            const base = (b * L + t) * D;
            for (let d = 0; d < D; d++) {
                let acc = w.bConv[d]!;
                for (let k = 0; k < K; k++) {
                    if (t >= k) acc += w.wConv[d * K + k]! * xConvIn[(b * L + t - k) * D + d]!;
                }
                convOut[base + d] = acc;
            }
        }
    }

    // 5. SiLU.
    const u = new Float32Array(M * D);
    for (let i = 0; i < M * D; i++) u[i] = silu(convOut[i]!);

    // 6. x_proj → Δ_raw, B, C (COLUMN slices of the (M, R+2N) projection).
    const xProj = new Float32Array(M * P);
    linearForward(u, w.wXProj, w.bXProj, M, D, P, xProj);
    const dtRaw = new Float32Array(M * R);
    const bRaw  = new Float32Array(M * N);
    const cRaw  = new Float32Array(M * N);
    for (let r = 0; r < M; r++) {
        for (let i = 0; i < R; i++) dtRaw[r * R + i] = xProj[r * P + i]!;
        for (let n = 0; n < N; n++) {
            bRaw[r * N + n] = xProj[r * P + R + n]!;
            cRaw[r * N + n] = xProj[r * P + R + N + n]!;
        }
    }

    // 7. dt_proj: expand Δ to the inner dimension.
    const deltaFull = new Float32Array(M * D);
    linearForward(dtRaw, w.wDtProj, w.bDtProj, M, R, D, deltaFull);

    // 8. Selective scan (S6). ZOH discretisation, exactly as the kernel does it.
    const dv = new Float32Array(M * D);
    for (let i = 0; i < M * D; i++) dv[i] = softplus(deltaFull[i]!);

    const aCont = new Float32Array(D * N);
    for (let i = 0; i < D * N; i++) {
        aCont[i] = -Math.exp(Math.min(A_LOG_CLAMP_HI, Math.max(A_LOG_CLAMP_LO, w.A_log[i]!)));
    }

    const aBar  = new Float32Array(M * D * N);
    const bBar  = new Float32Array(M * D * N);
    const h     = new Float32Array(M * D * N);
    const scanY = new Float32Array(M * D);

    for (let b = 0; b < B; b++) {
        for (let t = 0; t < L; t++) {
            const r = b * L + t;
            for (let d = 0; d < D; d++) {
                const dvv = dv[r * D + d]!;
                const uv  = u[r * D + d]!;
                let acc = w.D_vec[d]! * uv;
                for (let n = 0; n < N; n++) {
                    const ac = aCont[d * N + n]!;
                    const ab = Math.exp(dvv * ac);
                    const bb = ((ab - 1) / ac) * bRaw[r * N + n]!;
                    const idx = (r * D + d) * N + n;
                    aBar[idx] = ab;
                    bBar[idx] = bb;
                    const hPrev = t === 0 ? 0 : h[((r - 1) * D + d) * N + n]!;
                    const hv = ab * hPrev + bb * uv;
                    h[idx] = hv;
                    acc += cRaw[r * N + n]! * hv;
                }
                scanY[r * D + d] = acc;
            }
        }
    }

    // 9. Gate: y ⊗ SiLU(z).
    const siluZ = new Float32Array(M * D);
    const gated = new Float32Array(M * D);
    for (let i = 0; i < M * D; i++) {
        siluZ[i] = silu(z[i]!);
        gated[i] = scanY[i]! * siluZ[i]!;
    }

    // 10 + 11. Output projection, then the residual add.
    const output = new Float32Array(M * dModel);
    linearForward(gated, w.wOutProj, w.bOutProj, M, D, dModel, output);
    for (let i = 0; i < M * dModel; i++) output[i] = output[i]! + x[i]!;

    return {
        output,
        cache: { x, normInv, normOut, xConvIn, z, convOut, u, dtRaw, bRaw, cRaw, deltaFull, dv, aCont, aBar, bBar, h, scanY, siluZ, gated },
    };
}

/**
 * Exact CPU backward for one Mamba-1 block.
 *
 * @param dOut  dL/d(block output), (M, dModel)
 * @param grads accumulator — parameter gradients are ADDED into it
 * @returns dL/d(block input), (M, dModel)
 */
export function mamba1CpuBackward(
    dOut: Float32Array,
    w: Mamba1CpuWeights,
    cache: Mamba1CpuCache,
    dims: Mamba1CpuDims,
    grads: Mamba1CpuGrads,
): Float32Array {
    const { dModel, dState: N, dConv: K, dInner: D, dtRank: R, batch: B, seqLen: L } = dims;
    const M = B * L;
    const P = R + 2 * N;

    // 11. Residual: the block input receives dOut directly, plus whatever flows
    //     back through the mixer branch (accumulated at the very end).
    const dx = new Float32Array(M * dModel);
    dx.set(dOut);

    // 10. out_proj.
    const dGated = new Float32Array(M * D);
    linearBackward(dOut, cache.gated, w.wOutProj, M, D, dModel, dGated, grads.wOutProj, grads.bOutProj);

    // 9. Gate: gated = scanY * SiLU(z).
    const dScanY = new Float32Array(M * D);
    const dz     = new Float32Array(M * D);
    for (let i = 0; i < M * D; i++) {
        dScanY[i] = dGated[i]! * cache.siluZ[i]!;
        dz[i]     = dGated[i]! * cache.scanY[i]! * siluGrad(cache.z[i]!);
    }

    // 8. Selective scan, walked backwards in time.
    const du    = new Float32Array(M * D);
    const dBRaw = new Float32Array(M * N);
    const dCRaw = new Float32Array(M * N);
    const dDv   = new Float32Array(M * D);      // dL/d softplus(Δ)
    const dACont = new Float32Array(D * N);

    // dhNext[d*N+n] carries dL/dh_t propagated from step t+1 for the current batch row.
    const dhNext = new Float32Array(D * N);
    for (let b = 0; b < B; b++) {
        dhNext.fill(0);
        for (let t = L - 1; t >= 0; t--) {
            const r = b * L + t;
            for (let d = 0; d < D; d++) {
                const dY  = dScanY[r * D + d]!;
                const uv  = cache.u[r * D + d]!;
                const dvv = cache.dv[r * D + d]!;
                // Skip connection: y_t[d] += D_vec[d] * u_t[d].
                grads.D_vec[d] = grads.D_vec[d]! + dY * uv;
                du[r * D + d] = du[r * D + d]! + dY * w.D_vec[d]!;

                let ddvAcc = 0;
                for (let n = 0; n < N; n++) {
                    const idx = (r * D + d) * N + n;
                    const hv  = cache.h[idx]!;
                    const ab  = cache.aBar[idx]!;
                    const bb  = cache.bBar[idx]!;
                    const ac  = cache.aCont[d * N + n]!;
                    const cv  = cache.cRaw[r * N + n]!;

                    // y_t[d] = Σ_n C_t[n] * h_t[d,n]
                    dCRaw[r * N + n] = dCRaw[r * N + n]! + dY * hv;
                    const dh = dY * cv + dhNext[d * N + n]!;

                    // h_t = a_bar * h_{t-1} + b_bar * u_t
                    const hPrev = t === 0 ? 0 : cache.h[((r - 1) * D + d) * N + n]!;
                    const dABar = dh * hPrev;
                    const dBBar = dh * uv;
                    du[r * D + d] = du[r * D + d]! + dh * bb;
                    dhNext[d * N + n] = dh * ab;   // flows to step t-1

                    // a_bar = exp(dv * a_cont);  b_bar = (a_bar - 1) / a_cont * B_raw
                    const bv = cache.bRaw[r * N + n]!;
                    ddvAcc += dABar * ac * ab + dBBar * ab * bv;
                    dACont[d * N + n] = dACont[d * N + n]!
                        + dABar * dvv * ab
                        + dBBar * bv * ((dvv * ab * ac - (ab - 1)) / (ac * ac));
                    dBRaw[r * N + n] = dBRaw[r * N + n]! + dBBar * ((ab - 1) / ac);
                }
                dDv[r * D + d] = dDv[r * D + d]! + ddvAcc;
            }
        }
    }

    // a_cont = -exp(clamp(A_log)) ⇒ d a_cont / d A_log = a_cont inside the clamp,
    // and exactly 0 outside it (the kernel's output is constant there).
    for (let i = 0; i < D * N; i++) {
        const al = w.A_log[i]!;
        if (al <= A_LOG_CLAMP_LO || al >= A_LOG_CLAMP_HI) continue;
        grads.A_log[i] = grads.A_log[i]! + dACont[i]! * cache.aCont[i]!;
    }

    // softplus: dv = softplus(deltaFull)
    const dDeltaFull = new Float32Array(M * D);
    for (let i = 0; i < M * D; i++) dDeltaFull[i] = dDv[i]! * sigmoid(cache.deltaFull[i]!);

    // 7. dt_proj.
    const dDtRaw = new Float32Array(M * R);
    linearBackward(dDeltaFull, cache.dtRaw, w.wDtProj, M, R, D, dDtRaw, grads.wDtProj, grads.bDtProj);

    // 6. x_proj — re-assemble the (M, P) column slices, then one linear backward.
    const dXProj = new Float32Array(M * P);
    for (let r = 0; r < M; r++) {
        for (let i = 0; i < R; i++) dXProj[r * P + i] = dDtRaw[r * R + i]!;
        for (let n = 0; n < N; n++) {
            dXProj[r * P + R + n]     = dBRaw[r * N + n]!;
            dXProj[r * P + R + N + n] = dCRaw[r * N + n]!;
        }
    }
    const duFromXProj = new Float32Array(M * D);
    linearBackward(dXProj, cache.u, w.wXProj, M, D, P, duFromXProj, grads.wXProj, grads.bXProj);
    for (let i = 0; i < M * D; i++) du[i] = du[i]! + duFromXProj[i]!;

    // 5. SiLU.
    const dConvOut = new Float32Array(M * D);
    for (let i = 0; i < M * D; i++) dConvOut[i] = du[i]! * siluGrad(cache.convOut[i]!);

    // 4. Depthwise causal conv1d.
    const dXConvIn = new Float32Array(M * D);
    for (let b = 0; b < B; b++) {
        for (let t = 0; t < L; t++) {
            const r = b * L + t;
            for (let d = 0; d < D; d++) {
                const g = dConvOut[r * D + d]!;
                if (g === 0) continue;
                grads.bConv[d] = grads.bConv[d]! + g;
                for (let k = 0; k < K; k++) {
                    if (t < k) continue;
                    const src = (b * L + t - k) * D + d;
                    grads.wConv[d * K + k] = grads.wConv[d * K + k]! + g * cache.xConvIn[src]!;
                    dXConvIn[src] = dXConvIn[src]! + g * w.wConv[d * K + k]!;
                }
            }
        }
    }

    // 3. Un-split the (x, z) columns.
    const dInProj = new Float32Array(M * 2 * D);
    for (let r = 0; r < M; r++) {
        for (let d = 0; d < D; d++) {
            dInProj[r * 2 * D + d]     = dXConvIn[r * D + d]!;
            dInProj[r * 2 * D + D + d] = dz[r * D + d]!;
        }
    }

    // 2. in_proj.
    const dNormOut = new Float32Array(M * dModel);
    linearBackward(dInProj, cache.normOut, w.wInProj, M, dModel, 2 * D, dNormOut, grads.wInProj, grads.bInProj);

    // 1. RMSNorm — adds into the residual path already sitting in `dx`.
    rmsNormBackward(dNormOut, cache.x, w.normWeight, cache.normInv, M, dModel, dx, grads.normWeight, RMSNORM_EPS);

    return dx;
}
