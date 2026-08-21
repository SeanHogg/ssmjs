/**
 * cpu_ops.ts — the small set of exact CPU primitives the reference forward and
 * backward passes share (linear, RMSNorm, SiLU, softplus).
 *
 * Each one mirrors a WGSL kernel exactly, so a gradient computed here is the
 * gradient of the function the GPU actually evaluates. They are deliberately
 * plain loops over `Float32Array`: correctness first, and small enough that the
 * finite-difference check in `tests/backprop.test.ts` is meaningful.
 */

/** Epsilon used by `rmsnorm_forward` in `ACTIVATIONS_WGSL`. */
export const RMSNORM_EPS = 1e-6;

export function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
}

/** SiLU(x) = x * sigmoid(x) — mirrors `silu_forward`. */
export function silu(x: number): number {
    return x / (1 + Math.exp(-x));
}

/** d/dx SiLU(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x))). */
export function siluGrad(x: number): number {
    const s = sigmoid(x);
    return s * (1 + x * (1 - s));
}

/** Numerically stable softplus — mirrors the kernel's `max(x,0) + log1p(exp(-|x|))`. */
export function softplus(x: number): number {
    return Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
}

/**
 * Y = X @ W^T + b, mirroring `linear_forward`.
 *
 * @param X   (M, K)
 * @param W   (N, K) row-major
 * @param b   (N,)
 * @param out (M, N) — written, not accumulated
 */
export function linearForward(
    X: Float32Array, W: Float32Array, b: Float32Array,
    M: number, K: number, N: number, out: Float32Array,
): void {
    for (let r = 0; r < M; r++) {
        const xo = r * K;
        const yo = r * N;
        for (let n = 0; n < N; n++) {
            let acc = b[n]!;
            const wo = n * K;
            for (let k = 0; k < K; k++) acc += X[xo + k]! * W[wo + k]!;
            out[yo + n] = acc;
        }
    }
}

/**
 * Backward of {@link linearForward}. `dX` is written; `dW` and `db` are
 * ACCUMULATED into (so one parameter can collect gradient from several call
 * sites without a temporary).
 */
export function linearBackward(
    dY: Float32Array, X: Float32Array, W: Float32Array,
    M: number, K: number, N: number,
    dX: Float32Array, dW: Float32Array, db: Float32Array,
): void {
    dX.fill(0);
    for (let r = 0; r < M; r++) {
        const xo = r * K;
        const yo = r * N;
        for (let n = 0; n < N; n++) {
            const g = dY[yo + n]!;
            if (g === 0) continue;
            db[n] = db[n]! + g;
            const wo = n * K;
            for (let k = 0; k < K; k++) {
                dX[xo + k] = dX[xo + k]! + g * W[wo + k]!;
                dW[wo + k] = dW[wo + k]! + g * X[xo + k]!;
            }
        }
    }
}

/**
 * RMSNorm forward — mirrors `rmsnorm_forward`:
 *   inv[r] = 1 / sqrt(mean(x[r]²) + eps);  y[r,i] = x[r,i] * inv[r] * w[i]
 */
export function rmsNormForward(
    x: Float32Array, w: Float32Array, M: number, D: number,
    out: Float32Array, inv: Float32Array, eps: number = RMSNORM_EPS,
): void {
    for (let r = 0; r < M; r++) {
        const base = r * D;
        let sq = 0;
        for (let i = 0; i < D; i++) { const v = x[base + i]!; sq += v * v; }
        const iv = 1 / Math.sqrt(sq / D + eps);
        inv[r] = iv;
        for (let i = 0; i < D; i++) out[base + i] = x[base + i]! * iv * w[i]!;
    }
}

/**
 * RMSNorm backward. `dx` and `dw` are ACCUMULATED into — `dx` in particular is
 * usually the residual path, which already carries a gradient.
 */
export function rmsNormBackward(
    dY: Float32Array, x: Float32Array, w: Float32Array, inv: Float32Array,
    M: number, D: number,
    dx: Float32Array, dw: Float32Array, _eps: number = RMSNORM_EPS,
): void {
    for (let r = 0; r < M; r++) {
        const base = r * D;
        const iv = inv[r]!;
        let dot = 0;   // Σ_i (dY[r,i] * w[i]) * x[r,i]
        for (let i = 0; i < D; i++) {
            const g = dY[base + i]! * w[i]!;
            dot += g * x[base + i]!;
            dw[i] = dw[i]! + dY[base + i]! * x[base + i]! * iv;
        }
        const scale = (iv * iv * iv) / D;
        for (let i = 0; i < D; i++) {
            const g = dY[base + i]! * w[i]!;
            dx[base + i] = dx[base + i]! + g * iv - scale * dot * x[base + i]!;
        }
    }
}
