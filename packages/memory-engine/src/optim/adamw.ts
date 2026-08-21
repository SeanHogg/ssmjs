/**
 * adamw.ts — AdamW optimiser over a model's flat parameter list.
 *
 * Shared by every CPU-reference trainer in the engine (MoE FFN, the full
 * EvermindLM) so the optimiser maths lives in exactly one place. Operates on any
 * object exposing index-aligned `parameters()` / `gradients()` Float32Arrays.
 */

export interface OptimParam {
  data: Float32Array;
}

export interface OptimTarget {
  parameters(): OptimParam[];
  gradients(): OptimParam[];
}

/**
 * Optimizer-state sharding (ZeRO-1 analog). When set, this optimiser instance
 * OWNS only parameter tensors where `paramIndex % count === index`, and only
 * allocates the AdamW moments (m, v — 2× the parameter bytes, the dominant cost
 * of full fine-tuning per the cookbook's memory accounting) for its shard.
 *
 * Run `count` instances over the SAME model, one per `index`, and the union of
 * their steps equals one unsharded step — but each holds only 1/count of the
 * optimizer state. That is exactly how FSDP/ZeRO fits a larger model on the same
 * hardware; here it is the seam a multi-device trainer partitions on, and even
 * single-process it caps resident moment memory.
 */
export interface ShardSpec {
  /** This shard's rank, 0-based. */
  index: number;
  /** Total number of shards. */
  count: number;
}

export interface AdamWOptions {
  lr?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  weightDecay?: number;
  /** Optimizer-state sharding. Omit for the full (single-owner) optimiser. */
  shard?: ShardSpec;
}

export class AdamW {
  // Per-parameter moments, or null for tensors this shard does not own (no memory).
  private readonly m: (Float32Array | null)[] = [];
  private readonly v: (Float32Array | null)[] = [];
  private t = 0;
  private readonly opt: Required<Omit<AdamWOptions, "shard">>;
  private readonly shard: ShardSpec | null;

  constructor(
    private readonly target: OptimTarget,
    options: AdamWOptions = {},
  ) {
    this.opt = {
      lr: options.lr ?? 0.01,
      beta1: options.beta1 ?? 0.9,
      beta2: options.beta2 ?? 0.999,
      eps: options.eps ?? 1e-8,
      weightDecay: options.weightDecay ?? 0,
    };
    const shard = options.shard ?? null;
    if (shard && (shard.count <= 0 || shard.index < 0 || shard.index >= shard.count)) {
      throw new Error(`AdamW: invalid shard ${shard.index}/${shard.count}`);
    }
    this.shard = shard;
    const params = target.parameters();
    for (let p = 0; p < params.length; p++) {
      if (this._owns(p)) {
        this.m.push(new Float32Array(params[p]!.data.length));
        this.v.push(new Float32Array(params[p]!.data.length));
      } else {
        this.m.push(null);
        this.v.push(null);
      }
    }
  }

  /** Whether this shard owns (and updates) parameter tensor `p`. */
  private _owns(p: number): boolean {
    return this.shard === null || p % this.shard.count === this.shard.index;
  }

  /** One optimiser step from the currently-accumulated gradients (owned tensors only). */
  step(): void {
    this.t++;
    const { lr, beta1, beta2, eps, weightDecay } = this.opt;
    const params = this.target.parameters();
    const grads = this.target.gradients();
    const bc1 = 1 - Math.pow(beta1, this.t);
    const bc2 = 1 - Math.pow(beta2, this.t);
    for (let p = 0; p < params.length; p++) {
      if (!this._owns(p)) continue;
      const w = params[p]!.data;
      const g = grads[p]!.data;
      const m = this.m[p]!;
      const v = this.v[p]!;
      for (let i = 0; i < w.length; i++) {
        const gi = g[i]!;
        m[i] = beta1 * m[i]! + (1 - beta1) * gi;
        v[i] = beta2 * v[i]! + (1 - beta2) * gi * gi;
        const mh = m[i]! / bc1;
        const vh = v[i]! / bc2;
        w[i] = w[i]! - lr * (mh / (Math.sqrt(vh) + eps) + weightDecay * w[i]!);
      }
    }
  }

  /** Bytes of optimizer state this shard holds (m + v) — 1/count of the full state. */
  stateBytes(): number {
    let n = 0;
    for (const arr of this.m) if (arr) n += arr.byteLength;
    for (const arr of this.v) if (arr) n += arr.byteLength;
    return n;
  }
}

// ── The CPU mirror of WEIGHT_UPDATE_WGSL ─────────────────────────────────────

/** Hyperparameters of one {@link adamwUpdateInPlace} step. */
export interface AdamWKernelStep {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  /** beta1^t — the precomputed bias-correction term (matches the kernel uniform). */
  beta1_t: number;
  /** beta2^t. */
  beta2_t: number;
  /** Trust region: max |Δθ| per step. 0 disables the bound. */
  maxDelta: number;
  /** Global gradient-clip factor, applied to `grad` on the fly (1 = no clipping). */
  gradScale?: number;
}

/**
 * One AdamW step over a single parameter tensor — the exact CPU mirror of the
 * `adamw_update` entry point in `WEIGHT_UPDATE_WGSL`, including its two safety
 * properties:
 *
 *   • the **NaN/Inf guard** — a non-finite step is dropped rather than written
 *     into a weight, so one bad gradient cannot permanently poison the model;
 *   • the **trust region** — `maxDelta` bounds how far a single step can move a
 *     weight, which is what keeps repeated write-through adapts reversible.
 *
 * Decoupled weight decay (`θ * (1 - lr·wd)`), matching the kernel — deliberately
 * NOT the coupled form {@link AdamW} uses for the MoE/EvermindLM trainers.
 *
 * `param`, `m` and `v` are updated in place.
 */
export function adamwUpdateInPlace(
  param: Float32Array,
  grad: Float32Array,
  m: Float32Array,
  v: Float32Array,
  hp: AdamWKernelStep,
): void {
  const { lr, beta1, beta2, eps, weightDecay, beta1_t, beta2_t, maxDelta } = hp;
  const gradScale = hp.gradScale ?? 1;
  const bc1 = 1 - beta1_t;
  const bc2 = 1 - beta2_t;
  for (let i = 0; i < param.length; i++) {
    const g = grad[i]! * gradScale;
    const mNew = beta1 * m[i]! + (1 - beta1) * g;
    const vNew = beta2 * v[i]! + (1 - beta2) * g * g;
    m[i] = mNew;
    v[i] = vNew;

    let step = (lr * (mNew / bc1)) / (Math.sqrt(vNew / bc2) + eps);
    if (!Number.isFinite(step)) step = 0;
    if (maxDelta > 0) step = Math.min(maxDelta, Math.max(-maxDelta, step));

    param[i] = param[i]! * (1 - lr * weightDecay) - step;
  }
}
