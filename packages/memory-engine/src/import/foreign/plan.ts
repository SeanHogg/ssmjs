/**
 * import/foreign/plan.ts — the ONE executor for a foreign-checkpoint weight port.
 *
 * A port is expressed declaratively: every target parameter of this engine's
 * {@link HybridMambaModel} gets exactly one {@link PortRule} saying which source
 * tensor feeds it and how the bytes are reshaped. Architecture adapters
 * (`falcon_mamba.ts`, `codestral_mamba.ts`) only *describe* their rules — the
 * matching, validation and transform logic lives here once, so a new
 * architecture never re-implements it.
 *
 * The contract that makes a port trustworthy: NOTHING is silently dropped.
 *   • a target with no source in the checkpoint is a `fill` rule and is reported
 *     in {@link PortedCheckpoint.synthesisedTargets};
 *   • a source tensor no rule consumed is reported in
 *     {@link PortedCheckpoint.unmappedSources};
 *   • a source tensor only PARTLY consumed (e.g. Mamba-2's gate rows, which this
 *     engine's block has no branch for) is declared as a {@link PortDiscard};
 *   • a shape that doesn't line up throws, naming the tensor and both shapes.
 */

import type { NamedTensor } from "../../export/tensors.js";

// ── Transform operations ─────────────────────────────────────────────────────

/**
 * How one source tensor becomes one target buffer.
 *
 *  • `copy`                 — element-for-element; source and target shapes must
 *                             agree once size-1 axes are squeezed out. Linear
 *                             weights need no transpose: HF stores
 *                             `[out_features, in_features]` row-major and this
 *                             engine's LINEAR_FORWARD_WGSL reads `W` as `(N, K)`
 *                             = `[out, in]` row-major — the same layout.
 *  • `convTaps`             — a depthwise conv kernel `[C, 1, K]` (PyTorch
 *                             `nn.Conv1d(groups=C)`) → `[C, K]` with the taps
 *                             REVERSED. PyTorch pads left by `K-1` and cross-
 *                             correlates: `y[t] = Σ_j w[j]·x[t + j - (K-1)]`,
 *                             while conv1d.ts computes
 *                             `y[t] = Σ_k w[k]·x[t - k]`. Substituting
 *                             `k = K-1-j` shows the two agree only when the tap
 *                             axis is reversed.
 *  • `rowSlice`             — keep rows `[start, start+count)` of a rank-2
 *                             source. Used for Mamba-2's fused `in_proj`.
 *  • `fill`                 — no source: fill the target with a constant.
 *  • `negExpToSoftplusInv`  — `A_log` convention change. HF Mamba-2 uses
 *                             `A = -exp(A_log)`; this engine's SSD kernel uses
 *                             `A = -softplus(A_log)`. Re-express the same `A`:
 *                             `target = softplus⁻¹(exp(source))`.
 */
export type PortOp =
  | { kind: "copy" }
  | { kind: "convTaps" }
  | { kind: "rowSlice"; start: number; count: number }
  | { kind: "fill"; value: number }
  | { kind: "negExpToSoftplusInv" };

// ── Rules & plans ────────────────────────────────────────────────────────────

/** How ONE target parameter of this engine is produced. */
export interface PortRule {
  /** Target parameter name, exactly as `HybridMambaModel.parameters()` names it. */
  target: string;
  /** Expected target shape (row-major), for validation and reporting. */
  shape: number[];
  /**
   * Candidate source tensor names, in preference order — the first one present
   * in the checkpoint is used. Empty means the target is synthesised (`fill`),
   * because this architecture has no such tensor.
   */
  from: string[];
  op: PortOp;
  /** Why this isn't a plain copy / why there is no source. Required for `fill`. */
  note?: string;
}

/** A source tensor deliberately consumed only in part. */
export interface PortDiscard {
  source: string;
  /** How many source elements are left behind. */
  elements: number;
  reason: string;
}

/** The complete, architecture-specific description of a port. */
export interface PortPlan {
  /** Adapter id, e.g. `falcon_mamba`. */
  adapter: string;
  rules: PortRule[];
  discards: PortDiscard[];
}

/** A target that no source tensor feeds, and the constant used instead. */
export interface SynthesisedTarget {
  target: string;
  value: number;
  reason: string;
}

/** The result of running a {@link PortPlan} over a checkpoint's tensors. */
export interface PortedTensors {
  adapter: string;
  /** Target parameter name → buffer, sized exactly as the model expects. */
  weights: Map<string, Float32Array>;
  /** Checkpoint tensors that no rule consumed (e.g. an untied `lm_head.weight`). */
  unmappedSources: string[];
  /** Targets this architecture has no tensor for. */
  synthesisedTargets: SynthesisedTarget[];
  /** Source tensors consumed only in part. */
  discardedSources: PortDiscard[];
}

// ── Source-name normalisation ────────────────────────────────────────────────

/**
 * Mistral publishes the same Mamba-2 weights twice: the HF conversion
 * (`backbone.…`) and its native `consolidated.safetensors`, whose names carry a
 * `model.` prefix. Strip it so one rule table serves both. The
 * `embedding`/`embeddings` spelling difference is handled by the rules' `from`
 * alternatives, not here.
 */
export function normaliseSourceName(name: string): string {
  return name.startsWith("model.") ? name.slice("model.".length) : name;
}

// ── Execution ────────────────────────────────────────────────────────────────

/** Product of a shape's dimensions (`[]` ⇒ 1). */
function numel(shape: readonly number[]): number {
  return shape.reduce((n, d) => n * d, 1);
}

/** Shape with size-1 axes removed — `[C, 1, K]` and `[C, K]` compare equal. */
function squeeze(shape: readonly number[]): number[] {
  const kept = shape.filter((d) => d !== 1);
  return kept.length > 0 ? kept : [1];
}

function sameShape(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((d, i) => d === b[i]);
}

/**
 * `softplus⁻¹(y) = log(e^y − 1)`, evaluated so it neither overflows nor loses the
 * small-`y` tail. Above ~30 `log(expm1(y))` and `y` agree to well inside f32
 * precision, and `expm1` would have overflowed; below it, `expm1` keeps the
 * precision `exp(y) - 1` would have thrown away.
 */
function softplusInverse(y: number): number {
  if (!(y > 0)) {
    throw new Error(`import/foreign: softplus⁻¹ needs a positive input, got ${y}`);
  }
  if (y > 30) return y;
  return Math.log(Math.expm1(y));
}

/** Run one rule, returning the target buffer. Throws on any shape disagreement. */
function applyRule(rule: PortRule, source: NamedTensor | null): Float32Array {
  const want = numel(rule.shape);

  if (rule.op.kind === "fill") {
    return new Float32Array(want).fill(rule.op.value);
  }
  if (!source) {
    // Unreachable via executePortPlan (which checks first) — kept so a direct
    // caller gets a real message rather than a null dereference.
    throw new Error(`import/foreign: rule for "${rule.target}" has no source tensor`);
  }

  const where = `"${source.name}" → "${rule.target}"`;

  switch (rule.op.kind) {
    case "copy":
    case "negExpToSoftplusInv": {
      if (!sameShape(squeeze(source.shape), squeeze(rule.shape))) {
        throw new Error(
          `import/foreign: ${where} shape mismatch — checkpoint has [${source.shape.join(", ")}], ` +
            `this model expects [${rule.shape.join(", ")}]`,
        );
      }
      if (source.data.length !== want) {
        throw new Error(
          `import/foreign: ${where} has ${source.data.length} elements, expected ${want}`,
        );
      }
      if (rule.op.kind === "copy") return Float32Array.from(source.data);
      const out = new Float32Array(want);
      for (let i = 0; i < want; i++) out[i] = softplusInverse(Math.exp(source.data[i]!));
      return out;
    }

    case "convTaps": {
      const [channels, kernel] = squeeze(rule.shape) as [number, number];
      const src = squeeze(source.shape);
      if (!sameShape(src, [channels, kernel])) {
        throw new Error(
          `import/foreign: ${where} depthwise conv shape mismatch — checkpoint has ` +
            `[${source.shape.join(", ")}], this model expects [${rule.shape.join(", ")}] ` +
            `(a [channels, 1, kernel] PyTorch kernel squeezes to [channels, kernel])`,
        );
      }
      const out = new Float32Array(want);
      for (let c = 0; c < channels; c++) {
        for (let k = 0; k < kernel; k++) {
          // Reverse the tap axis — see PortOp.convTaps.
          out[c * kernel + k] = source.data[c * kernel + (kernel - 1 - k)]!;
        }
      }
      return out;
    }

    case "rowSlice": {
      if (source.shape.length !== 2) {
        throw new Error(
          `import/foreign: ${where} rowSlice needs a rank-2 source, got ` +
            `[${source.shape.join(", ")}]`,
        );
      }
      const [rows, cols] = source.shape as [number, number];
      const { start, count } = rule.op;
      if (start < 0 || start + count > rows) {
        throw new Error(
          `import/foreign: ${where} rowSlice [${start}, ${start + count}) is outside the ` +
            `checkpoint tensor's ${rows} rows — the source layout is not what this adapter expects`,
        );
      }
      if (!sameShape(squeeze(rule.shape), squeeze([count, cols]))) {
        throw new Error(
          `import/foreign: ${where} rowSlice yields [${count}, ${cols}] but this model expects ` +
            `[${rule.shape.join(", ")}]`,
        );
      }
      return Float32Array.from(source.data.subarray(start * cols, (start + count) * cols));
    }
  }
}

/**
 * Execute a plan against a checkpoint's tensors. Every rule must resolve; the
 * first failure throws naming the offending tensor. Sources no rule touched are
 * returned rather than ignored.
 */
export function executePortPlan(plan: PortPlan, tensors: NamedTensor[]): PortedTensors {
  const byName = new Map<string, NamedTensor>();
  for (const t of tensors) {
    const name = normaliseSourceName(t.name);
    byName.set(name, { ...t, name });
  }

  const weights = new Map<string, Float32Array>();
  const synthesisedTargets: SynthesisedTarget[] = [];
  const consumed = new Set<string>();

  for (const rule of plan.rules) {
    if (weights.has(rule.target)) {
      throw new Error(`import/foreign: plan "${plan.adapter}" defines "${rule.target}" twice`);
    }

    if (rule.from.length === 0) {
      if (rule.op.kind !== "fill") {
        throw new Error(
          `import/foreign: rule for "${rule.target}" names no source but is not a fill rule`,
        );
      }
      weights.set(rule.target, applyRule(rule, null));
      synthesisedTargets.push({
        target: rule.target,
        value: rule.op.value,
        reason: rule.note ?? "this architecture has no corresponding tensor",
      });
      continue;
    }

    const source = rule.from.map((n) => byName.get(n)).find((t): t is NamedTensor => t != null);
    if (!source) {
      throw new Error(
        `import/foreign: checkpoint is missing "${rule.from[0]}" (needed for "${rule.target}", ` +
          `expected shape [${rule.shape.join(", ")}])` +
          (rule.from.length > 1 ? `; also tried ${rule.from.slice(1).map((n) => `"${n}"`).join(", ")}` : ""),
      );
    }
    consumed.add(source.name);
    weights.set(rule.target, applyRule(rule, source));
  }

  const unmappedSources = [...byName.keys()].filter((n) => !consumed.has(n)).sort();

  return {
    adapter: plan.adapter,
    weights,
    unmappedSources,
    synthesisedTargets,
    discardedSources: plan.discards,
  };
}
