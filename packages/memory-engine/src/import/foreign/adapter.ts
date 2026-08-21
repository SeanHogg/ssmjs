/**
 * import/foreign/adapter.ts — what every foreign-architecture adapter provides,
 * plus the shared `config.json` readers so no adapter re-implements them.
 *
 * An adapter answers two questions about a published checkpoint:
 *   1. what {@link HybridMambaModelConfig} do these weights fit? and
 *   2. which {@link PortPlan} moves them onto this engine's parameters?
 * Everything downstream (detection, execution, upload) is architecture-agnostic.
 */

import type { HybridMambaModelConfig } from "../../model/mamba_model.js";
import type { PortPlan } from "./plan.js";

/** A checkpoint's parsed `config.json` (HF) or `params.json` (Mistral-native). */
export type ForeignConfig = Record<string, unknown>;

/** Everything a port needs to know about one checkpoint. */
export interface PortTarget {
  /** Construct a `HybridMambaModel` with this to receive the ported weights. */
  modelConfig: HybridMambaModelConfig;
  plan: PortPlan;
}

export interface ForeignMambaAdapter {
  /** Stable id, e.g. `falcon_mamba`. */
  readonly id: string;
  /** Human-readable label for errors and reports. */
  readonly label: string;
  /** `config.model_type` values this adapter claims. */
  readonly modelTypes: readonly string[];
  /** `config.architectures[]` entries this adapter claims. */
  readonly architectures: readonly string[];
  /** Read the checkpoint's geometry and build its port plan. */
  describe(config: ForeignConfig): PortTarget;
}

// ── config.json readers ──────────────────────────────────────────────────────

function fail(key: string, detail: string): never {
  throw new Error(`import/foreign: config field "${key}" ${detail}`);
}

/** A positive integer field, required. */
export function requireInt(config: ForeignConfig, key: string): number {
  const v = config[key];
  if (typeof v !== "number" || !Number.isInteger(v) || v <= 0) {
    fail(key, `must be a positive integer, got ${JSON.stringify(v)}`);
  }
  return v;
}

/** The first of `keys` present as a positive integer, else `undefined`. */
export function firstInt(config: ForeignConfig, keys: readonly string[]): number | undefined {
  for (const key of keys) {
    const v = config[key];
    if (typeof v === "number" && Number.isInteger(v) && v > 0) return v;
  }
  return undefined;
}

/**
 * The mixer's inner width.
 *
 * `intermediate_size` is authoritative and `expand` is NOT trustworthy:
 * `tiiuae/falcon-mamba-7b` ships `expand: 16` against `hidden_size: 4096` while
 * its weights are `d_inner = 8192`. Transformers only tolerates it because the
 * JSON's `intermediate_size` is re-applied over the derived value, so a port
 * that trusted `expand` would build a model 8× too wide. Fall back to
 * `expand * hidden_size` only when `intermediate_size` is absent.
 */
export function readInnerSize(config: ForeignConfig, dModel: number): number {
  const stated = firstInt(config, ["intermediate_size"]);
  if (stated !== undefined) return stated;
  const expand = firstInt(config, ["expand"]);
  if (expand === undefined) {
    fail("intermediate_size", "is absent and no usable \"expand\" was found");
  }
  return expand * dModel;
}

/**
 * `d_inner / d_model` as this engine's integer `expand` factor — the blocks size
 * themselves from it, so a non-integer ratio cannot be represented and must fail
 * loudly rather than silently truncate.
 */
export function expandFactor(dInner: number, dModel: number, label: string): number {
  if (dInner % dModel !== 0) {
    throw new Error(
      `import/foreign: ${label} has d_inner ${dInner} which is not a whole multiple of ` +
        `d_model ${dModel} — this engine's blocks are sized by an integer expand factor`,
    );
  }
  return dInner / dModel;
}

/**
 * `time_step_rank`, resolving HF's `"auto"` to `ceil(hidden_size / 16)` — the
 * same default `Mamba1Block` applies, so an `"auto"` checkpoint and this engine
 * agree without the caller doing anything.
 */
export function readTimeStepRank(config: ForeignConfig, dModel: number): number {
  const v = config["time_step_rank"];
  if (typeof v === "number" && Number.isInteger(v) && v > 0) return v;
  if (v === undefined || v === "auto") return Math.ceil(dModel / 16);
  fail("time_step_rank", `must be a positive integer or "auto", got ${JSON.stringify(v)}`);
}
