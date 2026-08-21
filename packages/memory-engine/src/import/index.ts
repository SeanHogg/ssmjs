/**
 * import/index.ts — the model-import (warm-start / weight-port) registry.
 *
 * The inverse of {@link ../export}: read a `.safetensors` checkpoint back into a
 * live {@link ../lm/evermind_lm.EvermindLM}. Round-trips Evermind's own exports,
 * and warm-starts a foreign SSM checkpoint via a `rename` map.
 *
 * `./foreign` is the same seam for checkpoints that were never ours: published
 * Mamba-1/Mamba-2 models, whose tensor names, layouts and `A_log`/conv
 * conventions are reconciled onto `HybridMambaModel`'s parameters by a
 * config-selected adapter.
 */

export { safetensorsToTensors } from "./safetensors.js";
export { importEvermind, importEvermindTensors, inferArchFromTensors } from "./evermind.js";
export type { ImportOptions } from "./evermind.js";

// ── Foreign checkpoints (Falcon-Mamba / Codestral-Mamba → HybridMambaModel) ────
export {
  FOREIGN_MAMBA_ADAPTERS,
  foreignMambaAdapterFor,
  portForeignMamba,
  portForeignMambaSafetensors,
  applyPortedWeights,
  executePortPlan,
  normaliseSourceName,
  falconMambaAdapter,
  codestralMambaAdapter,
} from "./foreign/index.js";
export type {
  ForeignConfig,
  ForeignMambaAdapter,
  PortTarget,
  PortDiscard,
  PortOp,
  PortPlan,
  PortRule,
  PortedCheckpoint,
  PortedTensors,
  SynthesisedTarget,
} from "./foreign/index.js";
