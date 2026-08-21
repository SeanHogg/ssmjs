/**
 * import/foreign/index.ts — the foreign-checkpoint weight-port surface.
 *
 * Warm-start this engine's `HybridMambaModel` from a published Mamba checkpoint:
 * pass the checkpoint's `config.json` and its tensors, get back the model config
 * to build plus a name-keyed set of weights, together with an explicit account of
 * anything that did not map cleanly.
 *
 * Supported: Falcon-Mamba (Mamba-1) and Codestral-Mamba (Mamba-2). Transformer
 * checkpoints are NOT portable at any tensor naming — distillation is the only
 * route, and {@link foreignMambaAdapterFor} says so when it rejects one.
 */

export {
  FOREIGN_MAMBA_ADAPTERS,
  foreignMambaAdapterFor,
  portForeignMamba,
  portForeignMambaSafetensors,
} from "./registry.js";
export type { PortedCheckpoint } from "./registry.js";

export { applyPortedWeights } from "./apply.js";

export { executePortPlan, normaliseSourceName } from "./plan.js";
export type {
  PortDiscard,
  PortOp,
  PortPlan,
  PortRule,
  PortedTensors,
  SynthesisedTarget,
} from "./plan.js";

export type { ForeignConfig, ForeignMambaAdapter, PortTarget } from "./adapter.js";

export { falconMambaAdapter } from "./falcon_mamba.js";
export { codestralMambaAdapter } from "./codestral_mamba.js";
