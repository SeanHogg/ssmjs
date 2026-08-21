/**
 * import/foreign/registry.ts — config-driven adapter selection and the public
 * port entry points.
 *
 * Given a checkpoint's `config.json`, pick the adapter by `model_type` /
 * `architectures` and run its plan. An architecture nobody claims fails loudly
 * with the list of what IS supported — and, for a transformer checkpoint, says
 * why no map could ever exist for it.
 */

import type { NamedTensor } from "../../export/tensors.js";
import type { HybridMambaModelConfig } from "../../model/mamba_model.js";
import { safetensorsToTensors } from "../safetensors.js";
import { executePortPlan, type PortDiscard, type PortedTensors, type SynthesisedTarget } from "./plan.js";
import { falconMambaAdapter } from "./falcon_mamba.js";
import { codestralMambaAdapter } from "./codestral_mamba.js";
import type { ForeignConfig, ForeignMambaAdapter } from "./adapter.js";

/** Every foreign SSM architecture this engine can warm-start from. */
export const FOREIGN_MAMBA_ADAPTERS: readonly ForeignMambaAdapter[] = [
  falconMambaAdapter,
  codestralMambaAdapter,
];

/** A completed port: the model to build, its weights, and what didn't line up. */
export interface PortedCheckpoint extends PortedTensors {
  /** Construct a `HybridMambaModel` with this to receive {@link PortedTensors.weights}. */
  modelConfig: HybridMambaModelConfig;
}

export type { PortDiscard, SynthesisedTarget };

// ── Detection ────────────────────────────────────────────────────────────────

function architecturesOf(config: ForeignConfig): string[] {
  const raw = config["architectures"];
  return Array.isArray(raw) ? raw.filter((a): a is string => typeof a === "string") : [];
}

function modelTypeOf(config: ForeignConfig): string | undefined {
  const raw = config["model_type"];
  return typeof raw === "string" ? raw : undefined;
}

/**
 * A checkpoint is a transformer when any declared architecture carries an
 * attention-stack marker. Recognised only to give a better error: attention
 * weights (Q/K/V/O, positional embeddings, per-token KV cache) have no
 * counterpart in a state-space mixer, so no rename or reshape can port them.
 */
function looksLikeTransformer(architectures: string[], modelType: string | undefined): boolean {
  const haystack = [...architectures, modelType ?? ""].join(" ").toLowerCase();
  if (haystack.includes("mamba") || haystack.includes("ssm")) return false;
  return /llama|mistral|qwen|gpt|phi|gemma|falcon(?!_?mamba)|starcoder|deepseek|olmo|attention|forcausallm/.test(
    haystack,
  );
}

function supportedList(): string {
  return FOREIGN_MAMBA_ADAPTERS.map(
    (a) => `${a.label} — architectures ${a.architectures.join(" / ")}, model_type ${a.modelTypes.join(" / ")}`,
  ).join("; ");
}

/**
 * The adapter that claims this checkpoint. Throws — never guesses — when no
 * adapter matches, because a wrong map produces a model that loads and then
 * generates noise.
 */
export function foreignMambaAdapterFor(config: ForeignConfig): ForeignMambaAdapter {
  const architectures = architecturesOf(config);
  const modelType = modelTypeOf(config);

  const byArch = FOREIGN_MAMBA_ADAPTERS.find((a) =>
    architectures.some((name) => a.architectures.includes(name)),
  );
  if (byArch) return byArch;

  const byType = modelType
    ? FOREIGN_MAMBA_ADAPTERS.find((a) => a.modelTypes.includes(modelType))
    : undefined;
  if (byType) return byType;

  const seen =
    architectures.length > 0
      ? `architectures [${architectures.join(", ")}]`
      : modelType
        ? `model_type "${modelType}"`
        : "a config declaring neither `architectures` nor `model_type`";

  if (looksLikeTransformer(architectures, modelType)) {
    throw new Error(
      `import/foreign: ${seen} is a TRANSFORMER checkpoint — its attention weights ` +
        `(q/k/v/o projections, positional embeddings) have no counterpart in a state-space ` +
        `mixer, so no weight port can exist for it at any tensor naming. Transformer coders ` +
        `(IQuest, NousCoder, Maincoder and the like) can only be brought across by ` +
        `DISTILLATION — train this engine's SSM against the transformer's outputs. ` +
        `Weight porting is supported only for: ${supportedList()}.`,
    );
  }

  throw new Error(
    `import/foreign: unsupported checkpoint — ${seen}. Supported: ${supportedList()}. ` +
      `A transformer checkpoint cannot be weight-ported at all; distillation is the only route.`,
  );
}

// ── Porting ──────────────────────────────────────────────────────────────────

/**
 * Port a foreign checkpoint's tensors onto this engine's parameters. The result
 * is device-free — {@link applyPortedWeights} uploads it into a live model.
 */
export function portForeignMamba(config: ForeignConfig, tensors: NamedTensor[]): PortedCheckpoint {
  const adapter = foreignMambaAdapterFor(config);
  const { modelConfig, plan } = adapter.describe(config);
  return { ...executePortPlan(plan, tensors), modelConfig };
}

/** Convenience: port straight from a `.safetensors` buffer. */
export function portForeignMambaSafetensors(
  config: ForeignConfig,
  bytes: Uint8Array,
): PortedCheckpoint {
  return portForeignMamba(config, safetensorsToTensors(bytes));
}
