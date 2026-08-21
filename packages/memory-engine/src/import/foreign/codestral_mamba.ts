/**
 * import/foreign/codestral_mamba.ts — weight port for Codestral-Mamba
 * (`mistralai/Mamba-Codestral-7B-v0.1`, `Mamba2ForCausalLM`,
 * `model_type: "mamba2"`, Mamba-2 / SSD).
 *
 * Source layout confirmed against the repo's published HF `config.json`, its
 * `model.safetensors.index.json` (579 tensors = 9 per layer × 64 + 3 globals),
 * the header of `consolidated.safetensors`, and
 * `transformers/models/mamba2/modeling_mamba2.py`:
 *
 *   backbone.embeddings.weight                    [vocab, d_model]
 *   backbone.layers.{N}.norm.weight               [d_model]
 *   backbone.layers.{N}.mixer.in_proj.weight      [2*d_inner + 2*n_groups*d_state + n_heads, d_model]
 *   backbone.layers.{N}.mixer.conv1d.weight       [conv_dim, 1, d_conv]
 *   backbone.layers.{N}.mixer.conv1d.bias         [conv_dim]
 *   backbone.layers.{N}.mixer.dt_bias             [n_heads]
 *   backbone.layers.{N}.mixer.A_log               [n_heads]
 *   backbone.layers.{N}.mixer.D                   [n_heads]
 *   backbone.layers.{N}.mixer.norm.weight         [d_inner]
 *   backbone.layers.{N}.mixer.out_proj.weight     [d_model, d_inner]
 *   backbone.norm_f.weight                        [d_model]
 *   lm_head.weight                                [vocab, d_model]
 *
 * where `conv_dim = d_inner + 2 * n_groups * d_state`. Mamba-2 has no `x_proj`
 * and no `dt_proj`: B, C and dt all come out of the fused `in_proj`.
 *
 * Three reconciliations against `Mamba2Block`:
 *
 *  1. **`in_proj` row order.** Upstream splits
 *     `[gate (d_inner), x|B|C (conv_dim), dt (n_heads)]`; this engine's fused
 *     projection is `[x (d_inner), B (n_groups*d_state), C (n_groups*d_state),
 *     dt (n_heads)]` — the SAME tail, minus the leading gate rows. So the port
 *     is one contiguous row slice starting at `d_inner`, and the gate rows are
 *     declared as a discard (see 3).
 *  2. **`A_log` convention.** Upstream uses `A = -exp(A_log)`; `ssd.ts` uses
 *     `A = -softplus(A_log)`. The port re-expresses the same `A` as
 *     `softplus⁻¹(exp(A_log))` so decay rates are preserved exactly, rather than
 *     copying the raw value and silently changing every head's decay.
 *  3. **The gate branch.** `Mamba2Block` has no `z` gate: it RMS-norms the scan
 *     output and projects out, where upstream computes
 *     `norm(y * silu(z))`. The `d_inner × d_model` gate rows therefore have no
 *     target. That is a REAL loss of capability, not a formatting detail, and it
 *     is reported as a `PortDiscard` on every layer so a caller sees it.
 *
 * Also mapped without transform: `mixer.norm.weight` → the block's inner
 * `normWeight` (applied ungated here, per 3), `backbone.layers.N.norm.weight` →
 * `preNormWeight`, `conv1d.{weight,bias}` (depthwise, taps reversed) → the fused
 * `[x, B, C]` conv, and `D` / `dt_bias` per head. Linear weights are
 * `[out, in]` row-major on both sides, so no transposes.
 *
 * `tie_word_embeddings: false`, so `lm_head.weight` is a real stored tensor with
 * no target here (`HybridMambaModel` ties its head to the embedding) — it is
 * reported as unmapped.
 */

import type { HybridMambaModelConfig, LayerSpec } from "../../model/mamba_model.js";
import type { PortDiscard, PortPlan, PortRule } from "./plan.js";
import {
  expandFactor,
  firstInt,
  readInnerSize,
  requireInt,
  type ForeignConfig,
  type ForeignMambaAdapter,
  type PortTarget,
} from "./adapter.js";

const ID = "codestral_mamba";

/** `backbone.embeddings` (HF) vs `backbone.embedding` (Mistral-native). */
const EMBEDDING_NAMES = ["backbone.embeddings.weight", "backbone.embedding.weight"];

interface Mamba2Geometry {
  vocabSize: number;
  dModel: number;
  numLayers: number;
  dInner: number;
  dState: number;
  dConv: number;
  nHeads: number;
  nGroups: number;
  chunkLen: number;
}

function rules(a: Mamba2Geometry): PortRule[] {
  const { vocabSize: V, dModel: E, numLayers: L, dInner: D, dState: N, dConv: K, nHeads: H, nGroups: G } = a;
  const convDim = D + 2 * G * N;
  /** Rows this engine keeps out of the fused projection: [x, B, C, dt]. */
  const keptRows = convDim + H;

  const out: PortRule[] = [
    { target: "embedding", shape: [V, E], from: EMBEDDING_NAMES, op: { kind: "copy" } },
  ];

  for (let i = 0; i < L; i++) {
    const m = `backbone.layers.${i}.mixer`;
    const t = `layer${i}`;
    out.push(
      {
        target: `${t}.wInProj`,
        shape: [keptRows, E],
        from: [`${m}.in_proj.weight`],
        // Drop the leading gate rows; keep [x, B, C, dt] verbatim.
        op: { kind: "rowSlice", start: D, count: keptRows },
        note: "upstream rows are [gate, x, B, C, dt]; this block has no gate branch",
      },
      {
        target: `${t}.wConv`,
        shape: [convDim, K],
        from: [`${m}.conv1d.weight`],
        op: { kind: "convTaps" },
        note: "depthwise [conv_dim, 1, d_conv] → [conv_dim, d_conv] with the tap axis reversed",
      },
      { target: `${t}.bConv`, shape: [convDim], from: [`${m}.conv1d.bias`], op: { kind: "copy" } },
      {
        target: `${t}.A_log`,
        shape: [H],
        from: [`${m}.A_log`],
        op: { kind: "negExpToSoftplusInv" },
        note: "upstream A = -exp(A_log); this engine's SSD kernel uses A = -softplus(A_log)",
      },
      { target: `${t}.dt_bias`, shape: [H], from: [`${m}.dt_bias`], op: { kind: "copy" } },
      { target: `${t}.D_vec`, shape: [H], from: [`${m}.D`], op: { kind: "copy" } },
      { target: `${t}.wOutProj`, shape: [E, D], from: [`${m}.out_proj.weight`], op: { kind: "copy" } },
      {
        target: `${t}.normWeight`,
        shape: [D],
        from: [`${m}.norm.weight`],
        op: { kind: "copy" },
        note: "upstream this gain belongs to a GATED RMSNorm; applied ungated here",
      },
      {
        target: `${t}.preNormWeight`,
        shape: [E],
        from: [`backbone.layers.${i}.norm.weight`],
        op: { kind: "copy" },
      },
    );
  }

  out.push({ target: "final_norm", shape: [E], from: ["backbone.norm_f.weight"], op: { kind: "copy" } });
  return out;
}

function discards(a: Mamba2Geometry): PortDiscard[] {
  const out: PortDiscard[] = [];
  for (let i = 0; i < a.numLayers; i++) {
    out.push({
      source: `backbone.layers.${i}.mixer.in_proj.weight`,
      elements: a.dInner * a.dModel,
      reason:
        "the leading d_inner gate (z) rows — Mamba2Block computes norm(y) rather than " +
        "norm(y * silu(z)), so it has no gate branch to receive them",
    });
  }
  return out;
}

export const codestralMambaAdapter: ForeignMambaAdapter = {
  id: ID,
  label: "Codestral-Mamba (Mamba-2 / SSD)",
  modelTypes: ["mamba2"],
  // `MambaCodestralForCausalLM` is not a transformers class — the published
  // config says `Mamba2ForCausalLM` — but it is a common mis-citation, so it is
  // accepted here rather than sending a correct checkpoint down the error path.
  architectures: ["Mamba2ForCausalLM", "MambaCodestralForCausalLM"],

  describe(config: ForeignConfig): PortTarget {
    const dModel = requireInt(config, "hidden_size");
    const numLayers = requireInt(config, "num_hidden_layers");
    const vocabSize = requireInt(config, "vocab_size");
    const dState = requireInt(config, "state_size");
    const dConv = requireInt(config, "conv_kernel");
    const nHeads = requireInt(config, "num_heads");
    const nGroups = firstInt(config, ["n_groups"]) ?? 1;
    const chunkLen = firstInt(config, ["chunk_size"]) ?? 256;
    const dInner = readInnerSize(config, dModel);
    const expand = expandFactor(dInner, dModel, "Codestral-Mamba");

    // Upstream's own invariant (`hidden_size * expand == num_heads * head_dim`);
    // Mamba2Block enforces the same divisibility, so check it here where the
    // message can name the config fields that disagree.
    const headDim = firstInt(config, ["head_dim"]);
    if (dInner % nHeads !== 0) {
      throw new Error(
        `import/foreign: Codestral-Mamba d_inner ${dInner} is not divisible by num_heads ${nHeads}`,
      );
    }
    if (headDim !== undefined && headDim * nHeads !== dInner) {
      throw new Error(
        `import/foreign: Codestral-Mamba config is inconsistent — head_dim ${headDim} × ` +
          `num_heads ${nHeads} = ${headDim * nHeads}, but d_inner is ${dInner}`,
      );
    }

    const layers: LayerSpec[] = Array.from({ length: numLayers }, () => ({ type: "mamba2" as const }));
    const modelConfig: HybridMambaModelConfig = {
      vocabSize,
      dModel,
      numLayers,
      layers,
      dState,
      dConv,
      expand,
      nHeads,
      nGroups,
      chunkLen,
    };

    const geometry: Mamba2Geometry = {
      vocabSize,
      dModel,
      numLayers,
      dInner,
      dState,
      dConv,
      nHeads,
      nGroups,
      chunkLen,
    };

    return { modelConfig, plan: { adapter: ID, rules: rules(geometry), discards: discards(geometry) } };
  },
};
