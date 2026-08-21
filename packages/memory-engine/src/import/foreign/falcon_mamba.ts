/**
 * import/foreign/falcon_mamba.ts — weight port for Falcon-Mamba
 * (`FalconMambaForCausalLM`, `model_type: "falcon_mamba"`, Mamba-1 / S6).
 *
 * Source layout confirmed against `tiiuae/falcon-mamba-7b`'s published
 * `config.json`, `model.safetensors.index.json` (643 tensors = 10 per layer × 64
 * + 3 globals) and the safetensors header of shard 1, plus
 * `transformers/models/falcon_mamba/modeling_falcon_mamba.py`:
 *
 *   backbone.embeddings.weight                       [vocab, d_model]
 *   backbone.layers.{N}.norm.weight                  [d_model]
 *   backbone.layers.{N}.mixer.in_proj.weight         [2*d_inner, d_model]
 *   backbone.layers.{N}.mixer.conv1d.weight          [d_inner, 1, d_conv]
 *   backbone.layers.{N}.mixer.conv1d.bias            [d_inner]
 *   backbone.layers.{N}.mixer.x_proj.weight          [dt_rank + 2*d_state, d_inner]
 *   backbone.layers.{N}.mixer.dt_proj.weight         [d_inner, dt_rank]
 *   backbone.layers.{N}.mixer.dt_proj.bias           [d_inner]
 *   backbone.layers.{N}.mixer.A_log                  [d_inner, d_state]
 *   backbone.layers.{N}.mixer.D                      [d_inner]
 *   backbone.layers.{N}.mixer.out_proj.weight        [d_model, d_inner]
 *   backbone.norm_f.weight                           [d_model]
 *   lm_head.weight                                   [vocab, d_model]
 *
 * Conventions that line up with `Mamba1Block` as-is, so most rules are copies:
 *   • `in_proj` rows are `[x, gate]` (`hidden_states, gate = …chunk(2, dim=1)`),
 *     and `Mamba1Block.forward` also takes the first half as `x` and the second
 *     as the gate `z`.
 *   • `x_proj` rows are `[dt, B, C]`, matching the block's split order.
 *   • `A = -exp(A_log)` in both — `selective_scan.ts` computes
 *     `a_cont = -exp(clamp(a_log, -10, 5))`.
 *   • `dt_proj.bias` is added inside `softplus(dt + bias)` upstream; here it is
 *     the `dt_proj` linear's bias and the scan applies `softplus` afterwards —
 *     algebraically the same value.
 *   • Linear weights are `[out, in]` row-major on both sides (no transpose).
 *
 * Fidelity caveats recorded on the plan rather than hidden:
 *   • `use_bias: false` upstream, so `in_proj` / `x_proj` / `out_proj` ship no
 *     bias; this engine's blocks always own one, and they are filled with zeros
 *     (an exact no-op, not an approximation).
 *   • Falcon-Mamba RMS-normalises `dt`, `B` and `C` inside the mixer. Those
 *     norms are `FalconMambaWeightlessRMSNorm` with `persistent=False` buffers,
 *     so they contribute NO tensors to the checkpoint — nothing is dropped from
 *     the file — but this engine's S6 scan has no such normalisation step, so a
 *     ported model is not numerically identical to the original.
 *   • `tie_word_embeddings: false`, so `lm_head.weight` is a real stored tensor.
 *     `HybridMambaModel` always ties its LM head to the embedding table, so that
 *     tensor has no target and is reported as unmapped.
 */

import type { HybridMambaModelConfig } from "../../model/mamba_model.js";
import type { PortPlan, PortRule } from "./plan.js";
import {
  expandFactor,
  readInnerSize,
  readTimeStepRank,
  requireInt,
  type ForeignConfig,
  type ForeignMambaAdapter,
  type PortTarget,
} from "./adapter.js";

const ID = "falcon_mamba";

/** `backbone.embeddings` (HF) vs `backbone.embedding` (Mamba-SSM native). */
const EMBEDDING_NAMES = ["backbone.embeddings.weight", "backbone.embedding.weight"];

function rules(a: {
  vocabSize: number;
  dModel: number;
  numLayers: number;
  dInner: number;
  dState: number;
  dConv: number;
  dtRank: number;
}): PortRule[] {
  const { vocabSize: V, dModel: E, numLayers: L, dInner: D, dState: N, dConv: K, dtRank: R } = a;
  const out: PortRule[] = [
    { target: "embedding", shape: [V, E], from: EMBEDDING_NAMES, op: { kind: "copy" } },
  ];

  for (let i = 0; i < L; i++) {
    const m = `backbone.layers.${i}.mixer`;
    const t = `layer${i}`;
    out.push(
      { target: `${t}.wInProj`, shape: [2 * D, E], from: [`${m}.in_proj.weight`], op: { kind: "copy" } },
      {
        target: `${t}.bInProj`,
        shape: [2 * D],
        from: [],
        op: { kind: "fill", value: 0 },
        note: "Falcon-Mamba sets use_bias=false — in_proj ships no bias; zeros are exact",
      },
      {
        target: `${t}.wConv`,
        shape: [D, K],
        from: [`${m}.conv1d.weight`],
        op: { kind: "convTaps" },
        note: "depthwise [d_inner, 1, d_conv] → [d_inner, d_conv] with the tap axis reversed",
      },
      { target: `${t}.bConv`, shape: [D], from: [`${m}.conv1d.bias`], op: { kind: "copy" } },
      { target: `${t}.wXProj`, shape: [R + 2 * N, D], from: [`${m}.x_proj.weight`], op: { kind: "copy" } },
      {
        target: `${t}.bXProj`,
        shape: [R + 2 * N],
        from: [],
        op: { kind: "fill", value: 0 },
        note: "x_proj is bias-free upstream; zeros are exact",
      },
      { target: `${t}.wDtProj`, shape: [D, R], from: [`${m}.dt_proj.weight`], op: { kind: "copy" } },
      { target: `${t}.bDtProj`, shape: [D], from: [`${m}.dt_proj.bias`], op: { kind: "copy" } },
      { target: `${t}.A_log`, shape: [D, N], from: [`${m}.A_log`], op: { kind: "copy" } },
      { target: `${t}.D_vec`, shape: [D], from: [`${m}.D`], op: { kind: "copy" } },
      { target: `${t}.wOutProj`, shape: [E, D], from: [`${m}.out_proj.weight`], op: { kind: "copy" } },
      {
        target: `${t}.bOutProj`,
        shape: [E],
        from: [],
        op: { kind: "fill", value: 0 },
        note: "out_proj is bias-free upstream; zeros are exact",
      },
      {
        target: `${t}.normWeight`,
        shape: [E],
        from: [`backbone.layers.${i}.norm.weight`],
        op: { kind: "copy" },
      },
    );
  }

  out.push({ target: "final_norm", shape: [E], from: ["backbone.norm_f.weight"], op: { kind: "copy" } });
  return out;
}

export const falconMambaAdapter: ForeignMambaAdapter = {
  id: ID,
  label: "Falcon-Mamba (Mamba-1 / S6)",
  modelTypes: ["falcon_mamba"],
  architectures: ["FalconMambaForCausalLM"],

  describe(config: ForeignConfig): PortTarget {
    const dModel = requireInt(config, "hidden_size");
    const numLayers = requireInt(config, "num_hidden_layers");
    const vocabSize = requireInt(config, "vocab_size");
    const dState = requireInt(config, "state_size");
    const dConv = requireInt(config, "conv_kernel");
    const dInner = readInnerSize(config, dModel);
    const expand = expandFactor(dInner, dModel, "Falcon-Mamba");
    const dtRank = readTimeStepRank(config, dModel);

    const modelConfig: HybridMambaModelConfig = {
      vocabSize,
      dModel,
      numLayers,
      dState,
      dConv,
      expand,
      // Every layer is a Mamba-1 mixer (the model's default schedule), but the
      // dt rank and conv bias are per-block settings the shorthand can't carry.
      defaultMamba1: { dtRank, biasConv: true },
    };

    const plan: PortPlan = {
      adapter: ID,
      rules: rules({ vocabSize, dModel, numLayers, dInner, dState, dConv, dtRank }),
      // Nothing is partially consumed: every Falcon-Mamba tensor this plan
      // touches is taken whole. `lm_head.weight` is untouched entirely and
      // surfaces through `unmappedSources`.
      discards: [],
    };

    return { modelConfig, plan };
  },
};
