/**
 * tests/text_embedder.test.ts — the CPU SSM embedding path.
 *
 * `EvermindLM.embed()` is what lets a HEADLESS host rank recall semantically: it
 * is the CPU counterpart of `HybridMambaModel.embed()` (mean-pool + L2-normalise
 * the hidden state the LM head consumes) and needs no WebGPU device at all. These
 * tests pin that contract, because a silently-wrong embedding degrades recall
 * without failing anything.
 */

import { EvermindLM, EvermindLMTrainer } from "../src/lm/evermind_lm";
import { EvermindTextEmbedder } from "../src/lm/text_embedder";
import { BPETokenizer } from "../src/tokenizer/bpe";

const CORPUS =
  "The memory layer stores facts as embeddings. " +
  "Agents recall facts before they act. " +
  "The planning loop retrieves context before generating.";

function lm(over: Partial<{ vocabSize: number; dModel: number; numLayers: number; hiddenDim: number; seed: number }> = {}) {
  return new EvermindLM({
    vocabSize: over.vocabSize ?? 24,
    dModel: over.dModel ?? 16,
    numLayers: over.numLayers ?? 2,
    hiddenDim: over.hiddenDim ?? 24,
    seed: over.seed ?? 7,
  });
}

function l2(v: Float32Array): number {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}

describe("EvermindLM.embed", () => {
  test("returns a dModel-wide, L2-normalised, deterministic vector", () => {
    const model = lm();
    const a = model.embed([3, 9, 1, 4]);
    const b = model.embed([3, 9, 1, 4]);
    expect(a).toHaveLength(16);
    expect(l2(a)).toBeCloseTo(1, 5);
    expect(Array.from(b)).toEqual(Array.from(a));
  });

  test("an empty sequence yields the zero vector rather than NaNs", () => {
    const v = lm().embed([]);
    expect(v).toHaveLength(16);
    expect(Array.from(v).every((x) => x === 0)).toBe(true);
  });

  test("different sequences embed differently", () => {
    const model = lm();
    expect(Array.from(model.embed([1, 2, 3]))).not.toEqual(Array.from(model.embed([7, 8, 9])));
  });

  /**
   * The contract shared with `HybridMambaModel.embed()`: pool the SAME hidden
   * state the tied head consumes, then L2-normalise. `forward()` exposes that
   * state as `cache.finalX`, so the two can be compared exactly — if `embed()`
   * ever pooled something else (pre-residual, post-head, per-layer) this fails.
   */
  test("is exactly the L2-normalised mean of forward()'s final hidden state", () => {
    const model = lm();
    const tokens = [2, 5, 5, 11, 0];
    const { cache } = model.forward(tokens);

    const dModel = model.config.dModel;
    const pooled = new Float32Array(dModel);
    for (const x of cache.finalX) for (let c = 0; c < dModel; c++) pooled[c]! += x[c]!;
    for (let c = 0; c < dModel; c++) pooled[c]! /= tokens.length;
    const norm = l2(pooled) || 1;
    for (let c = 0; c < dModel; c++) pooled[c]! /= norm;

    const got = model.embed(tokens);
    for (let c = 0; c < dModel; c++) expect(got[c]!).toBeCloseTo(pooled[c]!, 5);
  });

  test("evaluates exactly seqLen x layers block positions (no head projection)", () => {
    const model = lm({ numLayers: 3 });
    model.resetStats();
    model.embed([1, 2, 3, 4, 5, 6]);
    expect(model.positionsEvaluated).toBe(6 * 3);
  });

  test("embedText encodes through the codec then embeds", () => {
    const tok = new BPETokenizer();
    tok.train(CORPUS, { numMerges: 40 });
    const model = lm({ vocabSize: tok.vocabSize });
    expect(Array.from(model.embedText("agents recall facts", tok))).toEqual(
      Array.from(model.embed(tok.encode("agents recall facts"))),
    );
  });
});

describe("EvermindTextEmbedder", () => {
  function build(seed = 7) {
    const tok = new BPETokenizer();
    tok.train(CORPUS, { numMerges: 40 });
    const model = lm({ vocabSize: tok.vocabSize, seed });
    return { tok, model, embedder: new EvermindTextEmbedder(model, tok) };
  }

  test("exposes dModel-wide vectors and matches the model's own embedText", async () => {
    const { model, tok, embedder } = build();
    expect(embedder.dimensions).toBe(model.config.dModel);
    const v = await embedder.embed("agents recall facts");
    expect(Array.from(v)).toEqual(Array.from(model.embedText("agents recall facts", tok)));
  });

  test("the fingerprint is stable for one model and differs across models", () => {
    const a = build(7).embedder;
    const b = build(7).embedder;
    const c = build(99).embedder;
    expect(b.fingerprint).toBe(a.fingerprint);
    expect(c.fingerprint).not.toBe(a.fingerprint);
  });

  /**
   * Adapting the checkpoint MUST move the fingerprint: a persistent vector cache
   * keyed on it would otherwise serve vectors from the pre-adaptation model
   * forever, which is the silent way "recall stopped improving" happens.
   */
  test("adapting the model changes the fingerprint (so caches invalidate)", () => {
    const { tok, model } = build();
    const before = new EvermindTextEmbedder(model, tok).fingerprint;
    new EvermindLMTrainer(model, { lr: 0.05, epochs: 2 }).fit([tok.encode(CORPUS).slice(0, 24)]);
    expect(new EvermindTextEmbedder(model, tok).fingerprint).not.toBe(before);
  });

  test("an explicit fingerprint overrides the derived one", () => {
    const { tok, model } = build();
    expect(new EvermindTextEmbedder(model, tok, "project-evermind@42").fingerprint).toBe("project-evermind@42");
  });
});
