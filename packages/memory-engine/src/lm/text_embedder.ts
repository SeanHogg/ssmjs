/**
 * lm/text_embedder.ts — text → vector, on the CPU, from an `EvermindLM`.
 *
 * `EvermindLM.embed()` turns TOKENS into a `dModel` vector. Every consumer that
 * wants SEMANTIC RECALL wants it over TEXT, and needs the vector to be stable
 * across processes, so it also needs (a) the tokenizer that produced those ids
 * and (b) a way to tell whether a cached vector came from this exact model. This
 * class is that pairing, in one place, so no consumer re-derives it.
 *
 * It deliberately satisfies the SAME structural shape the memory runtime's
 * `embed(text)` seam expects (`MemoryStore.recallRanked(query, k, runtime)`
 * accepts anything with an async `embed(text): Float32Array`), so a headless host
 * can hand one of these straight to the memory layer and get embedding-ranked
 * recall with NO GPU — the CPU LM is the whole dependency.
 */

import type { EvermindLM, TextCodec } from "./evermind_lm.js";
import { crc32 } from "../utils/crc32.js";

/** The minimal text-embedder seam the memory/recall layers consume. */
export interface TextEmbedder {
  /** Embed `text` as an L2-normalised vector of length {@link dimensions}. */
  embed(text: string): Promise<Float32Array>;
  /** Vector width (the model's `dModel`). */
  readonly dimensions: number;
  /**
   * Identity of the model+tokenizer that produces these vectors. Vectors are only
   * comparable within one fingerprint, so a persistent cache MUST discard entries
   * stamped with a different one.
   */
  readonly fingerprint: string;
}

/** A tokenizer this embedder can encode through (the engine's `BPETokenizer` fits). */
export type EmbedderCodec = Pick<TextCodec, "encode">;

/**
 * Pairs an `EvermindLM` with the tokenizer whose ids it was trained on.
 *
 * Pure compute and stateless: caching belongs to whoever owns a lifetime (the
 * memory store's in-process LRU, or a host's on-disk vector cache), not here.
 */
export class EvermindTextEmbedder implements TextEmbedder {
  readonly dimensions: number;
  readonly fingerprint: string;

  constructor(
    private readonly model: EvermindLM,
    private readonly codec: EmbedderCodec,
    /** Override the derived fingerprint (e.g. with a published model version). */
    fingerprint?: string,
  ) {
    this.dimensions = model.config.dModel;
    this.fingerprint = fingerprint ?? derivedFingerprint(model);
  }

  /**
   * Embed `text`. Async only to match the seam — the work is synchronous CPU, so
   * a caller that wants the vector without a microtask can use
   * {@link embedSync}.
   */
  async embed(text: string): Promise<Float32Array> {
    return this.embedSync(text);
  }

  /** The synchronous form — encode with the codec, then run the CPU LM. */
  embedSync(text: string): Float32Array {
    return this.model.embedText(text, this.codec as TextCodec);
  }
}

/**
 * Model identity: architecture plus a CRC-32 over the tied embedding table.
 *
 * The embedding table is the one tensor every position's vector flows through, so
 * an adapted checkpoint (which is what a delta merge produces) always changes it —
 * making this a cheap but honest "are these vectors still comparable?" stamp.
 */
function derivedFingerprint(model: EvermindLM): string {
  const c = model.config;
  const bytes = new Uint8Array(model.emb.buffer, model.emb.byteOffset, model.emb.byteLength);
  const arch = [c.vocabSize, c.dModel, c.numLayers, c.convKernel, c.hiddenDim, c.numExperts, c.topK].join("-");
  return `evermind-lm/${arch}/${crc32(bytes).toString(16)}`;
}
