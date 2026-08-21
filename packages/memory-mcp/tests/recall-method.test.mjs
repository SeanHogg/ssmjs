/**
 * tests/recall-method.test.mjs — CPU SSM-embedding recall for the headless server.
 *
 * The stdio binary used to rank recall by word overlap on the assumption that the
 * embedding model meant a GPU. `EvermindLM` is pure CPU, so it does not. These
 * tests cover the three things that makes true:
 *
 *   1. the tool result NAMES its ranker, so a degrade to lexical is never silent;
 *   2. a `.evermind` package + tokenizer on disk becomes a working embedder, and
 *      the local backend then reports `method: 'embedding'`;
 *   3. the vectors are cached ACROSS PROCESSES — a per-session subprocess that
 *      re-embedded the whole store on first recall would cost more than the
 *      lexical ranking it replaces.
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { buildMemoryTools, CachedTextEmbedder, createEvermindEmbedder, createLocalMemoryStoreBackend } from "../dist/index.js";
import { EvermindLM, EvermindModelPackage, EvermindTextEmbedder, BPETokenizer } from "@seanhogg/builderforce-memory-engine";

const CORPUS = "agents recall facts before they act. the memory layer stores facts as embeddings.";

/** A tiny, untrained CPU model + its tokenizer — enough to prove the wiring. */
function cpuModel() {
  const tok = new BPETokenizer();
  tok.train(CORPUS, { numMerges: 40 });
  const lm = new EvermindLM({ vocabSize: tok.vocabSize, dModel: 16, numLayers: 2, hiddenDim: 24, seed: 7 });
  return { tok, lm };
}

function tmpdir(label) {
  return fs.mkdtempSync(path.join(os.tmpdir(), `bfmem-${label}-`));
}

async function recallText(backend, query = "facts") {
  const tool = buildMemoryTools(backend, {}).find((t) => t.name === "memory_recall");
  const res = await tool.handler({ query });
  return res.content[0].text;
}

// ── the tool names its ranker ─────────────────────────────────────────────────

test("a backend that cannot name its ranker produces no ranker line (back-compat)", async () => {
  const backend = { recall: async () => [{ key: "k", content: "v" }], get: async () => undefined, recallByTag: async () => [] };
  const text = await recallText(backend);
  assert.match(text, /k/);
  assert.ok(!text.includes("Ranked by:"), text);
});

test("memory_recall reports the embedding ranker when the backend used one", async () => {
  const backend = {
    recall: async () => [],
    recallRanked: async () => ({ hits: [{ key: "k", content: "v", score: 0.5 }], method: "embedding" }),
    get: async () => undefined,
    recallByTag: async () => [],
  };
  const text = await recallText(backend);
  assert.match(text, /Ranked by: semantic recall/);
});

test("memory_recall reports the lexical fallback, including when nothing matched", async () => {
  const backend = {
    recall: async () => [],
    recallRanked: async () => ({ hits: [], method: "lexical" }),
    get: async () => undefined,
    recallByTag: async () => [],
  };
  const text = await recallText(backend);
  assert.match(text, /No matching memories\./);
  assert.match(text, /Ranked by: lexical recall \(fallback/);
});

// ── the local backend, end to end ─────────────────────────────────────────────

test("the local backend ranks lexically with no embedder, and says so", async () => {
  const backend = await createLocalMemoryStoreBackend({ dbName: `mcp-lex-${Date.now()}` });
  await backend.remember({ key: "a", content: "agents recall facts before acting" });
  await backend.remember({ key: "b", content: "invoices reconciled in the ledger" });

  const ranked = await backend.recallRanked("recall facts", 2);
  assert.equal(ranked.method, "lexical");
  assert.equal(ranked.hits[0].key, "a");
  assert.ok(ranked.hits[0].score > 0, "the fused score must reach the tool layer");
  assert.match(await recallText(backend, "recall facts"), /Ranked by: lexical recall/);
});

test("the local backend ranks by CPU SSM embedding when one is supplied", async () => {
  const { tok, lm } = cpuModel();
  const backend = await createLocalMemoryStoreBackend({
    dbName: `mcp-emb-${Date.now()}`,
    runtime: new EvermindTextEmbedder(lm, tok),
  });
  await backend.remember({ key: "a", content: "agents recall facts before acting" });
  await backend.remember({ key: "b", content: "invoices reconciled in the ledger" });

  const ranked = await backend.recallRanked("recall facts", 2);
  assert.equal(ranked.method, "embedding");
  assert.match(await recallText(backend, "recall facts"), /Ranked by: semantic recall/);
});

// ── loading a checkpoint from disk ────────────────────────────────────────────

test("createEvermindEmbedder loads a .evermind package + tokenizer and matches the model", async () => {
  const dir = tmpdir("model");
  const { tok, lm } = cpuModel();
  const modelFile = path.join(dir, "memory.evermind");
  fs.writeFileSync(
    modelFile,
    Buffer.from(
      new Uint8Array(
        EvermindModelPackage.fromLM(lm, { name: "test", version: "1", card: { description: "test" } }).toBlob(),
      ),
    ),
  );
  fs.writeFileSync(`${modelFile}.tokenizer.json`, JSON.stringify(tok.toObject()));

  const embedder = await createEvermindEmbedder({ modelFile, cacheFile: path.join(dir, "vectors.json") });
  assert.ok(embedder, "the embedder must load from disk");
  assert.equal(embedder.dimensions, 16);
  // Loaded weights are the exported weights — the vector must be identical.
  assert.deepEqual(Array.from(await embedder.embed("recall facts")), Array.from(lm.embedText("recall facts", tok)));
});

test("createEvermindEmbedder degrades to null rather than throwing when there is no model", async () => {
  const dir = tmpdir("absent");
  assert.equal(await createEvermindEmbedder({ modelFile: path.join(dir, "nope.evermind") }), null);
});

// ── the persistent vector cache ───────────────────────────────────────────────

/** An embedder that counts how often it actually computed a vector. */
function countingEmbedder(fingerprint = "fp-1") {
  let calls = 0;
  return {
    dimensions: 3,
    fingerprint,
    calls: () => calls,
    embed: async (text) => {
      calls++;
      return Float32Array.from([text.length, 1, 0]);
    },
  };
}

test("vectors survive the process: a second embedder hydrates instead of recomputing", async () => {
  const cacheFile = path.join(tmpdir("cache"), "vectors.json");

  const first = countingEmbedder();
  const a = new CachedTextEmbedder(first, fs, { cacheFile, flushDelayMs: 0 });
  await a.embed("alpha");
  await a.embed("alpha"); // in-process cache hit
  assert.equal(first.calls(), 1);
  a.flush();
  assert.ok(fs.existsSync(cacheFile), "the cache must be written to disk");

  // A fresh instance — what a respawned stdio subprocess gets.
  const second = countingEmbedder();
  const b = new CachedTextEmbedder(second, fs, { cacheFile, flushDelayMs: 0 });
  assert.deepEqual(Array.from(await b.embed("alpha")), [5, 1, 0]);
  assert.equal(second.calls(), 0, "a persisted vector must not be recomputed");
});

test("a cache written by a different model is discarded, not mixed in", async () => {
  const cacheFile = path.join(tmpdir("fp"), "vectors.json");

  const a = new CachedTextEmbedder(countingEmbedder("fp-1"), fs, { cacheFile, flushDelayMs: 0 });
  await a.embed("alpha");
  a.flush();

  // Same text, ADAPTED model → the stored vector is no longer comparable.
  const adapted = countingEmbedder("fp-2");
  const b = new CachedTextEmbedder(adapted, fs, { cacheFile, flushDelayMs: 0 });
  await b.embed("alpha");
  assert.equal(adapted.calls(), 1, "a stale-fingerprint cache must be dropped");
});

test("a corrupt cache file is ignored rather than fatal", async () => {
  const cacheFile = path.join(tmpdir("corrupt"), "vectors.json");
  fs.writeFileSync(cacheFile, "{ not json");
  const inner = countingEmbedder();
  const e = new CachedTextEmbedder(inner, fs, { cacheFile, flushDelayMs: 0 });
  assert.deepEqual(Array.from(await e.embed("alpha")), [5, 1, 0]);
  assert.equal(inner.calls(), 1);
});

test("the cache is bounded — the least-recently-used vector is evicted", async () => {
  const inner = countingEmbedder();
  const e = new CachedTextEmbedder(inner, fs, { maxEntries: 2, flushDelayMs: 0 });
  await e.embed("aa");
  await e.embed("bbb");
  await e.embed("aa"); // touch: 'bbb' is now the LRU
  await e.embed("cccc"); // evicts 'bbb'
  assert.equal(inner.calls(), 3);
  await e.embed("aa");
  assert.equal(inner.calls(), 3, "'aa' must still be cached");
  await e.embed("bbb");
  assert.equal(inner.calls(), 4, "'bbb' must have been evicted");
});

// ── a self-contained .evermind needs no separate tokenizer file ───────────────
//
// `.evermind` packages can now EMBED their tokenizer (`PackageMeta.tokenizer`),
// which is checksummed and vocab-size-checked against the checkpoint at package
// time. That is strictly safer than a sibling file, which can silently be the
// WRONG vocabulary — so the loader must prefer it, and must not require the file.

test("an embedded tokenizer is used, with no tokenizer file on disk", async () => {
  const dir = tmpdir("selfcontained");
  const { tok, lm } = cpuModel();
  const modelFile = path.join(dir, "model.evermind");
  const pkg = EvermindModelPackage.fromLM(lm, {
    name: "m", version: "1", card: { description: "self-contained" }, tokenizer: tok,
  });
  fs.writeFileSync(modelFile, Buffer.from(pkg.toBlob()));
  assert.equal(fs.existsSync(`${modelFile}.tokenizer.json`), false);

  const embedder = await createEvermindEmbedder({ modelFile });
  assert.ok(embedder, "the loader must stand up from the package alone");

  // It must agree with the in-process model built from the same pair.
  const reference = new EvermindTextEmbedder(lm, tok);
  const got = await embedder.embed(CORPUS);
  const want = await reference.embed(CORPUS);
  assert.equal(got.length, want.length);
  for (let i = 0; i < want.length; i++) assert.ok(Math.abs(got[i] - want[i]) < 1e-6, `dim ${i}`);
  embedder.flush();
});

test("a package with NO embedded tokenizer still needs the file, and says no without one", async () => {
  const dir = tmpdir("nofile");
  const { tok, lm } = cpuModel();
  const modelFile = path.join(dir, "model.evermind");
  // Deliberately packaged WITHOUT a tokenizer — the pre-existing shape.
  const pkg = EvermindModelPackage.fromLM(lm, { name: "m", version: "1", card: { description: "bare" } });
  fs.writeFileSync(modelFile, Buffer.from(pkg.toBlob()));

  assert.equal(await createEvermindEmbedder({ modelFile }), null);

  fs.writeFileSync(`${modelFile}.tokenizer.json`, JSON.stringify(tok.toObject()));
  assert.ok(await createEvermindEmbedder({ modelFile }), "the sibling file still works");
});

test("the embedded tokenizer wins over a WRONG sibling file", async () => {
  const dir = tmpdir("precedence");
  const { tok, lm } = cpuModel();
  const modelFile = path.join(dir, "model.evermind");
  fs.writeFileSync(modelFile, Buffer.from(
    EvermindModelPackage.fromLM(lm, { name: "m", version: "1", card: { description: "c" }, tokenizer: tok }).toBlob(),
  ));
  // A sibling file trained on different text — a different vocabulary entirely.
  const wrong = new BPETokenizer();
  wrong.train("completely unrelated corpus about shipping containers and freight", { numMerges: 40 });
  fs.writeFileSync(`${modelFile}.tokenizer.json`, JSON.stringify(wrong.toObject()));

  const embedder = await createEvermindEmbedder({ modelFile });
  const reference = new EvermindTextEmbedder(lm, tok);
  const got = await embedder.embed(CORPUS);
  const want = await reference.embed(CORPUS);
  for (let i = 0; i < want.length; i++) assert.ok(Math.abs(got[i] - want[i]) < 1e-6, `dim ${i}`);
  embedder.flush();
});
