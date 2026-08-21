/**
 * Compaction + snapshot-freshness tests.
 *
 * Both cover the SAME race: the on-disk snapshot is shared with other processes
 * (the BuilderForce VS Code extension rewrites absorbed entries to stubs in place),
 * while this server holds the whole store in memory and re-snapshots on every
 * write. The first test proves the first-class path — `memory_compact` writes
 * through the live store, so memory and disk agree. The second proves the guard —
 * an EXTERNAL rewrite is re-hydrated, not clobbered, by the next `remember`.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { buildMemoryTools, createLocalMemoryStoreBackend } from "../dist/index.js";

/** A throwaway snapshot path; each test gets its own store + file. */
function tempSnapshot(name) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "bfmem-compact-"));
  return path.join(dir, `${name}.json`);
}

const readSnapshot = (file) => JSON.parse(fs.readFileSync(file, "utf8"));
const entryFor = (rows, key) => rows.find((e) => e.key === key);

// A realistic multi-line fact: compaction keeps the first line as the pointer, so the
// body has to be genuinely longer than that line for the shrink to be worth writing.
const FIRST = "The gateway mints tenant JWTs from the workspace key.";
const LONG = [
  FIRST,
  "",
  "Every adapter converts money to cents at its own edge before it reaches the ledger.",
  "Currency is normalised once, at the adapter boundary, and never inside the domain.",
].join("\n");

test("memory_compact rewrites through the live store and the snapshot agrees", async () => {
  const file = tempSnapshot("compact");
  const backend = await createLocalMemoryStoreBackend({ persistFile: file, dbName: "compact-1" });
  const tools = buildMemoryTools(backend);

  const compact = tools.find((t) => t.name === "memory_compact");
  assert.ok(compact, "memory_compact should be registered on a writable backend");

  await backend.remember({ key: "arch.money", content: LONG, tags: ["project"], importance: 0.8 });
  await backend.remember({ key: "arch.keep", content: LONG });

  const res = await compact.handler({ keys: ["arch.money"], version: 7 });
  assert.equal(res.isError, undefined);
  assert.match(res.content[0].text, /Compacted 1 memory\(ies\)/);

  // The LIVE store carries the stub — not just the file.
  const live = await backend.get("arch.money");
  assert.match(live.content, /^\[absorbed→Evermind v7\] The gateway mints tenant JWTs/);
  assert.deepEqual(live.tags, ["project"], "compaction must not reclassify the fact");
  assert.equal(live.importance, 0.8);

  // …and so does the snapshot, with the untouched entry intact.
  const rows = readSnapshot(file);
  assert.match(entryFor(rows, "arch.money").content, /^\[absorbed→Evermind v7\]/);
  assert.equal(entryFor(rows, "arch.keep").content, LONG);

  // Idempotent: a second pass finds nothing left to shrink.
  const again = await compact.handler({ keys: ["arch.money"], version: 8 });
  assert.match(again.content[0].text, /Compacted 0 memory\(ies\)/);
  assert.match(again.content[0].text, /already compacted/);
  assert.match(readSnapshot(file).find((e) => e.key === "arch.money").content, /v7/);

  // A later write must NOT resurrect the pre-compaction body.
  await backend.remember({ key: "arch.other", content: "unrelated" });
  assert.match(readSnapshot(file).find((e) => e.key === "arch.money").content, /^\[absorbed→Evermind v7\]/);
});

test("an external rewrite of the snapshot is re-hydrated, not clobbered, on the next remember", async () => {
  const file = tempSnapshot("external");
  const backend = await createLocalMemoryStoreBackend({ persistFile: file, dbName: "compact-2" });

  await backend.remember({ key: "fact.a", content: LONG });
  await backend.remember({ key: "fact.b", content: `${LONG} (b)` });
  await backend.remember({ key: "fact.gone", content: "to be deleted externally" });

  // Another process (the VS Code extension) compacts fact.a and drops fact.gone.
  const STUB = "[absorbed→Evermind v3] The gateway mints tenant JWTs";
  const external = readSnapshot(file)
    .filter((e) => e.key !== "fact.gone")
    .map((e) => (e.key === "fact.a" ? { ...e, content: STUB } : e));
  fs.writeFileSync(file, JSON.stringify(external, null, 2));

  // The server's next write must fold the external edit in first.
  await backend.remember({ key: "fact.c", content: "written after the external edit" });

  const rows = readSnapshot(file);
  assert.equal(entryFor(rows, "fact.a").content, STUB, "the external stub must survive our snapshot");
  assert.equal(entryFor(rows, "fact.b").content, `${LONG} (b)`);
  assert.equal(entryFor(rows, "fact.gone"), undefined, "an externally deleted key must not come back");
  assert.equal(entryFor(rows, "fact.c").content, "written after the external edit");

  // The in-memory store agrees with disk — that is the whole point.
  assert.equal((await backend.get("fact.a")).content, STUB);
  assert.equal(await backend.get("fact.gone"), undefined);
});

test("our own snapshot writes never trip the freshness watch", async () => {
  const file = tempSnapshot("loop");
  const backend = await createLocalMemoryStoreBackend({ persistFile: file, dbName: "compact-3" });

  await backend.remember({ key: "k1", content: "one" });
  const afterFirst = fs.statSync(file).mtimeMs;

  // Reads must not write. If ensureFresh treated our own output as external it
  // would re-hydrate (and a re-hydrate followed by a write would move mtime).
  await backend.recall("one", 3);
  await backend.get("k1");
  await backend.recallByTag("none", 3);
  assert.equal(fs.statSync(file).mtimeMs, afterFirst, "reads must leave the snapshot untouched");
  assert.equal((await backend.get("k1")).content, "one");
});
