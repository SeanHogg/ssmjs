#!/usr/bin/env node
/**
 * Keep `packages/memory-mcp/server.json` — the MCP Registry manifest — in step
 * with `package.json`.
 *
 * The registry rejects a manifest whose declared package version isn't on npm,
 * so the two versions must move together. `package.json` is the source of truth;
 * this script propagates it and CI verifies it, rather than trusting anyone to
 * remember a second bump.
 *
 * Usage:  node scripts/sync-mcp-registry.mjs [--check]
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { emitFiles } from "./lib/emit.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PKG_DIR = path.join(ROOT, "packages", "memory-mcp");
const CHECK = process.argv.includes("--check");

const pkg = JSON.parse(fs.readFileSync(path.join(PKG_DIR, "package.json"), "utf8"));
const manifest = JSON.parse(fs.readFileSync(path.join(PKG_DIR, "server.json"), "utf8"));

// Ownership check: the registry only accepts the manifest if the npm package
// itself claims the same server name via `mcpName`. Assert it here so a rename
// fails at build time rather than at publish time.
if (pkg.mcpName !== manifest.name) {
  console.error(`✗ package.json mcpName (${pkg.mcpName}) does not match server.json name (${manifest.name}).`);
  process.exit(1);
}

manifest.version = pkg.version;
for (const entry of manifest.packages ?? []) {
  if (entry.identifier === pkg.name) entry.version = pkg.version;
}

emitFiles(PKG_DIR, { "server.json": `${JSON.stringify(manifest, null, 2)}\n` }, {
  check: CHECK,
  label: "packages/memory-mcp/server.json",
  fixHint: "run `pnpm registry:sync` and commit the result",
});
