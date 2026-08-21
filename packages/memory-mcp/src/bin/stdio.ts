#!/usr/bin/env node
/**
 * `builderforce-memory-mcp` — stdio MCP server over the LOCAL MemoryStore.
 *
 * Env:
 *   BUILDERFORCE_MEMORY_DB        IndexedDB database name (default: MemoryStore default).
 *   BUILDERFORCE_MEMORY_READONLY  '1' to disable remember/forget tools.
 *   BUILDERFORCE_MEMORY_FILE      Absolute path to a JSON snapshot. Defaults to the
 *                                 shared per-machine store (~/.builderforce-memory/
 *                                 memory.json) so memory survives the respawn every
 *                                 MCP client does between sessions — without a
 *                                 snapshot fake-indexeddb loses everything on exit.
 *   BUILDERFORCE_MEMORY_MODEL     Absolute path to a `.evermind` package (an
 *                                 `evermind-lm`). When set, recall is ranked by SSM
 *                                 embedding cosine FUSED with the lexical ranking
 *                                 instead of lexically alone. Pure CPU — no GPU is
 *                                 involved; see src/embedding/evermind-embedder.ts.
 *   BUILDERFORCE_MEMORY_TOKENIZER Tokenizer JSON matching that checkpoint. Defaults
 *                                 to `<model>.tokenizer.json`.
 *   BUILDERFORCE_MEMORY_VECTORS   Persistent vector-cache path. Defaults beside the
 *                                 snapshot, so a respawned subprocess does not
 *                                 re-embed the whole store on its first recall.
 *   BUILDERFORCE_GATEWAY_URL      Gateway base URL (default https://api.builderforce.ai).
 *   BUILDERFORCE_API_KEY          `bfk_*` tenant key. When set, exposes the cost tools
 *                                 (token_usage, model_efficiency).
 *
 * With no model configured recall ranks lexically (BM25) and the tool result SAYS
 * so ("Ranked by: lexical recall (fallback …)"), so a degrade is never silent.
 */

import { createLocalMemoryStoreBackend } from "../backends/memory-store.js";
import { createRecallEmbedder } from "../embedding/index.js";
import { resolveMemoryFile } from "../install/server-spec.js";
import { runStdio } from "../transports/stdio.js";

const memoryFile = resolveMemoryFile();

const backend = await createLocalMemoryStoreBackend({
    dbName: process.env["BUILDERFORCE_MEMORY_DB"],
    persistFile: memoryFile,
    // Null when no model is configured (or it fails to load) — recall then stays
    // lexical rather than the server refusing to start over an optional model.
    runtime: (await createRecallEmbedder({ memoryFile })) ?? undefined,
});

await runStdio(backend, {
    writable: process.env["BUILDERFORCE_MEMORY_READONLY"] !== "1",
    // Optional gateway-backed cost tools (token_usage, model_efficiency). Only
    // exposed when an API key is present; URL defaults to the public gateway.
    gatewayUrl: process.env["BUILDERFORCE_GATEWAY_URL"] ?? "https://api.builderforce.ai",
    gatewayApiKey: process.env["BUILDERFORCE_API_KEY"],
});
