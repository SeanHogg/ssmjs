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
 *   BUILDERFORCE_GATEWAY_URL      Gateway base URL (default https://api.builderforce.ai).
 *   BUILDERFORCE_API_KEY          `bfk_*` tenant key. When set, exposes the cost tools
 *                                 (token_usage, model_efficiency).
 *
 * Recall is lexical (Jaccard) here — this headless binary does not stand up the
 * SSM runtime/GPU. For SSM-embedding recall, embed the package in-process and
 * pass `runtime` to createLocalMemoryStoreBackend (see README).
 */

import { createLocalMemoryStoreBackend } from "../backends/memory-store.js";
import { resolveMemoryFile } from "../install/server-spec.js";
import { runStdio } from "../transports/stdio.js";

const backend = await createLocalMemoryStoreBackend({
    dbName: process.env["BUILDERFORCE_MEMORY_DB"],
    persistFile: resolveMemoryFile(),
});

await runStdio(backend, {
    writable: process.env["BUILDERFORCE_MEMORY_READONLY"] !== "1",
    // Optional gateway-backed cost tools (token_usage, model_efficiency). Only
    // exposed when an API key is present; URL defaults to the public gateway.
    gatewayUrl: process.env["BUILDERFORCE_GATEWAY_URL"] ?? "https://api.builderforce.ai",
    gatewayApiKey: process.env["BUILDERFORCE_API_KEY"],
});
