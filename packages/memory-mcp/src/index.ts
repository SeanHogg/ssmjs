/**
 * @seanhogg/builderforce-memory-mcp — expose @seanhogg/builderforce-memory to MCP clients.
 *
 * One token-saving tool core over a pluggable MemoryBackend, three transports:
 *   - createMemoryMcpServer  → in-process Claude Agent SDK (type:"sdk")
 *   - runStdio               → stdio subprocess (any language)
 *   - createMemoryHttpHandler→ Streamable HTTP (multi-tenant / networked)
 */

// ── Seam ────────────────────────────────────────────────────────────────────
export type { MemoryBackend, RankedRecall, RecallHit, RecallMethod, RememberInput } from "./backend.js";

// -- CPU SSM-embedding recall (no GPU) --------------------------------------
// The headless servers rank recall by EvermindLM embedding cosine fused with the
// lexical ranking whenever a checkpoint is configured; exported so an embedding
// host can build the same embedder (and share its persistent vector cache).
export { createRecallEmbedder, createEvermindEmbedder, CachedTextEmbedder, defaultVectorCacheFile, MODEL_FILE_ENV, TOKENIZER_FILE_ENV, VECTOR_CACHE_ENV } from "./embedding/index.js";
export type { EvermindEmbedderOptions, PersistentTextEmbedder, TextEmbedderLike, RecallEmbedderEnv } from "./embedding/index.js";

// ── Local backend (IndexedDB via @seanhogg/builderforce-memory) ────────────────────────
export { MemoryStoreBackend, createLocalMemoryStoreBackend } from "./backends/memory-store.js";
export type { LocalBackendOptions } from "./backends/memory-store.js";

// ── Tool core ─────────────────────────────────────────────────────────────────
export { buildMemoryTools } from "./tools.js";
export type { MemoryTool, MemoryToolsOptions, ToolResult } from "./tools.js";

// ── Compaction (absorbed fact → one-line pointer stub) ───────────────────────
// Exported so an EXTERNAL compactor (the BuilderForce VS Code extension rewrites
// the snapshot file directly) can produce byte-identical stubs, keeping the two
// paths idempotent with respect to each other.
export { STUB_PREFIX, DEFAULT_STUB_CHARS, isStub, firstLine, memoryStub, planCompaction } from "./compaction.js";
export type { CompactionCandidate, CompactionWrite, CompactionSkip, CompactionPlan, CompactionOptions } from "./compaction.js";

// ── Transports ──────────────────────────────────────────────────────────────
export { createMemoryMcpServer } from "./transports/sdk.js";
export type { SdkServerOptions, SdkMcpServerConfig } from "./transports/sdk.js";

export { buildMcpServer } from "./transports/mcp-server.js";
export type { McpServerOptions } from "./transports/mcp-server.js";

export { runStdio } from "./transports/stdio.js";

export { createMemoryHttpHandler } from "./transports/http.js";
export type { HttpHandlerOptions } from "./transports/http.js";
export { hashToken, timingSafeEqualStr, buildTenantIndex, bearerToken, RateLimiter } from "./transports/auth.js";

// ── Multi-host installer (wire the stdio server into any MCP-capable agent) ───
export { buildServerSpec, defaultMemoryFile, resolveMemoryFile, MCP_PACKAGE, MCP_BIN, MEMORY_FILE_ENV, RUNTIME_PEERS } from "./install/server-spec.js";
export type { StdioServerSpec, ServerSpecOptions } from "./install/server-spec.js";
export { installMemoryServer } from "./install/install.js";
export type { InstallOptions, InstallResult, InstallStatus, HostSelector, FsLike } from "./install/install.js";
export { HOSTS, SERVER_KEY, findHost } from "./install/hosts.js";
export type { HostAdapter, HostEnv, ConfigFormat } from "./install/hosts.js";
// Claude Code "memory combo" — hooks (SessionStart/PreCompact/UserPromptSubmit/Stop)
// + companion skill that make memory self-driving (contextual recall + autonomous capture).
export { installClaudeCombo, bfmemHookSource, companionSkillMd, claudeComboPaths, pluginHooksConfig, HOOK_EVENTS } from "./install/claude-hooks.js";
export type { ClaudeComboPaths, HookConfig } from "./install/claude-hooks.js";
