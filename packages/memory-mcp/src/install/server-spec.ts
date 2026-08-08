/**
 * The canonical stdio launch spec for the builderforce-memory MCP server.
 *
 * Every host (Claude Code, Cursor, Windsurf, VS Code, Cline, Gemini/Codex CLI,
 * Claude Desktop) registers the SAME server — only the surrounding config
 * file/shape differs. This module is the single source of truth for HOW the
 * server is launched, so the per-host adapters never re-derive the command.
 */

import nodeOs from "node:os";
import nodePath from "node:path";

export interface StdioServerSpec {
    command: string;
    args: string[];
    env?: Record<string, string>;
}

/**
 * The default JSON snapshot path: ONE store per machine, so every MCP host
 * (Claude Code, Cursor, Claude Desktop, …) shares the same memory.
 *
 * Single source of truth for the installer, the stdio bin's fallback, and the
 * Claude Code plugin's hook script — a memory file the writer and the reader
 * disagree about is memory that silently disappears.
 */
export const MEMORY_FILE_ENV = "BUILDERFORCE_MEMORY_FILE";

export function defaultMemoryFile(homedir: string = nodeOs.homedir()): string {
    return nodePath.join(homedir, ".builderforce-memory", "memory.json");
}

/** The snapshot path in effect: explicit env wins, else the shared default. */
export function resolveMemoryFile(
    env: Record<string, string | undefined> = process.env,
    homedir?: string,
): string {
    return env[MEMORY_FILE_ENV] || defaultMemoryFile(homedir);
}

export interface ServerSpecOptions {
    /** Absolute path to the JSON snapshot that makes memory survive restarts. */
    memoryFile?: string;
    /** Register read-only (disables the remember/forget tools). */
    readonly?: boolean;
    /** Platform to target (command wrapping). Defaults to `process.platform`. */
    platform?: NodeJS.Platform;
    /**
     * Run a locally-built stdio bin via `node <path>` instead of the published
     * package via `npx` — for development against a checkout.
     */
    localBin?: string;
}

/** Published package + bin. The bin name matches the unscoped package name. */
export const MCP_PACKAGE = "@seanhogg/builderforce-memory-mcp";
export const MCP_BIN = "builderforce-memory-mcp";

/**
 * Runtime peers the stdio bin loads on demand. They are OPTIONAL peers of the
 * MCP package (so a custom-backend / HTTP-thin-client consumer needn't install
 * them), which means `npx` will NOT auto-pull them — the stdio launch lists
 * them explicitly. `-engine` arrives as `-memory`'s required peer; listed too
 * for determinism.
 */
export const RUNTIME_PEERS = [
    "@seanhogg/builderforce-memory",
    "@seanhogg/builderforce-memory-engine",
    "fake-indexeddb",
];

function specEnv(opts: ServerSpecOptions): Record<string, string> | undefined {
    const env: Record<string, string> = {};
    if (opts.memoryFile) env["BUILDERFORCE_MEMORY_FILE"] = opts.memoryFile;
    if (opts.readonly) env["BUILDERFORCE_MEMORY_READONLY"] = "1";
    return Object.keys(env).length > 0 ? env : undefined;
}

/**
 * Build the stdio launch spec. On Windows `npx` is a `.cmd` shim that an MCP
 * launcher must invoke through `cmd`; `node` + a local path needs no shell.
 */
export function buildServerSpec(opts: ServerSpecOptions = {}): StdioServerSpec {
    const platform = opts.platform ?? process.platform;
    const env = specEnv(opts);

    if (opts.localBin) {
        return { command: "node", args: [opts.localBin], ...(env ? { env } : {}) };
    }

    const npxArgs = ["-y", "-p", MCP_PACKAGE, ...RUNTIME_PEERS.flatMap((p) => ["-p", p]), MCP_BIN];

    if (platform === "win32") {
        return { command: "cmd", args: ["/c", "npx", ...npxArgs], ...(env ? { env } : {}) };
    }
    return { command: "npx", args: npxArgs, ...(env ? { env } : {}) };
}
