#!/usr/bin/env node
/**
 * builderforce-memory-install — wire the memory MCP server into any MCP-capable
 * agent (Claude Code, Claude Desktop, Cursor, Windsurf, VS Code, Cline, Gemini
 * CLI, Codex CLI), not just Claude Code.
 *
 *   npx -y @seanhogg/builderforce-memory-mcp                 # via the package's other bin
 *   builderforce-memory-install                              # auto-detect installed hosts
 *   builderforce-memory-install --host=cursor,windsurf       # explicit hosts
 *   builderforce-memory-install --host=all                   # every known host
 *   builderforce-memory-install --memory-file=/path/mem.json # shared store path
 *   builderforce-memory-install --readonly                   # recall-only
 *   builderforce-memory-install --local=/abs/dist/bin/stdio.js  # dev: local bin
 *
 * The store defaults to ~/.builderforce-memory/memory.json so every agent on
 * the machine shares one memory. Re-running is safe (idempotent + .bak backups).
 */

import os from "node:os";
import path from "node:path";
import nodeFs from "node:fs";
import { HOSTS } from "../install/hosts.js";
import { installMemoryServer, type HostSelector } from "../install/install.js";
import { installClaudeCombo } from "../install/claude-hooks.js";
import { defaultMemoryFile } from "../install/server-spec.js";

function argValue(name: string): string | undefined {
    const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
    return hit ? hit.slice(name.length + 3) : undefined;
}
const hasFlag = (name: string): boolean => process.argv.includes(`--${name}`);

if (hasFlag("help") || hasFlag("h")) {
    process.stdout.write(
        [
            "builderforce-memory-install — register the memory MCP server into MCP hosts.",
            "",
            "  --host=<a,b|all>     hosts to target (default: auto-detect installed)",
            "  --memory-file=<path> JSON snapshot path (default: ~/.builderforce-memory/memory.json)",
            "  --readonly           recall-only (no remember/forget)",
            "  --local=<path>       run a locally-built stdio bin via node (dev)",
            "",
            `Known hosts: ${HOSTS.map((h) => h.id).join(", ")}`,
            "",
        ].join("\n"),
    );
    process.exit(0);
}

const hostArg = argValue("host");
const hosts: HostSelector =
    hostArg === undefined || hostArg === "auto"
        ? "auto"
        : hostArg === "all"
          ? "all"
          : hostArg.split(",").map((s) => s.trim()).filter(Boolean);

const memoryFile = argValue("memory-file") ?? defaultMemoryFile();

let results;
try {
    results = installMemoryServer({
        hosts,
        memoryFile,
        readonly: hasFlag("readonly"),
        localBin: argValue("local"),
    });
} catch (err) {
    process.stderr.write(`✗ ${String((err as Error).message ?? err)}\n`);
    process.exit(1);
}

const icon: Record<string, string> = {
    installed: "✓",
    updated: "✓",
    skipped: "•",
    unsupported: "–",
    error: "✗",
};

let wrote = 0;
for (const r of results) {
    const where = r.path ? `  ${r.path}` : "";
    const why = r.detail ? `  (${r.detail})` : "";
    process.stdout.write(`${icon[r.status] ?? "?"} ${r.label}: ${r.status}${why}${where}\n`);
    if (r.status === "installed" || r.status === "updated") wrote += 1;
}

// ── Claude Code memory combo (hooks + companion skill) ───────────────────────
// Registering the server above gives Claude Code the TOOLS; this adds the
// self-driving behaviour (SessionStart digest, contextual recall, autonomous Stop
// capture). Only for Claude Code, only when present, and skippable via --no-hooks.
const claudeResult = results.find((r) => r.host === "claude-code");
const claudePresent = claudeResult && claudeResult.status !== "unsupported" && claudeResult.status !== "error";
if (claudePresent && !hasFlag("no-hooks")) {
    try {
        const combo = installClaudeCombo({
            fs: nodeFs,
            claudeDir: path.join(os.homedir(), ".claude"),
            memoryFile,
        });
        const what = combo.addedHooks.length ? `added ${combo.addedHooks.join(", ")}` : "already current";
        process.stdout.write(`✓ Claude Code memory combo: hooks ${what}; recall + autonomous capture wired\n`);
    } catch (err) {
        process.stderr.write(`✗ Claude Code hooks: ${String((err as Error).message ?? err)}\n`);
    }
}

if (results.length === 0) {
    process.stdout.write(
        "No MCP hosts detected. Pass --host=all to write configs anyway, or --host=cursor (etc.).\n",
    );
} else {
    process.stdout.write(
        `\nDone — ${wrote} config(s) written. Store: ${memoryFile}\nRestart the agent(s) to connect. Tools: memory_recall · memory_remember · memory_get · memory_recall_by_tag · memory_forget\n`,
    );
}
