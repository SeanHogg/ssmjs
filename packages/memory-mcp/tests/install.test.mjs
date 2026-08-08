// Unit tests for the multi-host installer. Uses an in-memory FsLike + a fake
// HostEnv so nothing touches a real machine. Run with `node --test` against dist.
import { test } from "node:test";
import assert from "node:assert/strict";
import {
    installMemoryServer, HOSTS, installClaudeCombo, bfmemHookSource,
    pluginHooksConfig, HOOK_EVENTS, defaultMemoryFile, resolveMemoryFile, MEMORY_FILE_ENV,
} from "../dist/index.js";

/**
 * Minimal in-memory filesystem implementing the installer's FsLike seam.
 * Separator-agnostic: all keys are normalised to "/" so the test reads the same
 * regardless of whether node:path emitted "\" (Windows) or "/" (POSIX).
 */
function memFs(seed = {}) {
    const N = (p) => p.replace(/\\/g, "/");
    const files = new Map(Object.entries(seed).map(([k, v]) => [N(k), v]));
    const dirs = new Set();
    const dirname = (p) => N(p).replace(/\/[^/]*$/, "");
    const seedDirs = (p) => { let d = dirname(p); while (d && !dirs.has(d)) { dirs.add(d); d = dirname(d); } };
    for (const p of files.keys()) seedDirs(p);
    return {
        files,
        dirs,
        get: (p) => files.get(N(p)),
        existsSync: (p) => files.has(N(p)) || dirs.has(N(p)),
        readFileSync: (p) => {
            if (!files.has(N(p))) throw new Error("ENOENT " + p);
            return files.get(N(p));
        },
        writeFileSync: (p, data) => { files.set(N(p), data); },
        mkdirSync: (p) => seedDirs(N(p) + "/_"),
        copyFileSync: (src, dest) => { files.set(N(dest), files.get(N(src))); },
    };
}

const linuxEnv = { homedir: "/home/u", platform: "linux", env: { XDG_CONFIG_HOME: "/home/u/.config" } };

test("--host=all writes every known host with the right config shape", () => {
    const fs = memFs();
    const results = installMemoryServer({
        hosts: "all",
        memoryFile: "/home/u/.builderforce-memory/memory.json",
        fs,
        hostEnv: linuxEnv,
    });
    assert.equal(results.length, HOSTS.length);
    assert.ok(results.every((r) => r.status === "installed"), JSON.stringify(results));

    // Cursor: json with mcpServers, no `type`, npx on linux.
    const cursor = JSON.parse(fs.get("/home/u/.cursor/mcp.json"));
    const entry = cursor.mcpServers["builderforce-memory"];
    assert.equal(entry.command, "npx");
    assert.equal(entry.type, undefined);
    assert.ok(entry.args.includes("@seanhogg/builderforce-memory-mcp"));
    assert.equal(entry.env.BUILDERFORCE_MEMORY_FILE, "/home/u/.builderforce-memory/memory.json");

    // VS Code: `servers` key + explicit type:"stdio".
    const vscode = JSON.parse(fs.get("/home/u/.config/Code/User/mcp.json"));
    assert.equal(vscode.servers["builderforce-memory"].type, "stdio");

    // Claude Code: mcpServers + type.
    const cc = JSON.parse(fs.get("/home/u/.claude.json"));
    assert.equal(cc.mcpServers["builderforce-memory"].type, "stdio");

    // Codex: TOML table.
    const toml = fs.get("/home/u/.codex/config.toml");
    assert.ok(toml.includes("[mcp_servers.builderforce-memory]"));
    assert.ok(toml.includes("BUILDERFORCE_MEMORY_FILE"));
});

test("re-running is idempotent (skipped, no .bak churn beyond first)", () => {
    const fs = memFs();
    const opts = { hosts: ["cursor"], memoryFile: "/home/u/m.json", fs, hostEnv: linuxEnv };
    const first = installMemoryServer(opts);
    assert.equal(first[0].status, "installed");
    const second = installMemoryServer(opts);
    assert.equal(second[0].status, "skipped");
});

test("changing the memory file updates the existing entry", () => {
    const fs = memFs();
    installMemoryServer({ hosts: ["cursor"], memoryFile: "/a.json", fs, hostEnv: linuxEnv });
    const r = installMemoryServer({ hosts: ["cursor"], memoryFile: "/b.json", fs, hostEnv: linuxEnv });
    assert.equal(r[0].status, "updated");
    assert.ok(fs.get("/home/u/.cursor/mcp.json").includes("/b.json"));
});

test("auto detects only hosts whose config dir/file already exists", () => {
    const fs = memFs({ "/home/u/.cursor/mcp.json": "{}" });
    const results = installMemoryServer({ hosts: "auto", memoryFile: "/m.json", fs, hostEnv: linuxEnv });
    const ids = results.map((r) => r.host);
    assert.ok(ids.includes("cursor"));
    assert.ok(!ids.includes("windsurf")); // ~/.codeium/windsurf absent → not detected
});

test("preserves existing servers in a host config", () => {
    const fs = memFs({
        "/home/u/.cursor/mcp.json": JSON.stringify({ mcpServers: { other: { command: "x", args: [] } } }),
    });
    installMemoryServer({ hosts: ["cursor"], memoryFile: "/m.json", fs, hostEnv: linuxEnv });
    const cfg = JSON.parse(fs.get("/home/u/.cursor/mcp.json"));
    assert.ok(cfg.mcpServers.other, "existing server must survive");
    assert.ok(cfg.mcpServers["builderforce-memory"], "new server added");
});

test("unknown host id throws", () => {
    assert.throws(() => installMemoryServer({ hosts: ["nope"], fs: memFs(), hostEnv: linuxEnv }));
});

test("claude combo: writes hook + skill and registers all four hooks (idempotent)", () => {
    const fs = memFs();
    const r = installClaudeCombo({ fs, claudeDir: "/home/u/.claude", memoryFile: "/home/u/.builderforce-memory/memory.json" });
    assert.deepEqual(r.addedHooks, ["SessionStart", "PreCompact", "UserPromptSubmit", "Stop"]);

    // Hook script written with all four modes baked in.
    const hook = fs.get("/home/u/.claude/builderforce-memory/bfmem-hook.mjs");
    for (const mode of ["--session", "--precompact", "--recall", "--capture"]) assert.ok(hook.includes(mode), `hook missing ${mode}`);
    // Companion skill written.
    assert.ok(fs.get("/home/u/.claude/skills/builderforce-memory/SKILL.md").includes("builderforce-memory"));

    // settings.json registers each event pointing at the hook script.
    const settings = JSON.parse(fs.get("/home/u/.claude/settings.json"));
    for (const ev of ["SessionStart", "PreCompact", "UserPromptSubmit", "Stop"]) {
        const cmd = settings.hooks[ev][0].hooks[0].command;
        assert.ok(cmd.includes("bfmem-hook.mjs"), `${ev} not wired`);
    }
    assert.ok(settings.hooks.UserPromptSubmit[0].hooks[0].command.includes("--recall"));
    assert.ok(settings.hooks.Stop[0].hooks[0].command.includes("--capture"));

    // Re-run: idempotent — no duplicate hook groups, nothing newly added.
    const again = installClaudeCombo({ fs, claudeDir: "/home/u/.claude", memoryFile: "/home/u/.builderforce-memory/memory.json" });
    assert.deepEqual(again.addedHooks, []);
    assert.equal(JSON.parse(fs.get("/home/u/.claude/settings.json")).hooks.Stop.length, 1);
});

test("claude combo: collapses duplicate hooks that differ only by path separator", () => {
    // Simulates the real-world drift where an earlier install wrote the command
    // with forward slashes and a later one used backslashes, so the naive
    // string-equality dedup missed it and BOTH fired every SessionStart —
    // doubling the memory injected into every session.
    const fwd = 'node "/home/u/.claude/builderforce-memory/bfmem-hook.mjs" --session';
    const back = 'node "\\home\\u\\.claude\\builderforce-memory\\bfmem-hook.mjs" --session';
    const fs = memFs({
        "/home/u/.claude/settings.json": JSON.stringify({
            hooks: {
                SessionStart: [
                    { hooks: [{ type: "command", command: fwd }] },
                    { hooks: [{ type: "command", command: back }] },
                ],
            },
        }),
    });
    installClaudeCombo({ fs, claudeDir: "/home/u/.claude", memoryFile: "/home/u/.builderforce-memory/memory.json" });
    const ss = JSON.parse(fs.get("/home/u/.claude/settings.json"));
    const sessionHooks = ss.hooks.SessionStart.filter((g) =>
        (g.hooks || []).some((h) => (h.command || "").includes("bfmem-hook.mjs") && (h.command || "").includes("--session")),
    );
    assert.equal(sessionHooks.length, 1, "duplicate SessionStart hooks must collapse to one");
});

test("claude combo: preserves pre-existing unrelated hooks", () => {
    const fs = memFs({
        "/home/u/.claude/settings.json": JSON.stringify({ hooks: { UserPromptSubmit: [{ hooks: [{ type: "command", command: "echo hi" }] }] } }),
    });
    installClaudeCombo({ fs, claudeDir: "/home/u/.claude", memoryFile: "/m.json" });
    const ups = JSON.parse(fs.get("/home/u/.claude/settings.json")).hooks.UserPromptSubmit;
    assert.ok(ups.some((g) => g.hooks[0].command === "echo hi"), "existing hook must survive");
    assert.ok(ups.some((g) => g.hooks[0].command.includes("--recall")), "recall hook added alongside");
});

test("bfmemHookSource bakes the memory file path in", () => {
    assert.ok(bfmemHookSource("/x/mem.json").includes('"/x/mem.json"'));
});

test("bfmemHookSource(null) resolves the snapshot at run time, matching defaultMemoryFile", () => {
    const src = bfmemHookSource(null);
    // A plugin ships one file to every machine, so no path may be baked in.
    assert.ok(!src.includes("/x/mem.json"));
    assert.ok(src.includes(`process.env["${MEMORY_FILE_ENV}"]`), "must honour the env override");
    // The inlined fallback and the shared helper must name the SAME file — a
    // reader and a writer that disagree is memory that silently disappears.
    const tail = defaultMemoryFile("/home/u").replace(/\\/g, "/").split("/home/u/")[1];
    assert.ok(src.includes(`path.join(os.homedir(), ${tail.split("/").map((s) => JSON.stringify(s)).join(", ")})`),
        "hook fallback drifted from defaultMemoryFile()");
});

test("resolveMemoryFile prefers the env override and otherwise persists by default", () => {
    assert.equal(resolveMemoryFile({ [MEMORY_FILE_ENV]: "/explicit.json" }), "/explicit.json");
    assert.equal(resolveMemoryFile({}, "/home/u"), defaultMemoryFile("/home/u"));
});

test("pluginHooksConfig covers every combo event and points at the plugin root", () => {
    const config = pluginHooksConfig();
    assert.deepEqual(Object.keys(config).sort(), HOOK_EVENTS.map((h) => h.event).sort());
    for (const { event, mode } of HOOK_EVENTS) {
        const command = config[event][0].hooks[0].command;
        assert.ok(command.includes("${CLAUDE_PLUGIN_ROOT}"), `${event} must resolve via the plugin root`);
        assert.ok(command.endsWith(mode), `${event} must run ${mode}`);
    }
});

test("windows wraps npx through cmd", () => {
    const fs = memFs();
    installMemoryServer({
        hosts: ["cursor"],
        memoryFile: "C:/m.json",
        fs,
        hostEnv: { homedir: "C:/Users/u", platform: "win32", env: { APPDATA: "C:/Users/u/AppData/Roaming" } },
    });
    const entry = JSON.parse(fs.get("C:/Users/u/.cursor/mcp.json")).mcpServers["builderforce-memory"];
    assert.equal(entry.command, "cmd");
    assert.equal(entry.args[0], "/c");
    assert.equal(entry.args[1], "npx");
});
