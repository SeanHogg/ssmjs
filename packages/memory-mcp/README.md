# @seanhogg/builderforce-memory-mcp

Expose [`@seanhogg/builderforce-memory`](../memory) to any MCP client. One token-saving
tool core over a pluggable `MemoryBackend`, three transports:

| Transport | Factory | Consumed as | Use when |
|---|---|---|---|
| **In-process (Claude Agent SDK)** | `createMemoryMcpServer(backend)` | the returned `type:"sdk"` object | the consuming product is TS and runs `@anthropic-ai/claude-agent-sdk` in-process — lowest latency |
| **stdio** | `runStdio(backend)` / `npx @seanhogg/builderforce-memory-mcp` | `{ type:"stdio", command, args }` | any language / separate process; decouples the SSM+IndexedDB deps from the consumer |
| **HTTP (Streamable)** | `createMemoryHttpHandler(backend, { authToken })` | `{ type:"http", url, headers }` | multi-tenant / networked (builderforce.ai hosting, remote agents) |

> **Technical report & peer review.** This transport is reviewed in the Evermind technical report's adversarial [`PEER-REVIEW.md`](../../publication/evermind/PEER-REVIEW.md). Note before exposing the HTTP handler publicly: the `authToken` check is a **non-constant-time** string compare (timing channel) and the handler shares **one `backend` with no per-tenant isolation** — a single token reads every tenant's facts. Despite the "multi-tenant" note above, real isolation (token→namespace mapping, constant-time compare, rate limiting) is **not yet implemented**. Tracked as `EVM-3` in the Builderforce.ai Consolidated Gap Register. **Resolved in v2026.6.34:** the HTTP handler now uses a constant-time token compare and per-tenant namespace isolation (see the resolution addendum in [`PEER-REVIEW.md`](../../publication/evermind/PEER-REVIEW.md)).

## Why this saves tokens

The point of an external memory store is **fetch-on-demand** instead of pinning
your whole memory file into every prompt. That only pays off if recall is
*selective*, so the caps are enforced server-side in [`tools.ts`](src/tools.ts):

- `memory_recall` returns a **ranked top-K** (default 5, hard-capped), never the store.
- Each entry's content is **truncated** (default 500 chars) before it hits context.
- There is **no "return everything" tool** — a dump-all is more expensive than inlining.
- `memory_compact` shrinks a fact that something durable has already ABSORBED down to a
  one-line pointer stub, so it stops being recalled in full. It writes **through the live
  store**, which is what keeps memory and the on-disk snapshot in agreement — see below.

Tool descriptions are prescriptive ("call this *before* answering when…") because
recent Claude models reach for tools more conservatively; the trigger condition in
the description is what drives should-call rate. Pair this with prompt caching (keep
the tool set identical across tenants so the cached prefix survives) and context
editing (prune stale recalled facts) for the full win.

## The seam

Everything is written against [`MemoryBackend`](src/backend.ts):

```ts
interface MemoryBackend {
  recall(query: string, topK: number): Promise<RecallHit[]>;  // semantic
  get(key: string): Promise<RecallHit | undefined>;           // exact
  recallByTag(tag: string, limit: number): Promise<RecallHit[]>;
  remember?(input: RememberInput): Promise<void>;             // optional → read-only backends omit
  forget?(key: string): Promise<void>;
}
```

Ship the local `MemoryStoreBackend` (IndexedDB via `@seanhogg/builderforce-memory`) today;
drop in a networked builderforce.ai adapter later with **zero** changes to tools
or transports.

## Quick start — in-process with the Claude Agent SDK

```ts
import { query } from "@anthropic-ai/claude-agent-sdk";
import { createMemoryMcpServer, createLocalMemoryStoreBackend } from "@seanhogg/builderforce-memory-mcp";

const backend = await createLocalMemoryStoreBackend();          // IndexedDB (fake-indexeddb in Node)
const memory  = await createMemoryMcpServer(backend);           // type:"sdk" config

for await (const msg of query({
  prompt: "What language does this user prefer? Check memory first.",
  options: {
    mcpServers: { builderforce_memory: memory },
    allowedTools: ["mcp__builderforce_memory__*"],             // auto-approve, no prompts
  },
})) {
  if (msg.type === "result" && msg.subtype === "success") console.log(msg.result);
}
```

### The snapshot is shared — the server watches it

With `persistFile` set, the store lives in memory and the JSON snapshot is the source of
truth across process lifetimes. Other processes edit that file too (the BuilderForce VS
Code extension rewrites absorbed entries to stubs in place), so before every operation the
backend stats the file and **re-hydrates from disk when it changed underneath** — otherwise
this server, still holding the pre-edit bodies, would re-snapshot over that work on its next
write. Its own writes are stamped (mtime + size + content hash) the moment they land, so the
steady-state cost is one `stat` and there is no write-loop.

### SSM-embedding recall — including on a headless host (no GPU)

With no model configured the local backend ranks recall **lexically** (BM25). Point it
at an Evermind checkpoint and the ordering becomes the SSM embedding's cosine ranking
**fused with** the lexical one (Reciprocal Rank Fusion — dense ⊕ sparse, so exact
tokens like identifiers and error codes are still caught). Every `memory_recall`
result names the ranker it used, so a degrade to lexical is never silent:

```
Ranked by: semantic recall (SSM embedding fused with lexical).
Ranked by: lexical recall (fallback - no embedding model loaded).
```

**The stdio/HTTP servers do not need a GPU for this.** `EvermindLM` is a pure-CPU
model and `EvermindLM.embed()` mean-pools + L2-normalises the same hidden state the
GPU `HybridMambaModel.embed()` does, to the same contract. Configure it with:

| Env | Meaning |
|-----|---------|
| `BUILDERFORCE_MEMORY_MODEL` | Path to a `.evermind` package holding an `evermind-lm`. |
| `BUILDERFORCE_MEMORY_TOKENIZER` | Tokenizer JSON for that checkpoint. Default `<model>.tokenizer.json`. |
| `BUILDERFORCE_MEMORY_VECTORS` | Vector-cache path. Defaults beside the memory snapshot. |

Vectors are cached **across process lifetimes** (stamped with the model's
fingerprint, so an adapted checkpoint invalidates them), because a per-session
subprocess that re-embedded the whole store on its first recall would cost more than
the lexical ranking it replaces.

In-process hosts pass an embedder directly — reuse an already-loaded GPU Evermind
runtime rather than standing up a second model:

```ts
const backend = await createLocalMemoryStoreBackend({ runtime: ssmMemoryService.runtime });
```

Either way, recall quality improves automatically as that model is adapted/distilled.

## One-shot install into any MCP host (not just Claude)

```bash
# Auto-detect installed agents and register the memory server into each:
npx -y -p @seanhogg/builderforce-memory-mcp builderforce-memory-install

npx -y -p @seanhogg/builderforce-memory-mcp builderforce-memory-install --host=all
npx -y -p @seanhogg/builderforce-memory-mcp builderforce-memory-install --host=cursor,windsurf
```

Writes the stdio server entry into each host's own config, idempotently and with
`.bak` backups. Supported hosts: **Claude Code, Claude Desktop, Cursor, Windsurf,
VS Code, Cline, Gemini CLI, Codex CLI**. Every agent points at one shared store
(`~/.builderforce-memory/memory.json` by default, override with `--memory-file`),
so a fact remembered in one agent is recalled in another. `--readonly` drops the
write tools; `--local=<dist/bin/stdio.js>` runs a checkout for development.

The same launch spec is exported for programmatic use:
`buildServerSpec()` / `installMemoryServer()` (see `src/install/`).

### Claude Code: the self-driving "memory combo"

Registering the server gives Claude Code the memory *tools*. For Claude Code the
installer ALSO wires hooks so memory is **self-driving instead of advisory** (skip
with `--no-hooks`):

| Hook | Behaviour |
|------|-----------|
| `SessionStart` | Inject a digest of the top durable memories. |
| `PreCompact` | Flush durable learnings before the window is summarised. |
| `UserPromptSubmit` | **Contextual recall** — score memories against the prompt and inject the relevant ones, so a stored fact is retrieved at the *decision point*, not only at session start. |
| `Stop` | **Autonomous capture** — if the user's last message was a durable instruction/correction and it wasn't consolidated this turn, BLOCK the stop with a directive to `memory_remember`. Capture flows through the MCP write path (never racing the snapshot); a `stop_hook_active` guard means at most one nudge per turn. |

All four point at one generated, dependency-free hook script
(`~/.claude/builderforce-memory/bfmem-hook.mjs`, four modes) plus a companion
skill. Idempotent + `.bak`-safe. Source of truth: `src/install/claude-hooks.ts`
(`installClaudeCombo` / `bfmemHookSource`).

## Quick start — stdio (any process / language)

```bash
npx -y @seanhogg/builderforce-memory-mcp        # serves the local MemoryStore over stdio
```

```ts
mcpServers: {
  builderforce_memory: { type: "stdio", command: "npx", args: ["-y", "@seanhogg/builderforce-memory-mcp"] },
}
```

Env: `BUILDERFORCE_MEMORY_DB` (db name), `BUILDERFORCE_MEMORY_READONLY=1` (drop write tools).

## Quick start — HTTP (multi-tenant)

```bash
BUILDERFORCE_MEMORY_TOKEN=secret PORT=8787 npx @seanhogg/builderforce-memory-mcp-http
```

```ts
mcpServers: {
  builderforce_memory: {
    type: "http",
    url: "http://localhost:8787",
    headers: { Authorization: "Bearer secret" },
  },
}
```

Or mount `createMemoryHttpHandler(backend, { authToken })` on your own Node/Express
server against any backend (including a future remote one).

## Custom / remote backend

Implement `MemoryBackend` and hand it to any transport:

```ts
import { createMemoryMcpServer, type MemoryBackend } from "@seanhogg/builderforce-memory-mcp";

const remote: MemoryBackend = {
  recall:      (q, k)   => bfClient.search(q, k),
  get:         (key)    => bfClient.get(key),
  recallByTag: (t, lim) => bfClient.byTag(t, lim),
  remember:    (input)  => bfClient.put(input),
  forget:      (key)    => bfClient.delete(key),
};
const server = await createMemoryMcpServer(remote);
```

## Dependencies

`@modelcontextprotocol/sdk` + `zod` are hard deps. `@anthropic-ai/claude-agent-sdk`
(SDK transport), `@seanhogg/builderforce-memory` + `fake-indexeddb` (local backend) are
**optional peers**, imported indirectly — install only what your transport/backend
needs.
