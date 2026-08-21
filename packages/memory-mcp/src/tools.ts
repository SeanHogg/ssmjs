/**
 * Framework-agnostic MCP tool definitions over a MemoryBackend.
 *
 * Defined once here, then registered into whichever server framework a
 * transport uses — the Claude Agent SDK's `tool()` (in-process) or the MCP
 * SDK's `registerTool()` (stdio/HTTP). Both accept the same (name, description,
 * zod raw shape, handler→CallToolResult) shape, so the handlers below are the
 * single source of truth.
 *
 * Token-saving is enforced HERE, server-side, regardless of what the model
 * asks for: recall is top-K (capped), each entry's content is truncated, and
 * there is deliberately no "return everything" tool. Moving memory out of the
 * prompt only saves tokens if recall is selective — a tool that dumps the whole
 * store back into context is more expensive than inlining it.
 */

import { z } from "zod";
import type { MemoryBackend, RankedRecall, RecallHit, RecallMethod, RememberInput } from "./backend.js";
import { DEFAULT_STUB_CHARS, planCompaction, type CompactionSkip } from "./compaction.js";

/** The MCP CallToolResult shape both server frameworks expect. */
export interface ToolResult {
    content: Array<{ type: "text"; text: string }>;
    isError?: boolean;
}

/** A framework-neutral tool: maps 1:1 onto Agent-SDK `tool()` and MCP `registerTool()`. */
export interface MemoryTool {
    name: string;
    description: string;
    /** Zod *raw shape* (e.g. `{ query: z.string() }`), not a ZodObject. */
    inputSchema: z.ZodRawShape;
    handler: (args: Record<string, unknown>) => Promise<ToolResult>;
}

export interface MemoryToolsOptions {
    /** Hard cap on entries any recall tool returns to the model. Default 5. */
    maxResults?: number;
    /** Max characters of each entry's content surfaced to the model. Default 500. */
    maxContentChars?: number;
    /** Expose write tools (remember/forget). Default true; forced false if the backend is read-only. */
    writable?: boolean;
    /**
     * BuilderForce gateway base URL (e.g. https://api.builderforce.ai). When set
     * together with {@link gatewayApiKey}, the cost/efficiency tools are exposed.
     */
    gatewayUrl?: string;
    /** A `bfk_*` tenant API key. Required (with {@link gatewayUrl}) for cost tools. */
    gatewayApiKey?: string;
}

const DEFAULT_MAX_RESULTS = 5;
const DEFAULT_MAX_CONTENT = 500;

function clip(s: string, n: number): string {
    return s.length <= n ? s : `${s.slice(0, n)}…`;
}

function ok(text: string): ToolResult {
    return { content: [{ type: "text", text }] };
}

function fail(text: string): ToolResult {
    return { content: [{ type: "text", text }], isError: true };
}

/**
 * How the ordering was produced, in one line the model can read.
 *
 * A semantic ranking that silently degraded to word overlap is indistinguishable
 * from one that did not - which is exactly how "recall got worse" stays invisible.
 * Naming the ranker (the same `embedding` / `lexical` vocabulary the gateway's
 * Evermind recall result reports) makes the degrade legible instead.
 */
const RANKER_LINE: Record<RecallMethod, string> = {
    embedding: "Ranked by: semantic recall (SSM embedding fused with lexical).",
    lexical: "Ranked by: lexical recall (fallback - no embedding model loaded).",
};

function renderHits(hits: RecallHit[], maxChars: number, method?: RecallMethod): string {
    const ranker = method ? `\n${RANKER_LINE[method]}` : "";
    if (hits.length === 0) return `No matching memories.${ranker}`;
    return (
        hits
            .map((h) => {
                const tags = h.tags?.length ? ` tags=[${h.tags.join(", ")}]` : "";
                const score = h.score != null ? ` score=${h.score.toFixed(3)}` : "";
                return `• ${h.key}${score}${tags}\n  ${clip(h.content, maxChars)}`;
            })
            .join("\n") + ranker
    );
}

/**
 * Recall through the backend's BEST available path: `recallRanked` when the backend
 * can name its ranker, plain `recall` otherwise. One helper, so every recall-shaped
 * tool reports the method identically - or stays silent identically.
 */
async function rankedRecall(
    backend: MemoryBackend,
    query: string,
    topK: number,
): Promise<Partial<RankedRecall> & { hits: RecallHit[] }> {
    if (backend.recallRanked) return backend.recallRanked(query, topK);
    return { hits: await backend.recall(query, topK) };
}

const SKIP_REASONS: Record<CompactionSkip["reason"], string> = {
    not_found: "not found",
    already_compacted: "already compacted",
    not_smaller: "already shorter than its stub",
};

/** One-line compaction receipt: what shrank, what it saved, and what was left alone. */
function renderCompaction(compacted: number, bytesSaved: number, skipped: CompactionSkip[]): string {
    const lines = [`Compacted ${compacted} memory(ies), ${bytesSaved} character(s) reclaimed.`];
    for (const s of skipped) lines.push(`• skipped ${s.key} — ${SKIP_REASONS[s.reason]}`);
    return lines.join("\n");
}

/**
 * Builds the memory tool set bound to `backend`. Write tools are included only
 * when `writable` is not false AND the backend actually implements them.
 */
export function buildMemoryTools(backend: MemoryBackend, opts: MemoryToolsOptions = {}): MemoryTool[] {
    const maxResults = Math.max(1, opts.maxResults ?? DEFAULT_MAX_RESULTS);
    const maxContent = Math.max(80, opts.maxContentChars ?? DEFAULT_MAX_CONTENT);
    const writable = opts.writable !== false;

    const tools: MemoryTool[] = [
        {
            name: "memory_recall",
            description:
                "Semantically recall the most relevant stored memories for a query. " +
                "Call this BEFORE answering whenever the task may depend on prior context, user " +
                "preferences, project decisions, or facts learned in earlier sessions — instead of " +
                "assuming that context is already in your prompt. Returns a small ranked set, not the " +
                "whole store.",
            inputSchema: {
                query: z.string().describe("What to look for — a question, topic, or keywords."),
                topK: z
                    .number()
                    .int()
                    .min(1)
                    .max(maxResults)
                    .optional()
                    .describe(`How many memories to return (max ${maxResults}).`),
            },
            handler: async (args) => {
                try {
                    const query = String(args["query"] ?? "");
                    if (!query.trim()) return fail("query is required.");
                    const k = Math.min(maxResults, Number(args["topK"] ?? maxResults));
                    const ranked = await rankedRecall(backend, query, k);
                    return ok(renderHits(ranked.hits.slice(0, maxResults), maxContent, ranked.method));
                } catch (err) {
                    return fail(`recall failed: ${String(err)}`);
                }
            },
        },
        {
            name: "memory_get",
            description:
                "Fetch a single memory by its exact key. Use when you already know the key " +
                "(e.g. one surfaced by memory_recall) and want its full, untruncated value.",
            inputSchema: {
                key: z.string().describe("The exact memory key."),
            },
            handler: async (args) => {
                try {
                    const key = String(args["key"] ?? "");
                    if (!key) return fail("key is required.");
                    const hit = await backend.get(key);
                    return hit ? ok(`• ${hit.key}\n  ${hit.content}`) : ok(`No memory found for key "${key}".`);
                } catch (err) {
                    return fail(`get failed: ${String(err)}`);
                }
            },
        },
        {
            name: "memory_recall_by_tag",
            description:
                "List memories carrying a given tag (e.g. 'user', 'project', 'decision'). " +
                "Use to pull a known category of context rather than searching semantically.",
            inputSchema: {
                tag: z.string().describe("The tag to filter by."),
                limit: z
                    .number()
                    .int()
                    .min(1)
                    .max(maxResults)
                    .optional()
                    .describe(`Max entries to return (max ${maxResults}).`),
            },
            handler: async (args) => {
                try {
                    const tag = String(args["tag"] ?? "");
                    if (!tag) return fail("tag is required.");
                    const limit = Math.min(maxResults, Number(args["limit"] ?? maxResults));
                    const hits = await backend.recallByTag(tag, limit);
                    return ok(renderHits(hits, maxContent));
                } catch (err) {
                    return fail(`recall_by_tag failed: ${String(err)}`);
                }
            },
        },
    ];

    if (writable && backend.remember) {
        tools.push({
            name: "memory_remember",
            description:
                "Persist a fact for future sessions. Call when you learn something durable and reusable: " +
                "a user preference, a project constraint, a decision and its rationale. Keep keys stable " +
                "and descriptive (e.g. 'user.preferred-language') so the same fact overwrites rather than " +
                "duplicating.",
            inputSchema: {
                key: z.string().describe("Stable, descriptive identifier; reusing a key overwrites it."),
                content: z.string().describe("The fact to store."),
                tags: z.array(z.string()).optional().describe("Optional grouping tags."),
                importance: z.number().min(0).max(1).optional().describe("Importance 0–1 (default 0.5)."),
                ttlMs: z.number().int().positive().optional().describe("Optional time-to-live in ms."),
            },
            handler: async (args) => {
                try {
                    const key = String(args["key"] ?? "");
                    const content = String(args["content"] ?? "");
                    if (!key || !content) return fail("key and content are required.");
                    await backend.remember!({
                        key,
                        content,
                        tags: args["tags"] as string[] | undefined,
                        importance: args["importance"] as number | undefined,
                        ttlMs: args["ttlMs"] as number | undefined,
                    });
                    return ok(`Remembered "${key}".`);
                } catch (err) {
                    return fail(`remember failed: ${String(err)}`);
                }
            },
        });
    }

    // memory_compact — the FIRST-CLASS compaction path. It writes through the live
    // store (get → plan → remember), so a durable backend's in-memory state and its
    // on-disk snapshot agree afterwards. Compacting the snapshot file behind the
    // server's back instead is what races: the server still holds the full bodies and
    // re-snapshots over them on its next write. Requires read + write, so it is
    // registered alongside the other write tools.
    if (writable && backend.remember) {
        tools.push({
            name: "memory_compact",
            description:
                "Shrink memories that have already been absorbed elsewhere (folded into a model, " +
                "summarised into a longer-lived note) down to a one-line pointer stub, reclaiming the " +
                "context they were eating. Call this AFTER something durable has learned the facts — " +
                "never as a way to tidy up, because the full body is gone afterwards. Already-compacted " +
                "keys are skipped, so repeating the call is safe.",
            inputSchema: {
                keys: z.array(z.string()).min(1).describe("Exact keys of the memories to compact."),
                stub: z
                    .string()
                    .optional()
                    .describe(
                        "Explicit replacement body applied to every listed key. Omit to generate " +
                            "'[absorbed→Evermind vN] <first line>' from each entry's own first line.",
                    ),
                version: z
                    .number()
                    .int()
                    .min(0)
                    .optional()
                    .describe("Version of the model that absorbed these facts; appears in the generated stub."),
                maxChars: z
                    .number()
                    .int()
                    .min(20)
                    .max(500)
                    .optional()
                    .describe(`Max characters of the generated pointer line (default ${DEFAULT_STUB_CHARS}).`),
            },
            handler: async (args) => {
                try {
                    const raw = args["keys"];
                    const keys = Array.isArray(raw) ? raw.map((k) => String(k)).filter(Boolean) : [];
                    if (keys.length === 0) return fail("keys is required.");

                    // ONE read pass: the hit carries both the body the plan reasons about
                    // and the tags/importance the rewrite must preserve.
                    const hits = new Map<string, RecallHit>();
                    for (const key of keys) {
                        const hit = await backend.get(key);
                        if (hit) hits.set(key, hit);
                    }

                    const plan = planCompaction(keys.map((key) => ({ key, content: hits.get(key)?.content })), {
                        stub: typeof args["stub"] === "string" ? (args["stub"] as string) : undefined,
                        version: args["version"] == null ? undefined : Number(args["version"]),
                        maxChars: args["maxChars"] == null ? undefined : Number(args["maxChars"]),
                    });

                    // Preserve tags/importance — compaction shortens a body, it does not
                    // reclassify the fact.
                    const writes: RememberInput[] = plan.writes.map((w) => {
                        const prev = hits.get(w.key);
                        return { key: w.key, content: w.content, tags: prev?.tags, importance: prev?.importance };
                    });
                    if (writes.length > 0) {
                        if (backend.rememberMany) await backend.rememberMany(writes);
                        else for (const w of writes) await backend.remember!(w);
                    }

                    return ok(renderCompaction(plan.writes.length, plan.bytesSaved, plan.skipped));
                } catch (err) {
                    return fail(`compact failed: ${String(err)}`);
                }
            },
        });
    }

    if (writable && backend.forget) {
        tools.push({
            name: "memory_forget",
            description: "Delete a memory by key. Use to remove a fact that is now wrong or obsolete.",
            inputSchema: {
                key: z.string().describe("The exact memory key to delete."),
            },
            handler: async (args) => {
                try {
                    const key = String(args["key"] ?? "");
                    if (!key) return fail("key is required.");
                    await backend.forget!(key);
                    return ok(`Forgot "${key}".`);
                } catch (err) {
                    return fail(`forget failed: ${String(err)}`);
                }
            },
        });
    }

    // Gateway-backed cost tools — only when BOTH a gateway URL and key are
    // present, so memory-only deployments are unchanged.
    if (opts.gatewayUrl && opts.gatewayApiKey) {
        appendGatewayCostTools(tools, opts.gatewayUrl.replace(/\/+$/, ""), opts.gatewayApiKey);
    }

    return tools;
}

/** Snapshot shape returned by GET /llm/v1/builder-insights. */
interface BuilderInsightsSnapshot {
    windowLabel?: string;
    todayTokens?: number;
    todayCostUsd?: number;
    dailyCapTokens?: number | null;
    pctOfDailyCap?: number | null;
    topModel?: { model: string; tokens: number } | null;
    costPerMergedPrUsd?: number | null;
    tip?: string | null;
}

/** Ranking shape returned by GET /llm/v1/model-analytics. */
interface ModelAnalytics {
    byAction?: Array<{
        actionType?: string;
        label?: string;
        models?: Array<{
            model: string;
            samples?: number;
            avgScore?: number;
            mergeRate?: number;
            avgCostMillicents?: number;
        }>;
    }>;
}

async function gatewayGet<T>(gatewayUrl: string, path: string, apiKey: string): Promise<T> {
    const res = await fetch(`${gatewayUrl}${path}`, {
        headers: { authorization: `Bearer ${apiKey}` },
    });
    if (!res.ok) throw new Error(`gateway ${path} → HTTP ${res.status}`);
    return (await res.json()) as T;
}

function appendGatewayCostTools(tools: MemoryTool[], gatewayUrl: string, apiKey: string): void {
    tools.push({
        name: "token_usage",
        description: "Current token spend + budget for the workspace (today).",
        inputSchema: {},
        handler: async () => {
            try {
                const s = await gatewayGet<BuilderInsightsSnapshot>(gatewayUrl, "/llm/v1/builder-insights", apiKey);
                const lines = [
                    `Token usage (${s.windowLabel ?? "today"}):`,
                    `• Tokens: ${(s.todayTokens ?? 0).toLocaleString()}`,
                    `• Cost: $${(s.todayCostUsd ?? 0).toFixed(2)}`,
                    `• % of daily cap: ${
                        s.pctOfDailyCap == null
                            ? "no cap"
                            : `${s.pctOfDailyCap}%${s.dailyCapTokens ? ` of ${s.dailyCapTokens.toLocaleString()}` : ""}`
                    }`,
                    `• Top model: ${s.topModel ? `${s.topModel.model} (${s.topModel.tokens.toLocaleString()} tok)` : "—"}`,
                ];
                if (s.costPerMergedPrUsd != null) lines.push(`• Cost / merged PR: $${s.costPerMergedPrUsd.toFixed(2)}`);
                if (s.tip) lines.push(`• Tip: ${s.tip}`);
                return ok(lines.join("\n"));
            } catch (err) {
                return fail(`token_usage failed: ${String(err)}`);
            }
        },
    });

    tools.push({
        name: "model_efficiency",
        description: "Which models performed best/cheapest for this workspace's recent work.",
        inputSchema: {},
        handler: async () => {
            try {
                const a = await gatewayGet<ModelAnalytics>(gatewayUrl, "/llm/v1/model-analytics", apiKey);
                const groups = a.byAction ?? [];
                if (groups.length === 0) return ok("No model efficiency data yet for this workspace.");
                const out: string[] = ["Model efficiency by action type (best first):"];
                for (const g of groups) {
                    out.push(`\n${g.label ?? g.actionType ?? "action"}:`);
                    for (const m of (g.models ?? []).slice(0, 3)) {
                        const cost = m.avgCostMillicents != null ? ` $${(m.avgCostMillicents / 100_000).toFixed(4)}/call` : "";
                        const merge = m.mergeRate != null ? ` merge=${Math.round(m.mergeRate * 100)}%` : "";
                        const score = m.avgScore != null ? ` score=${m.avgScore}` : "";
                        out.push(`  • ${m.model}${score}${merge}${cost} (n=${m.samples ?? 0})`);
                    }
                }
                return ok(out.join("\n"));
            } catch (err) {
                return fail(`model_efficiency failed: ${String(err)}`);
            }
        },
    });
}
