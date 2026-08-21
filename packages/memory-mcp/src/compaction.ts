/**
 * Compaction — turning an ABSORBED memory into a terse pointer stub.
 *
 * Once a fact has been folded into an Evermind (or any downstream learner) the
 * full body has done its job: keeping it verbatim in the store means every
 * subsequent recall pays for text the model already carries in its weights.
 * Compaction replaces the body with a one-line `[absorbed→Evermind vN] <first
 * line>` pointer, so the fact stops filling context while it stays findable.
 *
 * The rules live here — not in the tool handler and not in a transport — because
 * BOTH the in-process `memory_compact` tool and any external compactor (the
 * BuilderForce VS Code extension rewrites the snapshot directly) must produce
 * byte-identical stubs; otherwise re-running either one re-stubs the other's work.
 *
 * NOTE: `Builderforce.ai/clients/vscode/src/memorySnapshot.ts` is the extension-side
 * mirror of these three functions. They are deliberately kept in lockstep — the
 * marker below is the shared idempotency contract between the two processes.
 */

/**
 * Marker that opens every compacted stub — also the idempotency guard. An entry
 * whose content already starts with this is skipped, so compacting twice (or
 * from two processes) never double-stubs.
 */
export const STUB_PREFIX = "[absorbed→Evermind";

/** Default cap on the pointer line carried inside a stub. */
export const DEFAULT_STUB_CHARS = 140;

/** True when a body is already a compaction stub. */
export function isStub(content: string): boolean {
    return content.trimStart().startsWith(STUB_PREFIX);
}

/** The first non-empty line of a body, trimmed to `max` chars — the stub's pointer text. */
export function firstLine(content: string, max = DEFAULT_STUB_CHARS): string {
    const line =
        content
            .split(/\r?\n/)
            .map((l) => l.trim())
            .find((l) => l.length > 0) ?? "";
    return line.length > max ? `${line.slice(0, max - 1).trimEnd()}…` : line;
}

/** Build the terse stub that replaces an absorbed entry's body. */
export function memoryStub(content: string, version: number, maxChars = DEFAULT_STUB_CHARS): string {
    return `${STUB_PREFIX} v${version}] ${firstLine(content, maxChars)}`;
}

/** One entry considered for compaction. */
export interface CompactionCandidate {
    key: string;
    content: string;
}

/** A body rewrite the caller should apply. */
export interface CompactionWrite {
    key: string;
    /** The stub that replaces the entry's body. */
    content: string;
    /** Characters recovered by this rewrite. */
    bytesSaved: number;
}

/** Why a candidate was left alone. */
export interface CompactionSkip {
    key: string;
    reason: "not_found" | "already_compacted" | "not_smaller";
}

/** The decided plan: what to write, what to leave, and the total recovered. */
export interface CompactionPlan {
    writes: CompactionWrite[];
    skipped: CompactionSkip[];
    bytesSaved: number;
}

/** Options controlling how a stub body is produced. */
export interface CompactionOptions {
    /** Explicit replacement body. Omit to generate a pointer from each entry's own first line. */
    stub?: string;
    /** Evermind version that absorbed the facts; appears in a generated stub. Default 0. */
    version?: number;
    /** Max characters of a generated pointer line. Default {@link DEFAULT_STUB_CHARS}. */
    maxChars?: number;
}

/**
 * Decide the compaction for a set of candidates. Pure — no I/O — so the same
 * rules are unit-testable and reusable by any caller that can read and write
 * memories. `undefined` content means the key was not found.
 */
export function planCompaction(
    candidates: Array<CompactionCandidate | { key: string; content: undefined }>,
    opts: CompactionOptions = {},
): CompactionPlan {
    const version = opts.version ?? 0;
    const maxChars = Math.max(20, opts.maxChars ?? DEFAULT_STUB_CHARS);
    const writes: CompactionWrite[] = [];
    const skipped: CompactionSkip[] = [];
    let bytesSaved = 0;

    for (const c of candidates) {
        const content = c.content;
        if (typeof content !== "string" || !content.trim()) {
            skipped.push({ key: c.key, reason: "not_found" });
            continue;
        }
        if (isStub(content)) {
            skipped.push({ key: c.key, reason: "already_compacted" });
            continue;
        }
        const stub = opts.stub ?? memoryStub(content, version, maxChars);
        // Never grow an entry — a "compaction" that costs tokens is a regression.
        if (stub.length >= content.length) {
            skipped.push({ key: c.key, reason: "not_smaller" });
            continue;
        }
        const saved = content.length - stub.length;
        writes.push({ key: c.key, content: stub, bytesSaved: saved });
        bytesSaved += saved;
    }

    return { writes, skipped, bytesSaved };
}
