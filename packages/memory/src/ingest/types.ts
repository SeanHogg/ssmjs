/**
 * ingest/types.ts — the ingestion vocabulary.
 *
 * The pipeline handles STRUCTURED and UNSTRUCTURED sources through one path, which
 * is the point. A support ticket export (rows, typed columns, foreign keys) and a
 * policy PDF's text both end up as chunks with typed metadata; the difference is
 * which parser ran, not which pipeline. That is what makes a query like
 * "summarise open P1 tickets about billing" answerable — `priority` and `status`
 * survive as *filterable metadata* instead of being flattened into prose the
 * embedding has to recover.
 */

import type { MetadataRecord, MetadataValue } from '../vectorstore/types.js';

/** One row of a structured source (a DB result set, a CSV, a spreadsheet). */
export type StructuredRow = Record<string, MetadataValue>;

export type SourceContent =
    | { kind: 'text'; text: string; mediaType?: string }
    | { kind: 'rows'; rows: StructuredRow[]; rowIdField?: string };

export interface SourceDocument {
    /** Stable id in the ORIGIN system. Re-ingesting the same id replaces its chunks. */
    id: string;
    content: SourceContent;
    /** Owning tenant. Written to every chunk and enforced on every read. */
    tenantId: string;
    /**
     * Principal ids permitted to retrieve this document. Defaults to `['*']`
     * (tenant-public). Written to every chunk so ACL evaluation happens inside the
     * database query, not after the fact.
     */
    acl?: string[];
    /** Classification level; readers declare a ceiling via `AccessScope`. */
    sensitivity?: number;
    /** Arbitrary filterable metadata copied onto every chunk. */
    metadata?: MetadataRecord;
    /** Human-facing title, used in citations. */
    title?: string;
    /** Origin URI, used in citations. */
    uri?: string;
}

/** A parser's output: a span of text plus whatever structure it recovered. */
export interface ParsedSection {
    text: string;
    /** Section-scoped metadata — heading path, row id, JSON pointer, … */
    metadata?: MetadataRecord;
}

/**
 * Parsers are registry entries, so a new format (PDF text layer, DOCX, a bespoke
 * log) is registered rather than branched into the pipeline.
 */
export interface DocumentParser {
    readonly id: string;
    /** True when this parser handles the source. First match in registry order wins. */
    accepts(source: SourceDocument): boolean;
    parse(source: SourceDocument): ParsedSection[];
}

/** Identity of one stored chunk, for incremental sync. */
export interface ChunkFingerprint {
    id: string;
    /** Content hash — an unchanged hash means the chunk need not be re-embedded. */
    hash: string;
}

/**
 * Records which chunks each source produced last time.
 *
 * Supplying one enables `diff` sync: only changed chunks are re-embedded (the
 * expensive step) and only vanished chunks are deleted. Without it the pipeline
 * falls back to `replace`, which is correct but re-embeds everything.
 */
export interface IngestManifest {
    get(sourceId: string): Promise<ChunkFingerprint[] | undefined>;
    set(sourceId: string, fingerprints: ChunkFingerprint[]): Promise<void>;
    delete(sourceId: string): Promise<void>;
}

/** Batch embedder. Batched because per-text calls are the classic N+1 of RAG. */
export type BatchEmbedder = (texts: string[]) => Promise<Float32Array[]>;

export type SyncStrategy =
    /** Re-embed only changed chunks, delete only vanished ones. Needs a manifest. */
    | 'diff'
    /** Delete the source's existing chunks, then write the new set. */
    | 'replace'
    /** Write without deleting anything — for append-only corpora. */
    | 'append';

export interface IngestionReport {
    documents: number;
    sections: number;
    chunks: number;
    /** Chunks sent to the embedder — the cost driver. */
    embedded: number;
    upserted: number;
    /** Chunks whose hash was unchanged and were skipped entirely. */
    skippedUnchanged: number;
    /** Chunks deleted because they no longer exist in the source. */
    deletedStale: number;
    failures: Array<{ sourceId: string; error: string }>;
    durationMs: number;
    /** Trace id covering the whole run, when a tracer is attached. */
    traceId?: string;
}

/** Metadata keys the pipeline reserves on every chunk it writes. */
export const SOURCE_ID_FIELD = 'sourceId';
export const CHUNK_INDEX_FIELD = 'chunkIndex';
export const CONTENT_HASH_FIELD = 'contentHash';
export const TITLE_FIELD = 'title';
export const URI_FIELD = 'uri';
export const PARSER_FIELD = 'parser';
