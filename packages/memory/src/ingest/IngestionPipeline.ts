/**
 * ingest/IngestionPipeline.ts — source → parse → chunk → embed → upsert.
 *
 * The one-off script that loads a corpus into a vector database is easy. What
 * makes a pipeline PRODUCTION-grade is everything that happens on the second run:
 *
 *   • **Incremental sync.** Re-ingesting a 200k-document corpus must not re-embed
 *     200k documents because three of them changed. Chunks carry a content hash;
 *     with a manifest, unchanged hashes are skipped and vanished chunks deleted.
 *     Embedding is the cost line, so this is the difference between a pipeline that
 *     runs nightly and one that runs once.
 *   • **Deterministic ids.** A chunk's id is derived from its source id and its
 *     ordinal, so a re-run overwrites rather than duplicating. Random ids are why
 *     corpora silently triple in size.
 *   • **Batching and bounded concurrency.** Embedders and vector databases both
 *     rate-limit; an unbounded `Promise.all` over a corpus is a self-inflicted 429.
 *   • **Per-document failure isolation.** One malformed file fails that document
 *     and is reported, rather than aborting the run.
 *   • **Governance by construction.** Tenant, ACL and sensitivity are written onto
 *     every chunk during ingest, because a retrieval-time ACL that depends on
 *     ingest-time diligence is not an ACL.
 */

import { chunkText, type ChunkOptions } from '../retrieval/chunk.js';
import type { Tracer } from '../telemetry/Tracer.js';
import { estimateTokens } from '../telemetry/pricing.js';
import {
    ACL_FIELD,
    ACL_PUBLIC,
    SENSITIVITY_FIELD,
    TENANT_FIELD,
    type MetadataRecord,
    type VectorRecord,
    type VectorStore,
} from '../vectorstore/types.js';
import { ParserRegistry } from './parsers.js';
import {
    CHUNK_INDEX_FIELD,
    CONTENT_HASH_FIELD,
    PARSER_FIELD,
    SOURCE_ID_FIELD,
    TITLE_FIELD,
    URI_FIELD,
    type BatchEmbedder,
    type ChunkFingerprint,
    type DocumentParser,
    type IngestManifest,
    type IngestionReport,
    type SourceDocument,
    type SyncStrategy,
} from './types.js';

export interface IngestionPipelineOptions {
    store: VectorStore;
    embed: BatchEmbedder;
    /** Chunking parameters. Defaults to 1000 chars / 200 overlap. */
    chunk?: ChunkOptions;
    /** Custom parser set. Defaults to the builtin registry. */
    parsers?: ParserRegistry | readonly DocumentParser[];
    /** Enables `diff` sync — see {@link IngestManifest}. */
    manifest?: IngestManifest;
    /** Default `'diff'` when a manifest is supplied, `'replace'` otherwise. */
    syncStrategy?: SyncStrategy;
    /** Texts per embedder call. Default 64. */
    embedBatchSize?: number;
    /** Records per store upsert. Default 100. */
    upsertBatchSize?: number;
    /** Documents processed concurrently. Default 4. */
    concurrency?: number;
    /** Model label recorded on embedding spans, for cost attribution. */
    embeddingModel?: string;
    tracer?: Tracer;
}

export class IngestionPipeline {
    private readonly _store: VectorStore;
    private readonly _embed: BatchEmbedder;
    private readonly _chunkOpts: ChunkOptions;
    private readonly _parsers: ParserRegistry;
    private readonly _manifest: IngestManifest | undefined;
    private readonly _strategy: SyncStrategy;
    private readonly _embedBatch: number;
    private readonly _upsertBatch: number;
    private readonly _concurrency: number;
    private readonly _embeddingModel: string;
    private readonly _tracer: Tracer | undefined;

    constructor(opts: IngestionPipelineOptions) {
        this._store = opts.store;
        this._embed = opts.embed;
        this._chunkOpts = opts.chunk ?? {};
        this._parsers = opts.parsers instanceof ParserRegistry
            ? opts.parsers
            : new ParserRegistry(opts.parsers);
        this._manifest = opts.manifest;
        this._strategy = opts.syncStrategy ?? (opts.manifest ? 'diff' : 'replace');
        this._embedBatch = Math.max(1, opts.embedBatchSize ?? 64);
        this._upsertBatch = Math.max(1, opts.upsertBatchSize ?? 100);
        this._concurrency = Math.max(1, opts.concurrency ?? 4);
        this._embeddingModel = opts.embeddingModel ?? 'embedding';
        this._tracer = opts.tracer;
    }

    async ingest(sources: readonly SourceDocument[]): Promise<IngestionReport> {
        const started = Date.now();
        const report: IngestionReport = {
            documents: 0, sections: 0, chunks: 0, embedded: 0, upserted: 0,
            skippedUnchanged: 0, deletedStale: 0, failures: [], durationMs: 0,
        };

        const root = this._tracer?.startSpan('ingest.run', {
            kind: 'ingest',
            attributes: { 'ingest.sources': sources.length, 'ingest.strategy': this._strategy },
        });
        if (root) report.traceId = root.traceId;

        await mapWithConcurrency(sources, this._concurrency, async (source) => {
            try {
                const result = await this._ingestOne(source, root);
                report.documents += 1;
                report.sections += result.sections;
                report.chunks += result.chunks;
                report.embedded += result.embedded;
                report.upserted += result.upserted;
                report.skippedUnchanged += result.skippedUnchanged;
                report.deletedStale += result.deletedStale;
            } catch (err) {
                // One bad document must not abort a corpus-wide run; it is reported
                // with its id so the operator can requeue exactly that source.
                report.failures.push({
                    sourceId: source.id,
                    error: err instanceof Error ? err.message : String(err),
                });
            }
        });

        report.durationMs = Date.now() - started;
        root?.setAttributes({
            'ingest.chunks': report.chunks,
            'ingest.embedded': report.embedded,
            'ingest.skipped_unchanged': report.skippedUnchanged,
            'ingest.failures': report.failures.length,
        });
        root?.end(report.failures.length > 0 && report.documents === 0 ? 'error' : 'ok');
        await this._tracer?.flush();
        return report;
    }

    /** Removes every chunk a source produced — a GDPR/right-to-erasure primitive. */
    async forget(sourceId: string): Promise<number> {
        const deleted = await this._store.deleteByFilter({
            op: 'eq', field: SOURCE_ID_FIELD, value: sourceId,
        });
        await this._manifest?.delete(sourceId);
        return deleted;
    }

    private async _ingestOne(
        source: SourceDocument,
        parent?: import('../telemetry/Tracer.js').Span,
    ): Promise<{
        sections: number; chunks: number; embedded: number;
        upserted: number; skippedUnchanged: number; deletedStale: number;
    }> {
        const span = this._tracer?.startSpan('ingest.document', {
            kind: 'ingest',
            ...(parent ? { parent } : {}),
            attributes: { 'ingest.source_id': source.id, 'ingest.tenant': source.tenantId },
        });

        try {
            const parser = this._parsers.resolve(source);
            if (!parser) throw new Error(`No parser accepts source "${source.id}".`);

            const sections = parser.parse(source);
            const base = this._baseMetadata(source, parser.id);

            // Chunk index is global across sections so ids stay stable and unique
            // even when a section is added in the middle of a document.
            const pending: Array<{ record: VectorRecord; hash: string }> = [];
            let index = 0;
            for (const section of sections) {
                for (const chunk of chunkText(section.text, this._chunkOpts)) {
                    const hash = hashText(chunk.text);
                    const metadata: MetadataRecord = {
                        ...base,
                        ...(section.metadata ?? {}),
                        [CHUNK_INDEX_FIELD]: index,
                        [CONTENT_HASH_FIELD]: hash,
                    };
                    pending.push({
                        record: { id: `${source.id}#${index}`, text: chunk.text, metadata },
                        hash,
                    });
                    index += 1;
                }
            }

            const { toWrite, toDelete, skipped } = await this._planSync(source.id, pending);

            let embedded = 0;
            for (let i = 0; i < toWrite.length; i += this._embedBatch) {
                const batch = toWrite.slice(i, i + this._embedBatch);
                const vectors = await this._embedBatchTraced(batch.map((r) => r.text), span);
                batch.forEach((record, j) => {
                    const vector = vectors[j];
                    if (vector) record.vector = vector;
                });
                embedded += batch.length;
            }

            let upserted = 0;
            for (let i = 0; i < toWrite.length; i += this._upsertBatch) {
                const result = await this._store.upsert(toWrite.slice(i, i + this._upsertBatch));
                upserted += result.upserted;
            }

            let deletedStale = 0;
            if (toDelete.length > 0) {
                await this._store.delete(toDelete);
                deletedStale = toDelete.length;
            }

            await this._manifest?.set(
                source.id,
                pending.map(({ record, hash }): ChunkFingerprint => ({ id: record.id, hash })),
            );

            span?.setAttributes({
                'ingest.sections': sections.length,
                'ingest.chunks': pending.length,
                'ingest.embedded': embedded,
                'ingest.skipped_unchanged': skipped,
                'ingest.parser': parser.id,
            });
            span?.end('ok');

            return {
                sections: sections.length,
                chunks: pending.length,
                embedded,
                upserted,
                skippedUnchanged: skipped,
                deletedStale,
            };
        } catch (err) {
            span?.fail(err);
            throw err;
        }
    }

    /**
     * Decides what actually has to be written and deleted.
     *
     * `diff` compares content hashes against the manifest; `replace` clears the
     * source's chunks first (correct without a manifest, at the cost of re-embedding);
     * `append` writes everything and deletes nothing.
     */
    private async _planSync(
        sourceId: string,
        pending: Array<{ record: VectorRecord; hash: string }>,
    ): Promise<{ toWrite: VectorRecord[]; toDelete: string[]; skipped: number }> {
        if (this._strategy === 'diff' && this._manifest) {
            const previous = (await this._manifest.get(sourceId)) ?? [];
            const previousByHash = new Map(previous.map((f) => [f.id, f.hash]));
            const currentIds = new Set(pending.map((p) => p.record.id));

            const toWrite: VectorRecord[] = [];
            let skipped = 0;
            for (const { record, hash } of pending) {
                if (previousByHash.get(record.id) === hash) skipped += 1;
                else toWrite.push(record);
            }
            const toDelete = previous.map((f) => f.id).filter((id) => !currentIds.has(id));
            return { toWrite, toDelete, skipped };
        }

        if (this._strategy === 'replace') {
            await this._store.deleteByFilter({ op: 'eq', field: SOURCE_ID_FIELD, value: sourceId });
        }
        return { toWrite: pending.map((p) => p.record), toDelete: [], skipped: 0 };
    }

    private async _embedBatchTraced(
        texts: string[],
        parent?: import('../telemetry/Tracer.js').Span,
    ): Promise<Float32Array[]> {
        if (!this._tracer) return this._embed(texts);

        const span = this._tracer.startSpan('embed.batch', {
            kind: 'embedding',
            ...(parent ? { parent } : {}),
            attributes: { 'embedding.batch_size': texts.length },
        });
        try {
            const vectors = await this._embed(texts);
            span.recordUsage({
                model: this._embeddingModel,
                inputTokens: texts.reduce((sum, t) => sum + estimateTokens(t), 0),
                outputTokens: 0,
                estimated: true,
            });
            span.end('ok');
            return vectors;
        } catch (err) {
            span.fail(err);
            throw err;
        }
    }

    private _baseMetadata(source: SourceDocument, parserId: string): MetadataRecord {
        const metadata: MetadataRecord = {
            ...(source.metadata ?? {}),
            [TENANT_FIELD]: source.tenantId,
            // Defaulting to tenant-public keeps the ACL array NON-EMPTY, which is
            // what makes `accessFilter` fail closed: a record with no acl matches
            // nothing rather than everything.
            [ACL_FIELD]: source.acl?.length ? source.acl : [ACL_PUBLIC],
            [SOURCE_ID_FIELD]: source.id,
            [PARSER_FIELD]: parserId,
        };
        if (source.sensitivity !== undefined) metadata[SENSITIVITY_FIELD] = source.sensitivity;
        if (source.title) metadata[TITLE_FIELD] = source.title;
        if (source.uri) metadata[URI_FIELD] = source.uri;
        return metadata;
    }
}

/** In-process manifest — the default for a single-process or test deployment. */
export class InMemoryIngestManifest implements IngestManifest {
    private readonly _entries = new Map<string, ChunkFingerprint[]>();

    async get(sourceId: string): Promise<ChunkFingerprint[] | undefined> {
        return this._entries.get(sourceId);
    }

    async set(sourceId: string, fingerprints: ChunkFingerprint[]): Promise<void> {
        this._entries.set(sourceId, fingerprints);
    }

    async delete(sourceId: string): Promise<void> {
        this._entries.delete(sourceId);
    }

    get size(): number { return this._entries.size; }
}

/**
 * FNV-1a, 32-bit, rendered as hex.
 *
 * A content hash here answers "did this chunk change", not "was this chunk
 * tampered with", so a fast non-cryptographic hash is the right tool — and it is
 * synchronous, unlike WebCrypto, which keeps chunking a pure function.
 */
export function hashText(text: string): string {
    let hash = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) {
        hash ^= text.charCodeAt(i);
        hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    return hash.toString(16).padStart(8, '0');
}

/** Bounded-concurrency map — the reason a large corpus does not become a 429. */
export async function mapWithConcurrency<T>(
    items: readonly T[],
    limit: number,
    fn: (item: T, index: number) => Promise<void>,
): Promise<void> {
    let cursor = 0;
    const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
        while (cursor < items.length) {
            const index = cursor++;
            await fn(items[index] as T, index);
        }
    });
    await Promise.all(workers);
}
