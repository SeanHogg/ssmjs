/**
 * ingest — one pipeline for structured and unstructured sources, with incremental
 * sync so a nightly re-run re-embeds only what changed.
 */

export {
    IngestionPipeline,
    InMemoryIngestManifest,
    hashText,
    mapWithConcurrency,
} from './IngestionPipeline.js';
export type { IngestionPipelineOptions } from './IngestionPipeline.js';

export {
    ParserRegistry,
    BUILTIN_PARSERS,
    rowsParser,
    csvParser,
    jsonParser,
    markdownParser,
    htmlParser,
    plainTextParser,
    parseCsv,
    serializeRow,
} from './parsers.js';

export {
    SOURCE_ID_FIELD,
    CHUNK_INDEX_FIELD,
    CONTENT_HASH_FIELD,
    TITLE_FIELD,
    URI_FIELD,
    PARSER_FIELD,
} from './types.js';
export type {
    BatchEmbedder,
    ChunkFingerprint,
    DocumentParser,
    IngestManifest,
    IngestionReport,
    ParsedSection,
    SourceContent,
    SourceDocument,
    StructuredRow,
    SyncStrategy,
} from './types.js';
