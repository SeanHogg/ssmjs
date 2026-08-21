/**
 * vectorstore — the VectorStore port, its declarative filter algebra, and the
 * adapters. Tenant/ACL scope is compiled into every read, so multi-tenant RAG
 * fails closed rather than depending on each caller remembering to scope.
 */

export { MemoryVectorStore } from './MemoryVectorStore.js';
export type { MemoryVectorStoreOptions } from './MemoryVectorStore.js';

export { RestVectorStore } from './RestVectorStore.js';
export type { RestVectorStoreOptions } from './RestVectorStore.js';

export {
    getDialect,
    listDialects,
    registerDialect,
    evermindDialect,
    qdrantDialect,
    pineconeDialect,
    vertexAiDialect,
} from './dialects.js';
export type {
    VectorDialect,
    DialectContext,
    HttpRequest,
    TranslatedFilter,
} from './dialects.js';

export {
    accessFilter,
    combineFilters,
    matchesFilter,
    filterFields,
} from './filter.js';

export {
    TENANT_FIELD,
    ACL_FIELD,
    SENSITIVITY_FIELD,
    ACL_PUBLIC,
} from './types.js';
export type {
    AccessScope,
    ComparisonOp,
    MetadataFilter,
    MetadataRecord,
    MetadataValue,
    UpsertResult,
    VectorMatch,
    VectorQuery,
    VectorRecord,
    VectorStore,
} from './types.js';
