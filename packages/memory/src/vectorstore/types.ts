/**
 * vectorstore/types.ts — the VectorStore port.
 *
 * Enterprise deployments do not get to choose the vector database; the customer
 * already has one (pgvector in their Cloud SQL, Vertex AI Vector Search because
 * they are a GCP shop, Qdrant because platform standardised on it). So the store
 * is a PORT, and every adapter is infrastructure behind it. Retrieval, ingestion
 * and evaluation depend on this interface and never on a vendor.
 *
 * Two things make it enterprise-grade rather than a toy vector index:
 *
 *   1. **Filters are DATA, not predicates.** A `(record) => boolean` cannot be
 *      pushed down to a remote database — it forces fetch-everything-then-filter,
 *      which is both slow and a data-leak: rows the caller may not see are read
 *      into the process before being dropped. {@link MetadataFilter} is a small
 *      declarative algebra each adapter translates into its own query language.
 *
 *   2. **Access scope is first-class.** Multi-tenant RAG fails closed only if the
 *      tenant predicate is part of the query the database runs. {@link AccessScope}
 *      compiles to a filter that is AND-ed into every read, so "forgot to scope the
 *      query" is not a reachable state.
 */

export type MetadataValue = string | number | boolean | null | string[];

export type MetadataRecord = Record<string, MetadataValue>;

/** A stored unit of retrievable knowledge. */
export interface VectorRecord {
    id: string;
    /** The chunk text — returned with matches so a retriever can rerank on it. */
    text: string;
    /** Embedding. Optional so a store can embed server-side. */
    vector?: Float32Array;
    metadata: MetadataRecord;
}

export interface VectorMatch {
    id: string;
    text: string;
    /** Similarity in [0,1] — cosine for every adapter here. */
    score: number;
    metadata: MetadataRecord;
    vector?: Float32Array;
}

// ── Filter algebra ────────────────────────────────────────────────────────────

export type ComparisonOp = 'eq' | 'ne' | 'gt' | 'gte' | 'lt' | 'lte';

export type MetadataFilter =
    | { op: ComparisonOp; field: string; value: MetadataValue }
    /** Field equals any of `values` (scalar field). */
    | { op: 'in' | 'nin'; field: string; values: MetadataValue[] }
    /**
     * Field (an array, or a scalar treated as a one-element array) intersects
     * `values`. This is the ACL primitive — every mainstream vector database can
     * express "array field matches any of", which is why ACLs are modelled as an
     * array rather than as a bitmask or a nested object.
     */
    | { op: 'anyOf'; field: string; values: MetadataValue[] }
    /** Case-insensitive substring match on a string field. */
    | { op: 'contains'; field: string; value: string }
    | { op: 'exists'; field: string }
    // `and` and `or` are separate members rather than one `op: 'and' | 'or'`
    // variant so that a discriminant check narrows them away cleanly; a union-typed
    // discriminant is not removable by control flow, which forces casts on every
    // dialect that walks the tree.
    | { op: 'and'; filters: MetadataFilter[] }
    | { op: 'or'; filters: MetadataFilter[] }
    | { op: 'not'; filter: MetadataFilter };

/**
 * The non-composite half of the algebra — every variant that names a `field`.
 * Dialect translation works leaf-by-leaf, so having the type say so removes a
 * cast from every adapter.
 */
export type LeafFilter = Extract<MetadataFilter, { field: string }>;

// ── Access scope ──────────────────────────────────────────────────────────────

/** Metadata keys the access model reserves. Ingestion always populates them. */
export const TENANT_FIELD = 'tenantId';
export const ACL_FIELD = 'acl';
export const SENSITIVITY_FIELD = 'sensitivity';
/** ACL entry meaning "any principal within the tenant". */
export const ACL_PUBLIC = '*';

export interface AccessScope {
    /** Hard tenant boundary. Every read is AND-ed with an equality on this. */
    tenantId: string;
    /**
     * Group / role / user ids the caller holds. A record is visible when its `acl`
     * array intersects these, or contains {@link ACL_PUBLIC}. Omit for
     * tenant-wide access to public records only.
     */
    principals?: string[];
    /**
     * Highest classification level the caller may read (records carry a numeric
     * `sensitivity`). Omit for no ceiling.
     */
    maxSensitivity?: number;
}

// ── The port ──────────────────────────────────────────────────────────────────

export interface VectorQuery {
    /** Dense query embedding. Required unless the adapter embeds server-side. */
    vector?: Float32Array;
    /** Raw query text — used by `keywordSearch`, and by server-side embedders. */
    text?: string;
    /** Results to return. Default 10. */
    topK?: number;
    /** Declarative metadata predicate, pushed down to the store. */
    filter?: MetadataFilter;
    /** Tenant/ACL scope, AND-ed with `filter`. */
    scope?: AccessScope;
    /** Drop matches scoring below this. Default 0. */
    minScore?: number;
    /** Return stored vectors on matches (costs bandwidth). Default false. */
    includeVectors?: boolean;
}

export interface UpsertResult {
    upserted: number;
    /** Ids the store considered unchanged and skipped, when it can tell. */
    skipped: number;
}

/**
 * The port. `keywordSearch` is optional because not every vector database has a
 * lexical index; retrievers degrade to dense-only and SAY SO on the span rather
 * than silently returning worse results.
 */
export interface VectorStore {
    /** Adapter name, stamped on telemetry spans. */
    readonly name: string;

    upsert(records: readonly VectorRecord[]): Promise<UpsertResult>;
    query(query: VectorQuery): Promise<VectorMatch[]>;
    delete(ids: readonly string[]): Promise<number>;
    /** Bulk delete by predicate — how an incremental sync removes a stale source. */
    deleteByFilter(filter: MetadataFilter, scope?: AccessScope): Promise<number>;
    fetch(ids: readonly string[]): Promise<VectorRecord[]>;
    count(filter?: MetadataFilter, scope?: AccessScope): Promise<number>;

    /** Lexical/BM25 search, when the adapter supports one. */
    keywordSearch?(query: VectorQuery): Promise<VectorMatch[]>;
}
