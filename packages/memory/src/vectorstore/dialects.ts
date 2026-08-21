/**
 * vectorstore/dialects.ts — vendor wire shapes, as registry DATA.
 *
 * A dialect is everything that differs between vector databases: the URL shape,
 * the request bodies, the response parsing, and how much of the filter algebra the
 * vendor's query language can express. Supporting a new database is REGISTERING a
 * dialect, never editing a branch in the store — which is the whole reason the
 * adapter is one class over a registry rather than one class per vendor.
 *
 * `translateFilter` returns a `residual`: the part of the filter the vendor cannot
 * express. The store applies the residual locally AFTER the server has done the
 * part it could push down, and over-fetches to compensate. That is the honest
 * design — the alternative is silently dropping a clause, which on an ACL clause
 * is a cross-tenant leak.
 *
 * ⚠️ The three vendor dialects here are validated by unit tests that assert the
 * emitted request bodies and parse recorded response shapes; they have NOT been
 * run against live services in this repository (no credentials). The `evermind`
 * dialect is the canonical contract this package defines and owns.
 */

import type {
    LeafFilter,
    MetadataFilter,
    MetadataRecord,
    MetadataValue,
    UpsertResult,
    VectorMatch,
    VectorRecord,
} from './types.js';

export interface HttpRequest {
    /** Appended to the adapter's `baseUrl`. */
    path: string;
    method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
    body?: unknown;
}

export interface DialectContext {
    /** Collection / index / namespace name. */
    collection: string;
    /** Extra per-deployment values a dialect needs (project, deployedIndexId, …). */
    options: Record<string, string>;
}

export interface TranslatedFilter {
    /** The vendor-native filter object, or `undefined` when there is nothing to push. */
    pushed?: unknown;
    /** The clauses the vendor could not express — applied locally by the store. */
    residual?: MetadataFilter;
}

export interface VectorDialect {
    readonly id: string;
    /**
     * False when the vendor stores vectors only (Vertex AI Vector Search). The
     * store then hydrates `text` through its `textResolver`, and says so on the
     * telemetry span, rather than returning matches with empty text.
     */
    readonly storesText: boolean;
    readonly supportsKeywordSearch: boolean;
    /** False when `count` must be emulated by a capped query. */
    readonly supportsCount: boolean;
    /** False when a filtered bulk delete must be emulated by query-then-delete. */
    readonly supportsDeleteByFilter: boolean;

    translateFilter(filter: MetadataFilter): TranslatedFilter;

    upsert(ctx: DialectContext, records: readonly VectorRecord[]): HttpRequest;
    parseUpsert(json: unknown, sent: number): UpsertResult;

    query(
        ctx: DialectContext,
        args: { vector?: Float32Array; text?: string; topK: number; filter?: unknown; includeVectors: boolean },
    ): HttpRequest;
    parseMatches(json: unknown): VectorMatch[];

    deleteByIds(ctx: DialectContext, ids: readonly string[]): HttpRequest;
    fetch(ctx: DialectContext, ids: readonly string[]): HttpRequest;
    parseRecords(json: unknown): VectorRecord[];

    keywordSearch?(
        ctx: DialectContext,
        args: { text: string; topK: number; filter?: unknown },
    ): HttpRequest;
    deleteByFilter?(ctx: DialectContext, filter?: unknown): HttpRequest;
    count?(ctx: DialectContext, filter?: unknown): HttpRequest;
    parseCount?(json: unknown): number;
}

// ── helpers ───────────────────────────────────────────────────────────────────

const vecToArray = (v?: Float32Array): number[] | undefined => (v ? Array.from(v) : undefined);
const arrayToVec = (v: unknown): Float32Array | undefined =>
    Array.isArray(v) ? Float32Array.from(v as number[]) : undefined;

const asRecord = (v: unknown): Record<string, unknown> => (v && typeof v === 'object' ? v as Record<string, unknown> : {});
const asMetadata = (v: unknown): MetadataRecord => asRecord(v) as MetadataRecord;

/**
 * Splits a filter into the leaf clauses a dialect accepts and the rest.
 *
 * Only a top-level AND is split. Pushing half an OR to the server and keeping the
 * other half local would widen the server-side result set to *everything* that
 * matches either branch, so a disjunction (or a NOT) is sent to the local pass
 * whole. That is a recall cost, never a correctness one.
 */
function partition(
    filter: MetadataFilter,
    accepts: (leaf: LeafFilter) => boolean,
): { supported: LeafFilter[]; residual: MetadataFilter[] } {
    if (filter.op === 'and') {
        const supported: LeafFilter[] = [];
        const residual: MetadataFilter[] = [];
        for (const clause of filter.filters) {
            const split = partition(clause, accepts);
            supported.push(...split.supported);
            residual.push(...split.residual);
        }
        return { supported, residual };
    }
    if (filter.op === 'or' || filter.op === 'not') {
        return { supported: [], residual: [filter] };
    }
    return accepts(filter) ? { supported: [filter], residual: [] } : { supported: [], residual: [filter] };
}

function joinResidual(residual: MetadataFilter[]): MetadataFilter | undefined {
    if (residual.length === 0) return undefined;
    return residual.length === 1 ? residual[0] as MetadataFilter : { op: 'and', filters: residual };
}

// ── evermind: the contract this package defines ───────────────────────────────

/**
 * The canonical gateway contract. A thin service in front of ANY database can
 * implement it (the BuilderForce.ai gateway does), which keeps a customer's chosen
 * store behind their own network boundary while this package speaks one protocol.
 * The filter algebra crosses the wire unchanged, so nothing is ever residual.
 */
export const evermindDialect: VectorDialect = {
    id: 'evermind',
    storesText: true,
    supportsKeywordSearch: true,
    supportsCount: true,
    supportsDeleteByFilter: true,

    translateFilter: (filter) => ({ pushed: filter }),

    upsert: (ctx, records) => ({
        path: `/collections/${encodeURIComponent(ctx.collection)}/upsert`,
        method: 'POST',
        body: {
            records: records.map((r) => ({
                id: r.id,
                text: r.text,
                vector: vecToArray(r.vector),
                metadata: r.metadata,
            })),
        },
    }),
    parseUpsert: (json, sent) => {
        const body = asRecord(json);
        return {
            upserted: Number(body['upserted'] ?? sent) || 0,
            skipped: Number(body['skipped'] ?? 0) || 0,
        };
    },

    query: (ctx, args) => ({
        path: `/collections/${encodeURIComponent(ctx.collection)}/query`,
        method: 'POST',
        body: {
            vector: vecToArray(args.vector),
            text: args.text,
            topK: args.topK,
            filter: args.filter,
            includeVectors: args.includeVectors,
        },
    }),
    parseMatches: (json) => {
        const matches = asRecord(json)['matches'];
        if (!Array.isArray(matches)) return [];
        return matches.map((m) => {
            const row = asRecord(m);
            const match: VectorMatch = {
                id: String(row['id'] ?? ''),
                text: String(row['text'] ?? ''),
                score: Number(row['score']) || 0,
                metadata: asMetadata(row['metadata']),
            };
            const vector = arrayToVec(row['vector']);
            if (vector) match.vector = vector;
            return match;
        });
    },

    keywordSearch: (ctx, args) => ({
        path: `/collections/${encodeURIComponent(ctx.collection)}/keyword`,
        method: 'POST',
        body: { text: args.text, topK: args.topK, filter: args.filter },
    }),

    deleteByIds: (ctx, ids) => ({
        path: `/collections/${encodeURIComponent(ctx.collection)}/delete`,
        method: 'POST',
        body: { ids },
    }),
    deleteByFilter: (ctx, filter) => ({
        path: `/collections/${encodeURIComponent(ctx.collection)}/delete`,
        method: 'POST',
        body: { filter },
    }),

    fetch: (ctx, ids) => ({
        path: `/collections/${encodeURIComponent(ctx.collection)}/fetch`,
        method: 'POST',
        body: { ids },
    }),
    parseRecords: (json) => {
        const records = asRecord(json)['records'];
        if (!Array.isArray(records)) return [];
        return records.map((r) => {
            const row = asRecord(r);
            const record: VectorRecord = {
                id: String(row['id'] ?? ''),
                text: String(row['text'] ?? ''),
                metadata: asMetadata(row['metadata']),
            };
            const vector = arrayToVec(row['vector']);
            if (vector) record.vector = vector;
            return record;
        });
    },

    count: (ctx, filter) => ({
        path: `/collections/${encodeURIComponent(ctx.collection)}/count`,
        method: 'POST',
        body: { filter },
    }),
    parseCount: (json) => Number(asRecord(json)['count']) || 0,
};

// ── Qdrant ────────────────────────────────────────────────────────────────────

/** Maps one leaf onto a Qdrant field condition. */
function qdrantLeaf(leaf: LeafFilter): Record<string, unknown> | undefined {
    switch (leaf.op) {
        case 'eq':
            return { key: leaf.field, match: { value: leaf.value } };
        case 'in':
        case 'anyOf':
            return { key: leaf.field, match: { any: leaf.values } };
        case 'nin':
            return { key: leaf.field, match: { except: leaf.values } };
        case 'contains':
            return { key: leaf.field, match: { text: leaf.value } };
        case 'exists':
            return { is_empty: { key: leaf.field } };
        case 'gt': case 'gte': case 'lt': case 'lte':
            return typeof leaf.value === 'number'
                ? { key: leaf.field, range: { [leaf.op]: leaf.value } }
                : undefined;
        default:
            return undefined;
    }
}

export const qdrantDialect: VectorDialect = {
    id: 'qdrant',
    storesText: true,
    // Qdrant has full-text conditions but no BM25 ranking over the whole corpus,
    // so keyword search is declared unsupported rather than faked with a filter.
    supportsKeywordSearch: false,
    supportsCount: true,
    supportsDeleteByFilter: true,

    translateFilter: (filter) => {
        const { supported, residual } = partition(filter, (leaf) => {
            if (leaf.op === 'exists') return false;  // `is_empty` is the inverse; keep it local
            if (leaf.op === 'ne') return false;
            return qdrantLeaf(leaf) !== undefined;
        });
        const must = supported.map(qdrantLeaf).filter(Boolean);
        const out: TranslatedFilter = {};
        if (must.length > 0) out.pushed = { must };
        const rest = joinResidual(residual);
        if (rest) out.residual = rest;
        return out;
    },

    upsert: (ctx, records) => ({
        path: `/collections/${encodeURIComponent(ctx.collection)}/points`,
        method: 'PUT',
        body: {
            points: records.map((r) => ({
                id: r.id,
                vector: vecToArray(r.vector) ?? [],
                // Qdrant has no first-class text field; the chunk rides in the
                // payload beside its metadata under a reserved key.
                payload: { ...r.metadata, __text: r.text },
            })),
        },
    }),
    parseUpsert: (_json, sent) => ({ upserted: sent, skipped: 0 }),

    query: (ctx, args) => ({
        path: `/collections/${encodeURIComponent(ctx.collection)}/points/search`,
        method: 'POST',
        body: {
            vector: vecToArray(args.vector) ?? [],
            limit: args.topK,
            filter: args.filter,
            with_payload: true,
            with_vector: args.includeVectors,
        },
    }),
    parseMatches: (json) => {
        const result = asRecord(json)['result'];
        if (!Array.isArray(result)) return [];
        return result.map((r) => {
            const row = asRecord(r);
            const payload = asRecord(row['payload']);
            const { __text, ...metadata } = payload;
            const match: VectorMatch = {
                id: String(row['id'] ?? ''),
                text: String(__text ?? ''),
                score: Number(row['score']) || 0,
                metadata: metadata as MetadataRecord,
            };
            const vector = arrayToVec(row['vector']);
            if (vector) match.vector = vector;
            return match;
        });
    },

    deleteByIds: (ctx, ids) => ({
        path: `/collections/${encodeURIComponent(ctx.collection)}/points/delete`,
        method: 'POST',
        body: { points: ids },
    }),
    deleteByFilter: (ctx, filter) => ({
        path: `/collections/${encodeURIComponent(ctx.collection)}/points/delete`,
        method: 'POST',
        body: { filter },
    }),

    fetch: (ctx, ids) => ({
        path: `/collections/${encodeURIComponent(ctx.collection)}/points`,
        method: 'POST',
        body: { ids, with_payload: true, with_vector: true },
    }),
    parseRecords: (json) => {
        const result = asRecord(json)['result'];
        if (!Array.isArray(result)) return [];
        return result.map((r) => {
            const row = asRecord(r);
            const payload = asRecord(row['payload']);
            const { __text, ...metadata } = payload;
            const record: VectorRecord = {
                id: String(row['id'] ?? ''),
                text: String(__text ?? ''),
                metadata: metadata as MetadataRecord,
            };
            const vector = arrayToVec(row['vector']);
            if (vector) record.vector = vector;
            return record;
        });
    },

    count: (ctx, filter) => ({
        path: `/collections/${encodeURIComponent(ctx.collection)}/points/count`,
        method: 'POST',
        body: { filter, exact: true },
    }),
    parseCount: (json) => Number(asRecord(asRecord(json)['result'])['count']) || 0,
};

// ── Pinecone ──────────────────────────────────────────────────────────────────

function pineconeLeaf(leaf: LeafFilter): Record<string, unknown> | undefined {
    switch (leaf.op) {
        case 'eq':  return { [leaf.field]: { $eq: leaf.value } };
        case 'ne':  return { [leaf.field]: { $ne: leaf.value } };
        case 'gt':  return { [leaf.field]: { $gt: leaf.value } };
        case 'gte': return { [leaf.field]: { $gte: leaf.value } };
        case 'lt':  return { [leaf.field]: { $lt: leaf.value } };
        case 'lte': return { [leaf.field]: { $lte: leaf.value } };
        // `$in` matches when a list-valued field intersects, which is exactly the
        // ACL semantics `anyOf` describes.
        case 'in': case 'anyOf': return { [leaf.field]: { $in: leaf.values } };
        case 'nin': return { [leaf.field]: { $nin: leaf.values } };
        default: return undefined;
    }
}

export const pineconeDialect: VectorDialect = {
    id: 'pinecone',
    storesText: true,
    supportsKeywordSearch: false,
    supportsCount: false,
    supportsDeleteByFilter: true,

    translateFilter: (filter) => {
        const { supported, residual } = partition(filter, (leaf) => pineconeLeaf(leaf) !== undefined);
        const clauses = supported.map(pineconeLeaf).filter(Boolean) as Record<string, unknown>[];
        const out: TranslatedFilter = {};
        if (clauses.length === 1) out.pushed = clauses[0];
        else if (clauses.length > 1) out.pushed = { $and: clauses };
        const rest = joinResidual(residual);
        if (rest) out.residual = rest;
        return out;
    },

    upsert: (ctx, records) => ({
        path: '/vectors/upsert',
        method: 'POST',
        body: {
            namespace: ctx.collection,
            vectors: records.map((r) => ({
                id: r.id,
                values: vecToArray(r.vector) ?? [],
                metadata: { ...r.metadata, __text: r.text },
            })),
        },
    }),
    parseUpsert: (json, sent) => ({
        upserted: Number(asRecord(json)['upsertedCount'] ?? sent) || 0,
        skipped: 0,
    }),

    query: (ctx, args) => ({
        path: '/query',
        method: 'POST',
        body: {
            namespace: ctx.collection,
            vector: vecToArray(args.vector) ?? [],
            topK: args.topK,
            filter: args.filter,
            includeMetadata: true,
            includeValues: args.includeVectors,
        },
    }),
    parseMatches: (json) => {
        const matches = asRecord(json)['matches'];
        if (!Array.isArray(matches)) return [];
        return matches.map((m) => {
            const row = asRecord(m);
            const meta = asRecord(row['metadata']);
            const { __text, ...metadata } = meta;
            const match: VectorMatch = {
                id: String(row['id'] ?? ''),
                text: String(__text ?? ''),
                score: Number(row['score']) || 0,
                metadata: metadata as MetadataRecord,
            };
            const vector = arrayToVec(row['values']);
            if (vector) match.vector = vector;
            return match;
        });
    },

    deleteByIds: (ctx, ids) => ({
        path: '/vectors/delete',
        method: 'POST',
        body: { namespace: ctx.collection, ids },
    }),
    deleteByFilter: (ctx, filter) => ({
        path: '/vectors/delete',
        method: 'POST',
        body: { namespace: ctx.collection, filter },
    }),

    fetch: (ctx, ids) => ({
        path: '/vectors/fetch',
        method: 'POST',
        body: { namespace: ctx.collection, ids },
    }),
    parseRecords: (json) => {
        const vectors = asRecord(asRecord(json)['vectors']);
        return Object.entries(vectors).map(([id, raw]) => {
            const row = asRecord(raw);
            const meta = asRecord(row['metadata']);
            const { __text, ...metadata } = meta;
            const record: VectorRecord = {
                id,
                text: String(__text ?? ''),
                metadata: metadata as MetadataRecord,
            };
            const vector = arrayToVec(row['values']);
            if (vector) record.vector = vector;
            return record;
        });
    },
};

// ── Vertex AI Vector Search (GCP) ─────────────────────────────────────────────

/**
 * Vertex expresses filters as `restricts` (string namespaces with an allow list)
 * and `numericRestricts` (numeric comparisons), which is a close fit for this
 * algebra: `eq`/`in`/`anyOf` on strings become a restrict namespace, and numeric
 * comparisons become a numeric restrict. Anything else stays residual.
 *
 * Vertex stores vectors and restricts, NOT text — hence `storesText: false`. The
 * store hydrates chunk text through its `textResolver`, which in a real GCP
 * architecture reads from Firestore, BigQuery or GCS beside the index. Pretending
 * otherwise would return matches with empty text and an answer with no evidence.
 */
export const vertexAiDialect: VectorDialect = {
    id: 'vertex-ai',
    storesText: false,
    supportsKeywordSearch: false,
    supportsCount: false,
    supportsDeleteByFilter: false,

    translateFilter: (filter) => {
        const { supported, residual } = partition(filter, (leaf) => {
            if (leaf.op === 'eq') return typeof leaf.value === 'string' || typeof leaf.value === 'number';
            if (leaf.op === 'in' || leaf.op === 'anyOf') return leaf.values.every((v) => typeof v === 'string');
            if (leaf.op === 'gt' || leaf.op === 'gte' || leaf.op === 'lt' || leaf.op === 'lte') {
                return typeof leaf.value === 'number';
            }
            return false;
        });

        const restricts: Array<{ namespace: string; allowList: string[] }> = [];
        const numericRestricts: Array<Record<string, unknown>> = [];
        const VERTEX_NUMERIC_OP: Record<string, string> = {
            gt: 'GREATER', gte: 'GREATER_EQUAL', lt: 'LESS', lte: 'LESS_EQUAL', eq: 'EQUAL',
        };

        for (const leaf of supported) {
            if (leaf.op === 'in' || leaf.op === 'anyOf') {
                restricts.push({ namespace: leaf.field, allowList: leaf.values as string[] });
            } else if (leaf.op === 'eq' && typeof leaf.value === 'string') {
                restricts.push({ namespace: leaf.field, allowList: [leaf.value] });
            } else if ('value' in leaf && typeof leaf.value === 'number') {
                numericRestricts.push({
                    namespace: leaf.field,
                    op: VERTEX_NUMERIC_OP[leaf.op] as string,
                    valueDouble: leaf.value,
                });
            }
        }

        const out: TranslatedFilter = {};
        if (restricts.length || numericRestricts.length) out.pushed = { restricts, numericRestricts };
        const rest = joinResidual(residual);
        if (rest) out.residual = rest;
        return out;
    },

    upsert: (ctx, records) => ({
        path: `/v1/${ctx.options['index']}:upsertDatapoints`,
        method: 'POST',
        body: {
            datapoints: records.map((r) => ({
                datapointId: r.id,
                featureVector: vecToArray(r.vector) ?? [],
                restricts: stringRestricts(r.metadata),
                numericRestricts: numericRestrictsOf(r.metadata),
            })),
        },
    }),
    parseUpsert: (_json, sent) => ({ upserted: sent, skipped: 0 }),

    query: (ctx, args) => {
        const pushed = asRecord(args.filter);
        return {
            path: `/v1/${ctx.options['indexEndpoint']}:findNeighbors`,
            method: 'POST',
            body: {
                deployedIndexId: ctx.options['deployedIndexId'],
                returnFullDatapoint: args.includeVectors,
                queries: [{
                    datapoint: {
                        datapointId: 'query',
                        featureVector: vecToArray(args.vector) ?? [],
                        restricts: pushed['restricts'] ?? [],
                        numericRestricts: pushed['numericRestricts'] ?? [],
                    },
                    neighborCount: args.topK,
                }],
            },
        };
    },
    parseMatches: (json) => {
        const groups = asRecord(json)['nearestNeighbors'];
        if (!Array.isArray(groups)) return [];
        const neighbors = asRecord(groups[0])['neighbors'];
        if (!Array.isArray(neighbors)) return [];
        return neighbors.map((n) => {
            const row = asRecord(n);
            const datapoint = asRecord(row['datapoint']);
            const match: VectorMatch = {
                id: String(datapoint['datapointId'] ?? ''),
                text: '',
                // Vertex returns a DISTANCE; for the cosine measure similarity is
                // 1 - distance. Reported on the same [0,1] scale as every adapter.
                score: 1 - (Number(row['distance']) || 0),
                metadata: restrictsToMetadata(datapoint),
            };
            const vector = arrayToVec(datapoint['featureVector']);
            if (vector) match.vector = vector;
            return match;
        });
    },

    deleteByIds: (ctx, ids) => ({
        path: `/v1/${ctx.options['index']}:removeDatapoints`,
        method: 'POST',
        body: { datapointIds: ids },
    }),

    fetch: (ctx, ids) => ({
        path: `/v1/${ctx.options['indexEndpoint']}:readIndexDatapoints`,
        method: 'POST',
        body: { deployedIndexId: ctx.options['deployedIndexId'], ids },
    }),
    parseRecords: (json) => {
        const datapoints = asRecord(json)['datapoints'];
        if (!Array.isArray(datapoints)) return [];
        return datapoints.map((d) => {
            const row = asRecord(d);
            const record: VectorRecord = {
                id: String(row['datapointId'] ?? ''),
                text: '',
                metadata: restrictsToMetadata(row),
            };
            const vector = arrayToVec(row['featureVector']);
            if (vector) record.vector = vector;
            return record;
        });
    },
};

function stringRestricts(metadata: MetadataRecord): Array<{ namespace: string; allowList: string[] }> {
    const out: Array<{ namespace: string; allowList: string[] }> = [];
    for (const [namespace, value] of Object.entries(metadata)) {
        if (typeof value === 'string') out.push({ namespace, allowList: [value] });
        else if (Array.isArray(value)) out.push({ namespace, allowList: value });
    }
    return out;
}

function numericRestrictsOf(metadata: MetadataRecord): Array<Record<string, unknown>> {
    return Object.entries(metadata)
        .filter(([, v]) => typeof v === 'number')
        .map(([namespace, v]) => ({ namespace, valueDouble: v as number }));
}

function restrictsToMetadata(datapoint: Record<string, unknown>): MetadataRecord {
    const out: MetadataRecord = {};
    const restricts = datapoint['restricts'];
    if (Array.isArray(restricts)) {
        for (const r of restricts) {
            const row = asRecord(r);
            const list = Array.isArray(row['allowList']) ? row['allowList'] as string[] : [];
            const namespace = String(row['namespace'] ?? '');
            if (namespace) out[namespace] = list.length === 1 ? list[0] as string : list;
        }
    }
    const numeric = datapoint['numericRestricts'];
    if (Array.isArray(numeric)) {
        for (const r of numeric) {
            const row = asRecord(r);
            const namespace = String(row['namespace'] ?? '');
            const value = row['valueDouble'] ?? row['valueFloat'] ?? row['valueInt'];
            if (namespace && value !== undefined) out[namespace] = Number(value);
        }
    }
    return out;
}

// ── registry ──────────────────────────────────────────────────────────────────

const REGISTRY = new Map<string, VectorDialect>([
    [evermindDialect.id, evermindDialect],
    [qdrantDialect.id, qdrantDialect],
    [pineconeDialect.id, pineconeDialect],
    [vertexAiDialect.id, vertexAiDialect],
]);

/** Registers (or replaces) a dialect — how a customer adds their own database. */
export function registerDialect(dialect: VectorDialect): void {
    REGISTRY.set(dialect.id, dialect);
}

export function getDialect(id: string): VectorDialect {
    const dialect = REGISTRY.get(id);
    if (!dialect) {
        throw new Error(
            `Unknown vector dialect "${id}". Registered: ${[...REGISTRY.keys()].join(', ')}. ` +
            'Add your own with registerDialect().',
        );
    }
    return dialect;
}

export function listDialects(): string[] {
    return [...REGISTRY.keys()];
}
