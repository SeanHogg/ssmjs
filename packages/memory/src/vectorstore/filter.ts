/**
 * vectorstore/filter.ts — the ONE evaluator and the ONE scope compiler.
 *
 * Two rules this file exists to enforce:
 *
 *   • A filter means the same thing everywhere. The in-process store, the
 *     post-filter a remote adapter applies when its dialect cannot express a
 *     clause, the ingest de-duplicator and the eval harness all call
 *     {@link matchesFilter}. A second implementation would eventually disagree
 *     about `null` or about a missing field, and a filter that disagrees with
 *     itself across two code paths is a cross-tenant leak waiting for traffic.
 *
 *   • Access scope is compiled, never hand-written. {@link accessFilter} is the
 *     single definition of "what may this caller see", so a new adapter or a new
 *     query path inherits the rule instead of re-deriving it.
 */

import {
    ACL_FIELD,
    ACL_PUBLIC,
    SENSITIVITY_FIELD,
    TENANT_FIELD,
    type AccessScope,
    type MetadataFilter,
    type MetadataRecord,
    type MetadataValue,
} from './types.js';

/**
 * Compiles an access scope into a filter.
 *
 * The ACL clause is `anyOf(acl, [...principals, '*'])`. Records are required to
 * carry a non-empty `acl` (ingestion defaults it to `['*']`), which is what makes
 * the rule fail CLOSED: a record written without an ACL matches nothing rather
 * than matching everyone.
 */
export function accessFilter(scope: AccessScope): MetadataFilter {
    const clauses: MetadataFilter[] = [
        { op: 'eq', field: TENANT_FIELD, value: scope.tenantId },
        { op: 'anyOf', field: ACL_FIELD, values: [...(scope.principals ?? []), ACL_PUBLIC] },
    ];

    if (scope.maxSensitivity !== undefined) {
        clauses.push({ op: 'lte', field: SENSITIVITY_FIELD, value: scope.maxSensitivity });
    }

    return { op: 'and', filters: clauses };
}

/**
 * ANDs a caller filter with a scope, dropping either when absent.
 * Returns `undefined` only when neither is present — an unscoped query, which is
 * legitimate for single-tenant use but never happens by accident, because the
 * caller had to omit the scope explicitly.
 */
export function combineFilters(
    filter?: MetadataFilter,
    scope?: AccessScope,
): MetadataFilter | undefined {
    const scoped = scope ? accessFilter(scope) : undefined;
    if (filter && scoped) return { op: 'and', filters: [scoped, filter] };
    return filter ?? scoped;
}

/** Evaluates a filter against one record's metadata. */
export function matchesFilter(metadata: MetadataRecord, filter?: MetadataFilter): boolean {
    if (!filter) return true;

    switch (filter.op) {
        case 'and':
            return filter.filters.every((f) => matchesFilter(metadata, f));
        case 'or':
            return filter.filters.some((f) => matchesFilter(metadata, f));
        case 'not':
            return !matchesFilter(metadata, filter.filter);
        case 'exists':
            return metadata[filter.field] !== undefined && metadata[filter.field] !== null;
        case 'in':
            return filter.values.some((v) => scalarEquals(metadata[filter.field], v));
        case 'nin':
            return !filter.values.some((v) => scalarEquals(metadata[filter.field], v));
        case 'anyOf': {
            const actual = metadata[filter.field];
            if (actual === undefined || actual === null) return false;
            const held = Array.isArray(actual) ? actual : [actual];
            return held.some((h) => filter.values.some((v) => scalarEquals(h, v)));
        }
        case 'contains': {
            const actual = metadata[filter.field];
            if (typeof actual !== 'string') return false;
            return actual.toLowerCase().includes(filter.value.toLowerCase());
        }
        default:
            return compare(metadata[filter.field], filter.op, filter.value);
    }
}

function scalarEquals(a: MetadataValue | undefined, b: MetadataValue): boolean {
    if (Array.isArray(a) || Array.isArray(b)) {
        return Array.isArray(a) && Array.isArray(b)
            && a.length === b.length
            && a.every((v, i) => v === b[i]);
    }
    return a === b;
}

function compare(
    actual: MetadataValue | undefined,
    op: 'eq' | 'ne' | 'gt' | 'gte' | 'lt' | 'lte',
    expected: MetadataValue,
): boolean {
    if (op === 'eq') return scalarEquals(actual, expected);
    if (op === 'ne') return !scalarEquals(actual, expected);

    // Ordering comparisons on a missing or non-scalar field are false rather than
    // throwing: a record without the field simply does not satisfy `sensitivity <= n`.
    if (actual === undefined || actual === null) return false;
    if (Array.isArray(actual) || Array.isArray(expected) || expected === null) return false;
    if (typeof actual !== typeof expected) return false;
    if (typeof actual === 'boolean' || typeof expected === 'boolean') return false;

    switch (op) {
        case 'gt':  return actual > expected;
        case 'gte': return actual >= expected;
        case 'lt':  return actual < expected;
        default:    return actual <= expected;
    }
}

/**
 * Walks a filter tree, reporting every field it touches. Adapters use this to
 * decide up front whether a dialect can push the whole filter down, or whether a
 * post-filter pass is needed.
 */
export function filterFields(filter: MetadataFilter, out = new Set<string>()): Set<string> {
    switch (filter.op) {
        case 'and':
        case 'or':
            for (const f of filter.filters) filterFields(f, out);
            break;
        case 'not':
            filterFields(filter.filter, out);
            break;
        default:
            out.add(filter.field);
    }
    return out;
}
