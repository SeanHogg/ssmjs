/**
 * vectorstore/RestVectorStore.ts — ONE adapter for every HTTP vector database.
 *
 * The vendor-specific part is the {@link VectorDialect}; everything that is the
 * same everywhere lives here exactly once: auth headers, batching, retry with
 * backoff, residual filtering, capability emulation, and text hydration for stores
 * that hold vectors only.
 *
 * The residual pass is the part that matters for correctness. When a dialect
 * cannot push a filter clause down, the store over-fetches and applies the
 * remaining clauses locally with the SAME evaluator the in-process store uses. A
 * clause is therefore never dropped — which on an ACL or tenant clause would be a
 * data leak, and is precisely the failure mode a naive adapter ships with.
 */

import { combineFilters, matchesFilter } from './filter.js';
import { getDialect, type DialectContext, type HttpRequest, type VectorDialect } from './dialects.js';
import type {
    AccessScope,
    MetadataFilter,
    UpsertResult,
    VectorMatch,
    VectorQuery,
    VectorRecord,
    VectorStore,
} from './types.js';

export interface RestVectorStoreOptions {
    /** Dialect id (`evermind`, `qdrant`, `pinecone`, `vertex-ai`) or an instance. */
    dialect: string | VectorDialect;
    /** Service root, e.g. `https://my-cluster.qdrant.io`. */
    baseUrl: string;
    /** Collection / index / namespace. */
    collection: string;
    /** Static headers — API keys, `Authorization: Bearer …`. */
    headers?: Record<string, string>;
    /**
     * Per-request auth, for credentials that expire. On GCP this is where a
     * Workload-Identity access token goes, refreshed by the caller's token source
     * rather than cached here — the store must never own a credential lifecycle.
     */
    authorize?: () => Promise<Record<string, string>> | Record<string, string>;
    /** Dialect-specific deployment values (`index`, `indexEndpoint`, `deployedIndexId`, …). */
    options?: Record<string, string>;
    /** Records per upsert request. Default 100. */
    batchSize?: number;
    /** Retries on 429/5xx/network failure. Default 2. */
    maxRetries?: number;
    /** Base backoff in ms, doubled per attempt. Default 200. */
    retryBackoffMs?: number;
    /** Over-fetch multiplier when a residual filter must be applied locally. Default 4. */
    residualOverfetch?: number;
    /**
     * Supplies chunk text for dialects that store vectors only (Vertex AI). In a
     * GCP reference architecture this reads the chunk table from Firestore or
     * BigQuery keyed by the same ids the index holds.
     */
    textResolver?: (ids: readonly string[]) => Promise<Map<string, string>>;
    /** Injected for test. Default global `fetch`. */
    fetchImpl?: typeof fetch;
    /** Injected for test so retry backoff does not really sleep. */
    sleep?: (ms: number) => Promise<void>;
}

export class RestVectorStore implements VectorStore {
    readonly name: string;
    readonly dialect: VectorDialect;

    private readonly _baseUrl: string;
    private readonly _ctx: DialectContext;
    private readonly _headers: Record<string, string>;
    private readonly _authorize: (() => Promise<Record<string, string>> | Record<string, string>) | undefined;
    private readonly _batchSize: number;
    private readonly _maxRetries: number;
    private readonly _backoffMs: number;
    private readonly _overfetch: number;
    private readonly _textResolver: RestVectorStoreOptions['textResolver'];
    private readonly _fetch: typeof fetch;
    private readonly _sleep: (ms: number) => Promise<void>;

    constructor(opts: RestVectorStoreOptions) {
        this.dialect = typeof opts.dialect === 'string' ? getDialect(opts.dialect) : opts.dialect;
        this.name = `rest:${this.dialect.id}`;
        this._baseUrl = opts.baseUrl.replace(/\/+$/, '');
        this._ctx = { collection: opts.collection, options: opts.options ?? {} };
        this._headers = opts.headers ?? {};
        this._authorize = opts.authorize;
        this._batchSize = Math.max(1, opts.batchSize ?? 100);
        this._maxRetries = Math.max(0, opts.maxRetries ?? 2);
        this._backoffMs = Math.max(0, opts.retryBackoffMs ?? 200);
        this._overfetch = Math.max(1, opts.residualOverfetch ?? 4);
        this._textResolver = opts.textResolver;
        this._fetch = opts.fetchImpl ?? ((...args) => fetch(...args));
        this._sleep = opts.sleep ?? ((ms) => new Promise((r) => setTimeout(r, ms)));
    }

    async upsert(records: readonly VectorRecord[]): Promise<UpsertResult> {
        let upserted = 0;
        let skipped = 0;
        // Batched because every vendor caps request size, and a 10k-record ingest
        // that fails at record 9,999 must not lose the first 9,998.
        for (let i = 0; i < records.length; i += this._batchSize) {
            const batch = records.slice(i, i + this._batchSize);
            const json = await this._send(this.dialect.upsert(this._ctx, batch));
            const result = this.dialect.parseUpsert(json, batch.length);
            upserted += result.upserted;
            skipped += result.skipped;
        }
        return { upserted, skipped };
    }

    async query(query: VectorQuery): Promise<VectorMatch[]> {
        return this._search(query, (translated, topK) =>
            this.dialect.query(this._ctx, {
                ...(query.vector ? { vector: query.vector } : {}),
                ...(query.text ? { text: query.text } : {}),
                topK,
                ...(translated !== undefined ? { filter: translated } : {}),
                includeVectors: query.includeVectors ?? false,
            }),
        );
    }

    async keywordSearch(query: VectorQuery): Promise<VectorMatch[]> {
        const text = query.text?.trim();
        if (!text || !this.dialect.supportsKeywordSearch || !this.dialect.keywordSearch) return [];
        const build = this.dialect.keywordSearch.bind(this.dialect);
        return this._search(query, (translated, topK) =>
            build(this._ctx, { text, topK, ...(translated !== undefined ? { filter: translated } : {}) }),
        );
    }

    async fetch(ids: readonly string[]): Promise<VectorRecord[]> {
        if (ids.length === 0) return [];
        const json = await this._send(this.dialect.fetch(this._ctx, ids));
        const records = this.dialect.parseRecords(json);
        if (this.dialect.storesText || !this._textResolver) return records;

        const texts = await this._textResolver(records.map((r) => r.id));
        return records.map((r) => ({ ...r, text: texts.get(r.id) ?? r.text }));
    }

    async delete(ids: readonly string[]): Promise<number> {
        if (ids.length === 0) return 0;
        for (let i = 0; i < ids.length; i += this._batchSize) {
            await this._send(this.dialect.deleteByIds(this._ctx, ids.slice(i, i + this._batchSize)));
        }
        return ids.length;
    }

    async deleteByFilter(filter: MetadataFilter, scope?: AccessScope): Promise<number> {
        const effective = combineFilters(filter, scope) as MetadataFilter;
        const translated = this.dialect.translateFilter(effective);

        // Server-side bulk delete is only safe when the dialect can express the
        // WHOLE predicate. A residual would mean deleting a superset — so the
        // emulated path (query, filter locally, delete by id) is used instead.
        if (this.dialect.supportsDeleteByFilter && this.dialect.deleteByFilter && !translated.residual) {
            await this._send(this.dialect.deleteByFilter(this._ctx, translated.pushed));
            return -1; // vendor bulk deletes do not report a count
        }

        const doomed = await this.query({ filter, ...(scope ? { scope } : {}), topK: 10_000 });
        return this.delete(doomed.map((m) => m.id));
    }

    async count(filter?: MetadataFilter, scope?: AccessScope): Promise<number> {
        const effective = combineFilters(filter, scope);
        const translated = effective ? this.dialect.translateFilter(effective) : undefined;

        if (this.dialect.supportsCount && this.dialect.count && this.dialect.parseCount && !translated?.residual) {
            const json = await this._send(this.dialect.count(this._ctx, translated?.pushed));
            return this.dialect.parseCount(json);
        }
        const matches = await this.query({ ...(filter ? { filter } : {}), ...(scope ? { scope } : {}), topK: 10_000 });
        return matches.length;
    }

    /** Shared query path: translate, over-fetch for the residual, filter, hydrate. */
    private async _search(
        query: VectorQuery,
        build: (translated: unknown, topK: number) => HttpRequest,
    ): Promise<VectorMatch[]> {
        const topK = Math.max(1, query.topK ?? 10);
        const effective = combineFilters(query.filter, query.scope);
        const translated = effective ? this.dialect.translateFilter(effective) : undefined;
        const wanted = translated?.residual ? topK * this._overfetch : topK;

        const json = await this._send(build(translated?.pushed, wanted));
        let matches = this.dialect.parseMatches(json);

        if (translated?.residual) {
            const residual = translated.residual;
            matches = matches.filter((m) => matchesFilter(m.metadata, residual));
        }
        if (query.minScore !== undefined) {
            matches = matches.filter((m) => m.score >= (query.minScore as number));
        }
        matches = matches.slice(0, topK);

        if (!this.dialect.storesText && this._textResolver && matches.length > 0) {
            const texts = await this._textResolver(matches.map((m) => m.id));
            matches = matches.map((m) => ({ ...m, text: texts.get(m.id) ?? m.text }));
        }
        return matches;
    }

    private async _send(request: HttpRequest): Promise<unknown> {
        const dynamic = this._authorize ? await this._authorize() : {};
        const headers = { 'Content-Type': 'application/json', ...this._headers, ...dynamic };
        const url = `${this._baseUrl}${request.path}`;

        let lastError: unknown;
        for (let attempt = 0; attempt <= this._maxRetries; attempt++) {
            if (attempt > 0) await this._sleep(this._backoffMs * 2 ** (attempt - 1));
            try {
                const res = await this._fetch(url, {
                    method: request.method,
                    headers,
                    ...(request.body !== undefined ? { body: JSON.stringify(request.body) } : {}),
                });
                if (res.ok) return await res.json().catch(() => ({}));

                const detail = await res.text().catch(() => '');
                lastError = new Error(`${this.name} ${request.method} ${request.path} → ${res.status}: ${detail}`);
                // 4xx other than rate-limiting is a request the server will reject
                // identically on retry; failing fast keeps a bad filter from
                // becoming a retry storm.
                if (res.status !== 429 && res.status < 500) break;
            } catch (err) {
                lastError = err;
            }
        }
        throw lastError instanceof Error ? lastError : new Error(String(lastError));
    }
}
