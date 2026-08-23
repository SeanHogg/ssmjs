/**
 * telemetry/Tracer.ts — granular tracing for agent runs.
 *
 * A production agent failure is almost never "the LLM was wrong"; it is "the
 * retrieval returned nothing, so the LLM was asked to answer from an empty
 * context". You can only see that with a span tree covering the WHOLE request —
 * ingest, retrieval, each graph node, each model call — under one trace id.
 *
 * `now` and `newId` are injected so traces are deterministic under test; the
 * defaults use wall-clock and a random hex id.
 */

import { estimateCostUsd, DEFAULT_PRICE_BOOK, type PriceBook } from './pricing.js';
import { formatTraceparent, type TraceContext } from './traceparent.js';
import type {
    AttributeValue,
    LlmUsage,
    SpanData,
    SpanEvent,
    SpanExporter,
    SpanKind,
    SpanStatus,
} from './types.js';

export interface SpanOptions {
    kind?: SpanKind;
    attributes?: Record<string, AttributeValue>;
    /** Parent span. Omit for a root span (starts a new trace). */
    parent?: Span;
    /** Force a trace id (e.g. to continue a trace propagated from an HTTP header). */
    traceId?: string;
    /**
     * A parent span in ANOTHER process, decoded from a `traceparent` header. Adopts
     * its trace id, links to it as parent, and — crucially — obeys its sampling
     * flag, so the four hops of a cloud run are one tree or none, never a tree with
     * a hole where a hop re-rolled the dice. Ignored when `parent` is given (an
     * in-process parent is always the more specific answer).
     */
    remoteParent?: TraceContext;
}

export interface TracerOptions {
    /** Where finished spans go. Omit to buffer only (readable via `finished()`). */
    exporter?: SpanExporter;
    /** Rates used to price `llm` spans. Default {@link DEFAULT_PRICE_BOOK}. */
    pricing?: PriceBook;
    /** Clock, injected for determinism. Default `Date.now`. */
    now?: () => number;
    /** Id factory, injected for determinism. Default random hex. */
    newId?: (bytes: number) => string;
    /**
     * Export once this many spans have finished. Default 64. Set 1 to export
     * eagerly, when a downstream collector wants a live stream.
     */
    maxBatchSize?: number;
    /**
     * Fraction of ROOT traces recorded, 0..1. Default 1. Sampling is decided once
     * per trace and inherited by children, so a sampled trace is never a tree with
     * holes in it.
     */
    sampleRate?: number;
    /** Sampling dice, injected for test. Default `Math.random`. */
    random?: () => number;
}

/**
 * A live span. Mutating methods are no-ops once `end()`/`fail()` has run, so a
 * late callback (a stream resolving after a timeout) cannot corrupt a span that
 * has already been exported.
 */
export class Span {
    private readonly _tracer: Tracer;
    private readonly _data: SpanData;
    private _closed = false;
    private readonly _recorded: boolean;

    /** @internal — construct via {@link Tracer.startSpan}. */
    constructor(tracer: Tracer, data: SpanData, recorded: boolean) {
        this._tracer = tracer;
        this._data = data;
        this._recorded = recorded;
    }

    get traceId(): string { return this._data.traceId; }
    get spanId(): string { return this._data.spanId; }
    get name(): string { return this._data.name; }
    get closed(): boolean { return this._closed; }
    /** False when this trace was dropped by sampling — nothing will be exported. */
    get recorded(): boolean { return this._recorded; }

    /** A snapshot copy; mutating it does not affect the span. */
    get data(): SpanData {
        return { ...this._data, attributes: { ...this._data.attributes }, events: [...this._data.events] };
    }

    setAttribute(key: string, value: AttributeValue): this {
        if (!this._closed) this._data.attributes[key] = value;
        return this;
    }

    setAttributes(attrs: Record<string, AttributeValue | undefined>): this {
        for (const [k, v] of Object.entries(attrs)) {
            if (v !== undefined) this.setAttribute(k, v);
        }
        return this;
    }

    addEvent(name: string, attributes?: Record<string, AttributeValue>): this {
        if (this._closed) return this;
        const event: SpanEvent = { name, at: this._tracer.now() };
        if (attributes) event.attributes = attributes;
        this._data.events.push(event);
        return this;
    }

    /** Records token usage and prices it with the tracer's price book. */
    recordUsage(usage: LlmUsage): this {
        if (this._closed) return this;
        this._data.usage = usage;
        const cost = estimateCostUsd(usage, this._tracer.pricing);
        if (cost !== undefined) this._data.costUsd = cost;
        return this;
    }

    /** Marks time-to-first-token; call when a stream yields its first chunk. */
    recordFirstToken(): this {
        if (this._closed) return this;
        this._data.timeToFirstTokenMs = this._tracer.now() - this._data.startedAt;
        return this.addEvent('first-token');
    }

    /** Opens a child span inside this one (same trace, this span as parent). */
    child(name: string, opts: Omit<SpanOptions, 'parent' | 'traceId' | 'remoteParent'> = {}): Span {
        return this._tracer.startSpan(name, { ...opts, parent: this });
    }

    /**
     * This span's position, for handing to another process. The sampling flag is the
     * span's OWN `recorded` verdict, so the next hop inherits the decision already
     * taken rather than making a second, possibly contradictory one.
     */
    context(): TraceContext {
        return { traceId: this._data.traceId, spanId: this._data.spanId, sampled: this._recorded };
    }

    /**
     * The `traceparent` header value to send on an outbound request made inside this
     * span. Set it and the callee's root span becomes this span's child.
     */
    traceparent(): string {
        return formatTraceparent(this.context());
    }

    end(status: SpanStatus = 'ok'): SpanData {
        return this._close(status);
    }

    /** Ends the span as failed, recording the error message. */
    fail(error: unknown): SpanData {
        if (!this._closed) {
            this._data.error = error instanceof Error ? error.message : String(error);
        }
        return this._close('error');
    }

    private _close(status: SpanStatus): SpanData {
        if (this._closed) return this.data;
        this._closed = true;
        this._data.status = status;
        this._data.endedAt = this._tracer.now();
        this._data.durationMs = this._data.endedAt - this._data.startedAt;
        this._tracer._finish(this._data, this._recorded);
        return this.data;
    }
}

export class Tracer {
    readonly pricing: PriceBook;

    private readonly _exporter?: SpanExporter;
    private readonly _now: () => number;
    private readonly _newId: (bytes: number) => string;
    private readonly _maxBatch: number;
    private readonly _sampleRate: number;
    private readonly _random: () => number;

    private _buffer: SpanData[] = [];
    private readonly _finished: SpanData[] = [];
    /** traceId -> sampling decision, so children inherit the root's verdict. */
    private readonly _sampled = new Map<string, boolean>();

    constructor(opts: TracerOptions = {}) {
        if (opts.exporter) this._exporter = opts.exporter;
        this.pricing     = opts.pricing ?? DEFAULT_PRICE_BOOK;
        this._now        = opts.now ?? (() => Date.now());
        this._newId      = opts.newId ?? defaultIdFactory();
        this._maxBatch   = Math.max(1, opts.maxBatchSize ?? 64);
        this._sampleRate = clamp01(opts.sampleRate ?? 1);
        this._random     = opts.random ?? Math.random;
    }

    now(): number { return this._now(); }

    startSpan(name: string, opts: SpanOptions = {}): Span {
        // An in-process parent beats a remote one: if both are present the remote
        // header is stale context that the live parent already descends from.
        const remote = opts.parent ? undefined : opts.remoteParent;
        const traceId  = opts.parent?.traceId ?? remote?.traceId ?? opts.traceId ?? this._newId(16);
        const recorded = remote
            ? this._adoptSampling(traceId, remote.sampled)
            : this._decideSampling(traceId, Boolean(opts.parent));

        const data: SpanData = {
            traceId,
            spanId: this._newId(8),
            name,
            kind: opts.kind ?? 'internal',
            startedAt: this._now(),
            attributes: { ...(opts.attributes ?? {}) },
            events: [],
            status: 'unset',
        };
        if (opts.parent) data.parentSpanId = opts.parent.spanId;
        else if (remote) data.parentSpanId = remote.spanId;

        return new Span(this, data, recorded);
    }

    /**
     * Runs `fn` inside a span, ending it `ok` on resolve and `error` on throw.
     * Every subsystem uses this rather than hand-pairing start/end, so a thrown
     * error can never leak an un-ended span into the trace tree.
     */
    async withSpan<T>(name: string, opts: SpanOptions, fn: (span: Span) => Promise<T> | T): Promise<T> {
        const span = this.startSpan(name, opts);
        try {
            const out = await fn(span);
            span.end('ok');
            return out;
        } catch (err) {
            span.fail(err);
            throw err;
        }
    }

    /** All spans finished since construction (recorded ones only). */
    finished(): readonly SpanData[] { return this._finished; }

    /** Spans of one trace, in completion order. */
    trace(traceId: string): readonly SpanData[] {
        return this._finished.filter((s) => s.traceId === traceId);
    }

    /** Pushes any buffered spans to the exporter. */
    async flush(): Promise<void> {
        if (this._buffer.length === 0) return;
        const batch = this._buffer;
        this._buffer = [];
        await this._exporter?.export(batch);
    }

    async shutdown(): Promise<void> {
        await this.flush();
        await this._exporter?.shutdown?.();
    }

    /** @internal */
    _finish(data: SpanData, recorded: boolean): void {
        if (!recorded) return;
        this._finished.push(data);
        this._buffer.push(data);
        if (this._buffer.length >= this._maxBatch) void this.flush();
    }

    /**
     * Take the upstream hop's sampling verdict verbatim and record it against the
     * trace, so every later span in this process inherits it too. Head-based
     * sampling only works if it is decided ONCE, at the head.
     */
    private _adoptSampling(traceId: string, sampled: boolean): boolean {
        const known = this._sampled.get(traceId);
        if (known !== undefined) return known;
        this._sampled.set(traceId, sampled);
        return sampled;
    }

    private _decideSampling(traceId: string, hasParent: boolean): boolean {
        const known = this._sampled.get(traceId);
        if (known !== undefined) return known;
        // A child whose parent's decision is unknown defaults to recorded rather
        // than re-rolling the dice, which would tear a trace in half.
        const decision = hasParent || this._sampleRate >= 1 || this._random() < this._sampleRate;
        this._sampled.set(traceId, decision);
        return decision;
    }
}

function clamp01(n: number): number {
    return Number.isFinite(n) ? Math.min(1, Math.max(0, n)) : 1;
}

function defaultIdFactory(): (bytes: number) => string {
    return (bytes: number) => {
        let out = '';
        for (let i = 0; i < bytes; i++) {
            out += Math.floor(Math.random() * 256).toString(16).padStart(2, '0');
        }
        return out;
    };
}
