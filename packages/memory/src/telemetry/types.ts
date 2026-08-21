/**
 * telemetry/types.ts — the observability vocabulary shared by every enterprise
 * subsystem (ingest, retrieval, orchestration, eval, bridges).
 *
 * Enterprise buyers do not ask "is it fast", they ask "show me the span tree for
 * this request, the token count per model, and what it cost". That needs one
 * canonical record, not per-subsystem logging. A {@link SpanData} is that record:
 * a tracing span that ALSO carries LLM-native usage and cost, so a single
 * exporter pipeline feeds both a trace viewer and the metric aggregates.
 *
 * The shape is deliberately OTLP-compatible (trace/span ids, parent linkage,
 * attributes, events, status) so `toOtlpSpan()` is a pure rename rather than a
 * re-model — but nothing here depends on the OpenTelemetry SDK, which keeps the
 * package zero-dependency and browser-safe.
 */

/** Attribute values are kept primitive so a span serializes to JSON losslessly. */
export type AttributeValue = string | number | boolean;

export type SpanKind =
    | 'llm'
    | 'embedding'
    | 'retrieval'
    | 'ingest'
    | 'agent'
    | 'tool'
    | 'graph'
    | 'internal';

export type SpanStatus = 'unset' | 'ok' | 'error';

export interface SpanEvent {
    name: string;
    /** Epoch millis. */
    at: number;
    attributes?: Record<string, AttributeValue>;
}

/**
 * Token accounting for one model call.
 *
 * `cachedInputTokens` and `cacheWriteTokens` are split out because they are
 * billed at different multipliers than plain input (see {@link PriceBook}); a
 * cost model that folds them into `inputTokens` overstates spend by ~10x on a
 * cache-heavy workload, which is exactly the workload enterprises run.
 */
export interface LlmUsage {
    model: string;
    inputTokens: number;
    outputTokens: number;
    /** Input tokens served from the provider's prompt cache (billed at a discount). */
    cachedInputTokens?: number;
    /** Input tokens written INTO the provider's prompt cache (billed at a premium). */
    cacheWriteTokens?: number;
    /** True when the whole call was served locally (exact/semantic cache) — cost 0. */
    localCacheHit?: boolean;
    /** True when token counts are inferred rather than reported by the provider. */
    estimated?: boolean;
}

/** A completed (or in-flight) span, ready to export. */
export interface SpanData {
    traceId: string;
    spanId: string;
    parentSpanId?: string;
    name: string;
    kind: SpanKind;
    /** Epoch millis. */
    startedAt: number;
    /** Epoch millis; absent while the span is still open. */
    endedAt?: number;
    durationMs?: number;
    attributes: Record<string, AttributeValue>;
    events: SpanEvent[];
    status: SpanStatus;
    error?: string;
    /** Present on `llm` spans. */
    usage?: LlmUsage;
    /** Derived from `usage` + the tracer's price book, in USD. */
    costUsd?: number;
    /** Millis from request start to the first streamed token, when streaming. */
    timeToFirstTokenMs?: number;
}

/** Sink for finished spans. Implementations must not throw into the caller. */
export interface SpanExporter {
    export(spans: readonly SpanData[]): void | Promise<void>;
    /** Optional flush hook for buffered/remote exporters. */
    shutdown?(): void | Promise<void>;
}

/**
 * A bridge that reports what its last call actually consumed.
 *
 * Optional on {@link TransformerBridge} so third-party bridges stay valid; when
 * present, {@link InstrumentedBridge} prefers these provider-reported numbers
 * over its own character-based estimate.
 */
export interface BridgeCallInfo {
    usage?: LlmUsage;
    /** True when the call never reached the provider (exact or semantic cache hit). */
    cacheHit?: boolean;
    /** Which cache tier served it, when `cacheHit`. */
    cacheTier?: string;
}
