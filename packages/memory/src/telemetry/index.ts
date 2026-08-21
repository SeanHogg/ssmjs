/**
 * telemetry — granular tracing plus LLM-native metrics (tokens/sec, TTFT,
 * cost-per-request, cache-hit rate), built on ONE span stream so the trace view
 * and the metric view can never disagree.
 */

export { Tracer, Span } from './Tracer.js';
export type { TracerOptions, SpanOptions } from './Tracer.js';

export { MetricsRegistry, percentile } from './MetricsRegistry.js';
export type {
    LlmMetricSnapshot,
    KindMetricSnapshot,
    MetricsRegistryOptions,
} from './MetricsRegistry.js';

export { InstrumentedBridge } from './InstrumentedBridge.js';
export type { InstrumentedBridgeOptions } from './InstrumentedBridge.js';

export {
    InMemorySpanExporter,
    ConsoleSpanExporter,
    OtlpHttpSpanExporter,
    toOtlpPayload,
    toOtlpSpan,
} from './exporters.js';
export type {
    ConsoleSpanExporterOptions,
    OtlpHttpSpanExporterOptions,
} from './exporters.js';

export {
    estimateCostUsd,
    estimateTokens,
    resolveRate,
    DEFAULT_PRICE_BOOK,
    ANTHROPIC_PRICE_BOOK,
    EVERMIND_PRICE_BOOK,
    DEFAULT_CACHE_READ_MULTIPLIER,
    DEFAULT_CACHE_WRITE_MULTIPLIER,
} from './pricing.js';
export type { ModelRate, PriceBook } from './pricing.js';

export type {
    AttributeValue,
    SpanKind,
    SpanStatus,
    SpanEvent,
    SpanData,
    SpanExporter,
    LlmUsage,
    BridgeCallInfo,
} from './types.js';
