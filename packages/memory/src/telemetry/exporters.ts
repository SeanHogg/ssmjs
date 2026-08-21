/**
 * telemetry/exporters.ts — where finished spans go.
 *
 * Three sinks cover the deployment spectrum without a vendor dependency:
 *   • {@link InMemorySpanExporter} — tests and in-process dashboards.
 *   • {@link ConsoleSpanExporter}  — local development.
 *   • {@link OtlpHttpSpanExporter} — any OTLP/HTTP collector, which includes
 *     Google Cloud Trace, Datadog, Honeycomb, Grafana Tempo and the OpenTelemetry
 *     Collector. It speaks the protocol over `fetch` rather than importing the
 *     OpenTelemetry SDK, so the package stays zero-dependency and still runs in a
 *     browser or a Cloudflare Worker where the Node SDK cannot.
 */

import type { AttributeValue, SpanData, SpanExporter } from './types.js';

export class InMemorySpanExporter implements SpanExporter {
    private readonly _spans: SpanData[] = [];

    export(spans: readonly SpanData[]): void {
        this._spans.push(...spans);
    }

    get spans(): readonly SpanData[] { return this._spans; }

    /** Spans belonging to one trace, oldest first. */
    trace(traceId: string): SpanData[] {
        return this._spans.filter((s) => s.traceId === traceId);
    }

    reset(): void { this._spans.length = 0; }
}

export interface ConsoleSpanExporterOptions {
    /** Sink for the rendered lines. Default `console.log`. */
    write?: (line: string) => void;
    /** Include span attributes. Default false — one line per span. */
    verbose?: boolean;
}

export class ConsoleSpanExporter implements SpanExporter {
    private readonly _write: (line: string) => void;
    private readonly _verbose: boolean;

    constructor(opts: ConsoleSpanExporterOptions = {}) {
        this._write = opts.write ?? ((line) => console.log(line));
        this._verbose = opts.verbose ?? false;
    }

    export(spans: readonly SpanData[]): void {
        for (const span of spans) {
            const cost = span.costUsd !== undefined ? ` $${span.costUsd.toFixed(6)}` : '';
            const tokens = span.usage ? ` ${span.usage.inputTokens}in/${span.usage.outputTokens}out` : '';
            const status = span.status === 'error' ? ` ERROR: ${span.error ?? ''}` : '';
            this._write(
                `[${span.traceId.slice(0, 8)}] ${span.kind.padEnd(9)} ${span.name} ` +
                `${Math.round(span.durationMs ?? 0)}ms${tokens}${cost}${status}`,
            );
            if (this._verbose && Object.keys(span.attributes).length > 0) {
                this._write(`         ${JSON.stringify(span.attributes)}`);
            }
        }
    }
}

export interface OtlpHttpSpanExporterOptions {
    /** Collector endpoint, e.g. `https://otlp.example.com/v1/traces`. */
    url: string;
    /** Extra headers (auth token, GCP project header, …). */
    headers?: Record<string, string>;
    /** `service.name` reported on the resource. Default `evermind`. */
    serviceName?: string;
    /** Injected for test. Default global `fetch`. */
    fetchImpl?: typeof fetch;
    /**
     * Called when an export request fails. Default: swallow. Telemetry must never
     * take down the request it is measuring, so failures are reported, not thrown.
     */
    onError?: (err: unknown) => void;
}

/** OTLP span kind codes: UNSPECIFIED 0, INTERNAL 1, SERVER 2, CLIENT 3. */
const OTLP_CLIENT = 3;
const OTLP_INTERNAL = 1;

export class OtlpHttpSpanExporter implements SpanExporter {
    private readonly _url: string;
    private readonly _headers: Record<string, string>;
    private readonly _serviceName: string;
    private readonly _fetch: typeof fetch;
    private readonly _onError: (err: unknown) => void;

    constructor(opts: OtlpHttpSpanExporterOptions) {
        this._url = opts.url;
        this._headers = opts.headers ?? {};
        this._serviceName = opts.serviceName ?? 'evermind';
        this._fetch = opts.fetchImpl ?? ((...args) => fetch(...args));
        this._onError = opts.onError ?? (() => {});
    }

    async export(spans: readonly SpanData[]): Promise<void> {
        if (spans.length === 0) return;
        try {
            const res = await this._fetch(this._url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', ...this._headers },
                body: JSON.stringify(toOtlpPayload(spans, this._serviceName)),
            });
            if (!res.ok) this._onError(new Error(`OTLP export failed: ${res.status}`));
        } catch (err) {
            this._onError(err);
        }
    }
}

/** Wraps spans in the OTLP/HTTP JSON envelope. */
export function toOtlpPayload(spans: readonly SpanData[], serviceName = 'evermind'): unknown {
    return {
        resourceSpans: [{
            resource: { attributes: [otlpAttr('service.name', serviceName)] },
            scopeSpans: [{
                scope: { name: '@seanhogg/builderforce-memory' },
                spans: spans.map(toOtlpSpan),
            }],
        }],
    };
}

/**
 * Converts one span to the OTLP wire shape. The mapping is a rename, not a
 * re-model, because {@link SpanData} was designed against this schema.
 *
 * LLM attributes follow the OpenTelemetry GenAI semantic conventions
 * (`gen_ai.*`) so a stock trace viewer renders token counts without config.
 */
export function toOtlpSpan(span: SpanData): unknown {
    const attributes = Object.entries(span.attributes).map(([k, v]) => otlpAttr(k, v));

    if (span.usage) {
        attributes.push(
            otlpAttr('gen_ai.request.model', span.usage.model),
            otlpAttr('gen_ai.usage.input_tokens', span.usage.inputTokens),
            otlpAttr('gen_ai.usage.output_tokens', span.usage.outputTokens),
        );
        if (span.usage.cachedInputTokens !== undefined) {
            attributes.push(otlpAttr('gen_ai.usage.cached_input_tokens', span.usage.cachedInputTokens));
        }
    }
    if (span.costUsd !== undefined) attributes.push(otlpAttr('gen_ai.usage.cost_usd', span.costUsd));
    if (span.timeToFirstTokenMs !== undefined) {
        attributes.push(otlpAttr('gen_ai.server.time_to_first_token_ms', span.timeToFirstTokenMs));
    }

    const out: Record<string, unknown> = {
        traceId: span.traceId,
        spanId: span.spanId,
        name: span.name,
        kind: span.kind === 'llm' || span.kind === 'embedding' ? OTLP_CLIENT : OTLP_INTERNAL,
        startTimeUnixNano: msToNano(span.startedAt),
        endTimeUnixNano: msToNano(span.endedAt ?? span.startedAt),
        attributes,
        events: span.events.map((e) => ({
            name: e.name,
            timeUnixNano: msToNano(e.at),
            attributes: Object.entries(e.attributes ?? {}).map(([k, v]) => otlpAttr(k, v)),
        })),
        status: span.status === 'error'
            ? { code: 2, message: span.error ?? '' }
            : { code: span.status === 'ok' ? 1 : 0 },
    };
    if (span.parentSpanId) out['parentSpanId'] = span.parentSpanId;
    return out;
}

function otlpAttr(key: string, value: AttributeValue): unknown {
    if (typeof value === 'number') {
        return Number.isInteger(value)
            ? { key, value: { intValue: value } }
            : { key, value: { doubleValue: value } };
    }
    if (typeof value === 'boolean') return { key, value: { boolValue: value } };
    return { key, value: { stringValue: value } };
}

/** OTLP timestamps are nanoseconds since the epoch, as a decimal string. */
function msToNano(ms: number): string {
    return `${Math.round(ms)}000000`;
}
