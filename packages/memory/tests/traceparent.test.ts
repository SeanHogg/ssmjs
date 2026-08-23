/**
 * tests/traceparent.test.ts
 * W3C Trace Context propagation — the parse/format pair plus the Tracer wiring that
 * makes portal → api → relay → runtime ONE span tree instead of four.
 */

import { formatTraceparent, parseTraceparent } from '../src/telemetry/traceparent.js';
import { Tracer } from '../src/telemetry/Tracer.js';

const TRACE_ID = '4bf92f3577b34da6a3ce929d0e0e4736';
const SPAN_ID = '00f067aa0ba902b7';

// ── parseTraceparent ───────────────────────────────────────────────────────────

test('parses the canonical W3C example, sampled', () => {
    expect(parseTraceparent(`00-${TRACE_ID}-${SPAN_ID}-01`)).toEqual({
        traceId: TRACE_ID,
        spanId: SPAN_ID,
        sampled: true,
    });
});

test('reads the sampled flag as a BIT, not as equality with 01', () => {
    // Flags 0x03 = sampled plus a future bit set. Comparing the byte to '01' would
    // read this as "not sampled" and silently drop a trace the caller recorded.
    expect(parseTraceparent(`00-${TRACE_ID}-${SPAN_ID}-03`)?.sampled).toBe(true);
    expect(parseTraceparent(`00-${TRACE_ID}-${SPAN_ID}-00`)?.sampled).toBe(false);
});

test('accepts an unknown future version by reading the first four fields', () => {
    expect(parseTraceparent(`01-${TRACE_ID}-${SPAN_ID}-01-extra`)?.traceId).toBe(TRACE_ID);
});

test('rejects version ff, which the spec defines as invalid', () => {
    expect(parseTraceparent(`ff-${TRACE_ID}-${SPAN_ID}-01`)).toBeNull();
});

test('rejects a version-00 header with extra fields', () => {
    expect(parseTraceparent(`00-${TRACE_ID}-${SPAN_ID}-01-extra`)).toBeNull();
});

test('rejects all-zero ids — they mean "no trace", not a trace of zeros', () => {
    expect(parseTraceparent(`00-${'0'.repeat(32)}-${SPAN_ID}-01`)).toBeNull();
    expect(parseTraceparent(`00-${TRACE_ID}-${'0'.repeat(16)}-01`)).toBeNull();
});

test('rejects wrong-width, non-hex, absent and malformed headers', () => {
    expect(parseTraceparent(`00-${TRACE_ID.slice(0, 30)}-${SPAN_ID}-01`)).toBeNull();
    expect(parseTraceparent(`00-${'z'.repeat(32)}-${SPAN_ID}-01`)).toBeNull();
    expect(parseTraceparent(`00-${TRACE_ID}-${SPAN_ID}-zz`)).toBeNull();
    expect(parseTraceparent('garbage')).toBeNull();
    expect(parseTraceparent(null)).toBeNull();
    expect(parseTraceparent(undefined)).toBeNull();
});

test('normalizes case and surrounding whitespace', () => {
    expect(parseTraceparent(`  00-${TRACE_ID.toUpperCase()}-${SPAN_ID.toUpperCase()}-01 `)).toEqual({
        traceId: TRACE_ID,
        spanId: SPAN_ID,
        sampled: true,
    });
});

// ── formatTraceparent ──────────────────────────────────────────────────────────

test('formats a version-00 header and round-trips through the parser', () => {
    const header = formatTraceparent({ traceId: TRACE_ID, spanId: SPAN_ID, sampled: true });
    expect(header).toBe(`00-${TRACE_ID}-${SPAN_ID}-01`);
    expect(parseTraceparent(header)).toEqual({ traceId: TRACE_ID, spanId: SPAN_ID, sampled: true });
});

test('clears the sampled bit when the trace was dropped', () => {
    expect(formatTraceparent({ traceId: TRACE_ID, spanId: SPAN_ID, sampled: false })).toBe(
        `00-${TRACE_ID}-${SPAN_ID}-00`,
    );
});

test('pads short ids to the wire widths so the header stays legal', () => {
    const header = formatTraceparent({ traceId: 'abc', spanId: 'def', sampled: true });
    expect(header).toBe(`00-${'abc'.padStart(32, '0')}-${'def'.padStart(16, '0')}-01`);
    expect(parseTraceparent(header)).not.toBeNull();
});

// ── Tracer wiring ──────────────────────────────────────────────────────────────

/** Sequential ids so span/trace identity is assertable. */
function seqIds() {
    let n = 0;
    return (bytes: number) => String(++n).padStart(bytes * 2, 'a');
}

test('a remote parent adopts its trace id and links to its span id', () => {
    const tracer = new Tracer({ newId: seqIds() });
    const span = tracer.startSpan('api.handle', {
        remoteParent: { traceId: TRACE_ID, spanId: SPAN_ID, sampled: true },
    });
    span.end();

    const [data] = tracer.finished();
    expect(data!.traceId).toBe(TRACE_ID);
    expect(data!.parentSpanId).toBe(SPAN_ID);
});

test('an in-process parent wins over a remote one', () => {
    const tracer = new Tracer({ newId: seqIds() });
    const root = tracer.startSpan('root');
    const child = tracer.startSpan('child', {
        parent: root,
        remoteParent: { traceId: TRACE_ID, spanId: SPAN_ID, sampled: true },
    });
    expect(child.traceId).toBe(root.traceId);
    expect(child.data.parentSpanId).toBe(root.spanId);
});

test('the upstream sampling verdict is obeyed, not re-rolled', () => {
    // sampleRate 1 would record everything locally; the upstream said drop.
    const tracer = new Tracer({ newId: seqIds(), sampleRate: 1 });
    const span = tracer.startSpan('api.handle', {
        remoteParent: { traceId: TRACE_ID, spanId: SPAN_ID, sampled: false },
    });
    expect(span.recorded).toBe(false);
    span.end();
    expect(tracer.finished()).toHaveLength(0);
});

test('an upstream-sampled trace is recorded even when local sampling would drop it', () => {
    const tracer = new Tracer({ newId: seqIds(), sampleRate: 0, random: () => 0.99 });
    const span = tracer.startSpan('api.handle', {
        remoteParent: { traceId: TRACE_ID, spanId: SPAN_ID, sampled: true },
    });
    expect(span.recorded).toBe(true);
    span.end();
    expect(tracer.finished()).toHaveLength(1);
});

test('the adopted verdict is inherited by later spans of the same trace', () => {
    const tracer = new Tracer({ newId: seqIds(), sampleRate: 1 });
    tracer.startSpan('first', { remoteParent: { traceId: TRACE_ID, spanId: SPAN_ID, sampled: false } }).end();
    // A SECOND root on the same trace id (a retry, a second handler) must not
    // reappear in the export just because it had no remote header this time.
    const second = tracer.startSpan('second', { traceId: TRACE_ID });
    expect(second.recorded).toBe(false);
});

test('span.traceparent() emits the header the next hop parses back', () => {
    const tracer = new Tracer({ newId: seqIds() });
    const span = tracer.startSpan('portal.request');
    const parsed = parseTraceparent(span.traceparent());
    expect(parsed).toEqual({ traceId: span.traceId, spanId: span.spanId, sampled: true });
});

test('a dropped span propagates the cleared sampled bit downstream', () => {
    const tracer = new Tracer({ newId: seqIds(), sampleRate: 0, random: () => 0.5 });
    const span = tracer.startSpan('portal.request');
    expect(parseTraceparent(span.traceparent())?.sampled).toBe(false);
});

test('four hops chained by header land in ONE trace', () => {
    const hop = (header?: string) => {
        const tracer = new Tracer({ newId: seqIds() });
        const remote = parseTraceparent(header);
        const span = tracer.startSpan('hop', { ...(remote ? { remoteParent: remote } : {}) });
        span.end();
        return { tracer, span };
    };
    const portal = hop();
    const api = hop(portal.span.traceparent());
    const relay = hop(api.span.traceparent());
    const runtime = hop(relay.span.traceparent());

    const ids = new Set([portal, api, relay, runtime].map((h) => h.span.traceId));
    expect(ids.size).toBe(1);
    expect(runtime.span.data.parentSpanId).toBe(relay.span.spanId);
});
