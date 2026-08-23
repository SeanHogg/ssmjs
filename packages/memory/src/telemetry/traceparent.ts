/**
 * telemetry/traceparent.ts — W3C Trace Context propagation.
 *
 * A {@link Tracer} on its own produces one span tree per PROCESS. A cloud run is not
 * one process: portal → api → relay → runtime is four, and without propagation each
 * hop opens its own root trace, so the operator gets four unrelated trees and has to
 * correlate them by timestamp. Head-based propagation is what makes them one tree —
 * the caller serialises `{ traceId, spanId, sampled }` into a `traceparent` header,
 * the callee parses it and starts its root span as a CHILD of the remote span.
 *
 * The sampling flag travels with it on purpose: a sampling decision taken per-hop
 * would record hop 1 and drop hop 3, which is a tree with holes in it — the same
 * failure the in-process `_sampled` map exists to prevent, one network hop out.
 *
 * Pure string work, zero dependencies, no OpenTelemetry SDK — the wire format is a
 * 55-character ASCII string and re-implementing it costs less than the dependency.
 *
 * @see https://www.w3.org/TR/trace-context/
 */

/** A trace position received from (or sent to) another process. */
export interface TraceContext {
    /** 32 lowercase hex chars, never all-zero. */
    traceId: string;
    /** 16 lowercase hex chars, never all-zero — the REMOTE span, parent of ours. */
    spanId: string;
    /** The `sampled` flag (bit 0x01). False means the caller decided to drop the trace. */
    sampled: boolean;
}

/** The only trace-flag bit W3C defines today. */
const FLAG_SAMPLED = 0x01;

const TRACE_ID_LENGTH = 32;
const SPAN_ID_LENGTH = 16;

const HEX = /^[0-9a-f]+$/;

/** True when `value` is `length` lowercase hex chars and not entirely zeros. */
function isUsableId(value: string, length: number): boolean {
    return value.length === length && HEX.test(value) && /[1-9a-f]/.test(value);
}

/**
 * Parse a `traceparent` header, or `null` when it is absent or unusable.
 *
 * Rejecting rather than repairing is deliberate: a malformed header means the caller
 * is not actually in the trace we would be joining, and inventing a link to it
 * produces a tree that misattributes latency. `null` makes the callee start a fresh
 * root, which is at least true.
 *
 * Unknown future versions are accepted by reading the first four fields — that is
 * what the spec requires, so a later spec revision does not silently break every
 * hop. Version `ff` is invalid by definition and is rejected.
 */
export function parseTraceparent(header: string | null | undefined): TraceContext | null {
    if (typeof header !== 'string') return null;
    const parts = header.trim().toLowerCase().split('-');
    if (parts.length < 4) return null;

    const [version, traceId, spanId, flags] = parts as [string, string, string, string];
    if (version.length !== 2 || !HEX.test(version) || version === 'ff') return null;
    // Version 00 is exactly four fields; a longer header at version 00 is malformed.
    if (version === '00' && parts.length !== 4) return null;
    if (!isUsableId(traceId, TRACE_ID_LENGTH) || !isUsableId(spanId, SPAN_ID_LENGTH)) return null;
    if (flags.length !== 2 || !HEX.test(flags)) return null;

    return { traceId, spanId, sampled: (parseInt(flags, 16) & FLAG_SAMPLED) !== 0 };
}

/**
 * Serialise a trace position as a version-`00` `traceparent` header.
 *
 * Ids are normalised to the wire widths — left-padded when short, right-trimmed when
 * long — so a tracer configured with a non-standard id factory still emits a legal
 * header rather than one every downstream collector drops.
 */
export function formatTraceparent(ctx: TraceContext): string {
    const traceId = normaliseId(ctx.traceId, TRACE_ID_LENGTH);
    const spanId = normaliseId(ctx.spanId, SPAN_ID_LENGTH);
    const flags = (ctx.sampled ? FLAG_SAMPLED : 0).toString(16).padStart(2, '0');
    return `00-${traceId}-${spanId}-${flags}`;
}

/** Pad/trim an id to the wire width, dropping any non-hex byte to `0`. */
function normaliseId(value: string, length: number): string {
    const hex = (value ?? '').toLowerCase().replace(/[^0-9a-f]/g, '');
    return hex.length >= length ? hex.slice(0, length) : hex.padStart(length, '0');
}
