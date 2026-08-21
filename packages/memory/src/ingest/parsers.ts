/**
 * ingest/parsers.ts — format handlers, as a registry.
 *
 * Each parser turns a source into sections and recovers whatever STRUCTURE the
 * format carries, because structure that survives parsing becomes filterable
 * metadata, and metadata is what makes retrieval precise:
 *
 *   • Markdown keeps its heading path, so a chunk knows it lives under
 *     "Security › Data retention" and a citation can say so.
 *   • Rows keep every typed column, so `status = 'open'` is a database predicate
 *     rather than something the embedding has to imply.
 *   • JSON keeps its pointer, so a value can be traced back to its exact location.
 *
 * A parser that returned only text would throw all of that away and force the
 * retriever to recover it statistically — the single most common reason a RAG
 * pilot answers plausibly and cites nothing verifiable.
 */

import type { MetadataRecord, MetadataValue } from '../vectorstore/types.js';
import type { DocumentParser, ParsedSection, SourceDocument, StructuredRow } from './types.js';

const textOf = (source: SourceDocument): string =>
    source.content.kind === 'text' ? source.content.text : '';

const mediaTypeOf = (source: SourceDocument): string =>
    source.content.kind === 'text' ? (source.content.mediaType ?? 'text/plain').toLowerCase() : '';

// ── structured rows ───────────────────────────────────────────────────────────

/**
 * Serialises a row as `field: value` lines.
 *
 * Field names are kept in the text on purpose: an embedding of `"status: open"`
 * sits near a question about open items, whereas a bare `"open"` sits near
 * nothing useful. The typed values are ALSO emitted as metadata, so the same fact
 * is both semantically searchable and exactly filterable.
 */
export function serializeRow(row: StructuredRow): string {
    return Object.entries(row)
        .filter(([, v]) => v !== null && v !== undefined && v !== '')
        .map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join(', ') : String(v)}`)
        .join('\n');
}

export const rowsParser: DocumentParser = {
    id: 'rows',
    accepts: (source) => source.content.kind === 'rows',
    parse: (source) => {
        if (source.content.kind !== 'rows') return [];
        const { rows, rowIdField } = source.content;
        return rows.map((row, index) => {
            const metadata: MetadataRecord = { ...row };
            metadata['rowIndex'] = index;
            if (rowIdField && row[rowIdField] !== undefined) {
                metadata['rowId'] = String(row[rowIdField]);
            }
            return { text: serializeRow(row), metadata };
        });
    },
};

// ── CSV ───────────────────────────────────────────────────────────────────────

/**
 * RFC 4180 subset: quoted fields, doubled quotes, embedded newlines and commas.
 * Enough for the exports enterprises actually hand over, without a dependency.
 */
export function parseCsv(text: string, delimiter = ','): StructuredRow[] {
    const rows: string[][] = [];
    let field = '';
    let record: string[] = [];
    let quoted = false;

    for (let i = 0; i < text.length; i++) {
        const ch = text[i] as string;
        if (quoted) {
            if (ch === '"') {
                if (text[i + 1] === '"') { field += '"'; i++; }
                else quoted = false;
            } else field += ch;
            continue;
        }
        if (ch === '"') { quoted = true; continue; }
        if (ch === delimiter) { record.push(field); field = ''; continue; }
        if (ch === '\n' || ch === '\r') {
            if (ch === '\r' && text[i + 1] === '\n') i++;
            record.push(field);
            rows.push(record);
            record = [];
            field = '';
            continue;
        }
        field += ch;
    }
    if (field !== '' || record.length > 0) {
        record.push(field);
        rows.push(record);
    }

    const header = rows.shift();
    if (!header) return [];

    return rows
        .filter((r) => r.some((cell) => cell.trim() !== ''))
        .map((cells) => {
            const row: StructuredRow = {};
            header.forEach((name, i) => {
                row[name.trim()] = coerce(cells[i] ?? '');
            });
            return row;
        });
}

/**
 * Numbers and booleans are recovered from CSV text so range filters work.
 * A value that merely *looks* numeric but has a leading zero (an account number,
 * a zip code) stays a string — coercing it would silently destroy the identifier.
 */
function coerce(raw: string): MetadataValue {
    const value = raw.trim();
    if (value === '') return '';
    if (value === 'true') return true;
    if (value === 'false') return false;
    if (/^-?(0|[1-9]\d*)(\.\d+)?$/.test(value)) {
        const n = Number(value);
        if (Number.isFinite(n)) return n;
    }
    return value;
}

export const csvParser: DocumentParser = {
    id: 'csv',
    accepts: (source) => {
        const type = mediaTypeOf(source);
        return type.includes('csv') || type.includes('tab-separated');
    },
    parse: (source) => {
        const delimiter = mediaTypeOf(source).includes('tab-separated') ? '\t' : ',';
        const rows = parseCsv(textOf(source), delimiter);
        return rowsParser.parse({ ...source, content: { kind: 'rows', rows } });
    },
};

// ── JSON ──────────────────────────────────────────────────────────────────────

/**
 * An array of objects is treated as rows (the common export shape); anything else
 * is flattened to leaf paths so a value keeps its JSON pointer.
 */
export const jsonParser: DocumentParser = {
    id: 'json',
    accepts: (source) => mediaTypeOf(source).includes('json'),
    parse: (source) => {
        let parsed: unknown;
        try {
            parsed = JSON.parse(textOf(source));
        } catch {
            // Malformed JSON is still text worth indexing; degrading beats dropping
            // the document, and the parser name on the chunk records what happened.
            return plainTextParser.parse(source);
        }

        if (Array.isArray(parsed) && parsed.every((v) => v && typeof v === 'object' && !Array.isArray(v))) {
            return rowsParser.parse({
                ...source,
                content: { kind: 'rows', rows: parsed as StructuredRow[] },
            });
        }

        const leaves = flatten(parsed, '');
        if (leaves.length === 0) return [];
        return [{
            text: leaves.map(([path, value]) => `${path}: ${value}`).join('\n'),
            metadata: { leafCount: leaves.length },
        }];
    },
};

function flatten(value: unknown, path: string, out: Array<[string, string]> = []): Array<[string, string]> {
    if (value === null || typeof value !== 'object') {
        out.push([path || '/', String(value)]);
        return out;
    }
    if (Array.isArray(value)) {
        value.forEach((v, i) => flatten(v, `${path}/${i}`, out));
        return out;
    }
    for (const [key, v] of Object.entries(value as Record<string, unknown>)) {
        flatten(v, `${path}/${key}`, out);
    }
    return out;
}

// ── Markdown ──────────────────────────────────────────────────────────────────

export const markdownParser: DocumentParser = {
    id: 'markdown',
    accepts: (source) => {
        const type = mediaTypeOf(source);
        return type.includes('markdown') || type.includes('/md');
    },
    parse: (source) => {
        const lines = textOf(source).split('\n');
        const sections: ParsedSection[] = [];
        // Heading level -> text, so a chunk carries its full breadcrumb rather than
        // only its nearest heading.
        const path: string[] = [];
        let buffer: string[] = [];

        const flush = () => {
            const text = buffer.join('\n').trim();
            buffer = [];
            if (!text) return;
            const metadata: MetadataRecord = {};
            if (path.length) {
                metadata['headingPath'] = path.join(' › ');
                metadata['heading'] = path[path.length - 1] as string;
                metadata['headingDepth'] = path.length;
            }
            sections.push({ text, metadata });
        };

        let inFence = false;
        for (const line of lines) {
            if (/^\s*```/.test(line)) inFence = !inFence;
            const heading = inFence ? null : /^(#{1,6})\s+(.*)$/.exec(line);
            if (heading) {
                flush();
                const depth = (heading[1] as string).length;
                path.length = Math.min(path.length, depth - 1);
                path[depth - 1] = (heading[2] as string).trim();
                path.length = depth;
                continue;
            }
            buffer.push(line);
        }
        flush();
        return sections;
    },
};

// ── HTML ──────────────────────────────────────────────────────────────────────

export const htmlParser: DocumentParser = {
    id: 'html',
    accepts: (source) => mediaTypeOf(source).includes('html') || mediaTypeOf(source).includes('xml'),
    parse: (source) => {
        const raw = textOf(source);
        const titleMatch = /<title[^>]*>([\s\S]*?)<\/title>/i.exec(raw);

        const text = raw
            // script/style content is code, not prose — indexing it poisons recall.
            .replace(/<(script|style)[^>]*>[\s\S]*?<\/\1>/gi, ' ')
            .replace(/<!--[\s\S]*?-->/g, ' ')
            .replace(/<\/(p|div|section|article|li|tr|h[1-6])>/gi, '\n')
            .replace(/<br\s*\/?>/gi, '\n')
            .replace(/<[^>]+>/g, ' ')
            .replace(/&nbsp;/g, ' ')
            .replace(/&amp;/g, '&')
            .replace(/&lt;/g, '<')
            .replace(/&gt;/g, '>')
            .replace(/&quot;/g, '"')
            .replace(/&#39;/g, "'")
            .replace(/[ \t]+/g, ' ')
            .replace(/\n{3,}/g, '\n\n')
            .trim();

        if (!text) return [];
        const metadata: MetadataRecord = {};
        if (titleMatch?.[1]) metadata['htmlTitle'] = titleMatch[1].trim();
        return [{ text, metadata }];
    },
};

// ── plain text ────────────────────────────────────────────────────────────────

export const plainTextParser: DocumentParser = {
    id: 'text',
    accepts: (source) => source.content.kind === 'text',
    parse: (source) => {
        const text = textOf(source).trim();
        return text ? [{ text }] : [];
    },
};

// ── registry ──────────────────────────────────────────────────────────────────

/** Order matters: the first parser that accepts a source handles it. */
export const BUILTIN_PARSERS: readonly DocumentParser[] = [
    rowsParser,
    csvParser,
    jsonParser,
    markdownParser,
    htmlParser,
    plainTextParser,
];

export class ParserRegistry {
    private readonly _parsers: DocumentParser[];

    constructor(parsers: readonly DocumentParser[] = BUILTIN_PARSERS) {
        this._parsers = [...parsers];
    }

    /** Registers a parser at the FRONT, so it can override a builtin. */
    register(parser: DocumentParser): this {
        this._parsers.unshift(parser);
        return this;
    }

    resolve(source: SourceDocument): DocumentParser | undefined {
        return this._parsers.find((p) => p.accepts(source));
    }

    list(): string[] {
        return this._parsers.map((p) => p.id);
    }
}
