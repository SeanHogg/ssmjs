/**
 * VertexAIBridge — TransformerBridge for Google Cloud Vertex AI.
 *
 * GCP is where a large share of enterprise AI actually runs, usually because the
 * customer's data already lives in BigQuery and Cloud Storage and moving it is a
 * procurement problem, not a technical one. This bridge lets Evermind serve those
 * deployments without changing anything above the port.
 *
 * Two deliberate choices:
 *
 *   • **Auth is injected, never owned.** `getAccessToken` is supplied by the
 *     caller, which is what lets the same code run under Application Default
 *     Credentials locally, a Workload Identity service account on GKE, and an
 *     impersonated token in CI — and keeps this package free of the Google auth
 *     library (and of any credential lifecycle it has no business managing).
 *   • **Raw HTTP, like every other bridge here.** The package is zero-dependency
 *     and runs in a browser, a Worker and Node; adding a Node-only vendor SDK to
 *     one bridge would cost that portability for the whole package.
 *
 * Usage is reported from `usageMetadata`, so cost accounting on Vertex is measured
 * rather than estimated — with the caveat that Vertex is partner-billed at rates
 * that differ from the first-party API, so supply a matching `PriceBook`.
 */

import { SSMError } from '../errors/SSMError.js';
import type { LlmUsage } from '../telemetry/types.js';
import type { TransformerBridge, BridgeGenerateOptions, BridgeCallInfo } from './TransformerBridge.js';

export interface VertexAIBridgeOptions {
    /** GCP project id. */
    projectId: string;
    /** Region, or `'global'`. Default `'us-central1'`. */
    location?: string;
    /**
     * Publisher model id, e.g. `'gemini-2.5-pro'`, or a Claude model served through
     * Vertex. Default `'gemini-2.5-flash'`.
     */
    model?: string;
    /** Publisher owning the model. Default `'google'`; use `'anthropic'` for Claude. */
    publisher?: string;
    /**
     * Returns a current OAuth access token. Called per request, so the caller's
     * token source owns caching and refresh.
     */
    getAccessToken: () => Promise<string> | string;
    systemPrompt?: string;
    maxTokens?: number;
    /** Override the API host — for a private endpoint or a test double. */
    baseUrl?: string;
    fetchImpl?: typeof fetch;
}

export class VertexAIBridge implements TransformerBridge {
    readonly supportsStreaming = true as const;

    private readonly _projectId: string;
    private readonly _location: string;
    private readonly _model: string;
    private readonly _publisher: string;
    private readonly _getToken: () => Promise<string> | string;
    private readonly _systemPrompt: string;
    private readonly _maxTokens: number;
    private readonly _baseUrl: string;
    private readonly _fetch: typeof fetch;

    private _lastCall: BridgeCallInfo | undefined;

    constructor(opts: VertexAIBridgeOptions) {
        this._projectId = opts.projectId;
        this._location = opts.location ?? 'us-central1';
        this._model = opts.model ?? 'gemini-2.5-flash';
        this._publisher = opts.publisher ?? 'google';
        this._getToken = opts.getAccessToken;
        this._systemPrompt = opts.systemPrompt ?? '';
        this._maxTokens = opts.maxTokens ?? 1024;
        this._baseUrl = opts.baseUrl ?? defaultHost(this._location);
        this._fetch = opts.fetchImpl ?? ((...args) => fetch(...args));
    }

    get lastCall(): BridgeCallInfo | undefined {
        return this._lastCall;
    }

    async generate(prompt: string, opts: BridgeGenerateOptions = {}): Promise<string> {
        const res = await this._send(prompt, opts, false);
        const json = await res.json() as Record<string, unknown>;

        const text = readCandidateText(json);
        if (text === undefined) {
            throw new SSMError('BRIDGE_RESPONSE_INVALID', 'Unexpected Vertex AI response shape.');
        }

        this._lastCall = { usage: readVertexUsage(json['usageMetadata'], opts.model ?? this._model) };
        return text;
    }

    async *stream(prompt: string, opts: BridgeGenerateOptions = {}): AsyncIterable<string> {
        const res = await this._send(prompt, opts, true);
        if (!res.body) {
            throw new SSMError('BRIDGE_RESPONSE_INVALID', 'Vertex AI streaming response has no body.');
        }

        let usage: LlmUsage | undefined;
        for await (const event of parseSseJson(res.body)) {
            const text = readCandidateText(event);
            if (text) yield text;
            // Vertex repeats cumulative usage on each chunk; the last one wins.
            const reported = readVertexUsage(event['usageMetadata'], opts.model ?? this._model);
            if (reported) usage = reported;
        }
        this._lastCall = usage ? { usage } : {};
    }

    private async _send(prompt: string, opts: BridgeGenerateOptions, stream: boolean): Promise<Response> {
        const model = opts.model ?? this._model;
        const method = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
        const url =
            `${this._baseUrl}/v1/projects/${this._projectId}/locations/${this._location}` +
            `/publishers/${this._publisher}/models/${model}:${method}`;

        const system = opts.systemPrompt ?? this._systemPrompt;
        const body: Record<string, unknown> = {
            contents: [{ role: 'user', parts: [{ text: prompt }] }],
            generationConfig: {
                maxOutputTokens: opts.maxTokens ?? this._maxTokens,
                temperature: opts.temperature ?? 0.7,
                topP: opts.topP ?? 0.9,
            },
        };
        if (system) body['systemInstruction'] = { parts: [{ text: system }] };

        const token = await this._getToken();
        const res = await this._fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`,
            },
            body: JSON.stringify(body),
        });

        if (!res.ok) {
            const detail = await res.text().catch(() => '');
            throw new SSMError(
                'BRIDGE_REQUEST_FAILED',
                `Vertex AI returned ${res.status}: ${detail}`,
            );
        }
        return res;
    }
}

/**
 * Vertex AI text embeddings — the `:predict` surface on a publisher model.
 *
 * Exposed as a {@link BatchEmbedder}-shaped function so it drops straight into the
 * ingestion pipeline: on GCP the embedder, the vector index and the generator can
 * all be Vertex, and nothing above the ports changes.
 */
export interface VertexAIEmbedderOptions {
    projectId: string;
    location?: string;
    /** Embedding model. Default `'text-embedding-005'`. */
    model?: string;
    getAccessToken: () => Promise<string> | string;
    /**
     * Task type hint. `RETRIEVAL_DOCUMENT` when indexing, `RETRIEVAL_QUERY` when
     * searching — using the same hint for both measurably degrades recall, because
     * the model embeds questions and passages into deliberately different regions.
     */
    taskType?: 'RETRIEVAL_DOCUMENT' | 'RETRIEVAL_QUERY' | 'SEMANTIC_SIMILARITY' | 'CLASSIFICATION';
    baseUrl?: string;
    fetchImpl?: typeof fetch;
}

export class VertexAIEmbedder {
    private readonly _projectId: string;
    private readonly _location: string;
    private readonly _model: string;
    private readonly _getToken: () => Promise<string> | string;
    private readonly _taskType: string | undefined;
    private readonly _baseUrl: string;
    private readonly _fetch: typeof fetch;

    constructor(opts: VertexAIEmbedderOptions) {
        this._projectId = opts.projectId;
        this._location = opts.location ?? 'us-central1';
        this._model = opts.model ?? 'text-embedding-005';
        this._getToken = opts.getAccessToken;
        this._taskType = opts.taskType;
        this._baseUrl = opts.baseUrl ?? defaultHost(this._location);
        this._fetch = opts.fetchImpl ?? ((...args) => fetch(...args));
    }

    /** Embeds a batch. Shape matches `BatchEmbedder`, for the ingestion pipeline. */
    embedBatch = async (texts: string[]): Promise<Float32Array[]> => {
        if (texts.length === 0) return [];

        const url =
            `${this._baseUrl}/v1/projects/${this._projectId}/locations/${this._location}` +
            `/publishers/google/models/${this._model}:predict`;

        const token = await this._getToken();
        const res = await this._fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
            body: JSON.stringify({
                instances: texts.map((content) => ({
                    content,
                    ...(this._taskType ? { task_type: this._taskType } : {}),
                })),
            }),
        });

        if (!res.ok) {
            const detail = await res.text().catch(() => '');
            throw new SSMError('BRIDGE_REQUEST_FAILED', `Vertex AI embeddings returned ${res.status}: ${detail}`);
        }

        const json = await res.json() as { predictions?: Array<{ embeddings?: { values?: number[] } }> };
        const predictions = json.predictions ?? [];
        if (predictions.length !== texts.length) {
            // A short batch would silently misalign vectors with their chunks — the
            // worst kind of RAG bug, because retrieval still "works" and is wrong.
            throw new SSMError(
                'BRIDGE_RESPONSE_INVALID',
                `Vertex AI returned ${predictions.length} embeddings for ${texts.length} inputs.`,
            );
        }
        return predictions.map((p) => Float32Array.from(p.embeddings?.values ?? []));
    };

    /** Single-text convenience, for the retriever's query embedding. */
    embed = async (text: string): Promise<Float32Array> => {
        const [vector] = await this.embedBatch([text]);
        return vector ?? new Float32Array();
    };
}

function defaultHost(location: string): string {
    return location === 'global'
        ? 'https://aiplatform.googleapis.com'
        : `https://${location}-aiplatform.googleapis.com`;
}

function readCandidateText(json: Record<string, unknown>): string | undefined {
    const candidates = json['candidates'];
    if (!Array.isArray(candidates) || candidates.length === 0) return undefined;
    const parts = (candidates[0] as { content?: { parts?: Array<{ text?: string }> } }).content?.parts;
    if (!Array.isArray(parts)) return undefined;
    // Multi-part responses are concatenated; taking only parts[0] silently truncates.
    return parts.map((p) => p.text ?? '').join('');
}

function readVertexUsage(usage: unknown, model: string): LlmUsage | undefined {
    if (!usage || typeof usage !== 'object') return undefined;
    const u = usage as Record<string, unknown>;
    const cached = Number(u['cachedContentTokenCount']) || 0;
    const prompt = Number(u['promptTokenCount']) || 0;
    return {
        model,
        inputTokens: Math.max(0, prompt - cached),
        outputTokens: Number(u['candidatesTokenCount']) || 0,
        cachedInputTokens: cached,
    };
}

/** Parses an `alt=sse` stream into JSON events. */
async function* parseSseJson(body: ReadableStream<Uint8Array>): AsyncIterable<Record<string, unknown>> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split('\n');
            buffer = lines.pop() as string;
            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed.startsWith('data:')) continue;
                const payload = trimmed.slice(5).trim();
                if (!payload || payload === '[DONE]') continue;
                try {
                    yield JSON.parse(payload) as Record<string, unknown>;
                } catch {
                    // Skip malformed SSE frames rather than failing the stream.
                }
            }
        }
    } finally {
        reader.releaseLock();
    }
}
