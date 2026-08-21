import { afterEach, describe, expect, it, jest } from '@jest/globals';

import { AnthropicBridge } from '../src/bridges/AnthropicBridge.js';
import { OpenAIBridge } from '../src/bridges/OpenAIBridge.js';
import { CachingBridge } from '../src/bridges/CachingBridge.js';
import { SemanticCachingBridge } from '../src/bridges/SemanticCachingBridge.js';
import { VertexAIBridge, VertexAIEmbedder } from '../src/bridges/VertexAIBridge.js';
import { SSMError } from '../src/errors/SSMError.js';
import type { TransformerBridge } from '../src/bridges/TransformerBridge.js';

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

function mockFetch(payload: unknown, status = 200) {
    const impl = jest.fn(async () => new Response(JSON.stringify(payload), { status }));
    globalThis.fetch = impl as unknown as typeof fetch;
    return impl;
}

function sseResponse(frames: string[]): Response {
    const body = new ReadableStream<Uint8Array>({
        start(controller) {
            const encoder = new TextEncoder();
            for (const frame of frames) controller.enqueue(encoder.encode(frame));
            controller.close();
        },
    });
    return new Response(body, { status: 200 });
}

describe('AnthropicBridge usage capture', () => {
    it('splits input, cache read and cache write from the response', async () => {
        mockFetch({
            model: 'claude-haiku-4-5',
            content: [{ text: 'hello' }],
            usage: {
                input_tokens: 100,
                output_tokens: 20,
                cache_read_input_tokens: 4000,
                cache_creation_input_tokens: 500,
            },
        });

        const bridge = new AnthropicBridge({ apiKey: 'k' });
        expect(await bridge.generate('hi')).toBe('hello');

        expect(bridge.lastCall?.usage).toEqual({
            model: 'claude-haiku-4-5',
            inputTokens: 100,
            outputTokens: 20,
            cachedInputTokens: 4000,
            cacheWriteTokens: 500,
        });
    });

    it('reports no usage when the response omits it', async () => {
        mockFetch({ content: [{ text: 'hello' }] });
        const bridge = new AnthropicBridge({ apiKey: 'k', model: 'claude-opus-5' });
        await bridge.generate('hi');
        expect(bridge.lastCall?.usage).toBeUndefined();
    });

    it('folds message_start and message_delta usage while streaming', async () => {
        globalThis.fetch = (jest.fn(async () => sseResponse([
            'data: {"type":"message_start","message":{"model":"claude-opus-5","usage":{"input_tokens":50,"cache_read_input_tokens":10}}}\n',
            'data: {"type":"content_block_delta","delta":{"text":"one "}}\n',
            'data: {"type":"content_block_delta","delta":{"text":"two"}}\n',
            'data: {"type":"message_delta","usage":{"output_tokens":7}}\n',
        ])) as unknown) as typeof fetch;

        const bridge = new AnthropicBridge({ apiKey: 'k' });
        let text = '';
        for await (const chunk of bridge.stream('hi')) text += chunk;

        expect(text).toBe('one two');
        expect(bridge.lastCall?.usage).toEqual({
            model: 'claude-opus-5',
            inputTokens: 50,
            outputTokens: 7,
            cachedInputTokens: 10,
            cacheWriteTokens: 0,
        });
    });
});

describe('OpenAIBridge usage capture', () => {
    it('keeps cached and fresh prompt tokens disjoint', async () => {
        mockFetch({
            model: 'gpt-test',
            choices: [{ message: { content: 'hi' } }],
            usage: { prompt_tokens: 1000, completion_tokens: 25, prompt_tokens_details: { cached_tokens: 900 } },
        });

        const bridge = new OpenAIBridge({ apiKey: 'k' });
        await bridge.generate('hello');

        // OpenAI reports prompt_tokens INCLUSIVE of cached; the canonical shape
        // must not double-count them.
        expect(bridge.lastCall?.usage).toEqual({
            model: 'gpt-test',
            inputTokens: 100,
            outputTokens: 25,
            cachedInputTokens: 900,
        });
    });

    it('requests usage on streamed calls and reads it back', async () => {
        const impl = jest.fn(async () => sseResponse([
            'data: {"choices":[{"delta":{"content":"a"}}]}\n',
            'data: {"choices":[{"delta":{"content":"b"}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n',
            'data: [DONE]\n',
        ]));
        globalThis.fetch = impl as unknown as typeof fetch;

        const bridge = new OpenAIBridge({ apiKey: 'k', model: 'gpt-test' });
        let text = '';
        for await (const chunk of bridge.stream('hi')) text += chunk;

        expect(text).toBe('ab');
        const body = JSON.parse((impl.mock.calls[0]?.[1] as RequestInit).body as string);
        expect(body.stream_options).toEqual({ include_usage: true });
        expect(bridge.lastCall?.usage).toMatchObject({ inputTokens: 5, outputTokens: 2 });
    });

    it('omits stream_options on a non-streaming request', async () => {
        const impl = mockFetch({ choices: [{ message: { content: 'x' } }] });
        await new OpenAIBridge({ apiKey: 'k' }).generate('hi');
        const body = JSON.parse((impl.mock.calls[0]?.[1] as RequestInit).body as string);
        expect(body.stream_options).toBeUndefined();
    });
});

describe('caching bridges declare their hits', () => {
    function counting(reply: string): TransformerBridge & { calls: number } {
        let calls = 0;
        return {
            supportsStreaming: false,
            get calls() { return calls; },
            lastCall: { usage: { model: 'inner-model', inputTokens: 10, outputTokens: 5 } },
            async generate() { calls += 1; return reply; },
        };
    }

    it('CachingBridge reports a miss then an exact hit', async () => {
        const inner = counting('answer');
        const bridge = new CachingBridge(inner);

        await bridge.generate('same prompt', { model: 'm' });
        expect(bridge.lastCall).toMatchObject({ cacheHit: false });
        // A miss passes the inner bridge's measured usage through unchanged.
        expect(bridge.lastCall?.usage?.model).toBe('inner-model');

        await bridge.generate('same prompt', { model: 'm' });
        expect(inner.calls).toBe(1);
        expect(bridge.lastCall).toMatchObject({ cacheHit: true, cacheTier: 'exact' });
        expect(bridge.lastCall?.usage?.localCacheHit).toBe(true);
    });

    it('SemanticCachingBridge reports which tier answered', async () => {
        const inner = counting('answer');
        // Identical vectors for anything containing "retention" — a stand-in for
        // an on-device embedder, so the paraphrase hit is deterministic.
        const embed = async (t: string) =>
            Float32Array.from(t.toLowerCase().includes('retention') ? [1, 0] : [0, 1]);

        const bridge = new SemanticCachingBridge(inner, { embed, threshold: 0.9 });

        await bridge.generate('what is the retention policy');
        expect(bridge.lastCall).toMatchObject({ cacheHit: false });

        await bridge.generate('tell me about retention rules');
        expect(inner.calls).toBe(1);
        expect(bridge.lastCall?.cacheHit).toBe(true);
        expect(bridge.lastCall?.cacheTier).toBe('l1');
        expect(bridge.lastCall?.usage?.localCacheHit).toBe(true);
    });
});

describe('VertexAIBridge', () => {
    const opts = { projectId: 'p', getAccessToken: () => 'token' };

    it('posts to the regional endpoint with a bearer token and system instruction', async () => {
        const impl = jest.fn(async () => new Response(JSON.stringify({
            candidates: [{ content: { parts: [{ text: 'part one ' }, { text: 'part two' }] } }],
            usageMetadata: { promptTokenCount: 30, candidatesTokenCount: 8, cachedContentTokenCount: 10 },
        }), { status: 200 }));

        const bridge = new VertexAIBridge({
            ...opts, location: 'europe-west1', model: 'gemini-2.5-pro',
            systemPrompt: 'be terse', fetchImpl: impl as unknown as typeof fetch,
        });

        // Multi-part responses must be concatenated, not truncated to parts[0].
        expect(await bridge.generate('hello')).toBe('part one part two');

        const [url, init] = impl.mock.calls[0] as [string, RequestInit];
        expect(url).toBe(
            'https://europe-west1-aiplatform.googleapis.com/v1/projects/p/locations/europe-west1' +
            '/publishers/google/models/gemini-2.5-pro:generateContent',
        );
        expect((init.headers as Record<string, string>)['Authorization']).toBe('Bearer token');
        expect(JSON.parse(init.body as string).systemInstruction.parts[0].text).toBe('be terse');

        expect(bridge.lastCall?.usage).toEqual({
            model: 'gemini-2.5-pro',
            inputTokens: 20,          // promptTokenCount minus cached
            outputTokens: 8,
            cachedInputTokens: 10,
        });
    });

    it('uses the global host when the location is global', async () => {
        const impl = jest.fn(async () => new Response(JSON.stringify({
            candidates: [{ content: { parts: [{ text: 'x' }] } }],
        }), { status: 200 }));
        const bridge = new VertexAIBridge({ ...opts, location: 'global', fetchImpl: impl as unknown as typeof fetch });
        await bridge.generate('hi');
        expect((impl.mock.calls[0] as [string, RequestInit])[0]).toContain('https://aiplatform.googleapis.com/');
    });

    it('raises a typed error on a non-2xx response and on an unexpected shape', async () => {
        const failing = jest.fn(async () => new Response('permission denied', { status: 403 }));
        const bridge = new VertexAIBridge({ ...opts, fetchImpl: failing as unknown as typeof fetch });
        await expect(bridge.generate('hi')).rejects.toBeInstanceOf(SSMError);

        const odd = jest.fn(async () => new Response(JSON.stringify({ candidates: [] }), { status: 200 }));
        const oddBridge = new VertexAIBridge({ ...opts, fetchImpl: odd as unknown as typeof fetch });
        await expect(oddBridge.generate('hi')).rejects.toThrow(/Unexpected Vertex AI response shape/);
    });

    it('streams SSE frames and keeps the last cumulative usage', async () => {
        const impl = jest.fn(async () => sseResponse([
            'data: {"candidates":[{"content":{"parts":[{"text":"one "}]}}]}\n',
            'data: not-json\n',
            'data: {"candidates":[{"content":{"parts":[{"text":"two"}]}}],' +
            '"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":4}}\n',
            'data: [DONE]\n',
        ]));
        const bridge = new VertexAIBridge({ ...opts, fetchImpl: impl as unknown as typeof fetch });

        let text = '';
        for await (const chunk of bridge.stream('hi')) text += chunk;

        expect(text).toBe('one two');
        expect(bridge.lastCall?.usage).toMatchObject({ inputTokens: 12, outputTokens: 4 });
        expect((impl.mock.calls[0] as [string, RequestInit])[0]).toContain(':streamGenerateContent?alt=sse');
    });
});

describe('VertexAIEmbedder', () => {
    it('embeds a batch and passes the task type', async () => {
        const impl = jest.fn(async () => new Response(JSON.stringify({
            predictions: [{ embeddings: { values: [1, 2] } }, { embeddings: { values: [3, 4] } }],
        }), { status: 200 }));

        const embedder = new VertexAIEmbedder({
            projectId: 'p', getAccessToken: async () => 'tok',
            taskType: 'RETRIEVAL_DOCUMENT', fetchImpl: impl as unknown as typeof fetch,
        });

        const vectors = await embedder.embedBatch(['a', 'b']);
        expect(vectors[0]).toEqual(Float32Array.from([1, 2]));

        const body = JSON.parse((impl.mock.calls[0] as [string, RequestInit])[1].body as string);
        expect(body.instances[0].task_type).toBe('RETRIEVAL_DOCUMENT');
    });

    it('rejects a short batch rather than misaligning vectors with chunks', async () => {
        const impl = jest.fn(async () => new Response(JSON.stringify({
            predictions: [{ embeddings: { values: [1] } }],
        }), { status: 200 }));
        const embedder = new VertexAIEmbedder({
            projectId: 'p', getAccessToken: () => 't', fetchImpl: impl as unknown as typeof fetch,
        });

        await expect(embedder.embedBatch(['a', 'b'])).rejects.toThrow(/1 embeddings for 2 inputs/);
    });

    it('short-circuits an empty batch and surfaces transport errors', async () => {
        const impl = jest.fn(async () => new Response('quota', { status: 429 }));
        const embedder = new VertexAIEmbedder({
            projectId: 'p', getAccessToken: () => 't', fetchImpl: impl as unknown as typeof fetch,
        });

        expect(await embedder.embedBatch([])).toEqual([]);
        expect(impl).not.toHaveBeenCalled();
        await expect(embedder.embed('a')).rejects.toBeInstanceOf(SSMError);
    });
});
