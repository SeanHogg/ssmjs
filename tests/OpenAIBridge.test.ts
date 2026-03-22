/**
 * tests/OpenAIBridge.test.ts
 * Unit tests for OpenAIBridge and FetchBridge.
 * Uses jest.spyOn to mock globalThis.fetch — no real HTTP calls.
 */

import { jest } from '@jest/globals';
import { OpenAIBridge } from '../src/bridges/OpenAIBridge.js';
import { FetchBridge }  from '../src/bridges/FetchBridge.js';
import { SSMError }     from '../src/errors/SSMError.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeJsonResponse(body: unknown, status = 200): Response {
    return new Response(JSON.stringify(body), {
        status,
        headers: { 'Content-Type': 'application/json' },
    });
}

function makeSSEStream(lines: string[]): ReadableStream<Uint8Array> {
    const encoder = new TextEncoder();
    return new ReadableStream({
        start(controller) {
            for (const line of lines) {
                controller.enqueue(encoder.encode(line + '\n'));
            }
            controller.close();
        },
    });
}

function makeSseResponse(lines: string[], status = 200): Response {
    return new Response(makeSSEStream(lines), { status });
}

// ── generate() ───────────────────────────────────────────────────────────────

test('generate returns content from choices[0].message.content', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        makeJsonResponse({ choices: [{ message: { content: 'Hello!' } }] }),
    );

    const bridge = new OpenAIBridge({ apiKey: 'sk-test' });
    const result = await bridge.generate('Say hello');

    expect(result).toBe('Hello!');
    fetchSpy.mockRestore();
});

test('generate sends POST to /chat/completions', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        makeJsonResponse({ choices: [{ message: { content: 'ok' } }] }),
    );

    const bridge = new OpenAIBridge({ apiKey: 'sk-test', baseUrl: 'https://api.openai.com/v1' });
    await bridge.generate('ping');

    expect(fetchSpy).toHaveBeenCalledWith(
        'https://api.openai.com/v1/chat/completions',
        expect.objectContaining({ method: 'POST' }),
    );
    fetchSpy.mockRestore();
});

test('generate includes Authorization header', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        makeJsonResponse({ choices: [{ message: { content: 'ok' } }] }),
    );

    await new OpenAIBridge({ apiKey: 'sk-secret' }).generate('hi');

    const init = fetchSpy.mock.calls[0][1] as RequestInit;
    const headers = init.headers as Record<string, string>;
    expect(headers['Authorization']).toBe('Bearer sk-secret');
    fetchSpy.mockRestore();
});

test('generate includes system message when systemPrompt set', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        makeJsonResponse({ choices: [{ message: { content: 'ok' } }] }),
    );

    await new OpenAIBridge({ apiKey: 'key', systemPrompt: 'You are helpful.' }).generate('hi');

    const init  = fetchSpy.mock.calls[0][1] as RequestInit;
    const body  = JSON.parse(init.body as string);
    expect(body.messages[0]).toEqual({ role: 'system', content: 'You are helpful.' });
    expect(body.messages[1]).toEqual({ role: 'user',   content: 'hi' });
    fetchSpy.mockRestore();
});

test('generate omits system message when systemPrompt is empty', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        makeJsonResponse({ choices: [{ message: { content: 'ok' } }] }),
    );

    await new OpenAIBridge({ apiKey: 'key' }).generate('hi');

    const init = fetchSpy.mock.calls[0][1] as RequestInit;
    const body = JSON.parse(init.body as string);
    expect(body.messages).toHaveLength(1);
    expect(body.messages[0].role).toBe('user');
    fetchSpy.mockRestore();
});

test('generate throws SSMError(BRIDGE_REQUEST_FAILED) on non-OK response', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        new Response('Internal Server Error', { status: 500 }),
    );

    const bridge = new OpenAIBridge({ apiKey: 'key' });
    await expect(bridge.generate('hi')).rejects.toThrow(SSMError);
    await expect(bridge.generate('hi')).rejects.toMatchObject({ code: 'BRIDGE_REQUEST_FAILED' });
    fetchSpy.mockRestore();
});

test('generate throws SSMError(BRIDGE_RESPONSE_INVALID) when choices missing', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValue(
        makeJsonResponse({ result: 'unexpected' }),
    );

    const bridge = new OpenAIBridge({ apiKey: 'key' });
    await expect(bridge.generate('hi')).rejects.toMatchObject({ code: 'BRIDGE_RESPONSE_INVALID' });
    fetchSpy.mockRestore();
});

test('generate uses opts.model over constructor model', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        makeJsonResponse({ choices: [{ message: { content: 'ok' } }] }),
    );

    await new OpenAIBridge({ apiKey: 'key', model: 'gpt-4o' }).generate('hi', { model: 'o1' });

    const body = JSON.parse((fetchSpy.mock.calls[0][1] as RequestInit).body as string);
    expect(body.model).toBe('o1');
    fetchSpy.mockRestore();
});

// ── stream() ──────────────────────────────────────────────────────────────────

test('supportsStreaming is true', () => {
    expect(new OpenAIBridge({ apiKey: 'key' }).supportsStreaming).toBe(true);
});

test('stream yields tokens from SSE data lines', async () => {
    const sseLines = [
        'data: ' + JSON.stringify({ choices: [{ delta: { content: 'Hello' } }] }),
        'data: ' + JSON.stringify({ choices: [{ delta: { content: ' world' } }] }),
        'data: [DONE]',
    ];

    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        makeSseResponse(sseLines),
    );

    const bridge  = new OpenAIBridge({ apiKey: 'key' });
    const tokens: string[] = [];
    for await (const token of bridge.stream('Say hello')) {
        tokens.push(token);
    }

    expect(tokens).toEqual(['Hello', ' world']);
    fetchSpy.mockRestore();
});

test('stream ignores non-data SSE lines', async () => {
    const sseLines = [
        'event: message',
        'data: ' + JSON.stringify({ choices: [{ delta: { content: 'Hi' } }] }),
        'data: [DONE]',
    ];

    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        makeSseResponse(sseLines),
    );

    const tokens: string[] = [];
    for await (const token of new OpenAIBridge({ apiKey: 'key' }).stream('hi')) {
        tokens.push(token);
    }

    expect(tokens).toEqual(['Hi']);
    fetchSpy.mockRestore();
});

test('stream skips delta tokens with empty content', async () => {
    const sseLines = [
        'data: ' + JSON.stringify({ choices: [{ delta: {} }] }),
        'data: ' + JSON.stringify({ choices: [{ delta: { content: '' } }] }),
        'data: ' + JSON.stringify({ choices: [{ delta: { content: 'Real' } }] }),
        'data: [DONE]',
    ];

    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        makeSseResponse(sseLines),
    );

    const tokens: string[] = [];
    for await (const token of new OpenAIBridge({ apiKey: 'key' }).stream('hi')) {
        tokens.push(token);
    }

    expect(tokens).toEqual(['Real']);
    fetchSpy.mockRestore();
});

test('stream throws SSMError(BRIDGE_REQUEST_FAILED) on non-OK response', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        new Response('Bad Request', { status: 400 }),
    );

    const gen = new OpenAIBridge({ apiKey: 'key' }).stream('hi')[Symbol.asyncIterator]();
    await expect(gen.next()).rejects.toMatchObject({ code: 'BRIDGE_REQUEST_FAILED' });
    fetchSpy.mockRestore();
});

// ── FetchBridge ───────────────────────────────────────────────────────────────

test('FetchBridge uses custom baseUrl', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        makeJsonResponse({ choices: [{ message: { content: 'ok' } }] }),
    );

    const bridge = new FetchBridge({ baseUrl: 'http://localhost:11434/v1', model: 'llama3' });
    await bridge.generate('hi');

    expect(fetchSpy).toHaveBeenCalledWith(
        'http://localhost:11434/v1/chat/completions',
        expect.anything(),
    );
    fetchSpy.mockRestore();
});
