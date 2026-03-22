/**
 * tests/AnthropicBridge.test.ts
 * Unit tests for AnthropicBridge.
 * Uses jest.spyOn to mock globalThis.fetch — no real HTTP calls.
 */

import { jest } from '@jest/globals';
import { AnthropicBridge } from '../src/bridges/AnthropicBridge.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeJsonResponse(body: unknown, status = 200): Response {
    return new Response(JSON.stringify(body), {
        status,
        headers: { 'Content-Type': 'application/json' },
    });
}

// ── generate() ───────────────────────────────────────────────────────────────

test('generate returns content[0].text', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        makeJsonResponse({ content: [{ text: 'Hello from Claude' }] }),
    );

    const bridge = new AnthropicBridge({ apiKey: 'sk-ant-test' });
    const result = await bridge.generate('Say hello');

    expect(result).toBe('Hello from Claude');
    fetchSpy.mockRestore();
});

test('generate sends POST to Anthropic messages endpoint', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        makeJsonResponse({ content: [{ text: 'ok' }] }),
    );

    await new AnthropicBridge({ apiKey: 'sk-ant' }).generate('hi');

    expect(fetchSpy).toHaveBeenCalledWith(
        'https://api.anthropic.com/v1/messages',
        expect.objectContaining({ method: 'POST' }),
    );
    fetchSpy.mockRestore();
});

test('generate includes x-api-key and anthropic-version headers', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        makeJsonResponse({ content: [{ text: 'ok' }] }),
    );

    await new AnthropicBridge({ apiKey: 'mykey' }).generate('hi');

    const init    = fetchSpy.mock.calls[0][1] as RequestInit;
    const headers = init.headers as Record<string, string>;
    expect(headers['x-api-key']).toBe('mykey');
    expect(headers['anthropic-version']).toBe('2023-06-01');
    fetchSpy.mockRestore();
});

test('generate sets system as top-level field (not a message role)', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        makeJsonResponse({ content: [{ text: 'ok' }] }),
    );

    await new AnthropicBridge({ apiKey: 'key', systemPrompt: 'Be concise.' }).generate('hi');

    const init = fetchSpy.mock.calls[0][1] as RequestInit;
    const body = JSON.parse(init.body as string);
    expect(body.system).toBe('Be concise.');
    // No system role in messages array
    expect(body.messages.every((m: { role: string }) => m.role !== 'system')).toBe(true);
    fetchSpy.mockRestore();
});

test('generate omits system field when systemPrompt is empty', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        makeJsonResponse({ content: [{ text: 'ok' }] }),
    );

    await new AnthropicBridge({ apiKey: 'key' }).generate('hi');

    const body = JSON.parse((fetchSpy.mock.calls[0][1] as RequestInit).body as string);
    expect(body.system).toBeUndefined();
    fetchSpy.mockRestore();
});

test('generate body includes max_tokens', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        makeJsonResponse({ content: [{ text: 'ok' }] }),
    );

    await new AnthropicBridge({ apiKey: 'key', maxTokens: 256 }).generate('hi');

    const body = JSON.parse((fetchSpy.mock.calls[0][1] as RequestInit).body as string);
    expect(body.max_tokens).toBe(256);
    fetchSpy.mockRestore();
});

test('generate throws SSMError(BRIDGE_REQUEST_FAILED) on non-OK response', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValue(
        new Response('Unauthorized', { status: 401 }),
    );

    await expect(new AnthropicBridge({ apiKey: 'bad' }).generate('hi'))
        .rejects.toMatchObject({ code: 'BRIDGE_REQUEST_FAILED' });
    fetchSpy.mockRestore();
});

test('generate throws SSMError(BRIDGE_RESPONSE_INVALID) when content missing', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValue(
        makeJsonResponse({ error: 'unexpected' }),
    );

    await expect(new AnthropicBridge({ apiKey: 'key' }).generate('hi'))
        .rejects.toMatchObject({ code: 'BRIDGE_RESPONSE_INVALID' });
    fetchSpy.mockRestore();
});

// ── stream() ──────────────────────────────────────────────────────────────────

test('supportsStreaming is true', () => {
    expect(new AnthropicBridge({ apiKey: 'key' }).supportsStreaming).toBe(true);
});

test('stream yields tokens from content_block_delta events', async () => {
    const sseLines = [
        'data: ' + JSON.stringify({ type: 'message_start', message: {} }),
        'data: ' + JSON.stringify({ type: 'content_block_delta', delta: { text: 'Hello' } }),
        'data: ' + JSON.stringify({ type: 'content_block_delta', delta: { text: ' world' } }),
        'data: ' + JSON.stringify({ type: 'message_stop' }),
    ];

    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        new Response(
            new ReadableStream({
                start(c) {
                    const enc = new TextEncoder();
                    for (const l of sseLines) c.enqueue(enc.encode(l + '\n'));
                    c.close();
                },
            }),
            { status: 200 },
        ),
    );

    const tokens: string[] = [];
    for await (const token of new AnthropicBridge({ apiKey: 'key' }).stream('hi')) {
        tokens.push(token);
    }

    expect(tokens).toEqual(['Hello', ' world']);
    fetchSpy.mockRestore();
});

test('stream ignores non-content_block_delta events', async () => {
    const sseLines = [
        'data: ' + JSON.stringify({ type: 'ping' }),
        'data: ' + JSON.stringify({ type: 'content_block_delta', delta: { text: 'A' } }),
        'data: ' + JSON.stringify({ type: 'message_stop' }),
    ];

    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        new Response(
            new ReadableStream({
                start(c) {
                    const enc = new TextEncoder();
                    for (const l of sseLines) c.enqueue(enc.encode(l + '\n'));
                    c.close();
                },
            }),
            { status: 200 },
        ),
    );

    const tokens: string[] = [];
    for await (const token of new AnthropicBridge({ apiKey: 'key' }).stream('hi')) {
        tokens.push(token);
    }

    expect(tokens).toEqual(['A']);
    fetchSpy.mockRestore();
});

test('stream throws SSMError(BRIDGE_REQUEST_FAILED) on non-OK', async () => {
    const fetchSpy = jest.spyOn(globalThis, 'fetch').mockResolvedValueOnce(
        new Response('Forbidden', { status: 403 }),
    );

    const gen = new AnthropicBridge({ apiKey: 'key' }).stream('hi')[Symbol.asyncIterator]();
    await expect(gen.next()).rejects.toMatchObject({ code: 'BRIDGE_REQUEST_FAILED' });
    fetchSpy.mockRestore();
});
