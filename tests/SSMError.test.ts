/**
 * tests/SSMError.test.ts
 * Unit tests for the typed SSMError class.
 */

import { SSMError } from '../src/errors/SSMError.js';

test('SSMError is an instance of Error', () => {
    const err = new SSMError('UNKNOWN', 'something went wrong');
    expect(err).toBeInstanceOf(Error);
});

test('SSMError.name is "SSMError"', () => {
    const err = new SSMError('RUNTIME_DESTROYED', 'destroyed');
    expect(err.name).toBe('SSMError');
});

test('SSMError.message is set correctly', () => {
    const err = new SSMError('BRIDGE_REQUEST_FAILED', 'API returned 500');
    expect(err.message).toBe('API returned 500');
});

test('SSMError.code is set correctly', () => {
    const err = new SSMError('BRIDGE_RESPONSE_INVALID', 'bad shape');
    expect(err.code).toBe('BRIDGE_RESPONSE_INVALID');
});

test('SSMError.cause is set when provided', () => {
    const cause = new Error('original');
    const err   = new SSMError('DISTILL_FAILED', 'wrapped', cause);
    expect(err.cause).toBe(cause);
});

test('SSMError.cause is undefined when not provided', () => {
    const err = new SSMError('MEMORY_UNAVAILABLE', 'no IDB');
    expect(err.cause).toBeUndefined();
});

test('all SSMErrorCode variants can be constructed', () => {
    const codes = [
        'RUNTIME_DESTROYED',
        'BRIDGE_REQUEST_FAILED',
        'BRIDGE_RESPONSE_INVALID',
        'MEMORY_UNAVAILABLE',
        'DISTILL_FAILED',
        'UNKNOWN',
    ] as const;

    for (const code of codes) {
        const err = new SSMError(code, 'test');
        expect(err.code).toBe(code);
    }
});

test('SSMError can be caught as Error', () => {
    expect(() => {
        throw new SSMError('UNKNOWN', 'thrown');
    }).toThrow(Error);
});

test('SSMError can be caught with instanceof check', () => {
    let caught: unknown;
    try {
        throw new SSMError('RUNTIME_DESTROYED', 'gone');
    } catch (e) {
        caught = e;
    }
    expect(caught).toBeInstanceOf(SSMError);
    expect((caught as SSMError).code).toBe('RUNTIME_DESTROYED');
});
