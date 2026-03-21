/**
 * tests/MemoryStore.test.ts
 * Unit tests for MemoryStore — uses fake-indexeddb for in-memory IDB.
 */

// Install fake IndexedDB globals before importing MemoryStore
import 'fake-indexeddb/auto';

import { MemoryStore } from '../src/memory/MemoryStore.js';
import type { MemoryEntry } from '../src/memory/MemoryStore.js';

// ── Helpers ───────────────────────────────────────────────────────────────────

// Each test gets a unique DB name so stores are isolated
let _dbCounter = 0;
function uniqueStore() {
    return new MemoryStore({ dbName: `test-db-${_dbCounter++}` });
}

// ── remember / recall ─────────────────────────────────────────────────────────

test('recall returns undefined for a key that has never been stored', async () => {
    const store = uniqueStore();
    expect(await store.recall('missing')).toBeUndefined();
});

test('remember + recall round-trips a fact', async () => {
    const store = uniqueStore();
    await store.remember('lang', 'TypeScript');
    const entry = await store.recall('lang');
    expect(entry).toBeDefined();
    expect(entry!.content).toBe('TypeScript');
    expect(entry!.key).toBe('lang');
});

test('recall returns a MemoryEntry with a numeric timestamp', async () => {
    const store = uniqueStore();
    await store.remember('k', 'v');
    const entry = await store.recall('k') as MemoryEntry;
    expect(typeof entry.timestamp).toBe('number');
    expect(entry.timestamp).toBeGreaterThan(0);
});

test('remember overwrites an existing entry', async () => {
    const store = uniqueStore();
    await store.remember('key', 'first');
    await store.remember('key', 'second');
    const entry = await store.recall('key');
    expect(entry!.content).toBe('second');
});

// ── recallAll ─────────────────────────────────────────────────────────────────

test('recallAll returns empty array when no facts stored', async () => {
    const store = uniqueStore();
    expect(await store.recallAll()).toEqual([]);
});

test('recallAll returns all stored facts', async () => {
    const store = uniqueStore();
    await store.remember('a', 'alpha');
    await store.remember('b', 'beta');
    const all = await store.recallAll();
    expect(all).toHaveLength(2);
    expect(all.map(e => e.key).sort()).toEqual(['a', 'b']);
});

test('recallAll sorts newest first by timestamp', async () => {
    const store = uniqueStore();

    // Store two facts with a small delay so timestamps differ
    await store.remember('old', 'first');
    await new Promise(r => setTimeout(r, 5));
    await store.remember('new', 'second');

    const all = await store.recallAll();
    expect(all[0].key).toBe('new');
    expect(all[1].key).toBe('old');
});

// ── forget ────────────────────────────────────────────────────────────────────

test('forget removes a stored fact', async () => {
    const store = uniqueStore();
    await store.remember('x', 'val');
    await store.forget('x');
    expect(await store.recall('x')).toBeUndefined();
});

test('forget is a no-op for a key that does not exist', async () => {
    const store = uniqueStore();
    await expect(store.forget('nonexistent')).resolves.toBeUndefined();
});

// ── clear ─────────────────────────────────────────────────────────────────────

test('clear removes all stored facts', async () => {
    const store = uniqueStore();
    await store.remember('a', '1');
    await store.remember('b', '2');
    await store.clear();
    expect(await store.recallAll()).toHaveLength(0);
});

test('clear does not throw when store is already empty', async () => {
    const store = uniqueStore();
    await expect(store.clear()).resolves.toBeUndefined();
});

// ── saveWeights / loadWeights ─────────────────────────────────────────────────

test('saveWeights delegates to runtime.save with the weightsKey', async () => {
    const store   = new MemoryStore({ dbName: `test-db-${_dbCounter++}`, weightsKey: 'my-weights' });
    const runtime = { save: jest.fn().mockResolvedValue(undefined), load: jest.fn() };

    await store.saveWeights(runtime);
    expect(runtime.save).toHaveBeenCalledWith(
        expect.objectContaining({ key: 'my-weights' }),
    );
});

test('loadWeights delegates to runtime.load with the weightsKey', async () => {
    const store   = new MemoryStore({ dbName: `test-db-${_dbCounter++}`, weightsKey: 'my-weights' });
    const runtime = { save: jest.fn(), load: jest.fn().mockResolvedValue(false) };

    const result = await store.loadWeights(runtime);
    expect(runtime.load).toHaveBeenCalledWith(
        expect.objectContaining({ key: 'my-weights' }),
    );
    expect(result).toBe(false);
});

test('loadWeights returns false when runtime.load returns false', async () => {
    const store   = uniqueStore();
    const runtime = { save: jest.fn(), load: jest.fn().mockResolvedValue(false) };
    expect(await store.loadWeights(runtime)).toBe(false);
});
