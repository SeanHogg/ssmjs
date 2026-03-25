/**
 * persistence.ts – Storage helpers for the MambaSession layer.
 *
 * Supports three storage targets:
 *  - IndexedDB  (browser default)
 *  - Download   (Blob URL trigger)
 *  - File System Access API
 */

import { SessionError } from './errors.js';

const DB_NAME    = 'ssmjs-session';
const DB_VERSION = 1;
const STORE_NAME = 'checkpoints';

// ── IndexedDB helpers ─────────────────────────────────────────────────────────

function openDB(idb?: IDBFactory): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        const factory = idb ?? (typeof indexedDB !== 'undefined' ? indexedDB : undefined);
        if (!factory) {
            reject(new SessionError(
                'STORAGE_UNAVAILABLE',
                'IndexedDB is not available in this environment. Pass an idbFactory option (e.g. from fake-indexeddb) for Node.js support.',
            ));
            return;
        }

        const req = factory.open(DB_NAME, DB_VERSION);

        req.onupgradeneeded = (e) => {
            const db = (e.target as IDBOpenDBRequest).result;
            if (!db.objectStoreNames.contains(STORE_NAME)) {
                db.createObjectStore(STORE_NAME);
            }
        };

        req.onsuccess = () => resolve(req.result);
        req.onerror   = ()  => reject(new SessionError(
            'STORAGE_UNAVAILABLE',
            `Failed to open IndexedDB database "${DB_NAME}": ${req.error?.message ?? 'unknown error'}`,
            req.error,
        ));
    });
}

export async function saveToIndexedDB(key: string, buffer: ArrayBuffer, idb?: IDBFactory): Promise<void> {
    const db = await openDB(idb);
    return new Promise((resolve, reject) => {
        const tx    = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        const req   = store.put(buffer, key);

        req.onsuccess = () => resolve();
        req.onerror   = () => reject(new SessionError(
            'STORAGE_UNAVAILABLE',
            `Failed to write checkpoint to IndexedDB (key="${key}"): ${req.error?.message ?? 'unknown error'}`,
            req.error,
        ));
        tx.oncomplete = () => db.close();
    });
}

export async function loadFromIndexedDB(key: string, idb?: IDBFactory): Promise<ArrayBuffer | undefined> {
    const db = await openDB(idb);
    return new Promise((resolve, reject) => {
        const tx    = db.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);
        const req   = store.get(key);

        req.onsuccess = () => {
            db.close();
            resolve(req.result as ArrayBuffer | undefined);
        };
        req.onerror = () => reject(new SessionError(
            'STORAGE_UNAVAILABLE',
            `Failed to read checkpoint from IndexedDB (key="${key}"): ${req.error?.message ?? 'unknown error'}`,
            req.error,
        ));
    });
}

/** Milliseconds a download Blob URL is kept alive before being revoked. */
const DOWNLOAD_URL_TTL_MS = 10_000;

// ── Download helper ───────────────────────────────────────────────────────────

export async function triggerDownload(filename: string, buffer: ArrayBuffer): Promise<void> {
    const blob = new Blob([buffer], { type: 'application/octet-stream' });
    const url  = URL.createObjectURL(blob);

    const anchor       = document.createElement('a');
    anchor.href        = url;
    anchor.download    = filename;
    anchor.style.display = 'none';

    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);

    // Release the object URL after the TTL to allow the download to start
    setTimeout(() => URL.revokeObjectURL(url), DOWNLOAD_URL_TTL_MS);
}

// ── File System Access API helpers ────────────────────────────────────────────

export async function saveViaFileSystemAPI(filename: string, buffer: ArrayBuffer): Promise<void> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const win = window as any;
    if (typeof win.showSaveFilePicker !== 'function') {
        throw new SessionError(
            'STORAGE_UNAVAILABLE',
            'File System Access API (showSaveFilePicker) is not available in this browser.',
        );
    }

    const handle = await win.showSaveFilePicker({
        suggestedName: filename,
        types: [{ description: 'MambaSession Checkpoint', accept: { 'application/octet-stream': ['.bin'] } }],
    });
    const writable = await handle.createWritable();
    await writable.write(buffer);
    await writable.close();
}

export async function loadViaFileSystemAPI(): Promise<ArrayBuffer> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const win = window as any;
    if (typeof win.showOpenFilePicker !== 'function') {
        throw new SessionError(
            'STORAGE_UNAVAILABLE',
            'File System Access API (showOpenFilePicker) is not available in this browser.',
        );
    }

    const [handle] = await win.showOpenFilePicker({
        types: [{ description: 'MambaSession Checkpoint', accept: { 'application/octet-stream': ['.bin'] } }],
        multiple: false,
    });
    const file = await handle.getFile();
    return file.arrayBuffer();
}
