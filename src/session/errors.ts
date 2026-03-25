/**
 * errors.ts – Typed error class for the MambaSession layer (absorbed from MambaKit).
 */

export type MambaKitErrorCode =
    | 'GPU_UNAVAILABLE'          // navigator.gpu not present or adapter request failed
    | 'TOKENIZER_LOAD_FAILED'    // vocab.json or merges.txt could not be fetched/parsed
    | 'CHECKPOINT_FETCH_FAILED'  // checkpoint URL returned non-OK response after retries
    | 'CHECKPOINT_INVALID'       // loadWeights threw (bad magic, version, or size mismatch)
    | 'INPUT_TOO_SHORT'          // adapt() input encodes to fewer than 2 tokens
    | 'STORAGE_UNAVAILABLE'      // IndexedDB or File System Access API not available
    | 'SESSION_DESTROYED'        // method called after destroy()
    | 'UNKNOWN';                 // unexpected error (original in .cause)

export class MambaKitError extends Error {
    constructor(
        public readonly code: MambaKitErrorCode,
        message: string,
        public readonly cause?: unknown,
    ) {
        super(message);
        this.name = 'MambaKitError';
    }
}
