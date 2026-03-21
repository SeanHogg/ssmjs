/**
 * SSMError – typed error class for SSM.js.
 *
 * Mirrors MambaKitError in shape so callers can handle both with the same
 * `err.code` pattern. MambaKitError can propagate unchanged through
 * SSMRuntime.create() for GPU / tokenizer failures; SSM.js-level failures
 * throw SSMError.
 */

export type SSMErrorCode =
    | 'RUNTIME_DESTROYED'        // method called after destroy()
    | 'BRIDGE_REQUEST_FAILED'    // fetch to transformer API returned non-OK
    | 'BRIDGE_RESPONSE_INVALID'  // unexpected shape in transformer API response
    | 'MEMORY_UNAVAILABLE'       // IndexedDB unavailable or operation called without a MemoryStore
    | 'DISTILL_FAILED'           // distillation pipeline threw an unexpected error
    | 'UNKNOWN';                 // unexpected error (original in .cause)

export class SSMError extends Error {
    constructor(
        public readonly code: SSMErrorCode,
        message: string,
        public readonly cause?: unknown,
    ) {
        super(message);
        this.name = 'SSMError';
    }
}
