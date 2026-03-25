/**
 * SSMError – typed error class for SSM.js runtime-level failures.
 *
 * Session-layer failures (GPU init, tokenizer, checkpoint) throw SessionError.
 * Runtime-level failures (bridge, distillation, memory) throw SSMError.
 * Both carry a typed `code` discriminant for programmatic handling.
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
