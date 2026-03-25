/**
 * Tokenizer – pluggable tokenizer interface for MambaSession.
 *
 * Pass a custom implementation via `MambaSessionOptions.tokenizer` to bypass
 * the default Qwen2.5-Coder BPETokenizer.  This is useful for:
 *   - Unit-testing with a stub tokenizer (no network required)
 *   - Using a pre-built tokenizer from HuggingFace Transformers.js
 *   - Domain-specific vocabulary (e.g. code, biomedical, multilingual)
 */

export interface Tokenizer {
    /** Encodes `text` into a sequence of token IDs. */
    encode(text: string): number[];
    /** Decodes a sequence of token IDs back to a string. */
    decode(tokens: number[]): string;
    /** The total number of tokens in this tokenizer's vocabulary. */
    readonly vocabSize: number;
}
