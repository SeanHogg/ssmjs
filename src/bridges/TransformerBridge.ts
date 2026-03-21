/**
 * TransformerBridge – pluggable interface for any transformer LLM backend.
 *
 * Implementations (OpenAIBridge, AnthropicBridge, FetchBridge) satisfy this
 * interface and are passed to SSMRuntime to enable hybrid inference and
 * distillation.  The interface is structural — any object with the right
 * shape works, no base class required.
 */

export interface BridgeGenerateOptions {
    /** Max tokens to generate. Default per-adapter (typically 512). */
    maxTokens?    : number;
    /** Sampling temperature. Default per-adapter (typically 0.7). */
    temperature?  : number;
    /** Nucleus sampling p. Default per-adapter (typically 0.9). */
    topP?         : number;
    /** System prompt for this request, overriding the adapter's default. */
    systemPrompt? : string;
    /** Model string, overriding the adapter's default. */
    model?        : string;
}

export interface TransformerBridge {
    /**
     * Generates a completion for the given prompt.
     * Must resolve to the assistant's reply text only (not including the prompt).
     */
    generate(prompt: string, opts?: BridgeGenerateOptions): Promise<string>;

    /**
     * Streaming variant — yields tokens incrementally.
     * Check `supportsStreaming` before calling.
     */
    stream?(prompt: string, opts?: BridgeGenerateOptions): AsyncIterable<string>;

    /** True when this bridge supports the `stream()` method. */
    readonly supportsStreaming: boolean;
}
