/**
 * FetchBridge – generic OpenAI-compatible bridge for local or hosted endpoints.
 *
 * Works with Ollama, LM Studio, vLLM, llama.cpp server, or any service that
 * exposes a /chat/completions endpoint compatible with the OpenAI request
 * and response schema.
 */

import { OpenAIBridge } from './OpenAIBridge.js';
import type { BridgeGenerateOptions } from './TransformerBridge.js';

export interface FetchBridgeOptions {
    /** Base URL of the OpenAI-compatible server, e.g. 'http://localhost:1234/v1'. */
    baseUrl       : string;
    /** API key — many local servers require any non-empty string. Default: 'local'. */
    apiKey?       : string;
    /** Model name understood by the server. Default: 'default'. */
    model?        : string;
    /** Default system prompt. */
    systemPrompt? : string;
    /** Default max tokens. Default: 512. */
    maxTokens?    : number;
}

/**
 * FetchBridge is a thin re-configuration of OpenAIBridge pointed at a custom
 * base URL.  All streaming and request logic is inherited.
 */
export class FetchBridge extends OpenAIBridge {
    constructor(opts: FetchBridgeOptions) {
        super({
            apiKey       : opts.apiKey        ?? 'local',
            model        : opts.model         ?? 'default',
            baseUrl      : opts.baseUrl,
            systemPrompt : opts.systemPrompt,
            maxTokens    : opts.maxTokens,
        });
    }
}

export type { BridgeGenerateOptions };
