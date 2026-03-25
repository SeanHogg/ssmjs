/**
 * SSMAgent – high-level orchestration primitive.
 *
 * Combines SSMRuntime (inference + adaptation), MemoryStore (persistent facts),
 * and a conversation history manager into a single agent interface.
 *
 * Prompt format matches MambaChatbot so the SSM model can follow the same
 * token patterns it was trained on:
 *
 *   System: <systemPrompt>
 *   [Fact (<key>): <content>  ← injected by importance desc, filtered by tag/key]
 *   User: <message>
 *   Assistant: <message>
 *   ...
 *   User: <current input>
 *   Assistant:
 */

import type { AdaptOptions, AdaptResult } from '../session/index.js';
import type { SSMRuntime, GenerateOptions } from '../runtime/SSMRuntime.js';
import type { MemoryStore } from '../memory/MemoryStore.js';
import { SSMError } from '../errors/SSMError.js';

// ── Types ─────────────────────────────────────────────────────────────────────

export type MessageRole = 'user' | 'assistant' | 'system';

export interface AgentMessage {
    role   : MessageRole;
    content: string;
}

export interface SSMAgentOptions {
    /** The runtime to use for inference and adaptation. */
    runtime          : SSMRuntime;
    /** Optional memory store for persistent fact retrieval. */
    memory?          : MemoryStore;
    /** Default system prompt. Default: 'You are a helpful assistant.' */
    systemPrompt?    : string;
    /**
     * Max user+assistant turn pairs to include in context.
     * Oldest turns are dropped first.
     * Default: 20
     */
    maxHistoryTurns? : number;
    /**
     * When true, the agent serialises its conversation history to memory
     * under the `__history__` key on `destroy()`, and loads it back on
     * construction if the key is present.
     * Default: true
     */
    persistHistory?  : boolean;
}

export interface ThinkOptions extends GenerateOptions {
    /** Override the system prompt for this single turn only. */
    systemPrompt?: string;
    /**
     * Inject all recalled facts into the context for this turn.
     * Default: false — only facts whose keys appear in the input are injected.
     */
    injectAllFacts?: boolean;
}

/** Key used to persist conversation history in the MemoryStore. */
const HISTORY_KEY = '__history__';

// ── SSMAgent ──────────────────────────────────────────────────────────────────

export class SSMAgent {
    private readonly _runtime         : SSMRuntime;
    private readonly _memory          : MemoryStore | undefined;
    private readonly _systemPrompt    : string;
    private readonly _maxHistoryTurns : number;
    private readonly _persistHistory  : boolean;
    private _history: AgentMessage[] = [];

    constructor(opts: SSMAgentOptions) {
        this._runtime         = opts.runtime;
        this._memory          = opts.memory;
        this._systemPrompt    = opts.systemPrompt    ?? 'You are a helpful assistant.';
        this._maxHistoryTurns = opts.maxHistoryTurns ?? 20;
        this._persistHistory  = opts.persistHistory  ?? true;
    }

    /**
     * Initialises the agent, loading persisted history from memory if available.
     * Call this after construction when `persistHistory` is enabled and a memory
     * store is present.
     */
    async init(): Promise<void> {
        if (this._persistHistory && this._memory) {
            try {
                const entry = await this._memory.recall(HISTORY_KEY);
                if (entry) {
                    const parsed = JSON.parse(entry.content) as AgentMessage[];
                    if (Array.isArray(parsed)) {
                        this._history = parsed;
                    }
                }
            } catch {
                // Corrupted or missing history — start fresh
                this._history = [];
            }
        }
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /**
     * Sends a user message and returns the full assistant response.
     * Routes through InferenceRouter — may use SSM or transformer bridge.
     * Appends both user and assistant turns to history.
     */
    async think(input: string, opts: ThinkOptions = {}): Promise<string> {
        const { systemPrompt, injectAllFacts, ...generateOpts } = opts;
        const prompt = await this._buildPrompt(input, systemPrompt, injectAllFacts);

        const raw = await this._runtime.generate(prompt, {
            maxNewTokens: 200,
            temperature : 0.7,
            topK        : 50,
            topP        : 0.9,
            ...generateOpts,
        });

        // Trim any additional turns the model may have hallucinated
        const response = raw.split('\nUser:')[0].trim();

        this._appendHistory(input, response);
        return response;
    }

    /**
     * Streaming variant of `think()`.
     * Always uses the SSM path (consistent low-latency streaming).
     * Appends history after the stream completes.
     */
    async *thinkStream(input: string, opts: ThinkOptions = {}): AsyncIterable<string> {
        const { systemPrompt, injectAllFacts, bridgeOpts: _b, ...completeOpts } = opts;
        const prompt = await this._buildPrompt(input, systemPrompt, injectAllFacts);

        let full = '';
        for await (const token of this._runtime.stream(prompt, {
            maxNewTokens: 200,
            temperature : 0.7,
            topK        : 50,
            topP        : 0.9,
            ...completeOpts,
        })) {
            full += token;
            yield token;
        }

        const response = full.split('\nUser:')[0].trim();
        this._appendHistory(input, response);
    }

    // ── Adaptation ────────────────────────────────────────────────────────────

    /**
     * Fine-tunes the SSM on the provided text.
     * Pass-through to runtime.adapt().
     */
    async learn(data: string, opts?: AdaptOptions): Promise<AdaptResult> {
        return this._runtime.adapt(data, opts);
    }

    // ── Memory ────────────────────────────────────────────────────────────────

    /**
     * Stores a fact in the MemoryStore.
     * Throws SSMError('MEMORY_UNAVAILABLE') if no MemoryStore was provided.
     */
    async remember(key: string, fact: string): Promise<void> {
        if (!this._memory) {
            throw new SSMError(
                'MEMORY_UNAVAILABLE',
                'SSMAgent was constructed without a MemoryStore. Pass `memory` in SSMAgentOptions.',
            );
        }
        await this._memory.remember(key, fact);
    }

    /**
     * Retrieves a fact from the MemoryStore.
     * Returns `undefined` if key not found or no MemoryStore was provided.
     */
    async recall(key: string): Promise<string | undefined> {
        if (!this._memory) return undefined;
        const entry = await this._memory.recall(key);
        return entry?.content;
    }

    // ── History ───────────────────────────────────────────────────────────────

    /** Clears all conversation history. Does not affect MemoryStore. */
    clearHistory(): void {
        this._history = [];
    }

    /** Number of complete user+assistant turn pairs. */
    get turnCount(): number {
        return Math.floor(this._history.length / 2);
    }

    /** Read-only snapshot of the current conversation history. */
    get history(): readonly AgentMessage[] {
        return this._history;
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /**
     * Persists conversation history to memory (if `persistHistory` is true and
     * a MemoryStore is available) and destroys the underlying runtime.
     */
    async destroy(): Promise<void> {
        if (this._persistHistory && this._memory && this._history.length > 0) {
            try {
                await this._memory.remember(HISTORY_KEY, JSON.stringify(this._history));
            } catch {
                // Persistence failure is non-fatal
            }
        }
        this._runtime.destroy();
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private async _buildPrompt(
        input                : string,
        systemPromptOverride?: string,
        injectAllFacts?      : boolean,
    ): Promise<string> {
        const sys   = systemPromptOverride ?? this._systemPrompt;
        const lines : string[] = [`System: ${sys}`];

        // Inject relevant facts from MemoryStore, sorted by importance descending
        if (this._memory) {
            const facts = await this._memory.recallAll();
            const relevant = injectAllFacts
                ? facts
                : facts.filter(f => input.includes(f.key));

            // Sort by importance descending (missing importance defaults to 0.5)
            const sorted = relevant.slice().sort(
                (a, b) => (b.importance ?? 0.5) - (a.importance ?? 0.5),
            );

            for (const fact of sorted) {
                lines.push(`Fact (${fact.key}): ${fact.content}`);
            }
        }

        // Trim history to maxHistoryTurns pairs (oldest first)
        const maxMessages = this._maxHistoryTurns * 2;
        const trimmed = this._history.length > maxMessages
            ? this._history.slice(this._history.length - maxMessages)
            : this._history;

        for (const msg of trimmed) {
            const speaker = msg.role === 'user' ? 'User' : 'Assistant';
            lines.push(`${speaker}: ${msg.content}`);
        }

        lines.push(`User: ${input}`);
        lines.push('Assistant:');
        return lines.join('\n');
    }

    private _appendHistory(input: string, response: string): void {
        this._history.push({ role: 'user',      content: input    });
        this._history.push({ role: 'assistant', content: response });
    }
}
