/**
 * SSMRuntime – core runtime class for SSM.js.
 *
 * Wraps a MambaSession and adds:
 *   - Hybrid inference routing (SSM ↔ transformer bridge)
 *   - Unified generate / stream API
 *   - Perplexity-aware routing (InferenceRouter auto mode)
 *   - Thin save/load pass-throughs for MemoryStore integration
 *
 * GPU lifecycle is owned entirely by this class via the underlying session.
 */

import {
    MambaSession,
    type MambaSessionOptions,
    type CompleteOptions,
    type AdaptOptions,
    type AdaptResult,
    type SaveOptions,
    type LoadOptions,
    type CreateCallbacks,
} from '@seanhogg/mambakit';

import { InferenceRouter, type RoutingStrategy, type RouterContext } from '../router/InferenceRouter.js';
import type { TransformerBridge, BridgeGenerateOptions } from '../bridges/TransformerBridge.js';
import { SSMError } from '../errors/SSMError.js';

// ── Public types ──────────────────────────────────────────────────────────────

export type GenerateOptions = CompleteOptions & {
    /**
     * Options forwarded to the transformer bridge when the router selects it.
     * Ignored when SSM is selected.
     */
    bridgeOpts?: BridgeGenerateOptions;
};

export interface SSMRuntimeOptions {
    /**
     * Options forwarded verbatim to MambaSession.create().
     * Controls model size, SSM variant, tokenizer, and checkpoint URL.
     */
    session              : MambaSessionOptions;

    /**
     * Optional transformer bridge.  When absent, all requests go to the SSM
     * and distillation is unavailable.
     */
    bridge?              : TransformerBridge;

    /**
     * Routing strategy when a bridge is present.
     * Default: 'auto'
     */
    routingStrategy?     : RoutingStrategy;

    /**
     * Character length above which auto-routing prefers the transformer.
     * Default: 1200
     */
    longInputThreshold?  : number;

    /**
     * SSM perplexity above which auto-routing prefers the transformer.
     * Default: 80
     */
    perplexityThreshold? : number;

    /**
     * MambaSession.create() progress callbacks.
     */
    callbacks?           : CreateCallbacks;
}

// ── SSMRuntime ────────────────────────────────────────────────────────────────

export class SSMRuntime {
    private readonly _session  : MambaSession;
    private readonly _bridge   : TransformerBridge | undefined;
    private readonly _router   : InferenceRouter;
    private _destroyed = false;

    private constructor(
        session : MambaSession,
        bridge  : TransformerBridge | undefined,
        router  : InferenceRouter,
    ) {
        this._session = session;
        this._bridge  = bridge;
        this._router  = router;
    }

    // ── Factory ───────────────────────────────────────────────────────────────

    /**
     * Creates and initialises a new SSMRuntime.
     *
     * Delegates to MambaSession.create() — can throw MambaKitError for
     * GPU / tokenizer failures.
     */
    static async create(opts: SSMRuntimeOptions): Promise<SSMRuntime> {
        const session = await MambaSession.create(opts.session, opts.callbacks);

        const router = new InferenceRouter({
            strategy            : opts.routingStrategy,
            longInputThreshold  : opts.longInputThreshold,
            perplexityThreshold : opts.perplexityThreshold,
            hasBridge           : !!opts.bridge,
            // Pass perplexity probe as callback — avoids circular import
            perplexityProbe     : opts.bridge
                ? (text) => session.evaluate(text)
                : undefined,
        });

        return new SSMRuntime(session, opts.bridge, router);
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /**
     * Generates a full response for the given input.
     * Routes to SSM or transformer bridge based on the configured strategy.
     */
    async generate(input: string, opts: GenerateOptions = {}): Promise<string> {
        this._checkAlive();

        const decision = await this._router.route(input);

        if (decision === 'transformer' && this._bridge) {
            return this._bridge.generate(input, opts.bridgeOpts);
        }

        // SSM path — extract only CompleteOptions fields
        const { bridgeOpts: _, ...completeOpts } = opts;
        return this._session.complete(input, completeOpts);
    }

    /**
     * Streaming token generation.
     *
     * Routing note: streaming always uses the SSM path for consistent
     * latency characteristics.  Use `generate()` if transformer streaming
     * is needed (bridgeOpts.stream, handled by the bridge's own `stream()` method).
     */
    async *stream(input: string, opts: GenerateOptions = {}): AsyncIterable<string> {
        this._checkAlive();
        const { bridgeOpts: _, ...completeOpts } = opts;
        yield* this._session.completeStream(input, completeOpts);
    }

    /**
     * Hybrid streaming: routes to SSM or transformer bridge stream.
     * Falls back to `generate()` for bridges that don't support streaming.
     */
    async *streamHybrid(input: string, opts: GenerateOptions = {}): AsyncIterable<string> {
        this._checkAlive();

        const decision = await this._router.route(input);

        if (decision === 'transformer' && this._bridge) {
            if (this._bridge.supportsStreaming && this._bridge.stream) {
                yield* this._bridge.stream(input, opts.bridgeOpts);
            } else {
                yield await this._bridge.generate(input, opts.bridgeOpts);
            }
            return;
        }

        const { bridgeOpts: _, ...completeOpts } = opts;
        yield* this._session.completeStream(input, completeOpts);
    }

    // ── Adaptation ────────────────────────────────────────────────────────────

    /**
     * Fine-tunes the SSM on the provided text.
     * Pass-through to MambaSession.adapt().
     */
    async adapt(data: string, opts?: AdaptOptions): Promise<AdaptResult> {
        this._checkAlive();
        return this._session.adapt(data, opts);
    }

    /**
     * Evaluates SSM perplexity on the provided text.
     * Exposed for power users and InferenceRouter's perplexity probe.
     */
    async evaluate(text: string): Promise<number> {
        this._checkAlive();
        return this._session.evaluate(text);
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    /**
     * Saves SSM weights. Pass-through to MambaSession.save().
     * Accepts a `key` option used by MemoryStore to save under a custom key.
     */
    async save(opts?: SaveOptions & { key?: string }): Promise<void> {
        this._checkAlive();
        await this._session.save(opts);
    }

    /**
     * Loads SSM weights. Pass-through to MambaSession.load().
     * Returns false when no checkpoint exists.
     */
    async load(opts?: LoadOptions & { key?: string }): Promise<boolean> {
        this._checkAlive();
        return this._session.load(opts) as Promise<boolean>;
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /** The transformer bridge attached to this runtime, if any. */
    get bridge(): TransformerBridge | undefined { return this._bridge; }

    /** Whether this runtime has been destroyed. */
    get destroyed(): boolean { return this._destroyed; }

    /**
     * Escape hatch to the underlying MambaSession internals.
     * Use sparingly — the SSMRuntime API covers the common cases.
     */
    get internals() { return this._session.internals; }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /**
     * Destroys the underlying MambaSession and releases all GPU resources.
     * After calling this, all methods throw SSMError('RUNTIME_DESTROYED').
     */
    destroy(): void {
        this._session.destroy();
        this._destroyed = true;
    }

    private _checkAlive(): void {
        if (this._destroyed) {
            throw new SSMError('RUNTIME_DESTROYED', 'SSMRuntime has been destroyed.');
        }
    }
}
