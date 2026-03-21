/**
 * InferenceRouter – decides whether a given input should be handled by the
 * local SSM or routed to a transformer bridge.
 *
 * Three heuristics run in priority order (cheapest first):
 *   1. Complexity patterns  — regex against the input string (synchronous, zero cost)
 *   2. Input length         — char count proxy for token count (synchronous, zero cost)
 *   3. Perplexity probe     — async SSM evaluate(); only runs when 1 & 2 are inconclusive
 *
 * When no bridge is available the router always returns 'ssm'.
 */

export type RoutingStrategy = 'auto' | 'ssm' | 'transformer';
export type RoutingDecision = 'ssm' | 'transformer';

export interface RouterContext {
    /**
     * Cached SSM perplexity for this input if already computed by the caller.
     * Providing this skips the async perplexity probe in auto mode.
     */
    perplexity?: number;
}

export interface InferenceRouterOptions {
    /**
     * Routing strategy.
     * - 'auto'        : heuristic routing (default)
     * - 'ssm'         : always route to SSM
     * - 'transformer' : always route to transformer (no-op when bridge absent)
     */
    strategy?            : RoutingStrategy;

    /**
     * Input character length above which auto-routing prefers the transformer.
     * ~1200 chars ≈ 300 tokens — beyond this a large context model tends to
     * outperform a small SSM for most tasks.
     * Default: 1200
     */
    longInputThreshold?  : number;

    /**
     * SSM perplexity above which auto-routing falls back to the transformer.
     * High perplexity signals the model hasn't seen this topic.
     * Default: 80
     */
    perplexityThreshold? : number;

    /**
     * Whether a transformer bridge is currently attached to the runtime.
     * The router uses this to short-circuit to 'ssm' when no bridge is present.
     * Set by SSMRuntime when constructing the router.
     * Default: false
     */
    hasBridge?           : boolean;

    /**
     * Callback for computing SSM perplexity, provided by SSMRuntime.
     * Avoids a circular import between InferenceRouter and SSMRuntime.
     * Only called in 'auto' mode as a last-resort heuristic.
     */
    perplexityProbe?     : (text: string) => Promise<number>;
}

// ── Complexity-signal regex patterns ─────────────────────────────────────────
// These match inputs that benefit from a larger transformer's broader knowledge.

const COMPLEXITY_PATTERNS: RegExp[] = [
    /step[\s-]by[\s-]step/i,
    /\banalyze\b|\banalyse\b/i,
    /\bexplain\s+(why|how|in\s+detail)/i,
    /\bcompare\b.*\band\b/i,
    /\bcontrast\b/i,
    /\bpros?\s+and\s+cons?\b/i,
    /\bsummariz/i,
    /\bwrite\s+a\s+(detailed|comprehensive|complete|full)\b/i,
    /\bwhat\s+are\s+the\s+(key|main|top)\s+(difference|reason|factor)/i,
];

export class InferenceRouter {
    private readonly _strategy           : RoutingStrategy;
    private readonly _longInputThreshold : number;
    private readonly _perplexityThreshold: number;
    private readonly _hasBridge          : boolean;
    private readonly _perplexityProbe    : ((text: string) => Promise<number>) | undefined;

    constructor(opts: InferenceRouterOptions = {}) {
        this._strategy            = opts.strategy            ?? 'auto';
        this._longInputThreshold  = opts.longInputThreshold  ?? 1200;
        this._perplexityThreshold = opts.perplexityThreshold ?? 80;
        this._hasBridge           = opts.hasBridge           ?? false;
        this._perplexityProbe     = opts.perplexityProbe;
    }

    /**
     * Routes `input` to either 'ssm' or 'transformer'.
     *
     * Always returns 'ssm' when no bridge is attached, regardless of strategy.
     */
    async route(input: string, ctx: RouterContext = {}): Promise<RoutingDecision> {
        // No bridge → always SSM
        if (!this._hasBridge) return 'ssm';

        // Fixed-strategy overrides
        if (this._strategy === 'ssm')         return 'ssm';
        if (this._strategy === 'transformer') return 'transformer';

        // Auto heuristics ────────────────────────────────────────────────────

        // 1. Complexity patterns (synchronous)
        if (COMPLEXITY_PATTERNS.some(p => p.test(input))) return 'transformer';

        // 2. Input length (synchronous)
        if (input.length > this._longInputThreshold) return 'transformer';

        // 3. Perplexity probe (async) — only if probe function provided
        const perplexity = ctx.perplexity ??
            (this._perplexityProbe ? await this._perplexityProbe(input) : undefined);

        if (perplexity !== undefined && perplexity > this._perplexityThreshold) {
            return 'transformer';
        }

        return 'ssm';
    }
}
