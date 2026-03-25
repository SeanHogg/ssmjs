/**
 * InferenceRouter – decides whether a given input should be handled by the
 * local SSM or routed to a transformer bridge.
 *
 * Three heuristics run in priority order (cheapest first):
 *   1. Complexity patterns  — regex against the input string (synchronous, zero cost)
 *   2. Input length         — char count proxy for token count (synchronous, zero cost)
 *   3. Perplexity probe     — async SSM evaluate(); only runs when 1 & 2 are inconclusive
 *
 * When no bridge is available the router always returns target='ssm'.
 */

export type RoutingStrategy = 'auto' | 'ssm' | 'transformer';

/**
 * The structured result of a routing decision.
 */
export interface RoutingDecision {
    /** Which model should handle this input. */
    target     : 'ssm' | 'transformer';
    /** The primary heuristic that triggered this decision. */
    reason     : 'strategy' | 'complexity' | 'length' | 'perplexity' | 'no_bridge';
    /** Confidence score in the range 0–1. */
    confidence : number;
    /** Optional human-readable explanation. */
    details?   : string;
}

export interface RouterContext {
    /**
     * Cached SSM perplexity for this input if already computed by the caller.
     * Providing this skips the async perplexity probe in auto mode.
     */
    perplexity?: number;
}

export interface RoutingAuditEntry {
    timestamp  : number;
    inputLength: number;
    decision   : RoutingDecision;
    durationMs : number;
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

/** Maximum audit log entries to keep in memory. */
const MAX_AUDIT_LOG = 500;

export class InferenceRouter {
    private readonly _strategy           : RoutingStrategy;
    private readonly _longInputThreshold : number;
    private readonly _perplexityThreshold: number;
    private readonly _hasBridge          : boolean;
    private readonly _perplexityProbe    : ((text: string) => Promise<number>) | undefined;
    private readonly _auditLog           : RoutingAuditEntry[] = [];

    constructor(opts: InferenceRouterOptions = {}) {
        this._strategy            = opts.strategy            ?? 'auto';
        this._longInputThreshold  = opts.longInputThreshold  ?? 1200;
        this._perplexityThreshold = opts.perplexityThreshold ?? 80;
        this._hasBridge           = opts.hasBridge           ?? false;
        this._perplexityProbe     = opts.perplexityProbe;
    }

    /**
     * Routes `input` to either SSM or transformer and returns a RoutingDecision.
     *
     * Always returns target='ssm' when no bridge is attached, regardless of strategy.
     */
    async route(input: string, ctx: RouterContext = {}): Promise<RoutingDecision> {
        const startMs = Date.now();
        const decision = await this._decide(input, ctx);
        const durationMs = Date.now() - startMs;

        const auditEntry: RoutingAuditEntry = {
            timestamp  : Date.now(),
            inputLength: input.length,
            decision,
            durationMs,
        };
        this._auditLog.push(auditEntry);
        if (this._auditLog.length > MAX_AUDIT_LOG) {
            this._auditLog.shift();
        }

        return decision;
    }

    /** Returns a copy of the in-memory routing audit log (most recent last). */
    getAuditLog(): RoutingAuditEntry[] {
        return this._auditLog.slice();
    }

    // ── Private routing logic ─────────────────────────────────────────────────

    private async _decide(input: string, ctx: RouterContext): Promise<RoutingDecision> {
        // No bridge → always SSM
        if (!this._hasBridge) {
            return { target: 'ssm', reason: 'no_bridge', confidence: 1.0 };
        }

        // Fixed-strategy overrides
        if (this._strategy === 'ssm') {
            return { target: 'ssm', reason: 'strategy', confidence: 1.0 };
        }
        if (this._strategy === 'transformer') {
            return { target: 'transformer', reason: 'strategy', confidence: 1.0 };
        }

        // Auto heuristics ────────────────────────────────────────────────────

        // 1. Complexity patterns (synchronous)
        const matchedPattern = COMPLEXITY_PATTERNS.find(p => p.test(input));
        if (matchedPattern) {
            return {
                target    : 'transformer',
                reason    : 'complexity',
                confidence: 0.9,
                details   : `Matched pattern: ${matchedPattern.source}`,
            };
        }

        // 2. Input length (synchronous)
        if (input.length > this._longInputThreshold) {
            return {
                target    : 'transformer',
                reason    : 'length',
                confidence: 0.85,
                details   : `Input length ${input.length} exceeds threshold ${this._longInputThreshold}`,
            };
        }

        // 3. Perplexity probe (async) — only if probe function provided
        const perplexity = ctx.perplexity ??
            (this._perplexityProbe ? await this._perplexityProbe(input) : undefined);

        if (perplexity !== undefined && perplexity > this._perplexityThreshold) {
            return {
                target    : 'transformer',
                reason    : 'perplexity',
                confidence: Math.min(0.95, 0.5 + (perplexity - this._perplexityThreshold) / 200),
                details   : `SSM perplexity ${perplexity.toFixed(1)} exceeds threshold ${this._perplexityThreshold}`,
            };
        }

        return { target: 'ssm', reason: 'complexity', confidence: 0.8 };
    }
}
