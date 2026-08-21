/**
 * bench/types.ts — public types for the Evermind benchmarking harness.
 *
 * Benchmarking answers the question the marketplace (and the "beats a frozen
 * LLM" thesis) actually rests on: *how good is this model?* The harness measures
 * a trained model on held-out data with the standard language-model yardsticks —
 * perplexity, bits-per-token, and next-token accuracy — plus generation
 * throughput, and can A/B two models (e.g. a quantized vs full-precision build,
 * or a freshly-adapted vs prior checkpoint).
 *
 * The metrics operate on any object that can produce per-position logits from a
 * token sequence — the engine's `EvermindLM` satisfies {@link LogitsModel}
 * directly — so the harness is model-agnostic and dependency-free.
 */

/** The minimal surface a model must expose to be benchmarked: per-position logits. */
export interface LogitsModel {
  /** Run the model over a token sequence; returns one logit vector per position. */
  forward(tokens: number[]): { logits: Float32Array[] };
}

/** A model whose forward pass is asynchronous (e.g. a WebGPU backend). */
export interface AsyncLogitsModel {
  forward(tokens: number[]): Promise<{ logits: Float32Array[] }>;
}

/** Knobs for a benchmark run. */
export interface BenchmarkOptions {
  /** k for top-k next-token accuracy. Default 5. */
  topK?: number;
  /** Measure forward throughput (tokens/sec). Default true. */
  measureLatency?: boolean;
  /**
   * Monotonic clock in milliseconds, injectable for deterministic tests.
   * Defaults to `performance.now()` when available, else `Date.now()`.
   */
  now?: () => number;
}

/** The scorecard a benchmark run produces. */
export interface BenchmarkReport {
  /** Number of evaluated sequences. */
  sequences: number;
  /** Number of predicted positions (next-token targets) scored. */
  tokens: number;
  /** Mean next-token cross-entropy, in nats. Lower is better. */
  crossEntropy: number;
  /** Perplexity = exp(crossEntropy). Lower is better; 1.0 is perfect. */
  perplexity: number;
  /** Bits per token = crossEntropy / ln(2). Lower is better. */
  bitsPerToken: number;
  /** Fraction of positions where the argmax prediction was correct (0..1). */
  top1Accuracy: number;
  /** Fraction of positions where the true token was in the top-k (0..1). */
  topKAccuracy: number;
  /** The k used for {@link topKAccuracy}. */
  topK: number;
  /** Forward throughput in tokens/sec, when {@link BenchmarkOptions.measureLatency}. */
  tokensPerSecond?: number;
  /** Wall-clock spent in the forward passes (ms), when measured. */
  elapsedMs?: number;
}

/** The result of A/B-ing two models on the same eval set. */
export interface ComparisonReport {
  candidate: BenchmarkReport;
  baseline: BenchmarkReport;
  /** candidate.perplexity − baseline.perplexity (negative = candidate better). */
  perplexityDelta: number;
  /** candidate.perplexity / baseline.perplexity (<1 = candidate better). */
  perplexityRatio: number;
  /** candidate.top1Accuracy − baseline.top1Accuracy (positive = candidate better). */
  top1Delta: number;
  /** Which model won on perplexity (the primary metric). */
  winner: "candidate" | "baseline" | "tie";
  /** Human-readable one-liner. */
  summary: string;
}

/** Options for {@link trainAndBenchmark}: train a fresh EvermindLM, then score it. */
export interface TrainAndBenchmarkOptions {
  /** BPE merges to learn for the tokenizer. Default 100. */
  numMerges?: number;
  /** Model channel dimension. Default 32. */
  dModel?: number;
  /** Number of (conv + MoE) blocks. Default 2. */
  numLayers?: number;
  /** MoE expert FFN hidden width. Default 48. */
  hiddenDim?: number;
  /** Training epochs. Default 30. */
  epochs?: number;
  /** AdamW learning rate. Default 0.03. */
  lr?: number;
  /** Deterministic seed (model init + held-out split). Default 7. */
  seed?: number;
  /**
   * Fraction of sequences reserved for evaluation (never trained on). Default
   * 0.25. A real benchmark must score held-out data, so this is enforced > 0.
   */
  heldOutRatio?: number;
  /** k for top-k accuracy. Default 5. */
  topK?: number;
  /** Prompt used to capture a qualitative generation sample. Default "The". */
  prompt?: string;
}

/** {@link trainAndBenchmark} result: the held-out scorecard plus training context. */
export interface TrainAndBenchmarkResult extends BenchmarkReport {
  /** Sequences used for training. */
  trainSequences: number;
  /** Held-out sequences used for evaluation. */
  evalSequences: number;
  /** Final-epoch mean training loss (nats). */
  finalTrainLoss: number;
  /** First-epoch mean training loss (nats) — pairs with final to show the drop. */
  initialTrainLoss: number;
  /** Learned tokenizer vocabulary size. */
  vocabSize: number;
  /** A short greedy generation from {@link TrainAndBenchmarkOptions.prompt}. */
  sample: string;
}

// ── Adaptation cost ───────────────────────────────────────────────────────────

/**
 * Knobs for {@link benchmarkAdaptationCost} — the measured COST of on-prem
 * adaptation, as opposed to the quality metrics above.
 *
 * The defaults bracket what the on-prem host actually runs today
 * (`agent-runtime/src/infra/project-evermind-delta.ts`, mirrored by the cloud
 * coordinator): 64-token training windows, ONE epoch per contribution, over at
 * most 4000 characters of run text, with contributions under 20 characters
 * skipped before any work happens.
 */
export interface AdaptationCostOptions {
  /** Corpus the tokenizer is learned from and the adaptation trains on. */
  text?: string;
  /** Tokens per training window. Default `[32, 64, 128]` (production: 64). */
  windowTokens?: number[];
  /** Epochs per adaptation. Default `[1, 2, 4]` (production: 1). */
  epochs?: number[];
  /**
   * Fraction of incoming contributions skipped before any work, 0..1. The
   * on-prem skip is a floor on contribution length (`MIN_TEXT_CHARS`), so this is
   * that floor expressed as how often it fires. Default `[0, 0.5]`.
   */
  skipRates?: number[];
  /** Adaptation requests issued per grid point. Default 4. */
  requests?: number;
  /** BPE merges learned for the tokenizer. Default 120. */
  numMerges?: number;
  /** Model channel dimension. Default 32. */
  dModel?: number;
  /** Number of (conv + MoE) blocks. Default 2. */
  numLayers?: number;
  /** MoE expert FFN hidden width. Default 48. */
  hiddenDim?: number;
  /** AdamW learning rate for the adaptation fits. Default 0.01. */
  lr?: number;
  /** Deterministic seed for model init. Default 7. */
  seed?: number;
  /** Monotonic clock in ms, injectable for deterministic tests. */
  now?: () => number;
}

/** Measured cost at one (window, epochs, skipRate) point of the grid. */
export interface AdaptationCostPoint {
  /** Tokens per training window. */
  windowTokens: number;
  /** Epochs run per adaptation. */
  epochs: number;
  /** Fraction of requests skipped before any work. */
  skipRate: number;
  /** Adaptation requests issued at this point. */
  requests: number;
  /** Requests that actually ran a fit (the rest were skipped). */
  adapted: number;
  /** Training windows fitted, summed over the adaptations that ran. */
  sequences: number;
  /**
   * AdamW steps issued: one per window per epoch (the trainer's default, with
   * gradient accumulation pinned to 1 so the count is exact).
   */
  optimizerSteps: number;
  /**
   * Block positions pushed through the model, read from
   * `EvermindLM.positionsEvaluated`. Hardware-independent and exact — this is the
   * quantity a cost curve should be asserted on; wall time is what it costs HERE.
   */
  positionsEvaluated: number;
  /** Wall-clock milliseconds spent inside the fits. */
  elapsedMs: number;
  /** Milliseconds per 1000 evaluated positions — the machine's unit cost. */
  msPerKPosition: number;
  /** Wall-clock milliseconds per adaptation that actually ran (0 when all skipped). */
  msPerAdaptation: number;
}

/** The adaptation-cost scorecard: one point per (window, epochs, skipRate). */
export interface AdaptationCostReport {
  points: AdaptationCostPoint[];
  /** Tokens the source text produced (the adaptation's input size). */
  textTokens: number;
  /** Learned tokenizer vocabulary size. */
  vocabSize: number;
  /** Model channel dimension the grid was measured on. */
  dModel: number;
  /** Number of blocks the grid was measured on. */
  numLayers: number;
  /** MoE expert hidden width the grid was measured on. */
  hiddenDim: number;
  /** Trainable scalar parameters — what one optimiser step touches. */
  paramCount: number;
}
