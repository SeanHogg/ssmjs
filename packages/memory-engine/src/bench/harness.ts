/**
 * bench/harness.ts — run a benchmark and A/B two models.
 *
 * The harness drives a model over held-out token sequences, scores next-token
 * prediction with {@link metrics}, and produces a {@link BenchmarkReport}. It
 * also offers a one-call {@link trainAndBenchmark} that builds a fresh
 * `EvermindLM` from a corpus, holds out a slice, trains on the rest, and scores
 * — the path the Studio uses to benchmark a model in the browser.
 */

import { EvermindLM, EvermindLMTrainer } from "../lm/evermind_lm.js";
import { BPETokenizer } from "../tokenizer/bpe.js";
import { SeededRng } from "../utils/rng.js";
import { newAccumulator, scoreInto, perplexity, bitsPerToken } from "./metrics.js";
import type {
  AsyncLogitsModel,
  BenchmarkOptions,
  BenchmarkReport,
  ComparisonReport,
  LogitsModel,
  TrainAndBenchmarkOptions,
  TrainAndBenchmarkResult,
} from "./types.js";

/** A monotonic clock, preferring `performance.now()` when present. */
export function defaultNow(): number {
  const g = globalThis as { performance?: { now(): number } };
  return typeof g.performance?.now === "function" ? g.performance.now() : Date.now();
}

/** Next-token targets for a sequence: tokens[1..T-1] predicted from positions 0..T-2. */
function nextTokenTargets(seq: number[]): number[] {
  return seq.slice(1);
}

function finalize(
  acc: { tokens: number; ceSum: number; top1Hits: number; topKHits: number },
  sequences: number,
  topK: number,
  elapsedMs: number | undefined,
): BenchmarkReport {
  const meanCe = acc.tokens > 0 ? acc.ceSum / acc.tokens : 0;
  const report: BenchmarkReport = {
    sequences,
    tokens: acc.tokens,
    crossEntropy: meanCe,
    perplexity: perplexity(meanCe),
    bitsPerToken: bitsPerToken(meanCe),
    top1Accuracy: acc.tokens > 0 ? acc.top1Hits / acc.tokens : 0,
    topKAccuracy: acc.tokens > 0 ? acc.topKHits / acc.tokens : 0,
    topK,
  };
  if (elapsedMs !== undefined) {
    report.elapsedMs = elapsedMs;
    report.tokensPerSecond = elapsedMs > 0 ? (acc.tokens / elapsedMs) * 1000 : 0;
  }
  return report;
}

/**
 * Benchmark a model (synchronous forward) over held-out token sequences.
 * Sequences shorter than 2 tokens have no prediction target and are skipped.
 */
export function benchmarkModel(
  model: LogitsModel,
  sequences: number[][],
  opts: BenchmarkOptions = {},
): BenchmarkReport {
  const topK = opts.topK ?? 5;
  const measure = opts.measureLatency ?? true;
  const now = opts.now ?? defaultNow;
  const acc = newAccumulator();
  let scored = 0;
  let elapsed = 0;
  for (const seq of sequences) {
    if (seq.length < 2) continue;
    const t0 = measure ? now() : 0;
    const { logits } = model.forward(seq);
    if (measure) elapsed += now() - t0;
    scoreInto(acc, logits, nextTokenTargets(seq), topK);
    scored++;
  }
  return finalize(acc, scored, topK, measure ? elapsed : undefined);
}

/** Benchmark a model whose forward pass is asynchronous (e.g. a WebGPU backend). */
export async function benchmarkModelAsync(
  model: AsyncLogitsModel,
  sequences: number[][],
  opts: BenchmarkOptions = {},
): Promise<BenchmarkReport> {
  const topK = opts.topK ?? 5;
  const measure = opts.measureLatency ?? true;
  const now = opts.now ?? defaultNow;
  const acc = newAccumulator();
  let scored = 0;
  let elapsed = 0;
  for (const seq of sequences) {
    if (seq.length < 2) continue;
    const t0 = measure ? now() : 0;
    const { logits } = await model.forward(seq);
    if (measure) elapsed += now() - t0;
    scoreInto(acc, logits, nextTokenTargets(seq), topK);
    scored++;
  }
  return finalize(acc, scored, topK, measure ? elapsed : undefined);
}

/** A/B two models on the same eval set. Lower perplexity wins (the primary metric). */
export function compareModels(
  candidate: LogitsModel,
  baseline: LogitsModel,
  sequences: number[][],
  opts: BenchmarkOptions = {},
): ComparisonReport {
  return compareReports(benchmarkModel(candidate, sequences, opts), benchmarkModel(baseline, sequences, opts));
}

/** Build a {@link ComparisonReport} from two already-computed scorecards. */
export function compareReports(candidate: BenchmarkReport, baseline: BenchmarkReport): ComparisonReport {
  const perplexityDelta = candidate.perplexity - baseline.perplexity;
  const perplexityRatio = baseline.perplexity > 0 ? candidate.perplexity / baseline.perplexity : Infinity;
  const top1Delta = candidate.top1Accuracy - baseline.top1Accuracy;
  // A small relative band counts as a tie so noise doesn't flip the verdict.
  const tieBand = 0.005;
  let winner: ComparisonReport["winner"];
  if (Math.abs(perplexityRatio - 1) <= tieBand) winner = "tie";
  else winner = perplexityDelta < 0 ? "candidate" : "baseline";
  const pct = ((1 - perplexityRatio) * 100).toFixed(1);
  const summary =
    winner === "tie"
      ? `Tie: perplexity ${candidate.perplexity.toFixed(2)} vs ${baseline.perplexity.toFixed(2)}`
      : winner === "candidate"
        ? `Candidate wins: ${pct}% lower perplexity (${candidate.perplexity.toFixed(2)} vs ${baseline.perplexity.toFixed(2)})`
        : `Baseline wins: candidate ${(-Number(pct)).toFixed(1)}% higher perplexity (${candidate.perplexity.toFixed(2)} vs ${baseline.perplexity.toFixed(2)})`;
  return { candidate, baseline, perplexityDelta, perplexityRatio, top1Delta, winner, summary };
}

/** Split a corpus into next-token training sequences (one per sentence). */
export function corpusToSequences(corpus: string, codec: { encode(text: string): number[] }): number[][] {
  return corpus
    .split(/(?<=\.)\s+/)
    .map((s) => codec.encode(s.trim()))
    .filter((ids) => ids.length >= 2);
}

/** Benchmark an existing model against a raw-text corpus (tokenized per sentence). */
export function benchmarkText(
  model: LogitsModel,
  codec: { encode(text: string): number[] },
  corpus: string,
  opts: BenchmarkOptions = {},
): BenchmarkReport {
  return benchmarkModel(model, corpusToSequences(corpus, codec), opts);
}

/** Deterministic Fisher–Yates shuffle (in place) driven by a seeded RNG. */
function shuffle<T>(arr: T[], rng: SeededRng): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng.next() * (i + 1));
    const tmp = arr[i]!;
    arr[i] = arr[j]!;
    arr[j] = tmp;
  }
  return arr;
}

/**
 * Train a fresh `EvermindLM` on a corpus and benchmark it on a held-out slice —
 * the canonical "build + score a model from text" path. A real benchmark must
 * score data the model never trained on, so `heldOutRatio` is clamped to leave
 * at least one eval sequence and one train sequence.
 */
export function trainAndBenchmark(
  corpus: string,
  opts: TrainAndBenchmarkOptions = {},
): TrainAndBenchmarkResult {
  const seed = opts.seed ?? 7;
  const tok = new BPETokenizer();
  tok.train(corpus, { numMerges: opts.numMerges ?? 100 });

  const all = corpusToSequences(corpus, tok);
  if (all.length < 2) {
    throw new Error(
      `corpus produced ${all.length} trainable sequence(s); need at least 2 to hold out an eval split (add more sentences)`,
    );
  }

  // Deterministic split: shuffle by seed, reserve heldOutRatio for eval.
  const order = shuffle([...all], new SeededRng((seed >>> 0) || 1));
  const ratio = Math.min(0.9, Math.max(0.05, opts.heldOutRatio ?? 0.25));
  let evalCount = Math.round(order.length * ratio);
  evalCount = Math.max(1, Math.min(order.length - 1, evalCount));
  const evalSeqs = order.slice(0, evalCount);
  const trainSeqs = order.slice(evalCount);

  const model = new EvermindLM({
    vocabSize: tok.vocabSize,
    dModel: opts.dModel ?? 32,
    numLayers: opts.numLayers ?? 2,
    hiddenDim: opts.hiddenDim ?? 48,
    seed,
  });
  const epochs = opts.epochs ?? 30;
  const history = new EvermindLMTrainer(model, { lr: opts.lr ?? 0.03, epochs }).fit(trainSeqs);

  const report = benchmarkModel(model, evalSeqs, { topK: opts.topK ?? 5, measureLatency: true });
  const sample = model.generateText(opts.prompt ?? "The", tok, { maxNewTokens: 8, temperature: 0 });

  return {
    ...report,
    trainSequences: trainSeqs.length,
    evalSequences: evalSeqs.length,
    initialTrainLoss: history[0] ?? 0,
    finalTrainLoss: history.at(-1) ?? 0,
    vocabSize: tok.vocabSize,
    sample,
  };
}
