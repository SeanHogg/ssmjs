/**
 * bench — the Evermind benchmarking harness.
 *
 * Measures a trained model on held-out data (perplexity, bits-per-token,
 * next-token accuracy, throughput), A/Bs two models, offers a one-call
 * train-and-score path the Studio drives in the browser, and measures what one
 * on-prem ADAPTATION costs across the host's own window/epoch/skip knobs.
 */

export {
  benchmarkModel,
  benchmarkModelAsync,
  benchmarkText,
  compareModels,
  compareReports,
  corpusToSequences,
  trainAndBenchmark,
} from "./harness.js";

export {
  benchmarkAdaptationCost,
  formatAdaptationCostReport,
  tokenWindows,
  adaptationsAfterSkip,
} from "./adaptation.js";

export {
  argmax,
  topKIndices,
  perplexity,
  bitsPerToken,
  newAccumulator,
  scoreInto,
  LN2,
} from "./metrics.js";

export type {
  LogitsModel,
  AsyncLogitsModel,
  BenchmarkOptions,
  BenchmarkReport,
  ComparisonReport,
  TrainAndBenchmarkOptions,
  TrainAndBenchmarkResult,
  AdaptationCostOptions,
  AdaptationCostPoint,
  AdaptationCostReport,
} from "./types.js";
