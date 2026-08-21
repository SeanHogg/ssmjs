/**
 * bench/adaptation.ts — what does ON-PREM ADAPTATION COST?
 *
 * The rest of the harness measures how GOOD a model is. This measures how much
 * one write-through adaptation COSTS, which is the number an on-prem operator
 * needs before turning learning on: every run the host adapts on is CPU it is not
 * spending on the agent.
 *
 * It became measurable only once the gradient path was real. Before that a "fit"
 * was a forward pass plus an inert optimiser step, so any timing was a timing of
 * the wrong thing.
 *
 * The three knobs are the ones the host actually exposes
 * (`agent-runtime/src/infra/project-evermind-delta.ts`, and the cloud
 * coordinator's identical constants):
 *
 *   • WINDOW  — `ADAPT_WINDOW_TOKENS = 64`, the length of each training sequence
 *               the run text is chunked into;
 *   • EPOCHS  — `new EvermindLMTrainer(lm, { epochs: 1 })`, passes per contribution;
 *   • SKIP    — `MIN_TEXT_CHARS = 20`, the floor below which a contribution is
 *               dropped before any work happens; expressed here as how OFTEN the
 *               floor fires.
 *
 * Two quantities are reported, and the difference matters. `positionsEvaluated`
 * comes from the model's own counter: exact, integer, hardware-independent — the
 * thing a curve should be asserted on. `elapsedMs` is what those positions cost on
 * THIS machine, which is what an operator budgets with and what no assertion
 * should depend on.
 */

import { EvermindLM, EvermindLMTrainer } from "../lm/evermind_lm.js";
import { BPETokenizer } from "../tokenizer/bpe.js";
import { defaultNow } from "./harness.js";
import type { AdaptationCostOptions, AdaptationCostPoint, AdaptationCostReport } from "./types.js";

/**
 * Chunk token ids into fixed-length training windows, dropping any tail too short
 * to have a next-token target.
 *
 * A byte-for-byte mirror of the host's own `windows()` helper, so a benchmarked
 * fit sees exactly the batches a production adaptation would.
 */
export function tokenWindows(ids: number[], size: number): number[][] {
  const out: number[][] = [];
  const step = Math.max(1, Math.floor(size));
  for (let i = 0; i + 1 < ids.length; i += step) {
    const seq = ids.slice(i, i + step);
    if (seq.length >= 2) out.push(seq);
  }
  return out;
}

/**
 * How many of `requests` actually adapt at a given skip rate.
 *
 * Deterministic rather than sampled: a benchmark whose cost curve moved with an
 * RNG would be measuring the RNG.
 */
export function adaptationsAfterSkip(requests: number, skipRate: number): number {
  const rate = Math.min(1, Math.max(0, skipRate));
  return Math.round(requests * (1 - rate));
}

/** Default corpus — prose of roughly the shape a run-text contribution has. */
const DEFAULT_TEXT = [
  "The planning loop retrieves context before generating a response.",
  "Agents recall facts from the memory layer and act on them.",
  "Deployment runs on Cloudflare Workers and Durable Objects.",
  "Tools are gated by a capability registry and a policy pack.",
  "The coordinator merges weight deltas from every contributing host.",
  "A ticket moves through triage, execution, review and merge.",
  "Recall is ranked by embedding cosine fused with lexical scoring.",
  "The trainer adapts a private copy of the base checkpoint locally.",
].join(" ");

/** Total trainable scalars — what one optimiser step touches. */
function paramCountOf(model: EvermindLM): number {
  let n = 0;
  for (const p of model.parameters()) n += p.data.length;
  return n;
}

/**
 * Measure adaptation cost across a grid of (window, epochs, skip rate).
 *
 * Each grid point starts from a FRESH model, so the points are independent
 * measurements rather than a single model drifting through the grid (a diverged
 * checkpoint would otherwise make later points look different for reasons that
 * have nothing to do with cost).
 */
export function benchmarkAdaptationCost(opts: AdaptationCostOptions = {}): AdaptationCostReport {
  const text = opts.text ?? DEFAULT_TEXT;
  const windowTokens = opts.windowTokens ?? [32, 64, 128];
  const epochsGrid = opts.epochs ?? [1, 2, 4];
  const skipRates = opts.skipRates ?? [0, 0.5];
  const requests = Math.max(1, opts.requests ?? 4);
  const dModel = opts.dModel ?? 32;
  const numLayers = opts.numLayers ?? 2;
  const hiddenDim = opts.hiddenDim ?? 48;
  const seed = opts.seed ?? 7;
  const lr = opts.lr ?? 0.01;
  const now = opts.now ?? defaultNow;

  const tok = new BPETokenizer();
  tok.train(text, { numMerges: opts.numMerges ?? 120 });
  const ids = tok.encode(text);

  const newModel = (): EvermindLM =>
    new EvermindLM({ vocabSize: tok.vocabSize, dModel, numLayers, hiddenDim, seed });

  const points: AdaptationCostPoint[] = [];
  for (const w of windowTokens) {
    const windows = tokenWindows(ids, w);
    for (const epochs of epochsGrid) {
      for (const skipRate of skipRates) {
        const adapted = adaptationsAfterSkip(requests, skipRate);
        const model = newModel();
        model.resetStats();

        const t0 = now();
        for (let r = 0; r < adapted; r++) {
          // One contribution = one fit over the run text's windows, exactly as the
          // host does. accumSteps pinned to 1 so `optimizerSteps` is exact.
          new EvermindLMTrainer(model, { lr, epochs, accumSteps: 1 }).fit(windows);
        }
        const elapsedMs = now() - t0;

        const positionsEvaluated = model.positionsEvaluated;
        const sequences = windows.length * adapted;
        points.push({
          windowTokens: w,
          epochs,
          skipRate,
          requests,
          adapted,
          sequences,
          optimizerSteps: sequences * epochs,
          positionsEvaluated,
          elapsedMs,
          msPerKPosition: positionsEvaluated > 0 ? (elapsedMs / positionsEvaluated) * 1000 : 0,
          msPerAdaptation: adapted > 0 ? elapsedMs / adapted : 0,
        });
      }
    }
  }

  return {
    points,
    textTokens: ids.length,
    vocabSize: tok.vocabSize,
    dModel,
    numLayers,
    hiddenDim,
    paramCount: paramCountOf(newModel()),
  };
}

/** Render an {@link AdaptationCostReport} as a fixed-width table. */
export function formatAdaptationCostReport(report: AdaptationCostReport): string {
  const head =
    `Evermind adaptation cost — dModel=${report.dModel} layers=${report.numLayers} ` +
    `hidden=${report.hiddenDim} params=${report.paramCount.toLocaleString()} ` +
    `vocab=${report.vocabSize} textTokens=${report.textTokens}`;
  const cols = ["window", "epochs", "skip", "adapts", "seqs", "steps", "positions", "ms", "ms/adapt", "ms/1k pos"];
  const rows = report.points.map((p) => [
    String(p.windowTokens),
    String(p.epochs),
    p.skipRate.toFixed(2),
    `${p.adapted}/${p.requests}`,
    String(p.sequences),
    String(p.optimizerSteps),
    String(p.positionsEvaluated),
    p.elapsedMs.toFixed(1),
    p.msPerAdaptation.toFixed(1),
    p.msPerKPosition.toFixed(3),
  ]);
  const widths = cols.map((c, i) => Math.max(c.length, ...rows.map((r) => r[i]!.length)));
  const line = (cells: string[]): string => cells.map((c, i) => c.padStart(widths[i]!)).join("  ");
  return [head, "", line(cols), line(widths.map((w) => "-".repeat(w))), ...rows.map(line)].join("\n");
}
