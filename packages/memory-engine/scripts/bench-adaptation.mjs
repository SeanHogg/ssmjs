#!/usr/bin/env node
/**
 * scripts/bench-adaptation.mjs — run the adaptation-cost benchmark and print the
 * table.
 *
 *   pnpm --filter @seanhogg/builderforce-memory-engine bench:adapt
 *   node scripts/bench-adaptation.mjs --requests=8 --windows=64 --epochs=1,2
 *
 * Defaults bracket the on-prem settings the host actually runs (64-token windows,
 * one epoch per contribution). Numbers are wall-clock on THIS machine; the
 * `positions` column is hardware-independent, so it is the one to compare across
 * machines.
 */

import { benchmarkAdaptationCost, formatAdaptationCostReport } from "../dist/index.js";

const arg = (name) => {
  const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.slice(name.length + 3) : undefined;
};
const nums = (name) => {
  const v = arg(name);
  return v ? v.split(",").map(Number).filter((n) => Number.isFinite(n)) : undefined;
};

const report = benchmarkAdaptationCost({
  ...(nums("windows") ? { windowTokens: nums("windows") } : {}),
  ...(nums("epochs") ? { epochs: nums("epochs") } : {}),
  ...(nums("skips") ? { skipRates: nums("skips") } : {}),
  ...(arg("requests") ? { requests: Number(arg("requests")) } : {}),
  ...(arg("dModel") ? { dModel: Number(arg("dModel")) } : {}),
  ...(arg("layers") ? { numLayers: Number(arg("layers")) } : {}),
  ...(arg("hidden") ? { hiddenDim: Number(arg("hidden")) } : {}),
});

process.stdout.write(`${formatAdaptationCostReport(report)}\n`);
