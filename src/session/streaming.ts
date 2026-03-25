/**
 * streaming.ts – AsyncIterable token streaming adapter for the MambaSession layer.
 *
 * Wraps the step-by-step generation loop from MambaModel so that each
 * token is yielded immediately after sampling, enabling real-time streaming UIs.
 */

import type { HybridMambaModel, SamplingOptions } from '@seanhogg/mambacode.js';

/** Minimum effective temperature — avoids division by zero in sampling. */
const MIN_TEMPERATURE = 1e-7;

/**
 * Yields one token ID at a time, applying the same sampling logic as
 * `MambaModel.generate()` but yielding each step incrementally.
 */
export async function* tokenStream(
    model: HybridMambaModel,
    promptIds: number[],
    maxNewTokens: number,
    samplingOpts: SamplingOptions = {},
): AsyncGenerator<number> {
    const { temperature = 1.0, topK = 50, topP = 0.9 } = samplingOpts;
    const { vocabSize, eosId } = model.config;

    // ids is declared const because the variable itself is never reassigned;
    // ids.push() mutates the array contents, which const allows.
    const ids = [...promptIds];

    for (let step = 0; step < maxNewTokens; step++) {
        const { logits } = await model.forward(
            new Uint32Array(ids),
            1,
            ids.length,
        );

        // Only the last token position's logits are used for next-token prediction
        const lastLogits = logits.slice(
            (ids.length - 1) * vocabSize,
            ids.length * vocabSize,
        );

        const nextId = sampleToken(lastLogits, { temperature, topK, topP });
        ids.push(nextId);
        yield nextId;

        if (nextId === eosId) break;
    }
}

// ── Sampling (mirrors the private helper in mamba_model.js) ──────────────────

function sampleToken(
    logits: Float32Array,
    { temperature = 1.0, topK = 50, topP = 0.9 } = {},
): number {
    const n = logits.length;

    // Temperature scaling
    const scaled = new Float32Array(n);
    for (let i = 0; i < n; i++) scaled[i] = logits[i]! / Math.max(temperature, MIN_TEMPERATURE);

    // Softmax (numerically stable)
    let maxL = -Infinity;
    for (let i = 0; i < n; i++) if (scaled[i]! > maxL) maxL = scaled[i]!;
    let sumE = 0;
    const exps = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        exps[i] = Math.exp(scaled[i]! - maxL);
        sumE += exps[i]!;
    }

    // Top-K sort
    const indices = Array.from({ length: n }, (_, i) => i)
        .sort((a, b) => exps[b]! - exps[a]!);
    const topKIndices = indices.slice(0, topK);

    // Nucleus (Top-P) filtering
    let cumSum = 0;
    const nucleus: number[] = [];
    for (const idx of topKIndices) {
        cumSum += exps[idx]! / sumE;
        nucleus.push(idx);
        if (cumSum >= topP) break;
    }

    // Sample from nucleus
    let nucleusSum = 0;
    for (const idx of nucleus) nucleusSum += exps[idx]!;
    const threshold = Math.random() * nucleusSum;
    let acc = 0;
    for (const idx of nucleus) {
        acc += exps[idx]!;
        if (acc >= threshold) return idx;
    }
    return nucleus[nucleus.length - 1]!;
}
