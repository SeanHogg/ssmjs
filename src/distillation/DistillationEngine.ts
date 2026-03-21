/**
 * DistillationEngine – JS-only online knowledge distillation.
 *
 * The core insight: use a transformer as a *teacher* to generate high-quality
 * responses, then adapt the SSM *student* on those responses using WSLA.
 * This runs entirely in the browser with no Python or full-retraining required.
 *
 * Distillation flow:
 *   1. bridge.generate(input)  → teacher output
 *   2. runtime.adapt(teacherOutput, opts.adapt)  → SSM trains on it
 *   3. Return both results for inspection
 */

import type { AdaptOptions, AdaptResult } from '@seanhogg/mambakit';
import type { SSMRuntime } from '../runtime/SSMRuntime.js';
import type { TransformerBridge, BridgeGenerateOptions } from '../bridges/TransformerBridge.js';
import { SSMError } from '../errors/SSMError.js';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface DistillOptions {
    /**
     * Options forwarded to `runtime.adapt()`.
     * Default: { wsla: true, epochs: 3 }
     * WSLA is preferred because it is fast and targets the selective
     * projection rows — exactly the parameters that encode token routing.
     */
    adapt?    : AdaptOptions;

    /**
     * Options forwarded to `bridge.generate()`.
     */
    generate? : BridgeGenerateOptions;
}

export interface DistillResult {
    /** The input prompt that was distilled. */
    input        : string;
    /** The teacher's (transformer bridge) response to the input. */
    teacherOutput: string;
    /** The adapt() result from training the SSM on the teacher output. */
    adaptResult  : AdaptResult;
}

export interface DistillBatchResult {
    results    : DistillResult[];
    /** Total number of adapt epochs run across all inputs. */
    totalEpochs: number;
    /** Wall-clock time for the entire batch in milliseconds. */
    totalMs    : number;
}

// ── DistillationEngine ────────────────────────────────────────────────────────

export class DistillationEngine {
    private readonly _runtime: SSMRuntime;
    private readonly _bridge : TransformerBridge;

    /**
     * @param runtime The SSMRuntime whose SSM will be trained as the student.
     * @param bridge  The transformer bridge acting as teacher.
     *                A bridge must be provided — distillation requires one.
     */
    constructor(runtime: SSMRuntime, bridge: TransformerBridge) {
        this._runtime = runtime;
        this._bridge  = bridge;
    }

    /**
     * Runs a single distillation pass:
     *   1. Teacher generates a response for `input`
     *   2. SSM is adapted on the teacher's output (WSLA by default)
     *
     * The training signal is the teacher's full response — this teaches the
     * SSM what a good response to that prompt looks like, without requiring
     * labelled data or a loss function beyond the standard LM objective.
     */
    async distill(input: string, opts: DistillOptions = {}): Promise<DistillResult> {
        const adaptOpts: AdaptOptions = {
            wsla        : true,
            epochs      : 3,
            ...opts.adapt,
        };

        let teacherOutput: string;
        try {
            teacherOutput = await this._bridge.generate(input, opts.generate);
        } catch (err) {
            throw new SSMError(
                'DISTILL_FAILED',
                `Teacher bridge failed to generate for distillation: ${err instanceof Error ? err.message : String(err)}`,
                err,
            );
        }

        // Train the SSM on the teacher's output.
        // Prepend the input so the model learns the (prompt → response) mapping.
        const trainingText = `${input}\n${teacherOutput}`;

        let adaptResult: AdaptResult;
        try {
            adaptResult = await this._runtime.adapt(trainingText, adaptOpts);
        } catch (err) {
            throw new SSMError(
                'DISTILL_FAILED',
                `SSM adaptation failed during distillation: ${err instanceof Error ? err.message : String(err)}`,
                err,
            );
        }

        return { input, teacherOutput, adaptResult };
    }

    /**
     * Runs distillation for each input in sequence.
     * Aggregate statistics are returned alongside individual results.
     */
    async distillBatch(inputs: string[], opts: DistillOptions = {}): Promise<DistillBatchResult> {
        const startMs = Date.now();
        const results: DistillResult[] = [];
        let totalEpochs = 0;

        for (const input of inputs) {
            const result = await this.distill(input, opts);
            results.push(result);
            totalEpochs += result.adaptResult.epochCount;
        }

        return {
            results,
            totalEpochs,
            totalMs: Date.now() - startMs,
        };
    }
}
