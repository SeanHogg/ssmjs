/**
 * eval/EvalHarness.ts — run a dataset, grade it, and decide the gate.
 *
 * The harness measures cost and latency itself rather than trusting the target to
 * report them, because the number that matters is the one the caller experiences.
 * When a tracer is attached, per-case cost is read back from the SPANS that case
 * produced — the same numbers the production dashboard shows, so an eval budget
 * and a production bill are denominated identically.
 */

import { mapWithConcurrency } from '../ingest/IngestionPipeline.js';
import { percentile } from '../telemetry/MetricsRegistry.js';
import type { Tracer } from '../telemetry/Tracer.js';
import { DEFAULT_RAG_GRADERS } from './graders.js';
import type {
    CaseResult,
    EvalDataset,
    EvalGate,
    EvalReport,
    EvalTarget,
    GateVerdict,
    GradeResult,
    Grader,
} from './types.js';

export interface EvalHarnessOptions {
    graders?: readonly Grader[];
    gate?: EvalGate;
    /** Cases evaluated concurrently. Default 4. */
    concurrency?: number;
    /**
     * When supplied, per-case cost is summed from the spans of the case's trace.
     * The target must return its `traceId` for this to attribute anything.
     */
    tracer?: Tracer;
}

export class EvalHarness {
    private readonly _graders: readonly Grader[];
    private readonly _gate: EvalGate;
    private readonly _concurrency: number;
    private readonly _tracer: Tracer | undefined;

    constructor(opts: EvalHarnessOptions = {}) {
        this._graders = opts.graders ?? DEFAULT_RAG_GRADERS;
        this._gate = opts.gate ?? {};
        this._concurrency = Math.max(1, opts.concurrency ?? 4);
        this._tracer = opts.tracer;
    }

    async run(dataset: EvalDataset, target: EvalTarget): Promise<EvalReport> {
        const started = Date.now();
        const results: CaseResult[] = new Array(dataset.cases.length);

        await mapWithConcurrency(dataset.cases, this._concurrency, async (evalCase, index) => {
            const caseStarted = Date.now();
            let output;
            try {
                output = await target(evalCase);
            } catch (err) {
                // A thrown target is a data point, not a crashed suite — the run
                // must still produce a report covering every other case.
                output = { answer: '', error: err instanceof Error ? err.message : String(err) };
            }

            if (output.latencyMs === undefined) output.latencyMs = Date.now() - caseStarted;
            if (output.costUsd === undefined && this._tracer && output.traceId) {
                output.costUsd = this._traceCost(output.traceId);
            }

            const grades: GradeResult[] = [];
            for (const grader of this._graders) {
                try {
                    grades.push(await grader({ evalCase, output }));
                } catch (err) {
                    grades.push({
                        grader: 'grader-error',
                        score: 0,
                        passed: false,
                        detail: err instanceof Error ? err.message : String(err),
                    });
                }
            }

            const score = grades.length ? grades.reduce((s, g) => s + g.score, 0) / grades.length : 0;
            results[index] = {
                caseId: evalCase.id,
                output,
                grades,
                score,
                passed: grades.length > 0 && grades.every((g) => g.passed),
            };
        });

        return this._report(dataset.name, results, Date.now() - started);
    }

    private _traceCost(traceId: string): number {
        return (this._tracer as Tracer)
            .trace(traceId)
            .reduce((sum, span) => sum + (span.costUsd ?? 0), 0);
    }

    private _report(dataset: string, cases: CaseResult[], durationMs: number): EvalReport {
        const n = cases.length || 1;
        const latencies = cases.map((c) => c.output.latencyMs ?? 0);
        const costs = cases.map((c) => c.output.costUsd ?? 0);

        const graderTotals = new Map<string, { sum: number; count: number }>();
        for (const result of cases) {
            for (const g of result.grades) {
                const acc = graderTotals.get(g.grader) ?? { sum: 0, count: 0 };
                acc.sum += g.score;
                acc.count += 1;
                graderTotals.set(g.grader, acc);
            }
        }
        const graderScores: Record<string, number> = {};
        for (const [name, acc] of graderTotals) graderScores[name] = acc.sum / acc.count;

        const report: EvalReport = {
            dataset,
            cases,
            passRate: cases.filter((c) => c.passed).length / n,
            meanScore: cases.reduce((s, c) => s + c.score, 0) / n,
            graderScores,
            errorRate: cases.filter((c) => c.output.error).length / n,
            meanCostUsd: costs.reduce((s, c) => s + c, 0) / n,
            totalCostUsd: costs.reduce((s, c) => s + c, 0),
            p50LatencyMs: percentile(latencies, 0.5),
            p95LatencyMs: percentile(latencies, 0.95),
            durationMs,
            gate: { passed: true, failures: [], vacuous: true },
        };

        report.gate = evaluateGate(report, this._gate);
        return report;
    }
}

/** Applies a gate to a report. Pure, so a build can re-gate an archived report. */
export function evaluateGate(report: EvalReport, gate: EvalGate): GateVerdict {
    const failures: string[] = [];
    const thresholds = Object.entries(gate).filter(([, v]) => v !== undefined);

    if (gate.minPassRate !== undefined && report.passRate < gate.minPassRate) {
        failures.push(`pass rate ${pct(report.passRate)} < required ${pct(gate.minPassRate)}`);
    }
    if (gate.minMeanScore !== undefined && report.meanScore < gate.minMeanScore) {
        failures.push(`mean score ${report.meanScore.toFixed(3)} < required ${gate.minMeanScore}`);
    }
    if (gate.maxErrorRate !== undefined && report.errorRate > gate.maxErrorRate) {
        failures.push(`error rate ${pct(report.errorRate)} > allowed ${pct(gate.maxErrorRate)}`);
    }
    if (gate.maxMeanCostUsd !== undefined && report.meanCostUsd > gate.maxMeanCostUsd) {
        failures.push(`mean cost $${report.meanCostUsd.toFixed(6)} > allowed $${gate.maxMeanCostUsd}`);
    }
    if (gate.maxP95LatencyMs !== undefined && report.p95LatencyMs > gate.maxP95LatencyMs) {
        failures.push(`p95 latency ${Math.round(report.p95LatencyMs)}ms > allowed ${gate.maxP95LatencyMs}ms`);
    }
    for (const [grader, minimum] of Object.entries(gate.minGraderScore ?? {})) {
        const actual = report.graderScores[grader];
        if (actual === undefined) {
            failures.push(`gate names grader "${grader}", which did not run`);
        } else if (actual < minimum) {
            failures.push(`grader "${grader}" scored ${actual.toFixed(3)} < required ${minimum}`);
        }
    }

    return { passed: failures.length === 0, failures, vacuous: thresholds.length === 0 };
}

/** One-screen summary for CI logs. */
export function formatReport(report: EvalReport): string {
    const lines = [
        `Eval: ${report.dataset} — ${report.cases.length} cases in ${Math.round(report.durationMs)}ms`,
        `  pass rate     ${pct(report.passRate)}`,
        `  mean score    ${report.meanScore.toFixed(3)}`,
        `  error rate    ${pct(report.errorRate)}`,
        `  cost          $${report.totalCostUsd.toFixed(6)} total, $${report.meanCostUsd.toFixed(6)}/case`,
        `  latency       p50 ${Math.round(report.p50LatencyMs)}ms · p95 ${Math.round(report.p95LatencyMs)}ms`,
    ];

    for (const [grader, score] of Object.entries(report.graderScores)) {
        lines.push(`  ${grader.padEnd(20)} ${score.toFixed(3)}`);
    }

    if (report.gate.vacuous) {
        lines.push('  GATE: no thresholds configured — this suite proves nothing yet.');
    } else if (report.gate.passed) {
        lines.push('  GATE: PASS');
    } else {
        lines.push('  GATE: FAIL');
        for (const failure of report.gate.failures) lines.push(`    - ${failure}`);
    }

    const failed = report.cases.filter((c) => !c.passed);
    if (failed.length > 0) {
        lines.push(`  Failing cases (${failed.length}):`);
        for (const result of failed.slice(0, 10)) {
            const why = result.grades.filter((g) => !g.passed)
                .map((g) => `${g.grader}${g.detail ? ` (${g.detail})` : ''}`)
                .join(', ');
            lines.push(`    - ${result.caseId}: ${why}`);
        }
        if (failed.length > 10) lines.push(`    … and ${failed.length - 10} more`);
    }

    return lines.join('\n');
}

function pct(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
}
