import { describe, expect, it, jest } from '@jest/globals';

import { EvalHarness, evaluateGate, formatReport } from '../src/eval/EvalHarness.js';
import {
    citationValidity,
    costBudget,
    embeddingSimilarity,
    exactMatch,
    llmJudge,
    noError,
    retrievalRecall,
    substringChecks,
} from '../src/eval/graders.js';
import type { EvalCase, EvalDataset, EvalOutput, EvalReport } from '../src/eval/types.js';
import type { RetrievedPassage } from '../src/rag/EnterpriseRetriever.js';
import { Tracer } from '../src/telemetry/Tracer.js';

const CASE: EvalCase = { id: 'c1', input: 'how long?', expected: 'seven years' };

function passage(id: string, sourceId: string, text = 'text'): RetrievedPassage {
    return { id, text, score: 1, sourceId, metadata: { sourceId } };
}

const out = (over: Partial<EvalOutput> = {}): EvalOutput => ({ answer: '', ...over });

describe('graders', () => {
    it('exact-match normalises whitespace, case and terminal punctuation', async () => {
        expect((await exactMatch({ evalCase: CASE, output: out({ answer: ' Seven Years. ' }) })).passed).toBe(true);
        expect((await exactMatch({ evalCase: CASE, output: out({ answer: 'six years' }) })).passed).toBe(false);
        expect((await exactMatch({ evalCase: { id: 'x', input: 'q' }, output: out() })).passed).toBe(true);
    });

    it('scores required and forbidden substrings together', async () => {
        const evalCase: EvalCase = {
            id: 'c', input: 'q', mustContain: ['ERR-4021'], mustNotContain: ['ACCT-999'],
        };

        const good = await substringChecks({ evalCase, output: out({ answer: 'incident ERR-4021 resolved' }) });
        expect(good.passed).toBe(true);
        expect(good.score).toBe(1);

        const leaked = await substringChecks({ evalCase, output: out({ answer: 'ERR-4021 for ACCT-999' }) });
        expect(leaked.passed).toBe(false);
        expect(leaked.detail).toContain('forbidden present: ACCT-999');
        expect(leaked.score).toBeCloseTo(0.5, 6);

        const missing = await substringChecks({ evalCase, output: out({ answer: 'nothing useful' }) });
        expect(missing.detail).toContain('missing: ERR-4021');

        expect((await substringChecks({ evalCase: CASE, output: out() })).detail).toBe('no substring constraints');
    });

    it('measures retrieval recall independently of generation', async () => {
        const evalCase: EvalCase = { id: 'c', input: 'q', expectedSources: ['policy', 'billing'] };

        const partial = await retrievalRecall({
            evalCase, output: out({ passages: [passage('policy#0', 'policy')] }),
        });
        expect(partial.score).toBeCloseTo(0.5, 6);
        expect(partial.passed).toBe(false);
        expect(partial.detail).toContain('not retrieved: billing');

        const full = await retrievalRecall({
            evalCase,
            output: out({ passages: [passage('policy#0', 'policy'), passage('billing#0', 'billing')] }),
        });
        expect(full.passed).toBe(true);

        expect((await retrievalRecall({ evalCase: CASE, output: out() })).passed).toBe(true);
    });

    it('catches an invented citation that similarity scoring would miss', async () => {
        const passages = [passage('a#0', 'a'), passage('b#0', 'b')];

        const valid = await citationValidity({ evalCase: CASE, output: out({ answer: 'yes [1] and [2]', passages }) });
        expect(valid.passed).toBe(true);

        const invented = await citationValidity({ evalCase: CASE, output: out({ answer: 'per [7]', passages }) });
        expect(invented.passed).toBe(false);
        expect(invented.detail).toContain('out-of-range citations: 7');

        const uncited = await citationValidity({ evalCase: CASE, output: out({ answer: 'trust me', passages }) });
        expect(uncited.passed).toBe(false);
        expect(uncited.detail).toContain('cites nothing');

        const noEvidence = await citationValidity({ evalCase: CASE, output: out({ answer: 'nothing found' }) });
        expect(noEvidence.passed).toBe(true);

        const citesNothingRetrieved = await citationValidity({ evalCase: CASE, output: out({ answer: 'per [1]' }) });
        expect(citesNothingRetrieved.passed).toBe(false);
    });

    it('scores embedding similarity against the reference', async () => {
        const embed = async (text: string) =>
            Float32Array.from(text.includes('seven') ? [1, 0] : [0, 1]);
        const grader = embeddingSimilarity({ embed, threshold: 0.9 });

        expect((await grader({ evalCase: CASE, output: out({ answer: 'seven years' }) })).passed).toBe(true);
        expect((await grader({ evalCase: CASE, output: out({ answer: 'ten days' }) })).passed).toBe(false);
        expect((await grader({ evalCase: { id: 'x', input: 'q' }, output: out() })).passed).toBe(true);
    });

    it('parses an LLM judge verdict and FAILS on an unparseable one', async () => {
        const grader = llmJudge({ generate: async () => 'SCORE: 8\nREASON: accurate and cited' });
        const good = await grader({ evalCase: CASE, output: out({ answer: 'seven years [1]', passages: [passage('a#0', 'a')] }) });
        expect(good.passed).toBe(true);
        expect(good.score).toBeCloseTo(0.8, 6);
        expect(good.detail).toBe('accurate and cited');

        const low = llmJudge({ generate: async () => 'SCORE: 3', threshold: 7 });
        expect((await low({ evalCase: CASE, output: out({ answer: 'wrong' }) })).passed).toBe(false);

        // A judge outage must not read as a pass.
        const broken = llmJudge({ generate: async () => 'the judge is confused' });
        const verdict = await broken({ evalCase: CASE, output: out({ answer: 'x' }) });
        expect(verdict.passed).toBe(false);
        expect(verdict.score).toBe(0);
        expect(verdict.detail).toContain('no parseable score');
    });

    it('grades errors and cost budgets', async () => {
        expect((await noError({ evalCase: CASE, output: out({ error: 'boom' }) })).passed).toBe(false);
        expect((await noError({ evalCase: CASE, output: out() })).passed).toBe(true);

        const budget = costBudget(0.01);
        expect((await budget({ evalCase: CASE, output: out({ costUsd: 0.005 }) })).passed).toBe(true);
        expect((await budget({ evalCase: CASE, output: out({ costUsd: 0.5 }) })).passed).toBe(false);
        expect((await budget({ evalCase: CASE, output: out() })).detail).toBe('cost not measured');
    });
});

describe('EvalHarness', () => {
    const dataset: EvalDataset = {
        name: 'retention-suite',
        cases: [
            { id: 'good', input: 'q1', expectedSources: ['policy'] },
            { id: 'bad', input: 'q2', expectedSources: ['billing'] },
        ],
    };

    it('runs cases, grades them and aggregates per grader', async () => {
        const harness = new EvalHarness({ graders: [noError, retrievalRecall, citationValidity] });

        const report = await harness.run(dataset, async (evalCase) =>
            evalCase.id === 'good'
                ? { answer: 'answer [1]', passages: [passage('policy#0', 'policy')] }
                : { answer: 'answer [1]', passages: [passage('other#0', 'other')] });

        expect(report.passRate).toBeCloseTo(0.5, 6);
        expect(report.graderScores['retrieval-recall']).toBeCloseTo(0.5, 6);
        expect(report.graderScores['citation-validity']).toBe(1);
        expect(report.errorRate).toBe(0);
        expect(report.cases.map((c) => c.caseId)).toEqual(['good', 'bad']);
    });

    it('records a thrown target as a failing case rather than crashing the run', async () => {
        const harness = new EvalHarness({ graders: [noError] });
        const report = await harness.run(dataset, async (evalCase) => {
            if (evalCase.id === 'bad') throw new Error('target exploded');
            return { answer: 'ok' };
        });

        expect(report.errorRate).toBeCloseTo(0.5, 6);
        expect(report.cases[1]?.output.error).toBe('target exploded');
        expect(report.cases[1]?.passed).toBe(false);
    });

    it('records a thrown grader as a failed grade', async () => {
        const harness = new EvalHarness({
            graders: [() => { throw new Error('grader broke'); }],
        });
        const report = await harness.run({ name: 'x', cases: [CASE] }, async () => ({ answer: 'a' }));
        expect(report.cases[0]?.grades[0]?.grader).toBe('grader-error');
        expect(report.cases[0]?.grades[0]?.detail).toBe('grader broke');
    });

    it('attributes per-case cost from the case\'s trace', async () => {
        const tracer = new Tracer();
        const harness = new EvalHarness({ graders: [noError], tracer });

        const report = await harness.run({ name: 'costed', cases: [CASE] }, async () => {
            const span = tracer.startSpan('llm.generate', { kind: 'llm' });
            span.recordUsage({ model: 'claude-haiku-4-5', inputTokens: 1_000_000, outputTokens: 0 });
            span.end();
            return { answer: 'a', traceId: span.traceId };
        });

        expect(report.totalCostUsd).toBeCloseTo(1, 6);
        expect(report.meanCostUsd).toBeCloseTo(1, 6);
    });

    it('measures latency when the target does not report it', async () => {
        const harness = new EvalHarness({ graders: [noError] });
        const report = await harness.run({ name: 'x', cases: [CASE] }, async () => ({ answer: 'a' }));
        expect(report.cases[0]?.output.latencyMs).toBeGreaterThanOrEqual(0);
        expect(report.p95LatencyMs).toBeGreaterThanOrEqual(0);
    });

    it('respects the concurrency limit', async () => {
        let active = 0;
        let peak = 0;
        const harness = new EvalHarness({ graders: [noError], concurrency: 2 });
        const many: EvalDataset = {
            name: 'many',
            cases: Array.from({ length: 8 }, (_, i) => ({ id: `c${i}`, input: 'q' })),
        };

        await harness.run(many, async () => {
            active += 1;
            peak = Math.max(peak, active);
            await Promise.resolve();
            active -= 1;
            return { answer: 'a' };
        });
        expect(peak).toBeLessThanOrEqual(2);
    });
});

describe('gates', () => {
    const baseReport = (over: Partial<EvalReport> = {}): EvalReport => ({
        dataset: 'd', cases: [], passRate: 0.9, meanScore: 0.8,
        graderScores: { 'retrieval-recall': 0.7 }, errorRate: 0.05,
        meanCostUsd: 0.002, totalCostUsd: 0.02, p50LatencyMs: 100, p95LatencyMs: 400,
        durationMs: 10, gate: { passed: true, failures: [], vacuous: true },
        ...over,
    });

    it('declares an empty gate vacuous rather than implying a pass', () => {
        const verdict = evaluateGate(baseReport(), {});
        expect(verdict.passed).toBe(true);
        expect(verdict.vacuous).toBe(true);
    });

    it('reports each breached threshold by name', () => {
        const verdict = evaluateGate(baseReport(), {
            minPassRate: 0.95,
            minMeanScore: 0.9,
            maxErrorRate: 0.01,
            maxMeanCostUsd: 0.001,
            maxP95LatencyMs: 200,
            minGraderScore: { 'retrieval-recall': 0.85, 'citation-validity': 0.9 },
        });

        expect(verdict.passed).toBe(false);
        expect(verdict.vacuous).toBe(false);
        expect(verdict.failures).toHaveLength(7);
        expect(verdict.failures.join('\n')).toContain('pass rate 90.0%');
        expect(verdict.failures.join('\n')).toContain('did not run');
    });

    it('passes when every threshold is met', () => {
        const verdict = evaluateGate(baseReport(), {
            minPassRate: 0.8, minMeanScore: 0.5, maxErrorRate: 0.1,
            maxMeanCostUsd: 0.01, maxP95LatencyMs: 500,
            minGraderScore: { 'retrieval-recall': 0.5 },
        });
        expect(verdict).toMatchObject({ passed: true, vacuous: false, failures: [] });
    });

    it('formats a CI-readable report', async () => {
        const harness = new EvalHarness({
            graders: [noError, retrievalRecall],
            gate: { minPassRate: 1 },
        });
        const report = await harness.run(
            { name: 'suite', cases: [{ id: 'miss', input: 'q', expectedSources: ['policy'] }] },
            async () => ({ answer: 'a', passages: [] }),
        );

        const text = formatReport(report);
        expect(text).toContain('Eval: suite — 1 cases');
        expect(text).toContain('GATE: FAIL');
        expect(text).toContain('- miss: retrieval-recall');

        const vacuous = formatReport({ ...report, gate: { passed: true, failures: [], vacuous: true } });
        expect(vacuous).toContain('proves nothing yet');

        const clean = formatReport({
            ...report, cases: [], passRate: 1,
            gate: { passed: true, failures: [], vacuous: false },
        });
        expect(clean).toContain('GATE: PASS');
    });

    it('truncates a long failing-case list', async () => {
        const harness = new EvalHarness({ graders: [noError] });
        const report = await harness.run(
            { name: 'big', cases: Array.from({ length: 12 }, (_, i) => ({ id: `c${i}`, input: 'q' })) },
            async () => ({ answer: '', error: 'boom' }),
        );
        expect(formatReport(report)).toContain('and 2 more');
    });
});
