/**
 * eval/types.ts — dataset, grader and gate vocabulary.
 */

import type { RetrievedPassage } from '../rag/EnterpriseRetriever.js';

export interface EvalCase {
    id: string;
    input: string;
    /** Reference answer, for similarity and judge graders. */
    expected?: string;
    /**
     * Source ids that MUST be retrieved for this case to be answerable. This is
     * what makes a retrieval regression visible independently of generation.
     */
    expectedSources?: string[];
    /** Strings that must appear in the output (exact identifiers, figures). */
    mustContain?: string[];
    /** Strings that must NOT appear (leaked PII, a competitor's name, a refusal). */
    mustNotContain?: string[];
    metadata?: Record<string, unknown>;
}

export interface EvalDataset {
    name: string;
    cases: EvalCase[];
}

/** What the system under test produced for one case. */
export interface EvalOutput {
    answer: string;
    /** Passages the system retrieved, when it is a RAG system. */
    passages?: readonly RetrievedPassage[];
    /** Measured cost for this case, in USD. */
    costUsd?: number;
    latencyMs?: number;
    traceId?: string;
    error?: string;
}

/** Runs the system under test for one case. */
export type EvalTarget = (evalCase: EvalCase) => Promise<EvalOutput>;

export interface GradeResult {
    grader: string;
    /** 0..1. Graders that are inherently binary return 0 or 1. */
    score: number;
    passed: boolean;
    detail?: string;
}

export type Grader = (args: {
    evalCase: EvalCase;
    output: EvalOutput;
}) => Promise<GradeResult> | GradeResult;

export interface CaseResult {
    caseId: string;
    output: EvalOutput;
    grades: GradeResult[];
    /** Mean of the grade scores. */
    score: number;
    passed: boolean;
}

/**
 * The launch threshold. Every field is optional so a team can start with one
 * number and add the rest as the system matures — but a gate with no fields set
 * passes everything, and {@link EvalHarness.run} reports that explicitly rather
 * than implying the suite was meaningful.
 */
export interface EvalGate {
    /** Minimum share of cases that must pass. */
    minPassRate?: number;
    /** Minimum mean score across all cases. */
    minMeanScore?: number;
    /** Per-grader minimum mean score — catches a regression one dimension deep. */
    minGraderScore?: Record<string, number>;
    maxMeanCostUsd?: number;
    maxP95LatencyMs?: number;
    /** Maximum share of cases that may error outright. */
    maxErrorRate?: number;
}

export interface GateVerdict {
    passed: boolean;
    /** Human-readable reasons a gate failed, empty when it passed. */
    failures: string[];
    /** True when the gate declared no thresholds and therefore proves nothing. */
    vacuous: boolean;
}

export interface EvalReport {
    dataset: string;
    cases: CaseResult[];
    passRate: number;
    meanScore: number;
    /** Mean score per grader — where a regression actually happened. */
    graderScores: Record<string, number>;
    errorRate: number;
    meanCostUsd: number;
    totalCostUsd: number;
    p50LatencyMs: number;
    p95LatencyMs: number;
    durationMs: number;
    gate: GateVerdict;
}
