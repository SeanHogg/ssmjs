/**
 * eval/graders.ts — the scoring functions.
 *
 * Each is a plain function of (case, output), which keeps them composable, cheap
 * to unit-test, and free of hidden state. The RAG-specific pair — retrieval recall
 * and citation validity — are the ones that make a report actionable: together
 * they separate "retrieval failed", "the model ignored good evidence", and "the
 * model invented a citation", which are three different bugs with three different
 * owners.
 */

import { cosineSimilarity } from '../similarity/index.js';
import { SOURCE_ID_FIELD } from '../ingest/types.js';
import type { Grader, GradeResult } from './types.js';

const grade = (grader: string, score: number, passed: boolean, detail?: string): GradeResult => ({
    grader,
    score,
    passed,
    ...(detail ? { detail } : {}),
});

/** Normalised exact match — whitespace, case and terminal punctuation are noise. */
export const exactMatch: Grader = ({ evalCase, output }) => {
    if (evalCase.expected === undefined) return grade('exact-match', 1, true, 'no reference answer');
    const normalize = (s: string) => s.trim().toLowerCase().replace(/[.!?]+$/, '').replace(/\s+/g, ' ');
    const hit = normalize(output.answer) === normalize(evalCase.expected);
    return grade('exact-match', hit ? 1 : 0, hit);
};

/**
 * Required and forbidden substrings.
 *
 * The forbidden list is the one that earns its place in an enterprise suite: it is
 * how "must never mention a competitor", "must never echo an account number" and
 * "must not refuse this legitimate question" become failing tests rather than
 * incident reports.
 */
export const substringChecks: Grader = ({ evalCase, output }) => {
    const answer = output.answer.toLowerCase();
    const required = evalCase.mustContain ?? [];
    const forbidden = evalCase.mustNotContain ?? [];
    if (required.length === 0 && forbidden.length === 0) {
        return grade('substring', 1, true, 'no substring constraints');
    }

    const missing = required.filter((s) => !answer.includes(s.toLowerCase()));
    const leaked = forbidden.filter((s) => answer.includes(s.toLowerCase()));

    const checks = required.length + forbidden.length;
    const failures = missing.length + leaked.length;
    const score = checks > 0 ? (checks - failures) / checks : 1;

    const detail = [
        missing.length ? `missing: ${missing.join(', ')}` : '',
        leaked.length ? `forbidden present: ${leaked.join(', ')}` : '',
    ].filter(Boolean).join('; ');

    return grade('substring', score, failures === 0, detail || undefined);
};

export interface SimilarityGraderOptions {
    embed: (text: string) => Promise<Float32Array>;
    /** Cosine similarity at/above which the case passes. Default 0.85. */
    threshold?: number;
}

/**
 * Embedding similarity to the reference answer.
 *
 * Useful because a correct answer rarely matches the reference word for word, and
 * dangerous alone because a fluent WRONG answer sits close to a right one in
 * embedding space. Pair it with `substringChecks` on the load-bearing facts.
 */
export function embeddingSimilarity(opts: SimilarityGraderOptions): Grader {
    const threshold = opts.threshold ?? 0.85;
    return async ({ evalCase, output }) => {
        if (!evalCase.expected) return grade('similarity', 1, true, 'no reference answer');
        const [a, b] = await Promise.all([opts.embed(output.answer), opts.embed(evalCase.expected)]);
        const score = Math.max(0, Math.min(1, cosineSimilarity(a, b)));
        return grade('similarity', score, score >= threshold, `cosine ${score.toFixed(3)}`);
    };
}

/**
 * Retrieval recall: the share of `expectedSources` that made it into the passages.
 *
 * Graded before generation is even considered, because a system that retrieves
 * nothing relevant cannot be fixed by prompt work — and a generation-only report
 * will send a team to do exactly that.
 */
export const retrievalRecall: Grader = ({ evalCase, output }) => {
    const expected = evalCase.expectedSources ?? [];
    if (expected.length === 0) return grade('retrieval-recall', 1, true, 'no expected sources');

    const retrieved = new Set<string>();
    for (const passage of output.passages ?? []) {
        retrieved.add(passage.sourceId);
        const meta = passage.metadata[SOURCE_ID_FIELD];
        if (typeof meta === 'string') retrieved.add(meta);
    }

    const found = expected.filter((id) => retrieved.has(id));
    const score = found.length / expected.length;
    const missing = expected.filter((id) => !retrieved.has(id));

    return grade(
        'retrieval-recall',
        score,
        score === 1,
        missing.length ? `not retrieved: ${missing.join(', ')}` : undefined,
    );
};

/**
 * Every `[n]` citation must reference a passage that was actually retrieved.
 *
 * An answer citing `[7]` when six passages were supplied is fabricating its
 * evidence trail. That scores fine on similarity and fails an audit, which is
 * precisely why it is graded separately.
 */
export const citationValidity: Grader = ({ output }) => {
    const passages = output.passages ?? [];
    const cited = [...output.answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]));

    if (passages.length === 0) {
        return grade('citation-validity', cited.length === 0 ? 1 : 0, cited.length === 0,
            cited.length ? 'answer cites passages although none were retrieved' : 'no passages supplied');
    }
    if (cited.length === 0) {
        return grade('citation-validity', 0, false, 'answer cites nothing despite having evidence');
    }

    const invalid = cited.filter((n) => n < 1 || n > passages.length);
    const score = (cited.length - invalid.length) / cited.length;
    return grade(
        'citation-validity',
        score,
        invalid.length === 0,
        invalid.length ? `out-of-range citations: ${[...new Set(invalid)].join(', ')}` : undefined,
    );
};

export interface LlmJudgeOptions {
    /** Judge model. Use a DIFFERENT, stronger model than the one under test. */
    generate: (prompt: string, systemPrompt: string) => Promise<string>;
    /** What "good" means for this dataset. */
    rubric?: string;
    /** Minimum 0-10 judge score to pass. Default 7. */
    threshold?: number;
    graderName?: string;
}

export const DEFAULT_JUDGE_RUBRIC =
    'Score the answer 0-10 on: factual correctness against the reference, completeness, ' +
    'and whether every claim is supported by the passages shown. ' +
    'Penalise confident claims with no support. Reply with exactly one line: "SCORE: <0-10>", ' +
    'then one line: "REASON: <one sentence>".';

/**
 * LLM-as-judge.
 *
 * The judge must be a different (and at least as capable) model than the one under
 * test — a model grading its own output rates its own failure modes as successes,
 * which is how a suite reports 9/10 while users report the opposite.
 */
export function llmJudge(opts: LlmJudgeOptions): Grader {
    const threshold = opts.threshold ?? 7;
    const name = opts.graderName ?? 'llm-judge';

    return async ({ evalCase, output }) => {
        const evidence = (output.passages ?? [])
            .map((p, i) => `[${i + 1}] ${p.text}`)
            .join('\n\n');

        const prompt = [
            `Question: ${evalCase.input}`,
            evalCase.expected ? `Reference answer: ${evalCase.expected}` : '',
            evidence ? `Passages shown to the system:\n${evidence}` : '',
            `Answer under review:\n${output.answer}`,
        ].filter(Boolean).join('\n\n');

        const verdict = await opts.generate(prompt, opts.rubric ?? DEFAULT_JUDGE_RUBRIC);
        const match = /score\s*:\s*(\d+(?:\.\d+)?)/i.exec(verdict);
        const raw = match ? Number(match[1]) : NaN;

        if (!Number.isFinite(raw)) {
            // An unparseable verdict is a FAILED grade, not a passing default: a
            // silent 1.0 here would hide every judge outage behind a green suite.
            return grade(name, 0, false, `judge returned no parseable score: ${verdict.slice(0, 120)}`);
        }

        const score = Math.max(0, Math.min(1, raw / 10));
        const reason = /reason\s*:\s*(.+)/i.exec(verdict)?.[1]?.trim();
        return grade(name, score, raw >= threshold, reason);
    };
}

/** Fails a case whose target threw — an error is a result, not an absence of one. */
export const noError: Grader = ({ output }) =>
    grade('no-error', output.error ? 0 : 1, !output.error, output.error);

/** Per-case cost ceiling, for the cases a product cannot afford to get wrong slowly. */
export function costBudget(maxUsd: number): Grader {
    return ({ output }) => {
        if (output.costUsd === undefined) return grade('cost-budget', 1, true, 'cost not measured');
        const passed = output.costUsd <= maxUsd;
        return grade('cost-budget', passed ? 1 : 0, passed, `$${output.costUsd.toFixed(6)} vs $${maxUsd}`);
    };
}

/** The grader set that is right for most RAG suites out of the box. */
export const DEFAULT_RAG_GRADERS: readonly Grader[] = [
    noError,
    retrievalRecall,
    citationValidity,
    substringChecks,
];
