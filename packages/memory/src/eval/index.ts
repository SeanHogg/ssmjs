/**
 * eval — the gate between a demo and a launch.
 *
 * An enterprise AI project dies in one of two places: it never ships because
 * nobody can say whether it is good enough, or it ships and regresses silently
 * because "good enough" was a vibe rather than a number. Both are the same missing
 * artefact — a dataset, graders that produce comparable scores, and a THRESHOLD
 * that a build can fail on.
 *
 * The graders here deliberately include the two that RAG systems fail on and
 * generic LLM evals miss:
 *
 *   • {@link retrievalRecall} — did the right SOURCE come back at all? A generation
 *     score cannot distinguish "the model reasoned badly" from "the model was handed
 *     nothing", and those have opposite fixes.
 *   • {@link citationValidity} — does every citation point at a passage that was
 *     actually retrieved? An answer with invented citations scores well on
 *     similarity and is worse than useless in an audit.
 *
 * Cost and latency are graded alongside quality, because a launch decision needs
 * all three and a quality-only harness quietly ships an unaffordable system.
 */

export * from './types.js';
export * from './graders.js';
export * from './EvalHarness.js';
