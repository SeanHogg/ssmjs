/**
 * embedding — stand up the recall embedder for a headless server.
 *
 * Both out-of-process bins (stdio and HTTP) need the SAME decision: is an
 * embedding model configured, and if so where do its vectors get cached? That
 * decision lives here once, so the two servers cannot drift into ranking recall
 * differently from one another.
 */

import { createEvermindEmbedder, type PersistentTextEmbedder } from "./evermind-embedder.js";

export { createEvermindEmbedder, CachedTextEmbedder } from "./evermind-embedder.js";
export type { EvermindEmbedderOptions, PersistentTextEmbedder, TextEmbedderLike } from "./evermind-embedder.js";

/** Absolute path to a `.evermind` package (an `evermind-lm`) to embed recall with. */
export const MODEL_FILE_ENV = "BUILDERFORCE_MEMORY_MODEL";
/** Absolute path to the tokenizer JSON matching that checkpoint. */
export const TOKENIZER_FILE_ENV = "BUILDERFORCE_MEMORY_TOKENIZER";
/** Absolute path for the persistent vector cache; defaults beside the snapshot. */
export const VECTOR_CACHE_ENV = "BUILDERFORCE_MEMORY_VECTORS";

/**
 * Where vectors are cached by default: beside the memory snapshot, under the same
 * name. They belong together — the cache is only meaningful for the memories in
 * that snapshot, and pointing a second store at the same vectors would be a cache
 * for the wrong corpus.
 */
export function defaultVectorCacheFile(memoryFile: string): string {
    return memoryFile.replace(/\.json$/i, "") + ".vectors.json";
}

export interface RecallEmbedderEnv {
    env?: Record<string, string | undefined>;
    /** The snapshot path in effect — used to site the default vector cache. */
    memoryFile: string;
}

/**
 * The embedder this server should rank recall with, or `null` when none is
 * configured (recall then ranks lexically and says so).
 *
 * Opt-in by design: no model path, no model. A memory server must start and serve
 * recall on a host that has never downloaded a checkpoint.
 */
export async function createRecallEmbedder(opts: RecallEmbedderEnv): Promise<PersistentTextEmbedder | null> {
    const env = opts.env ?? process.env;
    const modelFile = env[MODEL_FILE_ENV];
    if (!modelFile) return null;
    return createEvermindEmbedder({
        modelFile,
        tokenizerFile: env[TOKENIZER_FILE_ENV],
        cacheFile: env[VECTOR_CACHE_ENV] || defaultVectorCacheFile(opts.memoryFile),
    });
}
