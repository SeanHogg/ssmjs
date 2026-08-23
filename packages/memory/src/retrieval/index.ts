/**
 * Retrieval layer — chunking, BM25, rank fusion, and the HybridRetriever.
 *
 * The classic RAG pieces the memory stack previously lacked (chunking, hybrid
 * dense+sparse search, reranking), implemented zero-dependency so they run in the
 * browser, Node, and the SSM runtime alike.
 */

export { chunkText } from './chunk.js';
export type { Chunk, ChunkOptions } from './chunk.js';

export { bm25Search, bm25Idf, bm25LengthNorm, bm25TermScore, BM25_DEFAULT_K1, BM25_DEFAULT_B } from './bm25.js';
export type { Bm25Doc, Bm25Hit, Bm25Options, Bm25Tokenizer } from './bm25.js';

export { reciprocalRankFusion, maximalMarginalRelevance } from './fusion.js';
export type { RankedList, FusedHit, MmrCandidate } from './fusion.js';

export { hybridRetrieve } from './HybridRetriever.js';
export type {
    RetrievalCandidate,
    HybridQuery,
    HybridRetrieveOptions,
    HybridHit,
} from './HybridRetriever.js';

export { HnswIndex, denseSearch } from './hnsw.js';
export type { HnswOptions, SearchHit } from './hnsw.js';
