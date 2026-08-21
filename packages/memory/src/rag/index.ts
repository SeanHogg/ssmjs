/**
 * rag — enterprise retrieval: hybrid, tenant/ACL scoped, cited, traced.
 */

export {
    EnterpriseRetriever,
    buildGroundedPrompt,
    GROUNDED_SYSTEM_PROMPT,
    NO_EVIDENCE_ANSWER,
} from './EnterpriseRetriever.js';
export type {
    EnterpriseRetrieverOptions,
    RetrieveOptions,
    RetrievalResult,
    RetrievalMode,
    RetrievedPassage,
    Reranker,
} from './EnterpriseRetriever.js';
