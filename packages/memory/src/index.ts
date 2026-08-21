/**
 * SSM.js – JavaScript-native AI runtime.
 *
 * Layer stack:
 *   MambaCode.js  →  WebGPU kernels (WGSL, Mamba-1/2/3 SSM math)
 *   SSM.js        →  Session layer + Runtime orchestration (this package)
 *
 * Quick start:
 *   import { SSM, AnthropicBridge } from 'ssmjs';
 *
 *   const ai = await SSM.create({
 *     session    : { modelSize: 'small' },
 *     bridge     : new AnthropicBridge({ apiKey: '...' }),
 *   });
 *
 *   await ai.adapt(myDocs);
 *   const answer = await ai.generate('How does MambaKit work?');
 */

// ── Session layer ─────────────────────────────────────────────────────────────
export { MambaSession }              from './session/index.js';
export { SessionError }              from './session/index.js';
export { MODEL_PRESETS, resolveLayerSchedule, resolveModelConfig } from './session/index.js';

export type { SessionErrorCode }     from './session/index.js';
export type { LayerSchedulePreset }  from './session/index.js';
export type {
    MambaSessionOptions,
    CompleteOptions,
    AdaptOptions,
    AdaptResult,
    SaveOptions,
    LoadOptions,
    StorageTarget,
    CreateProgressEvent,
    CreateStage,
    CreateCallbacks,
    SessionInternals,
    GpuMode,
    Tokenizer,
} from './session/index.js';

// ── Limbic system (trainable affective dynamics) ──────────────────────────────
export { LimbicSession } from './limbic/LimbicSession.js';
export type { LimbicSessionOptions, LimbicGpuMode } from './limbic/LimbicSession.js';
// Re-export the limbic engine primitives so consumers can use the model/trainer
// and the region schema directly from @seanhogg/builderforce-memory.
export {
    LimbicModel,
    LimbicTrainer,
    LIMBIC_DIM,
    LIMBIC_DIM_NAMES,
    LIMBIC_STATE_DIM,
    LIMBIC_BOUNDS,
    NEUTRAL_STATE,
    REGION,
    clampState,
    neutralState,
    stateToRecord,
    recordToState,
} from '@seanhogg/builderforce-memory-engine';
export type {
    LimbicModelConfig,
    LimbicForward,
    LimbicSample,
    LimbicTrainOptions,
    LimbicDimName,
    Region,
} from '@seanhogg/builderforce-memory-engine';

// ── Mixture-of-Experts (shared-expert hybrid — the Evermind generator's sparsity) ──
// Re-exported from the engine so consumers reach it from @seanhogg/builderforce-memory.
export {
    SharedExpertMoE,
    LoadBalanceAccumulator,
    DEFAULT_MOE_CONFIG,
    DEFAULT_MOE_SEED,
    MoETrainer,
    EvermindModelPackage,
} from '@seanhogg/builderforce-memory-engine';
export type {
    MoEConfig,
    MoEParam,
    RouteResult,
    MoESample,
    MoETrainOptions,
    MoEEpochResult,
    EvermindModelManifest,
    EvermindModelCard,
    PackageMeta,
    ValidationResult,
} from '@seanhogg/builderforce-memory-engine';
export { EvermindLM, EvermindLMTrainer } from '@seanhogg/builderforce-memory-engine';
export type { EvermindLMConfig, LMGenerateOptions, TextCodec } from '@seanhogg/builderforce-memory-engine';

// ── Runtime ───────────────────────────────────────────────────────────────────
export { SSMRuntime }    from './runtime/SSMRuntime.js';
export type { SSMRuntimeOptions, GenerateOptions } from './runtime/SSMRuntime.js';

// ── Bridges ───────────────────────────────────────────────────────────────────
export type { TransformerBridge, BridgeGenerateOptions } from './bridges/TransformerBridge.js';
export { OpenAIBridge }    from './bridges/OpenAIBridge.js';
export { VertexAIBridge, VertexAIEmbedder } from './bridges/VertexAIBridge.js';
export { AnthropicBridge } from './bridges/AnthropicBridge.js';
export { FetchBridge }     from './bridges/FetchBridge.js';
export { CachingBridge }   from './bridges/CachingBridge.js';
export { SemanticCachingBridge } from './bridges/SemanticCachingBridge.js';
export { ResponseCache, buildCacheKey } from './bridges/ResponseCache.js';
export type { BridgeCallInfo }         from './bridges/TransformerBridge.js';
export type { OpenAIBridgeOptions }    from './bridges/OpenAIBridge.js';
export type { VertexAIBridgeOptions, VertexAIEmbedderOptions } from './bridges/VertexAIBridge.js';
export type { AnthropicBridgeOptions } from './bridges/AnthropicBridge.js';
export type { FetchBridgeOptions }     from './bridges/FetchBridge.js';
export type { CachingBridgeOptions }   from './bridges/CachingBridge.js';
export type { SemanticCachingBridgeOptions } from './bridges/SemanticCachingBridge.js';
export type { ResponseCacheOptions }   from './bridges/ResponseCache.js';

// ── Semantic cache (embedding-keyed, L1 local + L2 shared) ─────────────────────
export { SemanticCache } from './cache/SemanticCache.js';
export { FetchSemanticCacheBackend } from './cache/FetchSemanticCacheBackend.js';
export type {
    Embedder,
    SemanticCacheBackend,
    SemanticCacheHit,
    SemanticCacheOptions,
    FetchSemanticCacheBackendOptions,
} from './cache/index.js';

// ── Similarity primitives ──────────────────────────────────────────────────────
export { cosineSimilarity, jaccardSimilarity, tokenize } from './similarity/index.js';

// ── Retrieval (chunking, BM25, rank fusion, hybrid RAG) ────────────────────────
export {
    chunkText,
    bm25Search,
    reciprocalRankFusion,
    maximalMarginalRelevance,
    hybridRetrieve,
} from './retrieval/index.js';
export type {
    Chunk,
    ChunkOptions,
    Bm25Doc,
    Bm25Hit,
    Bm25Options,
    RankedList,
    FusedHit,
    MmrCandidate,
    RetrievalCandidate,
    HybridQuery,
    HybridRetrieveOptions,
    HybridHit,
} from './retrieval/index.js';

// ── Telemetry (granular tracing + LLM-native metrics + cost-per-request) ──────
export {
    Tracer,
    Span,
    MetricsRegistry,
    InstrumentedBridge,
    InMemorySpanExporter,
    ConsoleSpanExporter,
    OtlpHttpSpanExporter,
    toOtlpPayload,
    toOtlpSpan,
    percentile,
    estimateCostUsd,
    estimateTokens,
    resolveRate,
    DEFAULT_PRICE_BOOK,
    ANTHROPIC_PRICE_BOOK,
    EVERMIND_PRICE_BOOK,
    DEFAULT_CACHE_READ_MULTIPLIER,
    DEFAULT_CACHE_WRITE_MULTIPLIER,
} from './telemetry/index.js';
export type {
    TracerOptions,
    SpanOptions,
    SpanData,
    SpanKind,
    SpanStatus,
    SpanEvent,
    SpanExporter,
    AttributeValue,
    LlmUsage,
    LlmMetricSnapshot,
    KindMetricSnapshot,
    MetricsRegistryOptions,
    InstrumentedBridgeOptions,
    ConsoleSpanExporterOptions,
    OtlpHttpSpanExporterOptions,
    ModelRate,
    PriceBook,
} from './telemetry/index.js';

// ── Vector store (port + adapters + tenant/ACL scoping) ───────────────────────
export {
    MemoryVectorStore,
    RestVectorStore,
    getDialect,
    listDialects,
    registerDialect,
    evermindDialect,
    qdrantDialect,
    pineconeDialect,
    vertexAiDialect,
    accessFilter,
    combineFilters,
    matchesFilter,
    filterFields,
    TENANT_FIELD,
    ACL_FIELD,
    SENSITIVITY_FIELD,
    ACL_PUBLIC,
} from './vectorstore/index.js';
export type {
    MemoryVectorStoreOptions,
    RestVectorStoreOptions,
    VectorDialect,
    DialectContext,
    TranslatedFilter,
    AccessScope,
    ComparisonOp,
    MetadataFilter,
    MetadataRecord,
    MetadataValue,
    UpsertResult,
    VectorMatch,
    VectorQuery,
    VectorRecord,
    VectorStore,
} from './vectorstore/index.js';

// ── Ingestion (structured + unstructured → vector store, incremental sync) ────
export {
    IngestionPipeline,
    InMemoryIngestManifest,
    ParserRegistry,
    BUILTIN_PARSERS,
    rowsParser,
    csvParser,
    jsonParser,
    markdownParser,
    htmlParser,
    plainTextParser,
    parseCsv,
    serializeRow,
    hashText,
    mapWithConcurrency,
    SOURCE_ID_FIELD,
    CHUNK_INDEX_FIELD,
    CONTENT_HASH_FIELD,
    TITLE_FIELD,
    URI_FIELD,
    PARSER_FIELD,
} from './ingest/index.js';
export type {
    IngestionPipelineOptions,
    BatchEmbedder,
    ChunkFingerprint,
    DocumentParser,
    IngestManifest,
    IngestionReport,
    ParsedSection,
    SourceContent,
    SourceDocument,
    StructuredRow,
    SyncStrategy,
} from './ingest/index.js';

// ── Enterprise RAG (hybrid, scoped, cited) ────────────────────────────────────
export {
    EnterpriseRetriever,
    buildGroundedPrompt,
    GROUNDED_SYSTEM_PROMPT,
    NO_EVIDENCE_ANSWER,
} from './rag/index.js';
export type {
    EnterpriseRetrieverOptions,
    RetrieveOptions,
    RetrievalResult,
    RetrievalMode,
    RetrievedPassage,
    Reranker,
} from './rag/index.js';

// ── Multi-agent orchestration (graph + ReAct/reflection/delegation) ───────────
export {
    AgentGraph,
    appendReducer,
    mergeReducer,
    sumReducer,
    unionReducer,
    InMemoryCheckpointer,
    KeyValueCheckpointer,
    createReactAgent,
    createReflectionAgent,
    createSupervisor,
    asWorker,
    parseReactTurn,
    renderToolCatalog,
    renderTranscript,
    REACT_PROTOCOL,
    DEFAULT_CRITIC_PROMPT,
    FINISH,
    START,
    END,
} from './orchestration/index.js';
export type {
    AgentGraphOptions,
    RunOptions,
    InMemoryCheckpointerOptions,
    KeyValueCheckpointerOptions,
    KeyValueLike,
    AgentTool,
    AgentTurn,
    AgentTurnRole,
    ParsedAction,
    ReactAgentOptions,
    ReactState,
    ReflectionAgentOptions,
    ReflectionState,
    SupervisorOptions,
    SupervisorState,
    Worker,
    ChannelReducer,
    ChannelSpec,
    ChannelSpecs,
    Checkpoint,
    Checkpointer,
    EdgeCondition,
    GraphEndReason,
    GraphEvent,
    GraphRunResult,
    GraphState,
    NodeContext,
    NodeFn,
} from './orchestration/index.js';

// ── Evaluation (the conception → launch gate) ─────────────────────────────────
export {
    EvalHarness,
    evaluateGate,
    formatReport,
    exactMatch,
    substringChecks,
    embeddingSimilarity,
    retrievalRecall,
    citationValidity,
    llmJudge,
    noError,
    costBudget,
    DEFAULT_RAG_GRADERS,
    DEFAULT_JUDGE_RUBRIC,
} from './eval/index.js';
export type {
    EvalHarnessOptions,
    EvalCase,
    EvalDataset,
    EvalGate,
    EvalOutput,
    EvalReport,
    EvalTarget,
    CaseResult,
    GateVerdict,
    GradeResult,
    Grader,
    LlmJudgeOptions,
    SimilarityGraderOptions,
} from './eval/index.js';

// ── Router ────────────────────────────────────────────────────────────────────
export { InferenceRouter } from './router/InferenceRouter.js';
export type {
    RoutingStrategy,
    RoutingDecision,
    RouterContext,
    InferenceRouterOptions,
    RoutingAuditEntry,
} from './router/InferenceRouter.js';

// ── Memory ────────────────────────────────────────────────────────────────────
export { MemoryStore }  from './memory/MemoryStore.js';
export type {
    MemoryEntry,
    MemoryStoreOptions,
    RememberOptions,
    FactType,
    RecallMethod,
    ScoredMemory,
    RankedRecall,
} from './memory/MemoryStore.js';

// ── Cognition (Evermind — Write-Through Cognition) ────────────────────────────
export {
    EvermindCognition,
    workspacePresenceGatherer,
    canonicalizeSubjectKey,
    normalizeSubjectKey,
    buildAliasTable,
    sanitizeRecalledFact,
    buildRecallContext,
    trustScore,
} from './cognition/index.js';
export type {
    EvermindCognitionOptions,
    WorkspacePresenceRule,
    Claim,
    CognitionFactStore,
    CommitResult,
    EvidenceContext,
    EvidenceGatherer,
    EvidenceResult,
    Verdict,
    AliasTable,
    RecalledFact,
    FactProvenance,
} from './cognition/index.js';

// ── Diagnostics (execution-output runner) ──────────────────────────────────────
export { runStackDiagnostic } from './diagnostics/index.js';
export type {
    StackStep,
    StackStepResult,
    StackContext,
    StackDiagnosticResult,
    RunStackOptions,
    StepStatus,
} from './diagnostics/index.js';

// ── Workflows (configurable templates + step registry → run + .evermind output) ─
export {
    runWorkflow,
    compileWorkflow,
    validateWorkflow,
    buildEvermindStackSteps,
    StepTypeRegistry,
    createDefaultRegistry,
    defaultStepRegistry,
    BUILTIN_STEPS,
    AGENTIC_SEVEN_LAYER,
    TRAIN_LLM,
    TEACH_CODE,
    WORKFLOW_TEMPLATES,
    getTemplate,
    cloneTemplate,
    analyzeCode,
    runJsCases,
} from './workflow/index.js';
export type {
    WorkflowConfig,
    WorkflowStepConfig,
    StepTypeInfo,
    StepFactory,
    WorkflowValidationError,
    RunWorkflowOptions,
    CodeAnalysis,
    CodeCase,
    CodeEvalResult,
} from './workflow/index.js';

// ── Publish (ship an engine model export to HF / local folder) ──────────────────
export { publishToHuggingFace, writeExportToDir } from './publish/index.js';
export type { HuggingFaceTarget, PublishOutcome, HubClient, FsLike } from './publish/index.js';

// ── Distillation ──────────────────────────────────────────────────────────────
export { DistillationEngine } from './distillation/DistillationEngine.js';
export type {
    DistillOptions,
    DistillResult,
    DistillBatchResult,
    DistillationLog,
    QualityGate,
    RehearsalOptions,
} from './distillation/DistillationEngine.js';

// ── Agent ─────────────────────────────────────────────────────────────────────
export { SSMAgent }  from './agent/SSMAgent.js';
export type { SSMAgentOptions, ThinkOptions, AgentMessage, MessageRole } from './agent/SSMAgent.js';

// ── Errors ────────────────────────────────────────────────────────────────────
export { SSMError }  from './errors/SSMError.js';
export type { SSMErrorCode } from './errors/SSMError.js';

// ── Top-level SSM namespace ───────────────────────────────────────────────────
// Allows the `SSM.create()` pattern from the spec:
//   const ai = await SSM.create({ session: { modelSize: 'nano' } });

import { SSMRuntime }          from './runtime/SSMRuntime.js';
import type { SSMRuntimeOptions } from './runtime/SSMRuntime.js';

export const SSM = {
    /**
     * Creates a new SSMRuntime.
     *
     * Shorthand for `SSMRuntime.create(opts)`.
     * Can throw `SessionError` for GPU / tokenizer failures during init.
     */
    create: (opts: SSMRuntimeOptions) => SSMRuntime.create(opts),
} as const;
