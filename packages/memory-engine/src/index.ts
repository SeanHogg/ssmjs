/**
 * MambaCode.js – Entry Point (v2.0.0)
 */

// ── Model classes ─────────────────────────────────────────────────────────────

export { HybridMambaModel, MambaModel } from './model/mamba_model.js';

// New block classes
export { Mamba1Block }   from './model/mamba1_block.js';
export { Mamba2Block }   from './model/mamba2_block.js';
export { Mamba3Block }   from './model/mamba3_block.js';
export { AttentionBlock } from './model/attention_block.js';

// Deprecated alias — kept until 3.0.0
export { MambaBlock } from './model/mamba1_block.js';

// ── Mixture-of-Experts (shared-expert hybrid sparsity) ─────────────────────────
export {
    SharedExpertMoE,
    LoadBalanceAccumulator,
    DEFAULT_MOE_CONFIG,
    DEFAULT_MOE_SEED,
    MoETrainer,
    EvermindModelPackage,
} from './moe/index.js';
export type {
    MoEConfig,
    MoEParam,
    RouteResult,
    MoESample,
    MoETrainOptions,
    MoEEpochResult,
    EvermindModelManifest,
    EvermindModelCard,
    EvermindModelType,
    EvermindModality,
    PackageMeta,
    ValidationResult,
} from './moe/index.js';

// ── EvermindLM (the generative model) + AdamW ──────────────────────────────────
export { EvermindLM, EvermindLMTrainer, DEFAULT_LM_CONFIG, DEFAULT_LM_SEED, logProbOfToken } from './lm/index.js';
// CPU text→vector embedder — SSM-embedding recall with no GPU.
export { EvermindTextEmbedder } from './lm/index.js';
export type { EvermindLMConfig, LMGenerateOptions, TextCodec, EvermindLMTrainOptions, EvermindLMDecodeState } from './lm/index.js';
export type { TextEmbedder, EmbedderCodec } from './lm/index.js';
export { AdamW, adamwUpdateInPlace } from './optim/adamw.js';
export type { AdamWOptions, OptimTarget, OptimParam, ShardSpec, AdamWKernelStep } from './optim/adamw.js';

// PEFT / efficient-training toolkit (LoRA, QLoRA, mixed precision, checkpointing).
export { LoRAAdapter, EvermindLMLoRA, quantizeBase } from './training/lora.js';
export type { LoRAConfig, LoRAFitOptions, BaseQuant } from './training/lora.js';
export { DynamicLossScaler, roundFp16, fp16View } from './training/mixed_precision.js';
export type { LossScalerOptions } from './training/mixed_precision.js';

// ── Training ──────────────────────────────────────────────────────────────────

export { MambaTrainer, cpuDimsFor, buildChunks, buildBatches, WSLA_MAX_DELTA } from './training/trainer.js';
export type { TrainOptions } from './training/trainer.js';
// The exact CPU forward + backward the trainer differentiates with. Exported so a
// host can compute gradients (or gradient-check the model) without a GPU.
export {
    cpuModelForward, cpuModelLoss, cpuModelBackward,
    zeroCpuModelGrads, toNamedGrads, namedWeights, cpuWeightsFromNamed,
    paramMatrixShape, DEFAULT_LORA_TARGETS, CPU_GRADIENT_LAYER_TYPES,
} from './training/model_cpu.js';
export type {
    CpuModelDims, CpuModelWeights, CpuModelGrads, CpuBackwardResult, CpuBackwardOptions, CpuGradientLayerType,
} from './training/model_cpu.js';
export {
    mamba1CpuForward, mamba1CpuBackward, zeroMamba1Grads,
    MAMBA1_PARAM_NAMES, A_LOG_CLAMP_LO, A_LOG_CLAMP_HI,
} from './training/mamba1_cpu.js';
export type {
    Mamba1CpuWeights, Mamba1CpuDims, Mamba1CpuCache, Mamba1CpuGrads, Mamba1CpuForwardResult,
} from './training/mamba1_cpu.js';
export {
    linearForward, linearBackward, rmsNormForward, rmsNormBackward,
    silu, siluGrad, softplus, sigmoid, RMSNORM_EPS,
} from './training/cpu_ops.js';
export {
    Tensor,
    backward,
    enableGrad,
    noGrad,
    clearTape,
    recordOperation,
    crossEntropyLoss,
    crossEntropyGrad,
} from './training/autograd.js';

// ── Tokenizer ─────────────────────────────────────────────────────────────────

export { BPETokenizer } from './tokenizer/bpe.js';
export type { BPETokenizerSpec } from './tokenizer/bpe.js';
export type { BPEEncodeOptions, PadSide, HuggingFaceTokenizerSpec, SpecialTokenOverrides } from './tokenizer/bpe.js';

// ── Modality codecs (media ⇄ tokens; lets EvermindLM generate video) ────────────

export {
    MultimodalVocab,
    VIDEO_BANK_INTRA,
    VIDEO_BANK_INTER,
    VideoRVQCodec,
    ImageRVQCodec,
    buildVideoSequence,
    generateVideo,
    generateImage,
} from './codec/index.js';
export type {
    MultimodalVocabConfig,
    TokenKind,
    VideoRVQConfig,
    Frame,
    Video,
} from './codec/index.js';

// ── Model export (the publishing step: ONNX / safetensors / GGUF / HF repo) ─────

export {
    exportEvermind,
    EXPORT_FORMATS,
    exportSafetensors,
    tensorsToSafetensors,
    exportOnnx,
    exportGguf,
    configJson,
    generationConfigJson,
    tokenizerJson,
    modelCardMarkdown,
    namedTensors,
    evermindTensorSpec,
    archOf,
    paramCount,
} from './export/index.js';
export type {
    ExportFormat,
    ExportFile,
    ExportResult,
    ExportOptions,
    HfMeta,
    NamedTensor,
    EvermindArch,
    TensorSpec,
} from './export/index.js';

// ── Model import (warm-start / weight-port: safetensors → EvermindLM) ────────────

export {
    safetensorsToTensors,
    importEvermind,
    importEvermindTensors,
    inferArchFromTensors,
} from './import/index.js';
export type { ImportOptions } from './import/index.js';

// ── Foreign weight port (Falcon-Mamba / Codestral-Mamba → HybridMambaModel) ────
// Transformer checkpoints are NOT portable — distillation is the only route, and
// `foreignMambaAdapterFor` rejects them saying so.

export {
    FOREIGN_MAMBA_ADAPTERS,
    foreignMambaAdapterFor,
    portForeignMamba,
    portForeignMambaSafetensors,
    applyPortedWeights,
    executePortPlan,
    normaliseSourceName,
    falconMambaAdapter,
    codestralMambaAdapter,
} from './import/index.js';
export type {
    ForeignConfig,
    ForeignMambaAdapter,
    PortTarget,
    PortDiscard,
    PortOp,
    PortPlan,
    PortRule,
    PortedCheckpoint,
    PortedTensors,
    SynthesisedTarget,
} from './import/index.js';

// ── Benchmarking (held-out perplexity / accuracy / throughput + A/B) ──────────

export {
    benchmarkModel,
    benchmarkModelAsync,
    benchmarkText,
    compareModels,
    compareReports,
    corpusToSequences,
    trainAndBenchmark,
    benchmarkAdaptationCost,
    formatAdaptationCostReport,
    tokenWindows,
    adaptationsAfterSkip,
    argmax as benchArgmax,
    topKIndices,
    perplexity,
    bitsPerToken,
    LN2,
} from './bench/index.js';
export type {
    LogitsModel,
    AsyncLogitsModel,
    BenchmarkOptions,
    BenchmarkReport,
    ComparisonReport,
    TrainAndBenchmarkOptions,
    TrainAndBenchmarkResult,
    AdaptationCostOptions,
    AdaptationCostPoint,
    AdaptationCostReport,
} from './bench/index.js';

// ── Checkpoint integrity (CRC-32 + trailer) ───────────────────────────────────

export { crc32, appendCrcTrailer, verifyCrcTrailer, CRC_TRAILER_MAGIC } from './utils/crc32.js';
export type { CrcCheck } from './utils/crc32.js';
export {
    computeRowDelta,
    applyRowDelta,
    serializeRowDelta,
    deserializeRowDelta,
    diffCheckpoints,
    applyCheckpointDiff,
} from './utils/delta.js';
export type { RowDelta } from './utils/delta.js';

// ── Seeded RNG (reproducible weight init) ─────────────────────────────────────

export { SeededRng, setInitSeed, randn, gaussianArray } from './utils/rng.js';

// ── Types ─────────────────────────────────────────────────────────────────────

export type {
    HybridMambaModelConfig,
    MambaModelConfig,
    ModelForwardResult,
    SamplingOptions,
    LayerSpec,
} from './model/mamba_model.js';

export type { SequenceLayer, LayerParam, LayerType, LayerForwardResult } from './model/sequence_layer.js';
export type { Mamba1BlockConfig, BlockParam, BlockCache, BlockForwardResult, MambaBlockConfig } from './model/mamba1_block.js';
export type { Mamba2BlockConfig, Mamba2Cache } from './model/mamba2_block.js';
export type { Mamba3BlockConfig, Mamba3Cache }  from './model/mamba3_block.js';
export type { AttentionBlockConfig, AttentionCache } from './model/attention_block.js';

// ── GPU utilities ─────────────────────────────────────────────────────────────

export {
    initWebGPU,
    createStorageBuffer,
    createEmptyStorageBuffer,
    createUniformBuffer,
    createComputePipeline,
    createBindGroup,
    dispatchKernel,
    readBuffer,
    uploadBuffer,
    cdiv,
    BufferPool,
} from './utils/gpu_utils.js';

// ── Quantization ──────────────────────────────────────────────────────────────

export {
    quantizeFp16,
    dequantizeFp16,
    floatToFp16,
    fp16ToFloat,
    quantizeInt8,
    dequantizeInt8,
    quantizeInt8PerChannel,
    dequantizeInt8PerChannel,
    estimateMemory,
} from './utils/quantization.js';

// ── WGSL kernel sources ───────────────────────────────────────────────────────

// Mamba-1 kernels (unchanged)
export { SELECTIVE_SCAN_FORWARD_WGSL, SELECTIVE_SCAN_BACKWARD_WGSL }
    from './kernels/selective_scan.js';
export { CONV1D_FORWARD_WGSL, CONV1D_BACKWARD_WGSL }
    from './kernels/conv1d.js';
export { LINEAR_FORWARD_WGSL, LINEAR_BACKWARD_WGSL }
    from './kernels/linear_projection.js';
export { COL_SLICE_WGSL, COL_SLICE_ENTRY, dispatchColumnSlice } from './kernels/slice.js';
export { WEIGHT_UPDATE_WGSL, GRAD_CLIP_WGSL }
    from './kernels/weight_update.js';
export { ACTIVATIONS_WGSL, ACTIVATIONS_BACKWARD_WGSL, SOFTMAX_FORWARD_WGSL, SOFTMAX_BACKWARD_WGSL }
    from './kernels/activations.js';

// Mamba-2 SSD kernels
export { SSD_FORWARD_WGSL, SSD_BACKWARD_WGSL }
    from './kernels/ssd.js';

// Mamba-3 complex SSD kernels
export { COMPLEX_SSD_FORWARD_WGSL, COMPLEX_SSD_BACKWARD_WGSL }
    from './kernels/complex_ssd.js';

// Attention kernels
export { ATTENTION_FORWARD_WGSL, ATTENTION_BACKWARD_WGSL, SOFTMAX_WGSL }
    from './kernels/attention.js';

// ── Limbic system (trainable affective dynamics) ──────────────────────────────

export {
    REGION,
    LIMBIC_DIM,
    LIMBIC_DIM_NAMES,
    LIMBIC_STATE_DIM,
    LIMBIC_BOUNDS,
    NEUTRAL_STATE,
    clampDim,
    clampState,
    neutralState,
    stateToRecord,
    recordToState,
    personalitySetpoint,
    LimbicModel,
    DEFAULT_LIMBIC_CONFIG,
    DEFAULT_LIMBIC_SEED,
    LimbicTrainer,
} from './limbic/index.js';
export type {
    Region,
    LimbicDimName,
    PersonalityTraits,
    LimbicModelConfig,
    LimbicForward,
    LimbicParam,
    LimbicSample,
    LimbicTrainOptions,
} from './limbic/index.js';

export { LIMBIC_AFFECT_WGSL } from './kernels/limbic_affect.js';

// ── Version ───────────────────────────────────────────────────────────────────

export const VERSION     = '2.0.0';
export const DESCRIPTION = 'MambaCode.js: WebGPU-accelerated Mamba-1/2/3 and Hybrid SSM for browser code models';
