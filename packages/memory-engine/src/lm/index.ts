/**
 * EvermindLM — the generative language model (the runnable "AI").
 */

export { EvermindLM, EvermindLMTrainer, DEFAULT_LM_CONFIG, DEFAULT_LM_SEED, logProbOfToken } from "./evermind_lm.js";
export type { EvermindLMConfig, LMGenerateOptions, TextCodec, EvermindLMTrainOptions, EvermindLMDecodeState } from "./evermind_lm.js";

// CPU text→vector embedder (semantic recall with no GPU).
export { EvermindTextEmbedder } from "./text_embedder.js";
export type { TextEmbedder, EmbedderCodec } from "./text_embedder.js";

// PEFT / efficient-training toolkit (LoRA, QLoRA, mixed precision).
export { LoRAAdapter, EvermindLMLoRA, quantizeBase } from "../training/lora.js";
export type { LoRAConfig, LoRAFitOptions, BaseQuant } from "../training/lora.js";
export { DynamicLossScaler, roundFp16, fp16View } from "../training/mixed_precision.js";
export type { LossScalerOptions } from "../training/mixed_precision.js";
