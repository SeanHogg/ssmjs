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

// ── Session layer (absorbed from @seanhogg/mambakit) ─────────────────────────
export { MambaSession }              from './session/index.js';
export { MambaKitError }             from './session/index.js';
export { MODEL_PRESETS, resolveLayerSchedule, resolveModelConfig } from './session/index.js';

export type { MambaKitErrorCode }    from './session/index.js';
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

// ── Runtime ───────────────────────────────────────────────────────────────────
export { SSMRuntime }    from './runtime/SSMRuntime.js';
export type { SSMRuntimeOptions, GenerateOptions } from './runtime/SSMRuntime.js';

// ── Bridges ───────────────────────────────────────────────────────────────────
export type { TransformerBridge, BridgeGenerateOptions } from './bridges/TransformerBridge.js';
export { OpenAIBridge }    from './bridges/OpenAIBridge.js';
export { AnthropicBridge } from './bridges/AnthropicBridge.js';
export { FetchBridge }     from './bridges/FetchBridge.js';
export type { OpenAIBridgeOptions }    from './bridges/OpenAIBridge.js';
export type { AnthropicBridgeOptions } from './bridges/AnthropicBridge.js';
export type { FetchBridgeOptions }     from './bridges/FetchBridge.js';

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
} from './memory/MemoryStore.js';

// ── Distillation ──────────────────────────────────────────────────────────────
export { DistillationEngine } from './distillation/DistillationEngine.js';
export type {
    DistillOptions,
    DistillResult,
    DistillBatchResult,
    DistillationLog,
    QualityGate,
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
     * Can throw `MambaKitError` for GPU / tokenizer failures during init.
     */
    create: (opts: SSMRuntimeOptions) => SSMRuntime.create(opts),
} as const;
