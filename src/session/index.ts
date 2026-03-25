/**
 * session/index.ts – barrel export for the MambaSession layer.
 *
 * This layer was originally published as @seanhogg/mambakit.
 * It is now an internal layer of @seanhogg/ssmjs.
 */

export { MambaSession }              from './session.js';
export { MambaKitError }             from './errors.js';
export { MODEL_PRESETS, resolveLayerSchedule, resolveModelConfig } from './presets.js';

export type { MambaKitErrorCode }    from './errors.js';
export type { LayerSchedulePreset }  from './presets.js';
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
} from './session.js';
export type { Tokenizer } from './tokenizer.js';
