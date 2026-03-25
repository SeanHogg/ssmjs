/**
 * presets.ts – Model size presets and config resolver for the MambaSession layer.
 */

import type { HybridMambaModelConfig, LayerSpec, LayerType } from '@seanhogg/mambacode.js';
import type { MambaSessionOptions } from './session.js';

// ── Model size presets ────────────────────────────────────────────────────────

/**
 * Pre-defined model size presets.
 * nHeads is used by Mamba-2/3 and Attention layers; ignored for Mamba-1.
 */
export const MODEL_PRESETS: Record<string, Partial<HybridMambaModelConfig>> = {
    nano  : { dModel: 128, numLayers:  4, dState: 16, dConv: 4, expand: 2, nHeads:  4 },
    small : { dModel: 256, numLayers:  6, dState: 16, dConv: 4, expand: 2, nHeads:  8 },
    medium: { dModel: 512, numLayers:  8, dState: 16, dConv: 4, expand: 2, nHeads:  8 },
    large : { dModel: 768, numLayers: 12, dState: 16, dConv: 4, expand: 2, nHeads: 12 },
};

const DEFAULT_PRESET = 'nano';

// ── Layer schedule presets ────────────────────────────────────────────────────

export type LayerSchedulePreset = 'jamba' | 'zamba';

/**
 * Resolve a layer schedule from a preset name, explicit array, or undefined.
 *
 *   'jamba' — Jamba-style: every 4th layer (index 3, 7, 11…) is attention, rest mamba2
 *   'zamba' — Zamba-style: every 6th layer (index 5, 11…) is attention, rest mamba3
 */
export function resolveLayerSchedule(
    schedule   : LayerSpec[] | LayerSchedulePreset | undefined,
    numLayers  : number,
    defaultType: LayerType,
): LayerSpec[] {
    if (!schedule) {
        return Array.from({ length: numLayers }, () => ({ type: defaultType }));
    }

    if (schedule === 'jamba') {
        return Array.from({ length: numLayers }, (_, i) => ({
            type: (i % 4 === 3 ? 'attention' : 'mamba2') as LayerType,
        }));
    }

    if (schedule === 'zamba') {
        return Array.from({ length: numLayers }, (_, i) => ({
            type: (i % 6 === 5 ? 'attention' : 'mamba3') as LayerType,
        }));
    }

    return schedule;
}

// ── Config resolution ─────────────────────────────────────────────────────────

/**
 * Resolves a fully-populated HybridMambaModelConfig from session options and
 * the actual tokenizer vocab size.
 *
 * Resolution order:
 *  1. Preset fields (default: 'nano')
 *  2. modelConfig overrides (only applied when modelSize === 'custom')
 *  3. vocabSize from the tokenizer
 *  4. mambaVersion → default layer type for schedule resolution
 *  5. layerSchedule → per-layer type array (preset string or explicit array)
 */
export function resolveModelConfig(
    options  : MambaSessionOptions,
    vocabSize: number,
): HybridMambaModelConfig {
    const presetName = options.modelSize === 'custom' || options.modelSize == null
        ? DEFAULT_PRESET
        : options.modelSize;

    const preset = MODEL_PRESETS[presetName] ?? MODEL_PRESETS[DEFAULT_PRESET]!;

    const overrides: Partial<HybridMambaModelConfig> =
        options.modelSize === 'custom' && options.modelConfig
            ? options.modelConfig
            : {};

    const dModel     = overrides.dModel     ?? preset.dModel     ?? 128;
    const numLayers  = overrides.numLayers  ?? preset.numLayers  ?? 4;
    const dState     = overrides.dState     ?? preset.dState     ?? 16;
    const dConv      = overrides.dConv      ?? preset.dConv      ?? 4;
    const expand     = overrides.expand     ?? preset.expand     ?? 2;
    const nHeads     = overrides.nHeads     ?? preset.nHeads     ?? 4;
    const nGroups    = overrides.nGroups    ?? preset.nGroups    ?? 1;
    const chunkLen   = overrides.chunkLen   ?? preset.chunkLen   ?? 256;
    const mimoGroup  = overrides.mimoGroup  ?? preset.mimoGroup  ?? 1;
    const eosId      = overrides.eosId      ?? preset.eosId      ?? -1;

    // Validate nHeads divides dModel for multi-head blocks
    if (dModel % nHeads !== 0) {
        throw new Error(
            `resolveModelConfig: dModel (${dModel}) must be divisible by nHeads (${nHeads}).`
        );
    }

    // Layer schedule
    const defaultType: LayerType = options.mambaVersion ?? 'mamba1';
    const layers = resolveLayerSchedule(options.layerSchedule, numLayers, defaultType);

    return {
        vocabSize,
        dModel,
        numLayers,
        dState,
        dConv,
        expand,
        nHeads,
        nGroups,
        chunkLen,
        mimoGroup,
        eosId,
        layers,
    };
}
