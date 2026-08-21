/**
 * tests/import_foreign.test.ts
 * The foreign-checkpoint weight port: Falcon-Mamba (Mamba-1) and
 * Codestral-Mamba (Mamba-2) tensor names, ranks and shapes → this engine's
 * HybridMambaModel parameters.
 *
 * The checkpoints themselves are multi-GB, so these run over SYNTHETIC tensors
 * built to the real published names/ranks/shapes at a toy width. What is being
 * verified is the map, not the download.
 */

import type { NamedTensor } from '../src/export/tensors';
import { tensorsToSafetensors } from '../src/export/safetensors';
import {
    FOREIGN_MAMBA_ADAPTERS,
    foreignMambaAdapterFor,
    portForeignMamba,
    portForeignMambaSafetensors,
} from '../src/import';

// ── Toy geometry (the real shapes, 512× narrower) ────────────────────────────

const E = 8;    // hidden_size
const L = 2;    // num_hidden_layers
const V = 12;   // vocab_size
const N = 4;    // state_size
const K = 3;    // conv_kernel
const D = 16;   // intermediate_size (d_inner) = expand 2
const R = 2;    // time_step_rank
const H = 4;    // num_heads      (Mamba-2)
const G = 2;    // n_groups       (Mamba-2)

const FALCON_CONFIG = {
    architectures: ['FalconMambaForCausalLM'],
    model_type: 'falcon_mamba',
    hidden_size: E,
    num_hidden_layers: L,
    vocab_size: V,
    state_size: N,
    conv_kernel: K,
    intermediate_size: D,
    expand: 2,
    time_step_rank: R,
    use_bias: false,
    use_conv_bias: true,
    tie_word_embeddings: false,
};

const CODESTRAL_CONFIG = {
    architectures: ['Mamba2ForCausalLM'],
    model_type: 'mamba2',
    hidden_size: E,
    num_hidden_layers: L,
    vocab_size: V,
    state_size: N,
    conv_kernel: K,
    intermediate_size: D,
    expand: 2,
    num_heads: H,
    head_dim: D / H,
    n_groups: G,
    chunk_size: 16,
    use_bias: false,
    use_conv_bias: true,
    tie_word_embeddings: false,
};

/** Deterministic, distinguishable values so a mis-slice can't accidentally pass. */
function ramp(shape: number[], seed: number): NamedTensor['data'] {
    const n = shape.reduce((a, b) => a * b, 1);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = seed + i / 1000;
    return out;
}

function tensors(spec: Array<[string, number[]]>): NamedTensor[] {
    return spec.map(([name, shape], i) => ({ name, shape, data: ramp(shape, i + 1) }));
}

function falconTensors(): NamedTensor[] {
    const spec: Array<[string, number[]]> = [['backbone.embeddings.weight', [V, E]]];
    for (let i = 0; i < L; i++) {
        const m = `backbone.layers.${i}.mixer`;
        spec.push(
            [`backbone.layers.${i}.norm.weight`, [E]],
            [`${m}.in_proj.weight`, [2 * D, E]],
            [`${m}.conv1d.weight`, [D, 1, K]],
            [`${m}.conv1d.bias`, [D]],
            [`${m}.x_proj.weight`, [R + 2 * N, D]],
            [`${m}.dt_proj.weight`, [D, R]],
            [`${m}.dt_proj.bias`, [D]],
            [`${m}.A_log`, [D, N]],
            [`${m}.D`, [D]],
            [`${m}.out_proj.weight`, [E, D]],
        );
    }
    spec.push(['backbone.norm_f.weight', [E]], ['lm_head.weight', [V, E]]);
    return tensors(spec);
}

/** Mamba-2's fused projection: [gate, x, B, C, dt]. */
const CONV_DIM = D + 2 * G * N;
const IN_PROJ_ROWS = D + CONV_DIM + H;

function codestralTensors(prefix = '', embeddingName = 'backbone.embeddings.weight'): NamedTensor[] {
    const spec: Array<[string, number[]]> = [[`${prefix}${embeddingName}`, [V, E]]];
    for (let i = 0; i < L; i++) {
        const m = `${prefix}backbone.layers.${i}.mixer`;
        spec.push(
            [`${prefix}backbone.layers.${i}.norm.weight`, [E]],
            [`${m}.in_proj.weight`, [IN_PROJ_ROWS, E]],
            [`${m}.conv1d.weight`, [CONV_DIM, 1, K]],
            [`${m}.conv1d.bias`, [CONV_DIM]],
            [`${m}.dt_bias`, [H]],
            [`${m}.A_log`, [H]],
            [`${m}.D`, [H]],
            [`${m}.norm.weight`, [D]],
            [`${m}.out_proj.weight`, [E, D]],
        );
    }
    spec.push([`${prefix}backbone.norm_f.weight`, [E]], [`${prefix}lm_head.weight`, [V, E]]);
    return tensors(spec);
}

function find(list: NamedTensor[], name: string): NamedTensor {
    const t = list.find((x) => x.name === name);
    if (!t) throw new Error(`test fixture missing ${name}`);
    return t;
}

/** Every parameter `HybridMambaModel.parameters()` would report, by name. */
function expectedTargets(perLayer: string[]): string[] {
    const names = ['embedding'];
    for (let i = 0; i < L; i++) for (const p of perLayer) names.push(`layer${i}.${p}`);
    names.push('final_norm');
    return names;
}

const MAMBA1_PARAMS = [
    'wInProj', 'bInProj', 'wConv', 'bConv', 'wXProj', 'bXProj',
    'wDtProj', 'bDtProj', 'A_log', 'D_vec', 'wOutProj', 'bOutProj', 'normWeight',
];
const MAMBA2_PARAMS = [
    'wInProj', 'wConv', 'bConv', 'A_log', 'dt_bias', 'D_vec',
    'wOutProj', 'normWeight', 'preNormWeight',
];

// ── Detection ────────────────────────────────────────────────────────────────

test('the config selects the adapter by architectures, then by model_type', () => {
    expect(foreignMambaAdapterFor(FALCON_CONFIG).id).toBe('falcon_mamba');
    expect(foreignMambaAdapterFor(CODESTRAL_CONFIG).id).toBe('codestral_mamba');
    // architectures absent → model_type decides
    expect(foreignMambaAdapterFor({ model_type: 'mamba2' }).id).toBe('codestral_mamba');
    // the commonly mis-cited Codestral class name is still accepted
    expect(foreignMambaAdapterFor({ architectures: ['MambaCodestralForCausalLM'] }).id)
        .toBe('codestral_mamba');
    expect(FOREIGN_MAMBA_ADAPTERS.map((a) => a.id)).toEqual(['falcon_mamba', 'codestral_mamba']);
});

test('an unknown architecture fails loudly and lists what IS supported', () => {
    const call = () => foreignMambaAdapterFor({ architectures: ['RwkvForCausalLM'], model_type: 'rwkv' });
    expect(call).toThrow(/RwkvForCausalLM/);
    expect(call).toThrow(/FalconMambaForCausalLM/);
    expect(call).toThrow(/Mamba2ForCausalLM/);
    expect(call).toThrow(/distillation/i);
});

test('a transformer checkpoint is rejected with distillation named as the only route', () => {
    // Transformer coders (IQuest / NousCoder / Maincoder …) are a different
    // architecture entirely — there is no tensor naming that ports attention.
    for (const arch of ['LlamaForCausalLM', 'Qwen3ForCausalLM', 'MistralForCausalLM']) {
        const call = () => foreignMambaAdapterFor({ architectures: [arch] });
        expect(call).toThrow(/TRANSFORMER/);
        expect(call).toThrow(/DISTILLATION/);
        expect(call).toThrow(/attention/i);
    }
});

// ── Falcon-Mamba (Mamba-1) ───────────────────────────────────────────────────

test('Falcon-Mamba populates every Mamba-1 target with the right shapes', () => {
    const src = falconTensors();
    const ported = portForeignMamba(FALCON_CONFIG, src);

    expect(ported.adapter).toBe('falcon_mamba');
    expect([...ported.weights.keys()]).toEqual(expectedTargets(MAMBA1_PARAMS));

    const sizes: Record<string, number> = {
        wInProj: 2 * D * E, bInProj: 2 * D, wConv: D * K, bConv: D,
        wXProj: (R + 2 * N) * D, bXProj: R + 2 * N, wDtProj: D * R, bDtProj: D,
        A_log: D * N, D_vec: D, wOutProj: E * D, bOutProj: E, normWeight: E,
    };
    expect(ported.weights.get('embedding')!.length).toBe(V * E);
    expect(ported.weights.get('final_norm')!.length).toBe(E);
    for (let i = 0; i < L; i++) {
        for (const [p, n] of Object.entries(sizes)) {
            expect(ported.weights.get(`layer${i}.${p}`)!.length).toBe(n);
        }
    }
});

test('Falcon-Mamba copies the [out, in] projections verbatim (no transpose)', () => {
    const src = falconTensors();
    const ported = portForeignMamba(FALCON_CONFIG, src);
    for (const [source, target] of [
        ['backbone.embeddings.weight', 'embedding'],
        ['backbone.layers.1.mixer.in_proj.weight', 'layer1.wInProj'],
        ['backbone.layers.1.mixer.x_proj.weight', 'layer1.wXProj'],
        ['backbone.layers.1.mixer.dt_proj.weight', 'layer1.wDtProj'],
        ['backbone.layers.1.mixer.out_proj.weight', 'layer1.wOutProj'],
        ['backbone.layers.1.mixer.A_log', 'layer1.A_log'],
        ['backbone.layers.1.mixer.D', 'layer1.D_vec'],
        ['backbone.layers.1.mixer.dt_proj.bias', 'layer1.bDtProj'],
        ['backbone.norm_f.weight', 'final_norm'],
    ]) {
        expect(Array.from(ported.weights.get(target!)!))
            .toEqual(Array.from(find(src, source!).data));
    }
});

test('the depthwise conv kernel is squeezed to [C, K] AND tap-reversed', () => {
    // PyTorch left-pads and cross-correlates (y[t] = Σ_j w[j]·x[t+j-(K-1)]);
    // conv1d.ts computes y[t] = Σ_k w[k]·x[t-k]. The tap axis must flip.
    const src = falconTensors();
    const ported = portForeignMamba(FALCON_CONFIG, src);
    const source = find(src, 'backbone.layers.0.mixer.conv1d.weight');
    const target = ported.weights.get('layer0.wConv')!;

    expect(source.shape).toEqual([D, 1, K]);
    expect(target.length).toBe(D * K);
    for (let c = 0; c < D; c++) {
        for (let k = 0; k < K; k++) {
            expect(target[c * K + k]).toBe(source.data[c * K + (K - 1 - k)]);
        }
    }
    // A real reversal, not a no-op copy.
    expect(Array.from(target)).not.toEqual(Array.from(source.data));
    // The conv bias, by contrast, is a plain copy.
    expect(Array.from(ported.weights.get('layer0.bConv')!))
        .toEqual(Array.from(find(src, 'backbone.layers.0.mixer.conv1d.bias').data));
});

test('Falcon-Mamba reports its bias-free projections instead of silently zeroing them', () => {
    const ported = portForeignMamba(FALCON_CONFIG, falconTensors());

    const synthesised = ported.synthesisedTargets.map((s) => s.target).sort();
    expect(synthesised).toEqual([
        'layer0.bInProj', 'layer0.bOutProj', 'layer0.bXProj',
        'layer1.bInProj', 'layer1.bOutProj', 'layer1.bXProj',
    ]);
    for (const s of ported.synthesisedTargets) {
        expect(s.value).toBe(0);
        expect(s.reason).toMatch(/bias/i);
        expect(Array.from(ported.weights.get(s.target)!).every((v) => v === 0)).toBe(true);
    }
    // Nothing is consumed in part.
    expect(ported.discardedSources).toEqual([]);
});

test('an untied lm_head has no target here and is reported, never dropped in silence', () => {
    // HybridMambaModel ties its LM head to the embedding table, so a checkpoint
    // with tie_word_embeddings:false carries a tensor we cannot place.
    expect(portForeignMamba(FALCON_CONFIG, falconTensors()).unmappedSources)
        .toEqual(['lm_head.weight']);
    expect(portForeignMamba(CODESTRAL_CONFIG, codestralTensors()).unmappedSources)
        .toEqual(['lm_head.weight']);
});

test('the model config a port targets matches the checkpoint geometry', () => {
    const falcon = portForeignMamba(FALCON_CONFIG, falconTensors()).modelConfig;
    expect(falcon).toMatchObject({
        vocabSize: V, dModel: E, numLayers: L, dState: N, dConv: K, expand: 2,
        defaultMamba1: { dtRank: R, biasConv: true },
    });
    expect(falcon.layers).toBeUndefined(); // all-mamba1 is the model's default schedule

    const codestral = portForeignMamba(CODESTRAL_CONFIG, codestralTensors()).modelConfig;
    expect(codestral).toMatchObject({
        vocabSize: V, dModel: E, numLayers: L, dState: N, dConv: K,
        expand: 2, nHeads: H, nGroups: G, chunkLen: 16,
    });
    expect(codestral.layers!.map((l) => l.type)).toEqual(['mamba2', 'mamba2']);
});

test('a bogus `expand` loses to `intermediate_size` (falcon-mamba-7b ships expand: 16)', () => {
    // tiiuae/falcon-mamba-7b declares expand:16 against hidden_size:4096 while
    // its weights are d_inner 8192. Trusting `expand` builds a model 8x too wide.
    const ported = portForeignMamba({ ...FALCON_CONFIG, expand: 16 }, falconTensors());
    expect(ported.modelConfig.expand).toBe(2);
    expect(ported.weights.get('layer0.wInProj')!.length).toBe(2 * D * E);

    // With intermediate_size absent, `expand` is all there is — and it must agree.
    const { intermediate_size: _drop, ...noInner } = FALCON_CONFIG as Record<string, unknown>;
    expect(portForeignMamba({ ...noInner, expand: 2 }, falconTensors()).modelConfig.expand).toBe(2);
});

// ── Codestral-Mamba (Mamba-2) ────────────────────────────────────────────────

test('Codestral-Mamba populates every Mamba-2 target with the right shapes', () => {
    const ported = portForeignMamba(CODESTRAL_CONFIG, codestralTensors());

    expect(ported.adapter).toBe('codestral_mamba');
    expect([...ported.weights.keys()]).toEqual(expectedTargets(MAMBA2_PARAMS));

    const sizes: Record<string, number> = {
        wInProj: (CONV_DIM + H) * E, wConv: CONV_DIM * K, bConv: CONV_DIM,
        A_log: H, dt_bias: H, D_vec: H, wOutProj: E * D, normWeight: D, preNormWeight: E,
    };
    for (let i = 0; i < L; i++) {
        for (const [p, n] of Object.entries(sizes)) {
            expect(ported.weights.get(`layer${i}.${p}`)!.length).toBe(n);
        }
    }
});

test('the fused in_proj drops the gate rows and keeps [x, B, C, dt] in order', () => {
    const src = codestralTensors();
    const ported = portForeignMamba(CODESTRAL_CONFIG, src);
    const source = find(src, 'backbone.layers.0.mixer.in_proj.weight');
    const target = ported.weights.get('layer0.wInProj')!;

    expect(source.shape).toEqual([IN_PROJ_ROWS, E]);
    expect(target.length).toBe((CONV_DIM + H) * E);
    // Row r of the target is row D+r of the source — the [x, B, C, dt] tail.
    expect(Array.from(target)).toEqual(Array.from(source.data.subarray(D * E)));
    // The gate rows really are gone.
    expect(target[0]).toBe(source.data[D * E]);
    expect(target[0]).not.toBe(source.data[0]);
});

test('the discarded gate rows are declared per layer, with their element count', () => {
    const ported = portForeignMamba(CODESTRAL_CONFIG, codestralTensors());
    expect(ported.discardedSources).toHaveLength(L);
    for (let i = 0; i < L; i++) {
        const d = ported.discardedSources[i]!;
        expect(d.source).toBe(`backbone.layers.${i}.mixer.in_proj.weight`);
        expect(d.elements).toBe(D * E);
        expect(d.reason).toMatch(/gate/i);
    }
    // The gate is the ONLY thing Mamba-2 synthesises nothing for; every other
    // target has a real source tensor.
    expect(ported.synthesisedTargets).toEqual([]);
});

test('A_log is re-expressed for the SSD kernel: softplus(target) === exp(source)', () => {
    // Upstream A = -exp(A_log); ssd.ts uses A = -softplus(A_log). A raw copy
    // would change every head's decay rate.
    const src = codestralTensors();
    const ported = portForeignMamba(CODESTRAL_CONFIG, src);
    const source = find(src, 'backbone.layers.0.mixer.A_log').data;
    const target = ported.weights.get('layer0.A_log')!;

    const softplus = (x: number) => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
    for (let h = 0; h < H; h++) {
        // Relative, not absolute: the target rides in a Float32Array, so the
        // round-trip is exact only to f32 precision (~1e-7 relative).
        expect(softplus(target[h]!) / Math.exp(source[h]!)).toBeCloseTo(1, 6);
    }
    // Not a copy — the transform actually fired.
    expect(Array.from(target)).not.toEqual(Array.from(source));

    // dt_bias and D ride across untouched.
    expect(Array.from(ported.weights.get('layer0.dt_bias')!))
        .toEqual(Array.from(find(src, 'backbone.layers.0.mixer.dt_bias').data));
    expect(Array.from(ported.weights.get('layer0.D_vec')!))
        .toEqual(Array.from(find(src, 'backbone.layers.0.mixer.D').data));
});

test('the gated-RMSNorm gain and the pre-block norm land on distinct targets', () => {
    const src = codestralTensors();
    const ported = portForeignMamba(CODESTRAL_CONFIG, src);
    expect(Array.from(ported.weights.get('layer1.normWeight')!))
        .toEqual(Array.from(find(src, 'backbone.layers.1.mixer.norm.weight').data));
    expect(Array.from(ported.weights.get('layer1.preNormWeight')!))
        .toEqual(Array.from(find(src, 'backbone.layers.1.norm.weight').data));
});

test("Mistral's native naming (model. prefix, singular `embedding`) ports identically", () => {
    const hf = portForeignMamba(CODESTRAL_CONFIG, codestralTensors());
    const native = portForeignMamba(
        CODESTRAL_CONFIG,
        codestralTensors('model.', 'backbone.embedding.weight'),
    );
    expect([...native.weights.keys()]).toEqual([...hf.weights.keys()]);
    expect(Array.from(native.weights.get('embedding')!))
        .toEqual(Array.from(hf.weights.get('embedding')!));
    expect(native.unmappedSources).toEqual(['lm_head.weight']);
});

// ── Failure modes ────────────────────────────────────────────────────────────

test('a wrong-shape tensor is rejected, naming the tensor and both shapes', () => {
    const bad = falconTensors().map((t) =>
        t.name === 'backbone.layers.0.mixer.in_proj.weight'
            ? { ...t, shape: [D, E], data: ramp([D, E], 99) }
            : t,
    );
    const call = () => portForeignMamba(FALCON_CONFIG, bad);
    expect(call).toThrow(/in_proj\.weight/);
    expect(call).toThrow(/layer0\.wInProj/);
    expect(call).toThrow(new RegExp(`\\[${2 * D}, ${E}\\]`));
});

test('a rank-2 conv kernel is rejected — the depthwise axis must be there', () => {
    const bad = codestralTensors().map((t) =>
        t.name === 'backbone.layers.0.mixer.conv1d.weight'
            ? { ...t, shape: [CONV_DIM, K + 1], data: ramp([CONV_DIM, K + 1], 7) }
            : t,
    );
    expect(() => portForeignMamba(CODESTRAL_CONFIG, bad)).toThrow(/conv1d\.weight/);
});

test('a short in_proj is rejected rather than sliced past its end', () => {
    const bad = codestralTensors().map((t) =>
        t.name === 'backbone.layers.0.mixer.in_proj.weight'
            ? { ...t, shape: [CONV_DIM, E], data: ramp([CONV_DIM, E], 5) }
            : t,
    );
    expect(() => portForeignMamba(CODESTRAL_CONFIG, bad)).toThrow(/rowSlice|outside/);
});

test('a missing tensor names itself and the target that needed it', () => {
    const dropped = falconTensors().filter((t) => t.name !== 'backbone.layers.1.mixer.A_log');
    const call = () => portForeignMamba(FALCON_CONFIG, dropped);
    expect(call).toThrow(/backbone\.layers\.1\.mixer\.A_log/);
    expect(call).toThrow(/layer1\.A_log/);
});

test('an inconsistent head geometry is rejected before any tensor is touched', () => {
    expect(() => portForeignMamba({ ...CODESTRAL_CONFIG, head_dim: 3 }, codestralTensors()))
        .toThrow(/head_dim/);
    expect(() => portForeignMamba({ ...CODESTRAL_CONFIG, num_heads: 5, head_dim: 5 }, codestralTensors()))
        .toThrow(/divisible by num_heads/);
});

// ── Public surface ───────────────────────────────────────────────────────────

test('a port runs straight off a .safetensors buffer', () => {
    const bytes = tensorsToSafetensors(falconTensors());
    const ported = portForeignMambaSafetensors(FALCON_CONFIG, bytes);
    expect([...ported.weights.keys()]).toEqual(expectedTargets(MAMBA1_PARAMS));
    expect(Array.from(ported.weights.get('embedding')!))
        .toEqual(Array.from(find(falconTensors(), 'backbone.embeddings.weight').data));
});

test('the port is reachable from the package entry point (not an orphan module)', async () => {
    const pkg = await import('../src/index');
    expect(typeof pkg.portForeignMamba).toBe('function');
    expect(typeof pkg.portForeignMambaSafetensors).toBe('function');
    expect(typeof pkg.foreignMambaAdapterFor).toBe('function');
    expect(typeof pkg.applyPortedWeights).toBe('function');
    expect(pkg.FOREIGN_MAMBA_ADAPTERS.map((a) => a.id)).toEqual(['falcon_mamba', 'codestral_mamba']);
});
