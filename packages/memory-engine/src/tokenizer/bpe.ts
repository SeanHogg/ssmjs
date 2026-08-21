/**
 * bpe.ts – Browser-side Byte Pair Encoding (BPE) tokenizer.
 */

export interface BPEEncodeOptions {
  addBos?: boolean;
  addEos?: boolean;
}

export type PadSide = 'right' | 'left';

/** A Hugging Face `tokenizer.json` (or a bare `{ vocab, merges }`) for import. */
export interface HuggingFaceTokenizerSpec {
  model?: { vocab?: Record<string, number>; merges?: Array<string | [string, string]> };
  vocab?: Record<string, number>;
  merges?: Array<string | [string, string]>;
}

/** Override the special-token strings to match an imported tokenizer. */
export interface SpecialTokenOverrides {
  bos?: string;
  eos?: string;
  pad?: string;
  unk?: string;
}

function buildByteEncoder(): Map<number, string> {
    const enc = new Map<number, string>();
    const ranges: [number, number][] = [
        [0x21, 0x7E],
        [0xA1, 0xAC],
        [0xAE, 0xFF],
    ];
    let n = 0;
    for (const [lo, hi] of ranges) {
        for (let b = lo; b <= hi; b++) {
            enc.set(b, String.fromCodePoint(b));
        }
    }
    for (let b = 0; b < 256; b++) {
        if (!enc.has(b)) {
            enc.set(b, String.fromCodePoint(256 + n));
            n++;
        }
    }
    return enc;
}

const BYTE_ENCODER = buildByteEncoder();
const BYTE_DECODER = new Map([...BYTE_ENCODER].map(([k, v]) => [v, k]));

const PRE_TOKENIZE_RE =
    /(?:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

/** A round-trippable snapshot of a {@link BPETokenizer} — see `toObject`/`deserialize`. */
export interface BPETokenizerSpec {
    vocab: Record<string, number>;
    /** Merge rules in rank order ("a b" strings). */
    merges: string[];
    specials?: { bos?: string; eos?: string; pad?: string; unk?: string };
}

export class BPETokenizer {
    vocab: Map<string, number>;
    idToToken: Map<number, string>;
    merges: Map<string, number>;
    bosToken: string;
    eosToken: string;
    padToken: string;
    unkToken: string;
    bosId: number | null;
    eosId: number | null;
    padId: number | null;

    constructor() {
        this.vocab      = new Map();
        this.idToToken  = new Map();
        this.merges     = new Map();
        this.bosToken   = '<|im_start|>';
        this.eosToken   = '<|im_end|>';
        this.padToken   = '<|endoftext|>';
        this.unkToken   = '<unk>';
        this.bosId      = null;
        this.eosId      = null;
        this.padId      = null;
    }

    async load(vocab: string | Record<string, number>, merges: string | string[]): Promise<void> {
        let vocabObj: Record<string, number>;
        if (typeof vocab === 'string') {
            const res = await fetch(vocab);
            vocabObj = await res.json() as Record<string, number>;
        } else {
            vocabObj = vocab;
        }
        this.vocab     = new Map(Object.entries(vocabObj).map(([k, v]) => [k, Number(v)]));
        this.idToToken = new Map([...this.vocab].map(([k, v]) => [v, k]));

        let mergeLines: string[];
        if (typeof merges === 'string') {
            const res = await fetch(merges);
            const txt = await res.text();
            mergeLines = txt.split('\n').filter(l => l && !l.startsWith('#'));
        } else {
            mergeLines = merges;
        }
        this.merges = new Map();
        mergeLines.forEach((line, rank) => {
            this.merges.set(line.trim(), rank);
        });

        this.bosId = this.vocab.get(this.bosToken) ?? null;
        this.eosId = this.vocab.get(this.eosToken) ?? null;
        this.padId = this.vocab.get(this.padToken) ?? null;
    }

    /**
     * Learn a byte-level BPE vocabulary + merges from a text corpus, then load
     * them into this tokenizer. Makes the tokenizer self-contained (no external
     * vocab file): train on your data, then `encode`/`decode` round-trips real
     * text and frequent sequences compress to single tokens. The base vocabulary
     * always covers all 256 byte symbols, so any input is representable.
     */
    train(corpus: string | string[], opts: { numMerges?: number; minPairFreq?: number } = {}): void {
        const numMerges = opts.numMerges ?? 200;
        const minPairFreq = opts.minPairFreq ?? 2;
        const texts = Array.isArray(corpus) ? corpus : [corpus];

        // Byte-encode each pre-token into a symbol sequence; count word frequencies.
        const wordFreq = new Map<string, number>();
        for (const text of texts) {
            for (const word of text.match(PRE_TOKENIZE_RE) ?? []) {
                const bytes = new TextEncoder().encode(word);
                const byteStr = Array.from(bytes).map((b) => BYTE_ENCODER.get(b) ?? '?').join('');
                wordFreq.set(byteStr, (wordFreq.get(byteStr) ?? 0) + 1);
            }
        }
        const words = [...wordFreq].map(([w, freq]) => ({ syms: [...w], freq }));

        // Greedy BPE: repeatedly merge the most frequent adjacent symbol pair.
        const merges: string[] = [];
        for (let m = 0; m < numMerges; m++) {
            const pairCounts = new Map<string, number>();
            for (const { syms, freq } of words) {
                for (let i = 0; i < syms.length - 1; i++) {
                    const pair = syms[i]! + ' ' + syms[i + 1]!;
                    pairCounts.set(pair, (pairCounts.get(pair) ?? 0) + freq);
                }
            }
            let best = '';
            let bestCount = 0;
            for (const [pair, count] of pairCounts) {
                if (count > bestCount) {
                    best = pair;
                    bestCount = count;
                }
            }
            if (bestCount < minPairFreq) break;
            merges.push(best);
            const sp = best.indexOf(' ');
            const a = best.slice(0, sp);
            const b = best.slice(sp + 1);
            const merged = a + b;
            for (const w of words) {
                const out: string[] = [];
                for (let i = 0; i < w.syms.length; i++) {
                    if (i < w.syms.length - 1 && w.syms[i] === a && w.syms[i + 1] === b) {
                        out.push(merged);
                        i++;
                    } else {
                        out.push(w.syms[i]!);
                    }
                }
                w.syms = out;
            }
        }

        // Assemble the vocabulary: specials, every byte symbol (full coverage), then merges.
        const vocabObj: Record<string, number> = {};
        let id = 0;
        for (const special of [this.unkToken, this.bosToken, this.eosToken, this.padToken]) {
            if (!(special in vocabObj)) vocabObj[special] = id++;
        }
        for (const ch of BYTE_ENCODER.values()) {
            if (!(ch in vocabObj)) vocabObj[ch] = id++;
        }
        for (const pair of merges) {
            const merged = pair.split(' ').join('');
            if (!(merged in vocabObj)) vocabObj[merged] = id++;
        }
        this.loadFromObjects(vocabObj, merges);
    }

    loadFromObjects(vocabObj: Record<string, number>, mergeArr: string[]): void {
        this.vocab     = new Map(Object.entries(vocabObj).map(([k, v]) => [k, Number(v)]));
        this.idToToken = new Map([...this.vocab].map(([k, v]) => [v, k]));
        this.merges    = new Map(mergeArr.map((m, i) => [m, i]));
        this.bosId = this.vocab.get(this.bosToken) ?? null;
        this.eosId = this.vocab.get(this.eosToken) ?? null;
        this.padId = this.vocab.get(this.padToken) ?? null;
    }

    /**
     * Seed this tokenizer from an existing Hugging Face `tokenizer.json` — the
     * *import-merges* path (vs `train`, which learns a fresh vocab from a corpus).
     *
     * BPE merges + vocab are portable DATA, not architecture: a proven code
     * tokenizer (GPT-2 / Llama / StarCoder family) gives Evermind a battle-tested
     * code vocabulary on day one. Those tokenizers use the same GPT-2 byte→unicode
     * mapping this class already implements ({@link BYTE_ENCODER}), so the imported
     * vocab is directly compatible with `encode`/`decode`/`_bpe`.
     *
     * Accepts the full `tokenizer.json` object (reads `.model.vocab` / `.model.merges`)
     * or a bare `{ vocab, merges }`. `merges` may be the classic `"a b"` strings or
     * the newer `["a", "b"]` pair arrays. Special-token strings can be overridden to
     * match the source tokenizer (e.g. `<|endoftext|>`, `<s>`, `</s>`).
     */
    loadHuggingFace(spec: HuggingFaceTokenizerSpec, specials: SpecialTokenOverrides = {}): void {
        const model = (spec.model ?? spec) as { vocab?: Record<string, number>; merges?: Array<string | [string, string]> };
        const vocabObj = model.vocab ?? (spec as HuggingFaceTokenizerSpec).vocab;
        const rawMerges = model.merges ?? (spec as HuggingFaceTokenizerSpec).merges;
        if (!vocabObj || typeof vocabObj !== 'object') {
            throw new Error('loadHuggingFace: missing model.vocab (expected a Hugging Face tokenizer.json)');
        }
        if (!Array.isArray(rawMerges)) {
            throw new Error('loadHuggingFace: missing model.merges (expected a BPE tokenizer.json)');
        }
        const mergeArr = rawMerges.map((m) => (Array.isArray(m) ? `${m[0]} ${m[1]}` : String(m)));
        if (specials.bos) this.bosToken = specials.bos;
        if (specials.eos) this.eosToken = specials.eos;
        if (specials.pad) this.padToken = specials.pad;
        if (specials.unk) this.unkToken = specials.unk;
        this.loadFromObjects(vocabObj, mergeArr);
    }

    /**
     * The tokenizer as a plain, JSON-serialisable object: vocab, merges and the
     * four special-token strings.
     *
     * This is what lets an `.evermind` package EMBED its tokenizer instead of
     * expecting a consumer to source the matching vocab separately — token ids are
     * meaningless without the exact vocabulary that produced them, so shipping a
     * checkpoint without one is shipping half an artifact.
     */
    toObject(): BPETokenizerSpec {
        return {
            vocab: Object.fromEntries(this.vocab),
            merges: [...this.merges.entries()].sort((a, b) => a[1] - b[1]).map(([m]) => m),
            specials: { bos: this.bosToken, eos: this.eosToken, pad: this.padToken, unk: this.unkToken },
        };
    }

    /** Restore a tokenizer from {@link toObject}. */
    loadFromSpec(spec: BPETokenizerSpec): void {
        const sp = spec.specials;
        if (sp?.bos) this.bosToken = sp.bos;
        if (sp?.eos) this.eosToken = sp.eos;
        if (sp?.pad) this.padToken = sp.pad;
        if (sp?.unk) this.unkToken = sp.unk;
        this.loadFromObjects(spec.vocab, spec.merges);
    }

    /** UTF-8 JSON bytes of {@link toObject} — the `.evermind` tokenizer section. */
    serialize(): ArrayBuffer {
        const bytes = new TextEncoder().encode(JSON.stringify(this.toObject()));
        // Copy into a standalone ArrayBuffer (the view may be a slice of a pool).
        const out = new ArrayBuffer(bytes.byteLength);
        new Uint8Array(out).set(bytes);
        return out;
    }

    /** Inverse of {@link serialize}. */
    static deserialize(buffer: ArrayBuffer): BPETokenizer {
        const spec = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer))) as BPETokenizerSpec;
        if (!spec || typeof spec.vocab !== 'object' || !Array.isArray(spec.merges)) {
            throw new Error('BPETokenizer.deserialize: not a serialised BPE tokenizer (expected { vocab, merges })');
        }
        const tok = new BPETokenizer();
        tok.loadFromSpec(spec);
        return tok;
    }

    /**
     * Fetch and import a Hugging Face `tokenizer.json` by URL — the IDE's
     * "import merges from a repo" path. Resolves once the tokenizer is loaded.
     */
    async loadHuggingFaceUrl(url: string, specials: SpecialTokenOverrides = {}): Promise<void> {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`loadHuggingFaceUrl: ${res.status} fetching ${url}`);
        const spec = (await res.json()) as HuggingFaceTokenizerSpec;
        this.loadHuggingFace(spec, specials);
    }

    encode(text: string, opts: BPEEncodeOptions = {}): number[] {
        const words = text.match(PRE_TOKENIZE_RE) ?? [];
        const ids: number[]   = [];

        if (opts.addBos && this.bosId !== null) ids.push(this.bosId);

        for (const word of words) {
            const bytes    = new TextEncoder().encode(word);
            const byteStr  = Array.from(bytes).map(b => BYTE_ENCODER.get(b) ?? '?').join('');
            const bpeTokens = this._bpe(byteStr);

            for (const tok of bpeTokens) {
                const id = this.vocab.get(tok);
                if (id !== undefined) {
                    ids.push(id);
                } else {
                    for (const ch of tok) {
                        const cid = this.vocab.get(ch);
                        if (cid !== undefined) ids.push(cid);
                    }
                }
            }
        }

        if (opts.addEos && this.eosId !== null) ids.push(this.eosId);
        return ids;
    }

    decode(ids: number[]): string {
        let byteStr = '';
        for (const id of ids) {
            const tok = this.idToToken.get(id);
            if (tok !== undefined) byteStr += tok;
        }
        const bytes = new Uint8Array(
            [...byteStr].map(ch => BYTE_DECODER.get(ch) ?? ch.codePointAt(0) ?? 0)
        );
        try {
            return new TextDecoder('utf-8').decode(bytes);
        } catch {
            return byteStr;
        }
    }

    _bpe(word: string): string[] {
        if (this.vocab.has(word)) return [word];

        let symbols = [...word];

        while (symbols.length > 1) {
            let bestRank = Infinity;
            let bestIdx  = -1;

            for (let i = 0; i < symbols.length - 1; i++) {
                const pair = symbols[i] + ' ' + symbols[i + 1];
                const rank = this.merges.get(pair);
                if (rank !== undefined && rank < bestRank) {
                    bestRank = rank;
                    bestIdx  = i;
                }
            }

            if (bestIdx === -1) break;

            const merged = symbols[bestIdx]! + symbols[bestIdx + 1]!;
            symbols = [
                ...symbols.slice(0, bestIdx),
                merged,
                ...symbols.slice(bestIdx + 2),
            ];
        }

        return symbols;
    }

    padOrTruncate(ids: number[], maxLen: number, side: PadSide = 'right'): number[] {
        if (ids.length >= maxLen) return ids.slice(0, maxLen);
        const padId = this.padId ?? 0;
        const pad   = new Array<number>(maxLen - ids.length).fill(padId);
        return side === 'right' ? [...ids, ...pad] : [...pad, ...ids];
    }

    get vocabSize(): number { return this.vocab.size; }
}
