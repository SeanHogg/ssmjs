/**
 * tests/package_tokenizer.test.ts — the embedded tokenizer section.
 *
 * An `.evermind` package used to ship a checkpoint with no vocabulary. Token ids
 * are meaningless without the exact vocab that produced them, so a consumer had
 * to source the matching tokenizer separately — and could pair the wrong one with
 * it and get confident nonsense. The package can now carry its own.
 *
 * Backward compatibility is the load-bearing part: every artifact published
 * before this section existed must still parse byte for byte.
 */

import { EvermindModelPackage } from '../src/moe/moe_package';
import { EvermindLM } from '../src/lm/evermind_lm';
import { BPETokenizer } from '../src/tokenizer/bpe';
import { SharedExpertMoE } from '../src/moe/moe_model';

const CARD = { description: 'test model' };

function tokenizer(extra: string[] = []): BPETokenizer {
    const tok = new BPETokenizer();
    tok.train(['the quick brown fox jumps over the lazy dog', 'the dog barks', ...extra].join('\n'), { numMerges: 24 });
    return tok;
}

function lmFor(tok: BPETokenizer): EvermindLM {
    return new EvermindLM({ vocabSize: tok.vocabSize, dModel: 8, numLayers: 1, hiddenDim: 12, numExperts: 2, topK: 1, seed: 7 });
}

describe('BPETokenizer round-trip', () => {
    test('serialize / deserialize preserves encode and decode exactly', () => {
        const tok = tokenizer();
        const back = BPETokenizer.deserialize(tok.serialize());
        expect(back.vocabSize).toBe(tok.vocabSize);
        for (const text of ['the quick brown fox', 'lazy dog', 'unseen words here']) {
            expect(back.encode(text)).toEqual(tok.encode(text));
            expect(back.decode(tok.encode(text))).toBe(tok.decode(tok.encode(text)));
        }
    });

    test('merge ranks survive the round-trip in order', () => {
        const tok = tokenizer();
        const spec = tok.toObject();
        const back = BPETokenizer.deserialize(tok.serialize());
        expect(back.toObject().merges).toEqual(spec.merges);
        expect(back.toObject().specials).toEqual(spec.specials);
    });

    test('deserialize rejects something that is not a tokenizer', () => {
        const junk = new TextEncoder().encode(JSON.stringify({ hello: 'world' }));
        expect(() => BPETokenizer.deserialize(junk.buffer as ArrayBuffer)).toThrow(/not a serialised BPE tokenizer/);
    });
});

describe('.evermind package with an embedded tokenizer', () => {
    test('an LM package carries its tokenizer through toBlob / fromBlob', () => {
        const tok = tokenizer();
        const pkg = EvermindModelPackage.fromLM(lmFor(tok), { name: 'm', version: '1', card: CARD, tokenizer: tok });
        const back = EvermindModelPackage.fromBlob(pkg.toBlob());

        expect(back.validate()).toEqual({ ok: true, errors: [] });
        expect(back.manifest.tokenizerFormat).toBe('BPE0');
        expect(back.manifest.tokenizerVocabSize).toBe(tok.vocabSize);

        const loaded = back.loadTokenizer();
        expect(loaded).not.toBeNull();
        expect(loaded!.encode('the quick brown fox')).toEqual(tok.encode('the quick brown fox'));
        // And the model still loads alongside it.
        expect(back.loadLM().config.vocabSize).toBe(tok.vocabSize);
    });

    test('a package without a tokenizer still loads, and reports none', () => {
        const tok = tokenizer();
        const pkg = EvermindModelPackage.fromLM(lmFor(tok), { name: 'm', version: '1', card: CARD });
        const back = EvermindModelPackage.fromBlob(pkg.toBlob());
        expect(back.validate().ok).toBe(true);
        expect(back.manifest.tokenizerFormat).toBeUndefined();
        expect(back.loadTokenizer()).toBeNull();
        expect(back.loadLM().config.vocabSize).toBe(tok.vocabSize);
    });

    test('a bare MoE package can embed a tokenizer too', () => {
        const tok = tokenizer();
        const moe = new SharedExpertMoE({ modelDim: 8, hiddenDim: 12, numExperts: 2, topK: 1 });
        const back = EvermindModelPackage.fromBlob(
            EvermindModelPackage.fromModel(moe, { name: 'moe', version: '1', card: CARD, tokenizer: tok }).toBlob(),
        );
        expect(back.validate().ok).toBe(true);
        expect(back.loadTokenizer()!.vocabSize).toBe(tok.vocabSize);
        expect(back.loadModel()).toBeInstanceOf(SharedExpertMoE);
    });

    test('a tokenizer whose vocab disagrees with the model is refused at package time', () => {
        const tok = tokenizer();
        const wrong = new EvermindLM({ vocabSize: tok.vocabSize + 1, dModel: 8, numLayers: 1, hiddenDim: 12, numExperts: 2, topK: 1, seed: 7 });
        expect(() => EvermindModelPackage.fromLM(wrong, { name: 'm', version: '1', card: CARD, tokenizer: tok }))
            .toThrow(/tokenizer vocabSize/);
    });

    test('a corrupted tokenizer section fails validation instead of decoding nonsense', () => {
        const tok = tokenizer();
        const blob = EvermindModelPackage.fromLM(lmFor(tok), { name: 'm', version: '1', card: CARD, tokenizer: tok }).toBlob();
        const bytes = new Uint8Array(blob);
        bytes[bytes.length - 3] = bytes[bytes.length - 3]! ^ 0xff;   // flip a byte in the tokenizer section
        const back = EvermindModelPackage.fromBlob(blob);
        const v = back.validate();
        expect(v.ok).toBe(false);
        expect(v.errors.join(' ')).toMatch(/tokenizer checksum mismatch/);
        expect(() => back.loadTokenizer()).toThrow(/tokenizer checksum mismatch/);
        // A damaged package refuses EVERY load path, matching how a corrupt codec
        // section already behaves: an artifact is intact or it is not.
        expect(() => back.loadLM()).toThrow(/tokenizer checksum mismatch/);
    });

    test('the checkpoint is byte-identical whether or not a tokenizer is embedded', () => {
        const tok = tokenizer();
        const lm = lmFor(tok);
        const withTok = EvermindModelPackage.fromBlob(
            EvermindModelPackage.fromLM(lm, { name: 'm', version: '1', card: CARD, tokenizer: tok }).toBlob(),
        );
        const without = EvermindModelPackage.fromBlob(
            EvermindModelPackage.fromLM(lm, { name: 'm', version: '1', card: CARD }).toBlob(),
        );
        expect(new Uint8Array(withTok.checkpoint)).toEqual(new Uint8Array(without.checkpoint));
        expect(withTok.manifest.checksum).toBe(without.manifest.checksum);
    });
});
