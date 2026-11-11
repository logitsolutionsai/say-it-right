"""The lexicon in each TTS engine's own pronunciation syntax (checked 2026-11-11; see references/tts-engines.md).

    text        plain respellings (any engine; the only option for Qwen3-TTS, Chatterbox, Phonon, Pocket, XTTS, Dia)
    ssml        <phoneme alphabet="ipa"> / <sub alias> (Azure, Google Cloud incl. Chirp 3 HD sync, Amazon Polly)
    elevenlabs  inline /IPA/ for researched words, respellings for the rest (eleven_v4; other models: use a PLS
                dictionary via export-pls, or text)
    cartesia    <<p|h|o|n|e|s>> inline IPA (Sonic 3 family): one symbol per |, a primary stress on every word
    kokoro      [word](/phonemes/) in misaki's symbols (Kokoro / misaki English)
    piper       [[ phonemes ]] in espeak-ng's en-us style (Piper)
    pls         a W3C Pronunciation Lexicon file (ElevenLabs dictionaries, Azure <lexicon>, Polly PutLexicon)

For English voices every phoneme format starts from the same US-English phones (_us_words): a sound English lacks
becomes the nearest English one, and a symbol with no English neighbour is a ValueError rather than an input the
engine rejects (Azure: HTTP 400) or silently drops (Kokoro).
"""
from __future__ import annotations

import html
import re
import unicodedata

from . import ipa as I
from .lexicon import Entry, Rendered, Term, render

FORMATS = ("text", "ssml", "elevenlabs", "cartesia", "kokoro", "piper")

_NOT_XML = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")   # XML 1.0 can't hold these, escaped or not


def _x(s: str, quote: bool = True) -> str:
    """Text for an XML document: characters XML forbids become a space, the rest is escaped."""
    return html.escape(_NOT_XML.sub(" ", s), quote=quote)

# Dictionary IPA -> the US-English phone set Azure's en-US voices accept (an unknown phone is an HTTP 400) and
# Cartesia's English table uses: no length marks, ɹ for r, r-coloured ɝ/ɚ, British ɒ as ɑ, and foreign sounds
# as the nearest English one (the target IPA in the lexicon keeps the original; only the engine input changes).
# Order matters: ties and ligatures first so an affricate stays one phone, then the r-coloured vowels.
_US = [("͡", ""), ("͜", ""), ("ʧ", "tʃ"), ("ʤ", "dʒ"), ("ʦ", "ts"), ("ʣ", "dz"), ("ʨ", "tʃ"), ("ʥ", "dʒ"),
       ("ʈʂ", "tʃ"), ("ɖʐ", "dʒ"), ("əʊ", "oʊ"), ("eə", "ɛə"), ("ɪər", "ɪɹ"), ("ʊər", "ʊɹ"), ("ɛər", "ɛɹ"),
       ("ei", "eɪ"), ("ej", "eɪ"), ("ɛj", "eɪ"), ("ou", "oʊ"), ("ai", "aɪ"), ("aj", "aɪ"), ("au", "aʊ"),
       ("oi", "ɔɪ"), ("oɪ", "ɔɪ"), ("oj", "ɔɪ"), ("ɔj", "ɔɪ"),
       ("ɜːr", "ɝ"), ("ɜr", "ɝ"), ("əːr", "ɝ"), ("ər", "ɚ"), ("ɜ", "ɝ"),
       ("ː", ""), ("ˑ", ""), ("ɡ", "g"),
       ("r", "ɹ"), ("ɾ", "ɹ"), ("ʁ", "ɹ"), ("ʀ", "ɹ"), ("ɒ", "ɑ"), ("ɐ", "ə"), ("x", "k"), ("χ", "k"), ("ç", "h"),
       ("ɣ", "g"), ("β", "v"), ("ɸ", "f"), ("ʔ", ""), ("y", "u"), ("ʏ", "ʊ"), ("ø", "ʊ"), ("œ", "ʊ"), ("ɲ", "nj"),
       ("ʎ", "lj"), ("ɨ", "ɪ"), ("ɯ", "u"), ("ɬ", "l"), ("ɕ", "ʃ"), ("ʂ", "ʃ"), ("ʑ", "ʒ"), ("ʐ", "ʒ"),
       ("ʋ", "v"), ("ɫ", "l"), ("ʍ", "hw"), ("ã", "ɑn"), ("ɑ̃", "ɑn"), ("ɛ̃", "æn"), ("ɔ̃", "ɔn"), ("œ̃", "ʌn")]
_LENGTH = ("ː", "ˑ")


def _us(ipa_str: str, keep_long: bool = False) -> str:
    s = unicodedata.normalize("NFC", ipa_str.strip().strip("/[] "))
    s = re.sub("[˥˦˧˨˩ꜜꜛʼ()]", "", s)                        # tone letters, ejectives; (r) is said in the US
    s = re.sub("(.)\u0329", r"ə\1", s)                     # a syllabic consonant (bʌtn̩) is schwa + consonant
    for a, b in _US:
        if not (keep_long and a in _LENGTH):
            s = s.replace(a, b)
    s = re.sub(r"e(?!ɪ)", "eɪ", s)
    s = re.sub(r"o(?!ʊ)", "oʊ", s)
    return s.replace("ɹɹ", "ɹ")


def us_english(ipa_str: str) -> str:
    """IPA as US-English phones only (see _US, _NEAREST), stress marks where the source has them. A lone e/o
    becomes eɪ/oʊ, the nearest English vowels. ValueError for a sound with no English neighbour."""
    return " ".join("".join(t if isinstance(t, str) else "ɹ" if t.sym == "r" else t.sym for t in w)
                    for w in _us_words(ipa_str))


# What Azure's en-US phone table lists (learn.microsoft.com/azure/ai-services/speech-service/speech-ssml-phonetic-sets),
# in ipa.py's spelling (g and r; each format writes ɡ/ɹ its own way). Polly's and Google's en-US tables hold the same
# sounds; Cartesia's and misaki's are subsets handled in their own functions.
_US_VOWELS = frozenset("i ɪ eɪ ɛ æ ɑ ɔ ʊ oʊ u ʌ aɪ aʊ ɔɪ ə ɝ ɚ".split())
_US_PHONES = _US_VOWELS | frozenset("p b t d k g m n ŋ f v θ ð s z ʃ ʒ h tʃ dʒ l r j w".split())

# Sounds _US leaves alone, as the nearest English phone(s); "" drops the sound. Applied per phone, after
# ipa.phones() has removed aspiration, ties and the other diacritics no English phone set has.
_NEAREST = {"a": "ɑ", "e": "eɪ", "o": "oʊ", "ɘ": "ə", "ɵ": "ʊ", "ɞ": "ʌ",
            "ɤ": "ʌ", "ɶ": "æ", "ʉ": "u", "ᵻ": "ɪ", "c": "k", "ɟ": "g", "q": "k", "ɢ": "g", "ʈ": "t", "ɖ": "d",
            "ɳ": "n", "ɭ": "l", "ɽ": "r", "ɻ": "r", "ɺ": "l", "ħ": "h", "ɦ": "h", "ʕ": "", "ɱ": "m", "ɴ": "n",
            "ɥ": "w", "ɰ": "w", "ʝ": "j", "ɮ": "l", "ʟ": "l", "ɓ": "b", "ɗ": "d", "ɠ": "g", "ʙ": "b", "ɧ": "ʃ"}


def _nearest(sym: str, ipa_str: str) -> list[str]:
    if sym in _US_PHONES:
        return [sym]
    if sym in _NEAREST:
        return _NEAREST[sym].split()
    if len(sym) > 1:                                       # ts, pf, British ɪə (no r is added): its parts
        return [s for c in sym for s in _nearest(c, ipa_str)]
    raise ValueError(f"{sym!r} (U+{ord(sym):04X}) in /{ipa_str.strip('/[] ')}/ has no US-English sound to stand in "
                     "for it; write the entry's `ipa` with the nearest English sounds")


def _us_words(ipa_str: str, keep_long: bool = False) -> list[list]:
    """IPA -> words -> tokens (ipa.Phone or a ˈ ˌ . mark) where every phone is in _US_PHONES. Raises ValueError
    for a symbol that can't be mapped, so no format emits something its engine doesn't know."""
    out = []
    for w in I.phones(_us(ipa_str, keep_long)):
        toks = []
        for i, t in enumerate(w):
            if isinstance(t, str):
                toks.append(t)
                continue
            toks += [I.Phone(sym=s, vowel=s in _US_VOWELS, long=t.long) for s in _nearest(t.sym, ipa_str)]
            nxt = next((x for x in w[i + 1:] if isinstance(x, I.Phone)), None)
            if t.nasal and not (nxt and nxt.sym in ("m", "n", "ŋ")):
                toks.append(I.Phone(sym="n", vowel=False))  # a nasal vowel is vowel + n to an English ear
        if any(isinstance(t, I.Phone) for t in toks):
            out.append(toks)
    if not out:
        raise ValueError(f"no sounds in the IPA /{ipa_str.strip('/[] ')}/")
    return out


def _syllables(word: list) -> str:
    """One word with every syllable marked off: "ʃɪ.ˈvɔn". Azure reads a stress mark as stressing the syllable
    after it only when all syllables are marked; otherwise it stresses the one before ("təmeɪˈtoʊ" = tə-MAY-toʊ).
    Google and Polly document "." as an optional syllable boundary, so the same string is right for all three."""
    return ".".join(("ˈ" if s[0].stress == 1 else "ˌ" if s[0].stress == 2 else "")
                    + "".join("ɹ" if p.sym == "r" else p.sym for p in s) for s in I.syllabify(word))


def _phoneme_tags(t: Term, english: bool) -> str:
    """<phoneme> for one term, one tag per written word: Google reads a tag as "a single word", ElevenLabs too."""
    written = re.split(r"(\s+)", t.shown)                  # words at even indexes, the spaces between at odd ones
    phs = [_syllables(w) for w in _us_words(t.ipa)] if english else t.ipa.strip("/[] ").split()
    n = (len(written) + 1) // 2
    if n == 1:
        phs = [".".join(phs)]                              # one written word said as several: still one tag
    if len(phs) != n:
        raise ValueError(f"the IPA has {len(phs)} words for {n} written ones; give it one word per written word")
    return "".join(f'<phoneme alphabet="ipa" ph="{_x(phs[i // 2])}">{_x(part)}</phoneme>'
                   if i % 2 == 0 else _x(part) for i, part in enumerate(written))


def _located(r: Rendered):
    """(offset in r.shown, term) for each term, in order. Term.at is the offset when the lexicon recorded it;
    without it the term is searched for, as a whole word first: "AI" is not the AI inside "MAIN"."""
    pos = 0
    for t in r.terms:
        at = getattr(t, "at", -1)
        if not (at >= pos and r.shown.startswith(t.shown, at)):
            m = re.compile(rf"(?<![\w-]){re.escape(t.shown)}(?![\w-])").search(r.shown, pos)
            at = m.start() if m else r.shown.find(t.shown, pos)
        if at >= 0:
            yield at, t
            pos = at + len(t.shown)


def _named(t: Term, fmt) -> str:
    """fmt(t), with the script's word in front of any error so the user knows which lexicon entry to fix."""
    try:
        return fmt(t)
    except ValueError as e:
        raise ValueError(f"{t.shown}: {e}") from None


def ssml(script: str, table: dict[str, Entry], lang: str = "en-US", voice: str | None = None,
         bare: bool = False) -> str:
    """A complete SSML document. Azure needs version, xmlns, xml:lang and a <voice>; Google and Polly take a bare
    <speak> (bare=True) and no <voice> (Polly has no such tag; Google's wants a Google voice name). English voices
    get US-English phones in <phoneme>, in Azure's spelling (ASCII g) with every syllable marked."""
    r = render(script, table)
    english = lang.lower().startswith("en")
    body, pos = [], 0
    for at, t in _located(r):
        body.append(_x(r.shown[pos:at], quote=False))
        if t.ipa:
            body.append(_named(t, lambda t: _phoneme_tags(t, english)))
        else:
            body.append(f'<sub alias="{_x(t.said)}">{_x(t.shown)}</sub>')
        pos = at + len(t.shown)
    body.append(_x(r.shown[pos:], quote=False))
    inner = "".join(body)
    if voice:
        inner = f'<voice name="{_x(voice)}">{inner}</voice>'
    if bare:
        return f"<speak>{inner}</speak>"
    return (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{_x(lang)}">'
            f"{inner}</speak>")


def _stress_before_vowel(word: list, spell) -> str:
    """espeak/misaki style: the stress mark sits right before the syllable's vowel, not at the syllable's start.
    spell(phone) is the engine's own spelling of one phone."""
    out, pending = [], ""
    for t in word:
        if isinstance(t, str):
            pending = t if t in "ˈˌ" else pending
            continue
        if t.vowel and pending:
            out.append(pending)
            pending = ""
        out.append(spell(t))
    return "".join(out)


def _checked(s: str, allowed: frozenset, engine: str, ipa_str: str) -> str:
    """A guard on the mapping tables: nothing outside the engine's own symbols leaves this module."""
    bad = sorted(set(s) - allowed - {" "})
    if bad:
        raise ValueError(f"{bad[0]!r} (U+{ord(bad[0]):04X}) from /{ipa_str.strip('/[] ')}/ is not a symbol {engine} "
                         "knows")
    return s


# misaki's US-English symbols (github.com/hexgrad/misaki EN_PHONES.md). It has no ɝ/ɚ: "her" is hɜɹ, "butter" bˈʌɾəɹ.
_MISAKI = {"eɪ": "A", "aɪ": "I", "aʊ": "W", "ɔɪ": "Y", "oʊ": "O", "dʒ": "ʤ", "tʃ": "ʧ", "r": "ɹ", "g": "ɡ",
           "ɝ": "ɜɹ", "ɚ": "əɹ"}
_MISAKI_US = frozenset("AIWYObdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹʃʊʌʒʤʧθᵊᵻɾˈˌ")


def misaki(ipa_str: str) -> str:
    """IPA -> misaki's US-English phone set (Kokoro). Kokoro silently drops anything outside its vocabulary, so
    every sound is first made a US-English one (a nasal vowel keeps its n, British əʊ is O, not the British Q)."""
    s = " ".join(_stress_before_vowel(w, lambda p: _MISAKI.get(p.sym, p.sym)) for w in _us_words(ipa_str))
    return _checked(s, _MISAKI_US, "misaki (Kokoro)", ipa_str)


# What espeak-ng's en-us voice writes, the only symbols an English Piper voice was trained on: ɑ ɔ ɜ are always
# long, the NURSE vowel is ɜː and the lettER vowel ɚ, i is short only where the source is (happy), u is uː.
_ESPEAK = {"ɑ": "ɑː", "ɔ": "ɔː", "ɝ": "ɜː", "u": "uː", "g": "ɡ", "r": "ɹ"}
_ESPEAK_US = frozenset("pbtdkɡmnŋfvθðszʃʒhlɹjwiɪeɛæɑɔʊuʌaoəɜɚːˈˌ")


def _espeak(ipa_str: str) -> str:
    """espeak-ng's en-us IPA style (Piper): stress before the vowel, ɡ and ɹ, its vowel lengths."""
    def spell(p):
        return "iː" if p.sym == "i" and p.long else _ESPEAK.get(p.sym, p.sym)
    s = " ".join(_stress_before_vowel(w, spell) for w in _us_words(ipa_str, keep_long=True))
    return _checked(s, _ESPEAK_US, "espeak-ng en-us (Piper)", ipa_str)


# Sonic 3's whole English inventory (docs.cartesia.ai/build-with-cartesia/capability-guides/phonemes): "any other
# symbol ... makes a pronunciation invalid". No ʌ (write ə), no ɚ and no ə|ɹ (write ɝ), ɡ and ɹ not g and r.
_CARTESIA = frozenset("p b t d k ɡ tʃ dʒ f v θ ð s z ʃ ʒ h m n ŋ l ɹ j w "
                      "i ɪ ɛ æ ɑ ɔ ʊ u ə ɝ eɪ aɪ ɔɪ aʊ oʊ ˈ ˌ".split())
_CARTESIA_SYM = {"ʌ": "ə", "ɚ": "ɝ", "g": "ɡ", "r": "ɹ"}


def cartesia(ipa_str: str) -> str:
    """<<s|ˈ|ɛ|ɡ|ɛ|d|i>>: one symbol per |, only Sonic 3's 41 symbols, stress before the stressed syllable's
    onset, and a primary stress on every word (one-syllable words too)."""
    words = []
    for w in _us_words(ipa_str):
        syms = []
        for t in w:
            if t == ".":                                   # no syllable dots in Cartesia's notation
                continue
            s = t if isinstance(t, str) else _CARTESIA_SYM.get(t.sym, t.sym)
            if s == "ɹ":                                   # ə|ɹ and ɝ|ɹ are written ɝ, also across a stress mark:
                k = len(syms) - 2 if syms and syms[-1] in ("ˈ", "ˌ") else len(syms) - 1   # garage is ɡ|ɝ|ˈ|ɑ|ʒ
                if k >= 0 and syms[k] in ("ə", "ɝ"):
                    syms[k] = "ɝ"
                    continue
            syms.append(s)
        if "ˈ" not in syms:
            syms.insert(0, "ˈ")
        bad = [s for s in syms if s not in _CARTESIA]
        if bad:
            raise ValueError(f"{bad[0]!r} from /{ipa_str.strip('/[] ')}/ is not one of Cartesia Sonic 3's symbols")
        words.append("<<" + "|".join(syms) + ">>")
    return " ".join(words)


def _with(r: Rendered, script_terms_fmt) -> str:
    """Rebuild the shown script, swapping each term for script_terms_fmt(term)."""
    out, pos = [], 0
    for at, t in _located(r):
        out.append(r.shown[pos:at])
        out.append(_named(t, script_terms_fmt))
        pos = at + len(t.shown)
    out.append(r.shown[pos:])
    return "".join(out)


def to_engine(script: str, table: dict[str, Entry], fmt: str, opts: dict | None = None) -> str:
    r = render(script, table)
    if fmt == "text":
        return r.spoken
    if fmt == "ssml":
        return ssml(script, table, **(opts or {}))
    if fmt == "elevenlabs":
        # one pair of slashes per word: ElevenLabs' phoneme input "only works for individual words"
        return _with(r, lambda t: " ".join(f"/{w}/" for w in t.ipa.strip("/[] ").split()) if t.ipa else t.said)
    if fmt == "cartesia":
        return _with(r, lambda t: cartesia(t.ipa) if t.ipa else t.said)
    if fmt == "kokoro":
        return _with(r, lambda t: f"[{t.shown}](/{misaki(t.ipa)}/)" if t.ipa else t.said)
    if fmt == "piper":
        return _with(r, lambda t: f"[[ {_espeak(t.ipa)} ]]" if t.ipa else t.said)
    raise ValueError(f"unknown format {fmt!r}; one of {', '.join(FORMATS)}")


def pls(table: dict[str, Entry], lang: str = "en-US") -> str:
    """The lexicon as PLS 1.0: <phoneme> (IPA) for researched entries, <alias> (the respelling) for the rest.

    A multi-word key with IPA becomes one lexeme per word ("Xi Jinping" -> Xi, Jinping): Azure's lexicon can't
    give a phrase a phoneme and allows no white space in one, and ElevenLabs applies phonemes word by word. The
    cost is that each word is then said that way on its own too. If the IPA isn't one word per written word the
    entry falls back to an <alias> with its respelling, which all three engines accept for a phrase."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<lexicon version="1.0" xmlns="http://www.w3.org/2005/01/pronunciation-lexicon"',
             '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
             '         xsi:schemaLocation="http://www.w3.org/2005/01/pronunciation-lexicon '
             'http://www.w3.org/TR/2007/CR-pronunciation-lexicon-20071212/pls.xsd"',
             f'         alphabet="ipa" xml:lang="{_x(lang)}">']
    done = set()

    def lexeme(grapheme: str, tag: str, value: str) -> None:
        if grapheme not in done:                           # engines use the first lexeme for a grapheme
            done.add(grapheme)
            lines.extend(["  <lexeme>", f"    <grapheme>{_x(grapheme)}</grapheme>",
                          f"    <{tag}>{_x(value)}</{tag}>", "  </lexeme>"])
    for key, e in table.items():
        said, written = (re.sub(r"[/\[\]]", "", e.ipa).split() if e.ipa else []), key.split()
        if not said:
            lexeme(key, "alias", e.say)
        elif len(written) == 1:
            lexeme(key, "phoneme", ".".join(said))         # never white space inside <phoneme> (Azure)
        elif len(said) == len(written):
            for word, ph in zip(written, said):
                if word not in table:                      # a word with its own entry keeps that entry
                    lexeme(word, "phoneme", ph)
        else:
            lexeme(key, "alias", e.say)
    lines.append("</lexicon>")
    return "\n".join(lines) + "\n"
