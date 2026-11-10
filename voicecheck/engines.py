"""The lexicon in each TTS engine's own pronunciation syntax (researched 2026-11-10; see references/tts-engines.md).

    text        plain respellings (any engine; the only option for Qwen3-TTS, Chatterbox, Phonon, Pocket, XTTS, Dia)
    ssml        <phoneme alphabet="ipa"> / <sub alias> (Azure, Google Cloud incl. Chirp 3 HD sync, Amazon Polly)
    elevenlabs  inline /IPA/ for researched words, respellings for the rest (eleven_v4; other models: use a PLS
                dictionary via export-pls, or text)
    cartesia    <<p|h|o|n|e|s>> inline IPA (Sonic 3 family): one symbol per |, a primary stress on every word
    kokoro      [word](/phonemes/) in misaki's symbols (Kokoro / misaki English)
    piper       [[ phonemes ]] in espeak-ng style (Piper)
    pls         a W3C Pronunciation Lexicon file (ElevenLabs dictionaries, Azure <lexicon>, Polly PutLexicon)
"""
from __future__ import annotations

import html
import re
import unicodedata

from . import ipa as I
from .lexicon import Entry, Rendered, render

FORMATS = ("text", "ssml", "elevenlabs", "cartesia", "kokoro", "piper")

# Dictionary IPA -> the US-English phone set Azure's en-US voices accept (an unknown phone is an HTTP 400) and
# Cartesia's English table uses: no length marks, ɹ for r, r-coloured ɝ/ɚ, British ɒ as ɑ, and foreign sounds
# as the nearest English one (the target IPA in the lexicon keeps the original; only the engine input changes).
_US = [("ei", "eɪ"), ("ej", "eɪ"), ("ɛj", "eɪ"), ("ou", "oʊ"), ("ai", "aɪ"), ("aj", "aɪ"), ("au", "aʊ"),
       ("oi", "ɔɪ"), ("oj", "ɔɪ"), ("ɔj", "ɔɪ"), ("ɜːr", "ɝ"), ("ɜr", "ɝ"), ("əːr", "ɝ"), ("ər", "ɚ"), ("ɜ", "ɝ"), ("ː", ""), ("ˑ", ""), ("ɡ", "g"),
       ("r", "ɹ"), ("ɾ", "ɹ"), ("ʁ", "ɹ"), ("ʀ", "ɹ"), ("ɒ", "ɑ"), ("ɐ", "ə"), ("x", "k"), ("χ", "k"), ("ç", "h"),
       ("ɣ", "g"), ("β", "v"), ("ɸ", "f"), ("ʔ", ""), ("y", "u"), ("ʏ", "ʊ"), ("ø", "ʊ"), ("œ", "ʊ"), ("ɲ", "nj"),
       ("ʎ", "lj"), ("ɨ", "ɪ"), ("ɯ", "u"), ("ɬ", "l"), ("ɕ", "ʃ"), ("ʂ", "ʃ"), ("ʑ", "ʒ"), ("ʐ", "ʒ"),
       ("ʋ", "v"), ("ɫ", "l"), ("ʍ", "hw"), ("ã", "ɑn"), ("ɑ̃", "ɑn"), ("ɛ̃", "æn"), ("ɔ̃", "ɔn"), ("œ̃", "ʌn")]


def us_english(ipa_str: str) -> str:
    """IPA as US-English phones only (see _US). A lone e/o becomes eɪ/oʊ, the nearest English vowels."""
    s = unicodedata.normalize("NFC", ipa_str.strip().strip("/[] "))
    for a, b in _US:
        s = s.replace(a, b)
    s = re.sub(r"e(?!ɪ)", "eɪ", s)
    s = re.sub(r"o(?!ʊ)", "oʊ", s)
    return s.replace("ɹɹ", "ɹ")


def ssml(script: str, table: dict[str, Entry], lang: str = "en-US", voice: str | None = None,
         bare: bool = False) -> str:
    """A complete SSML document. Azure needs version, xmlns, xml:lang and a <voice>; Google and Polly accept the
    same document, or a bare <speak> with bare=True. English voices get US-English phones in <phoneme>."""
    r = render(script, table)
    english = lang.lower().startswith("en")
    body, rest = [], r.shown
    for t in r.terms:
        k = rest.find(t.shown)
        if k < 0:
            continue
        body.append(html.escape(rest[:k], quote=False))
        if t.ipa:
            ph = us_english(t.ipa) if english else t.ipa.strip("/[] ")
            body.append(f'<phoneme alphabet="ipa" ph="{html.escape(ph)}">{html.escape(t.shown)}</phoneme>')
        else:
            body.append(f'<sub alias="{html.escape(t.said)}">{html.escape(t.shown)}</sub>')
        rest = rest[k + len(t.shown):]
    body.append(html.escape(rest, quote=False))
    inner = "".join(body)
    if voice:
        inner = f'<voice name="{html.escape(voice)}">{inner}</voice>'
    if bare:
        return f"<speak>{inner}</speak>"
    return (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{html.escape(lang)}">'
            f"{inner}</speak>")


def _stress_before_vowel(ipa_str: str) -> str:
    """espeak/misaki style: the stress mark sits right before the syllable's vowel, not at the syllable's start."""
    words = []
    for w in I.phones(ipa_str):
        out, pending = [], ""
        for t in w:
            if isinstance(t, str):
                pending = t if t in "ˈˌ" else pending
                continue
            if t.vowel and pending:
                out.append(pending)
                pending = ""
            out.append(t.sym + ("ː" if t.long else ""))
        words.append("".join(out))
    return " ".join(words)


_MISAKI = [("eɪ", "A"), ("aɪ", "I"), ("aʊ", "W"), ("ɔɪ", "Y"), ("oʊ", "O"), ("əʊ", "Q"), ("dʒ", "ʤ"), ("tʃ", "ʧ"),
           ("r", "ɹ"), ("g", "ɡ"), ("ɛə", "ɛɹ"), ("ɪə", "ɪɹ"), ("ʊə", "ʊɹ"), ("x", "k"), ("e", "A"), ("o", "O"),
           ("ɒ", "ɑ"), ("ɐ", "ə"), ("y", "u"), ("ø", "ə"), ("œ", "ə")]


def misaki(ipa_str: str) -> str:
    """IPA -> misaki's English phone set (Kokoro). Symbols misaki lacks are mapped to the nearest English sound,
    because Kokoro silently drops anything outside its vocabulary."""
    s = _stress_before_vowel(ipa_str).replace("ː", "")
    for a, b in _MISAKI:
        s = s.replace(a, b)
    return s


def _espeak(ipa_str: str) -> str:
    """espeak-ng's English IPA style (Piper): stress before the vowel, ɡ and ɹ."""
    return _stress_before_vowel(ipa_str).replace("g", "ɡ").replace("r", "ɹ")


def cartesia(ipa_str: str) -> str:
    """<<s|ˈ|ɛ|ɡ|ɛ|d|i>>: one symbol per |, no length marks, stress before the stressed syllable's onset, and a
    primary stress on every word (one-syllable words too)."""
    words = []
    for w in I.phones(us_english(ipa_str)):
        syms = []
        if not any(t == "ˈ" for t in w if isinstance(t, str)):
            syms.append("ˈ")
        for t in w:
            if isinstance(t, str):
                if t in "ˈˌ":
                    syms.append(t)
                continue
            syms.append(t.sym.replace("g", "ɡ").replace("r", "ɹ"))
        words.append("<<" + "|".join(syms) + ">>")
    return " ".join(words)


def _with(r: Rendered, script_terms_fmt) -> str:
    """Rebuild the shown script, swapping each term for script_terms_fmt(term)."""
    out, rest = [], r.shown
    for t in r.terms:
        k = rest.find(t.shown)
        if k < 0:
            continue
        out.append(rest[:k])
        out.append(script_terms_fmt(t))
        rest = rest[k + len(t.shown):]
    out.append(rest)
    return "".join(out)


def to_engine(script: str, table: dict[str, Entry], fmt: str, opts: dict | None = None) -> str:
    r = render(script, table)
    if fmt == "text":
        return r.spoken
    if fmt == "ssml":
        return ssml(script, table, **(opts or {}))
    if fmt == "elevenlabs":
        return _with(r, lambda t: f"/{t.ipa.strip('/[] ')}/" if t.ipa else t.said)
    if fmt == "cartesia":
        return _with(r, lambda t: cartesia(t.ipa) if t.ipa else t.said)
    if fmt == "kokoro":
        return _with(r, lambda t: f"[{t.shown}](/{misaki(t.ipa)}/)" if t.ipa else t.said)
    if fmt == "piper":
        return _with(r, lambda t: f"[[ {_espeak(t.ipa)} ]]" if t.ipa else t.said)
    raise ValueError(f"unknown format {fmt!r}; one of {', '.join(FORMATS)}")


def pls(table: dict[str, Entry], lang: str = "en-US") -> str:
    """The lexicon as PLS 1.0: <phoneme> (IPA) for researched entries, <alias> (the respelling) for the rest."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<lexicon version="1.0" xmlns="http://www.w3.org/2005/01/pronunciation-lexicon"',
             '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
             '         xsi:schemaLocation="http://www.w3.org/2005/01/pronunciation-lexicon '
             'http://www.w3.org/TR/2007/CR-pronunciation-lexicon-20071212/pls.xsd"',
             f'         alphabet="ipa" xml:lang="{html.escape(lang)}">']
    for key, e in table.items():
        lines.append("  <lexeme>")
        lines.append(f"    <grapheme>{html.escape(key)}</grapheme>")
        if e.ipa:
            ph = html.escape(re.sub(r"[/\[\]]", "", e.ipa).strip())
            lines.append(f"    <phoneme>{ph}</phoneme>")
        else:
            lines.append(f"    <alias>{html.escape(e.say)}</alias>")
        lines.append("  </lexeme>")
    lines.append("</lexicon>")
    return "\n".join(lines) + "\n"
