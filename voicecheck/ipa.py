"""IPA: split it into phones, compare two pronunciations, and respell one for a voice that only reads text.

Respelling follows Wikipedia's pronunciation respelling key (Help:Pronunciation_respelling_key), adapted for TTS:
schwa is "uh" (no engine reads "ə"), syllables are joined by hyphens into one token, and only the stressed syllable of
a multi-syllable word may be capitalised (an all-caps one-syllable chunk can be spelled out as an initialism).

    respell("ˈsɛɡɛdi", name=True)        -> "Seg-eh-dee"
    respell("ʃɪˈvɔːn", style="stress")   -> "shiv-AWN"
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------- phones

_VOWEL_CHARS = set("aeiouyæɑɒɐɔəɘɛɜɞɪɨʉʊʌɯɤøœɶʏɵɚɝᵻ")
_MULTI = sorted(["eɪ", "aɪ", "ɔɪ", "aʊ", "oʊ", "əʊ", "ɪə", "ɛə", "ʊə", "eə", "ɑɪ", "ɒɪ", "ɐʊ", "ei", "ai", "oi", "au",
                 "ou", "ɛj", "ej", "aj", "ɑj", "ɔj", "oj", "uj", "tʃ", "dʒ", "ts", "dz", "tɕ", "dʑ", "ʈʂ", "ɖʐ", "pf", "tʂ"], key=len, reverse=True)
_DROP = set("ʰʷˠˤ̚ˀ̥̬̩̯̰̤̪̺̻̹̜̟̠̈̽̆̊͡‿͜|‖()[]/-0123456789?^\"")   # aspiration, ties, tone digits...


@dataclass
class Phone:
    sym: str                   # base symbol(s), e.g. "ɛ", "eɪ", "tʃ"
    vowel: bool
    long: bool = False         # followed by ː
    nasal: bool = False        # combining tilde (French an, on)
    stress: int = 0            # 1 primary, 2 secondary: set on a syllable's first phone by syllabify()
    palatal: bool = False      # ʲ


def _is_vowel(sym: str) -> bool:
    return sym[0] in _VOWEL_CHARS


def clean(ipa: str) -> str:
    """Strip the wrapping /.../ or [...], normalise ɡ/g and r-variants spelled differently by different sources."""
    s = unicodedata.normalize("NFD", ipa.strip()).strip("/[] ")
    return s.replace("ɡ", "g").replace("'", "ˈ").replace(",", "ˌ").replace("ɹ", "r").replace(":", "ː")


def phones(ipa: str) -> list[list]:
    """IPA -> words -> tokens; a token is a Phone or a boundary mark ("ˈ", "ˌ", ".")."""
    out = []
    for word in clean(ipa).split():
        toks, i = [], 0
        while i < len(word):
            c = word[i]
            if c in "ˈˌ.":
                toks.append(c)
                i += 1
                continue
            if c in "ːˑ":
                if toks and isinstance(toks[-1], Phone):
                    toks[-1].long = True
                i += 1
                continue
            if c == "̃":                                   # combining tilde: nasal vowel
                if toks and isinstance(toks[-1], Phone):
                    toks[-1].nasal = True
                i += 1
                continue
            if c == "ʲ":
                if toks and isinstance(toks[-1], Phone):
                    toks[-1].palatal = True
                i += 1
                continue
            if c in _DROP or unicodedata.combining(c):
                i += 1
                continue
            sym = next((m for m in _MULTI if word.startswith(m, i)), c)
            toks.append(Phone(sym=sym, vowel=_is_vowel(sym)))
            i += len(sym)
        if toks:
            out.append(toks)
    return out


# ---------------------------------------------------------------- syllables

_ONSETS = {"pl", "pr", "bl", "br", "tr", "dr", "kl", "kr", "gl", "gr", "fl", "fr", "θr", "ʃr", "sp", "st", "sk", "sm",
           "sn", "sl", "sw", "kw", "tw", "dw", "gw", "θw", "spl", "spr", "str", "skr", "skw", "pj", "bj", "kj", "mj",
           "fj", "vj", "hj", "nj", "lj", "sv", "ʃt", "ʃp", "ʃm", "ʃn", "ʃl", "ʃv", "ts", "tsv", "zv", "zm"}


def syllabify(word: list) -> list[list[Phone]]:
    """Split one word's tokens into syllables. Marks in the IPA (ˈ ˌ .) are used where present; elsewhere each
    vowel gets the longest legal onset before it (maximal onset)."""
    sylls, cur, stress = [], [], 0
    for t in word:
        if isinstance(t, str):
            if cur:
                sylls.append((cur, stress))
            cur, stress = [], (1 if t == "ˈ" else 2 if t == "ˌ" else 0)
        else:
            cur.append(t)
    if cur:
        sylls.append((cur, stress))
    out = []
    for seg, st in sylls:                                  # a marked chunk may still hold several vowels
        nuclei = [i for i, p in enumerate(seg) if p.vowel]
        if len(nuclei) <= 1:
            out.append((seg, st))
            continue
        cuts = []
        for a, b in zip(nuclei, nuclei[1:]):
            cons = seg[a + 1:b]
            k = 0                                          # how many of the consonants start the next syllable
            for n in range(len(cons), 0, -1):
                cl = "".join(p.sym for p in cons[-n:])
                if n == 1 or cl in _ONSETS:
                    k = n
                    break
            cuts.append(b - k if cons else b)
        prev = 0
        for j, c in enumerate(cuts + [len(seg)]):
            out.append((seg[prev:c], st if j == 0 else 0))
            prev = c
    result = []
    for seg, st in out:
        if seg:
            seg[0].stress = st
            result.append(seg)
    return result


# ---------------------------------------------------------------- respelling

_CHECKED = {"ɛ", "ɪ", "æ", "ʌ", "ɒ", "ʊ", "e", "ə"}           # short vowels English spelling won't end a syllable on

_V = {  # vowel -> (closed syllable, open syllable)
    "æ": ("a", "a"), "a": ("ah", "ah"), "ɑ": ("ah", "ah"), "ɐ": ("uh", "uh"), "ɒ": ("o", "o"), "ɔ": ("aw", "aw"),
    "ɛ": ("e", "eh"), "e": ("e", "ay"), "ɪ": ("i", "ih"), "i": ("ee", "ee"), "ʌ": ("u", "uh"), "ə": ("u", "uh"),
    "ʊ": ("uu", "uu"), "u": ("oo", "oo"), "o": ("oh", "oh"), "ɜ": ("ur", "ur"), "ɝ": ("ur", "ur"), "ɚ": ("er", "er"),
    "y": ("ew", "ew"), "ʏ": ("uu", "uu"), "ø": ("ur", "ur"), "œ": ("ur", "ur"), "ɨ": ("ih", "ih"), "ɯ": ("oo", "oo"),
    "ɵ": ("ur", "ur"), "ɘ": ("uh", "uh"), "ɤ": ("uh", "uh"), "ɞ": ("ur", "ur"), "ᵻ": ("i", "ih"), "ʉ": ("oo", "oo"),
    "eɪ": ("ay", "ay"), "ei": ("ay", "ay"), "aɪ": ("y", "y"), "ai": ("y", "y"), "ɑɪ": ("y", "y"), "ɔɪ": ("oy", "oy"),
    "oi": ("oy", "oy"), "ɒɪ": ("oy", "oy"), "aʊ": ("ow", "ow"), "au": ("ow", "ow"), "ɐʊ": ("ow", "ow"),
    "oʊ": ("oh", "oh"), "əʊ": ("oh", "oh"), "ou": ("oh", "oh"), "ɪə": ("eer", "eer"), "ɛə": ("air", "air"),
    "eə": ("air", "air"), "ʊə": ("oor", "oor"), "ɛj": ("ay", "ay"), "ej": ("ay", "ay"), "aj": ("y", "y"),
    "ɑj": ("y", "y"), "ɔj": ("oy", "oy"), "oj": ("oy", "oy"), "uj": ("ooy", "ooy"),
}
_V_LONG = {"ɔ": "aw", "ɑ": "ah", "a": "ah", "i": "ee", "u": "oo", "ɜ": "ur", "e": "ay", "o": "oh", "ɛ": "eh"}
_V_R = {"ɛ": "air", "e": "air", "æ": "arr", "ɑ": "ar", "a": "ar", "ɔ": "or", "o": "or", "ɒ": "or", "ɪ": "eer",
        "i": "eer", "ʊ": "oor", "u": "oor", "ʌ": "ur", "ə": "er", "ɜ": "ur", "aɪ": "ire", "aʊ": "our"}
_NASAL = {"ɑ": "ahn", "a": "ahn", "ɛ": "an", "e": "an", "ɔ": "on", "o": "on", "œ": "un", "ə": "un"}
_C = {
    "tʃ": "ch", "dʒ": "j", "ʃ": "sh", "ʒ": "zh", "θ": "th", "ð": "dh", "ŋ": "ng", "j": "y", "x": "kh", "χ": "kh",
    "ç": "kh", "ɲ": "ny", "ʎ": "ly", "ɣ": "g", "ʁ": "r", "ʀ": "r", "r": "r", "ɾ": "r", "ɫ": "l", "ʔ": "", "β": "v", "ɸ": "f",
    "ɕ": "sh", "ʂ": "sh", "ʑ": "zh", "ʐ": "zh", "c": "ky", "ɟ": "gy", "g": "g", "ts": "ts", "dz": "dz", "tɕ": "ch",
    "dʑ": "j", "ʈʂ": "ch", "tʂ": "ch", "ɖʐ": "j", "pf": "pf", "ɬ": "hl", "ɦ": "h", "ʋ": "v", "ɥ": "w", "ʍ": "wh",
    "ɱ": "m", "ɳ": "n", "ɭ": "l", "ɻ": "r", "ʈ": "t", "ɖ": "d", "q": "k", "ɢ": "g", "ħ": "h", "ʕ": "", "ʝ": "y",
}


def _vowel(p: Phone, closed: bool, r_follows: bool) -> str:
    if p.nasal and p.sym in _NASAL:
        return _NASAL[p.sym]
    if r_follows and p.sym in _V_R:
        return _V_R[p.sym]
    if p.long and p.sym in _V_LONG:
        return _V_LONG[p.sym]
    c, o = _V.get(p.sym, (p.sym, p.sym))
    return c if closed else o


def _consonant(p: Phone, nxt: str) -> str:
    s = _C.get(p.sym, p.sym)
    if s == "g" and nxt[:1] in ("e", "i", "y"):
        s = "gh"                                            # keep it hard: "ghee", not "jee"
    if p.palatal and s and s[-1] != "y":
        s += "y"
    return s


def _move_checked(sylls: list[list[Phone]]) -> None:
    """A stressed short vowel can't end an English syllable: 'ˈsɛ.ɡɛ.di' respells as Seg-eh-dee, not Seh-geh-dee.
    Move the next syllable's single onset consonant back when this syllable is stressed or the next one is (shiv-AWN),
    except p/t/k before a stressed vowel (Wikipedia's rule: puh-TAH-toh). Between two unstressed syllables the vowel
    is written open instead (the "eh" of Seg-eh-dee)."""
    for i in range(len(sylls) - 1):
        a, b = sylls[i], sylls[i + 1]
        if not a or not a[-1].vowel or a[-1].sym not in _CHECKED or a[-1].long:
            continue
        if not (a[0].stress or b[0].stress):
            continue
        onset = []
        for p in b:
            if p.vowel:
                break
            onset.append(p)
        if len(b) == len(onset) or not onset:
            continue
        if len(onset) > 1:                                 # ˈdʒɛ.fri -> Jef-ree: a stressed short vowel takes one
            if a[0].stress != 1:
                continue
            st = b[0].stress
            a.append(b.pop(0))
            b[0].stress = st
            continue
        if onset[0].sym in ("p", "t", "k") and b[0].stress == 1:
            continue
        if onset[0].sym in ("j", "w", "h"):
            continue
        st = b[0].stress
        a.append(b.pop(0))
        b[0].stress = st


def respell_word(word: list, style: str = "plain", name: bool = False) -> str:
    sylls = syllabify(word)
    _move_checked(sylls)
    parts = []
    for si, syl in enumerate(sylls):
        out = []
        for k, p in enumerate(syl):
            if p.vowel:
                rest = syl[k + 1:]
                closed = any(not q.vowel for q in rest)
                r_next = bool(rest) and rest[0].sym == "r" and (len(rest) > 1 or si == len(sylls) - 1 or
                                                                  not sylls[si + 1] or not sylls[si + 1][0].vowel)
                out.append(_vowel(p, closed, r_next))
                if r_next:
                    syl[k + 1] = Phone(sym="", vowel=False)    # the r went into the vowel's spelling
            else:
                nxt = ""                                       # the spelling right after, inside this syllable
                if k + 1 < len(syl) and syl[k + 1].vowel:
                    q = syl[k + 1]
                    nxt = _vowel(q, any(not x.vowel for x in syl[k + 2:]), False)
                out.append(_consonant(p, nxt) if p.sym else "")
        s = "".join(out)
        last = next((q.sym for q in reversed(syl) if q.sym), "")
        if last in ("s", "f") and re.fullmatch(r"[^aeiouy]?[aeiou][sf]", s.lower()):
            s += s[-1]                     # "of" reads "uv" and "is" reads "iz": write off, iss (Wikipedia's ss rule)
        if s:
            parts.append((s, syl[0].stress if syl else 0))
    text = "-".join(s for s, _ in parts)
    if style == "stress" and len(parts) > 1:
        text = "-".join(s.upper() if st == 1 else s for s, st in parts)
    if name and text:
        text = text[0].upper() + text[1:]
    return text


def respell(ipa: str, style: str = "plain", name: bool = False) -> str:
    """IPA -> a respelling a plain-text TTS reads. style "plain" (Seg-eh-dee) or "stress" (SEG-eh-dee); name=True
    capitalises each word's first letter."""
    return " ".join(respell_word(w, style, name) for w in phones(ipa))


def candidates(ipa: str, name: bool = False) -> list[str]:
    """A few respellings to try in turn when the first one isn't said right: plain, stress-marked, and spaced."""
    plain = respell(ipa, "plain", name)
    out = [plain, respell(ipa, "stress", name), plain.replace("-", " ")]
    return list(dict.fromkeys(x for x in out if x))


# ---------------------------------------------------------------- comparing two pronunciations

# Vowels: (height 0 close .. 3 open, backness 0 front .. 2 back, rounded)
_VF = {"i": (0, 0, 0), "y": (0, 0, 1), "ɪ": (.5, .3, 0), "ʏ": (.5, .3, 1), "e": (1, 0, 0), "ø": (1, 0, 1),
       "ɛ": (2, 0, 0), "œ": (2, 0, 1), "æ": (2.5, 0, 0), "a": (3, .5, 0), "ɶ": (3, 0, 1), "ɨ": (0, 1, 0),
       "ʉ": (0, 1, 1), "ɘ": (1, 1, 0), "ɵ": (1, 1, 1), "ə": (1.5, 1, 0), "ɜ": (2, 1, 0), "ɞ": (2, 1, 1),
       "ɐ": (2.5, 1, 0), "ɯ": (0, 2, 0), "u": (0, 2, 1), "ʊ": (.5, 1.7, 1), "ɤ": (1, 2, 0), "o": (1, 2, 1),
       "ʌ": (2, 2, 0), "ɔ": (2, 2, 1), "ɑ": (3, 2, 0), "ɒ": (3, 2, 1), "ᵻ": (.5, 1, 0), "ɚ": (1.5, 1, 0),
       "ɝ": (2, 1, 0)}
# Consonants: (place 0 lips .. 8 glottis, manner 0 stop, 1 affricate, 2 fricative, 3 nasal, 4 lateral, 4.5 rhotic,
# 5 approximant, voiced)
_CF = {"p": (0, 0, 0), "b": (0, 0, 1), "m": (0, 3, 1), "ɸ": (0, 2, 0), "β": (0, 2, 1), "f": (1, 2, 0), "v": (1, 2, 1),
       "ʋ": (1, 5, 1), "ɱ": (1, 3, 1), "pf": (1, 1, 0), "θ": (2, 2, 0), "ð": (2, 2, 1), "t": (3, 0, 0), "d": (3, 0, 1),
       "n": (3, 3, 1), "s": (3, 2, 0), "z": (3, 2, 1), "l": (3, 4, 1), "ɫ": (3, 4, 1), "r": (3, 4.5, 1),
       "ts": (3, 1, 0), "dz": (3, 1, 1), "ɬ": (3, 2.5, 0), "ʃ": (4, 2, 0), "ʒ": (4, 2, 1), "tʃ": (4, 1, 0),
       "dʒ": (4, 1, 1), "ʂ": (4.5, 2, 0), "ʐ": (4.5, 2, 1), "ʈ": (4.5, 0, 0), "ɖ": (4.5, 0, 1), "ɳ": (4.5, 3, 1),
       "ɭ": (4.5, 4, 1), "ɻ": (4.5, 4.5, 1), "tʂ": (4.5, 1, 0), "ʈʂ": (4.5, 1, 0), "ɖʐ": (4.5, 1, 1),
       "ɕ": (4.7, 2, 0), "ʑ": (4.7, 2, 1), "tɕ": (4.7, 1, 0), "dʑ": (4.7, 1, 1), "c": (5, 0, 0), "ɟ": (5, 0, 1),
       "ɲ": (5, 3, 1), "ç": (5, 2, 0), "ʝ": (5, 2, 1), "j": (5, 5, 1), "ʎ": (5, 4, 1), "ɥ": (5, 5, 1),
       "k": (6, 0, 0), "g": (6, 0, 1), "ŋ": (6, 3, 1), "x": (6, 2, 0), "ɣ": (6, 2, 1), "w": (6, 5, 1),
       "ʍ": (6, 2, 0), "q": (7, 0, 0), "ɢ": (7, 0, 1), "χ": (7, 2, 0), "ʁ": (7, 4.5, 1), "ʀ": (7, 4.5, 1),
       "ħ": (7.5, 2, 0), "ʕ": (7.5, 2, 1), "h": (8, 2, 0), "ɦ": (8, 2, 1), "ʔ": (8, 0, 0)}
_CF["ɾ"] = (3, 0.5, 1)                                     # a flap: the d of American "ladder", the r of Spanish "pero"
_RHOTIC = {"r", "ʁ", "ʀ", "ɻ", "R"}
_WEAK = {"ə", "ɪ", "ʊ", "ᵻ", "ɐ", "h", "ʔ", "j", "w", "R"}    # cheap to drop or add: reduced vowels, glides, and
                                                           # R, an r after a vowel (rhotic vs non-rhotic accents)
_NEAR = {frozenset(p): c for p, c in [(("ɾ", "d"), .15), (("ɾ", "t"), .2), (("ɾ", "r"), .2), (("R", "r"), .1),
                                      (("j", "i"), .3), (("w", "u"), .3), (("j", "ɪ"), .3), (("w", "ʊ"), .3),
                                      (("ɫ", "l"), .05), (("ʋ", "v"), .15), (("ʋ", "w"), .2),
                                      # the cot-caught and father-bother mergers: accents, not mispronunciations
                                      (("ɔ", "ɑ"), .12), (("ɒ", "ɑ"), .1), (("ɒ", "ɔ"), .1), (("a", "ɑ"), .12),
                                      (("ɛ", "e"), .08)]}
_SPLIT = {"eɪ": "eɪ", "ei": "eɪ", "aɪ": "aɪ", "ai": "aɪ", "ɑɪ": "ɑɪ", "ɔɪ": "ɔɪ", "oi": "ɔɪ", "ɒɪ": "ɒɪ",
          "aʊ": "aʊ", "au": "aʊ", "ɐʊ": "ɐʊ", "oʊ": "oʊ", "ou": "oʊ", "əʊ": "əʊ", "ɪə": "ɪə", "ɛə": "ɛə", "eə": "eə",
          "ʊə": "ʊə", "ɛj": "ɛj", "ej": "ej", "aj": "aj", "ɑj": "ɑj", "ɔj": "ɔj", "oj": "oj", "uj": "uj"}


_RCOLOR = {"ɚ": ("ə", "R"), "ɝ": ("ɜ", "R")}


def units(ipa_or_phones) -> list[tuple]:
    """Comparable units [(symbol, nasal, stress)]: length dropped, diphthongs split into their two parts (so eɪ
    against a plain e costs only the glide), an r after a vowel written R (weak), r-coloured vowels split (ɚ = ə R).
    stress: 1 for the vowel of a ˈ syllable, 0 for other vowels when the IPA marks stress, None when it doesn't."""
    words = phones(ipa_or_phones if isinstance(ipa_or_phones, str) else " ".join(ipa_or_phones))
    out = []
    for w in words:
        marked = any(t == "ˈ" for t in w if isinstance(t, str))
        cur = 0 if marked else None
        flat = []
        for t in w:
            if isinstance(t, str):
                cur = (1 if t == "ˈ" else 0) if marked else None
                continue
            if not t.sym:
                continue
            st = cur if t.vowel else None
            if t.sym in _SPLIT:
                flat += [(c, t.nasal, st, True) for c in _SPLIT[t.sym]]
            elif t.sym in _RCOLOR:
                v, r = _RCOLOR[t.sym]
                flat += [(v, t.nasal, st, True), (r, False, None, False)]
            else:
                flat.append((t.sym, t.nasal, st, t.vowel))
        for k, (sym, nasal, st, vowel) in enumerate(flat):
            if sym == "r" and k and flat[k - 1][3] and not (k + 1 < len(flat) and flat[k + 1][3]):
                sym = "R"
            out.append((sym, nasal, st))
    return out


def _indel(u) -> float:
    return 0.5 if u[0] in _WEAK else (0.8 if u[0] in _VF else 1.0)


def _sub(a, b) -> float:
    """a: a target unit (its stress counts), b: a heard unit."""
    (x, nx, st), (y, ny, _) = a, b
    extra = 0.2 if nx != ny else 0.0
    if x == y:
        return extra
    near = _NEAR.get(frozenset((x, y)))
    if near is not None:
        return near + extra
    if x in _VF and y in _VF:
        (h1, b1, r1), (h2, b2, r2) = _VF[x], _VF[y]
        cost = min(0.7, 0.14 * abs(h1 - h2) + 0.14 * abs(b1 - b2) + 0.12 * abs(r1 - r2))
        if st == 0:
            cost = min(cost, 0.3)              # an unstressed vowel's colour varies by accent and recogniser
        elif st == 1:
            cost = min(1.0, cost * 1.5)        # the stressed vowel is what makes kar-PAH-thee not KAR-puh-thee
        return cost + extra
    if x in _CF and y in _CF:
        if x in _RHOTIC and y in _RHOTIC:
            return 0.1
        (p1, m1, v1), (p2, m2, v2) = _CF[x], _CF[y]
        return min(1.0, min(0.5, 0.12 * abs(p1 - p2)) + (0.0 if m1 == m2 else 0.25 + 0.05 * abs(m1 - m2)) +
                   (0.2 if v1 != v2 else 0.0))
    return 1.0


@dataclass
class Comparison:
    score: float               # 1 = the heard sounds are the target; 0 = nothing in common
    target: str
    heard: str
    diff: str                  # e.g. "s=s ɛ=ɛ g=g ɛ~e d=d i=i" (= same, ~ close, ! different, + extra, - missing)


def compare(target_ipa: str, heard) -> Comparison:
    """How well `heard` (IPA, or a list of phones from the phoneme ear) matches `target_ipa`. Feature-weighted edit
    distance: ɛ for e or a flapped r costs little, s for k costs a lot, a dropped schwa costs half. Heard sounds
    before and after the target are free (a cut-out window carries a little of the neighbouring words), so the
    score says whether the target was said, not what surrounds it."""
    t, h = units(target_ipa), units(heard)
    n, m = len(t), len(h)
    if not n:
        return Comparison(1.0, target_ipa, " ".join(u[0] for u in h), "")
    INF = float("inf")
    D = [[0.0] * (m + 1)] + [[INF] * (m + 1) for _ in range(n)]
    B = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + _indel(t[i - 1])
        B[i][0] = "-"
        for j in range(1, m + 1):
            opts = ((D[i - 1][j - 1] + _sub(t[i - 1], h[j - 1]), "s"), (D[i - 1][j] + _indel(t[i - 1]), "-"),
                    (D[i][j - 1] + _indel(h[j - 1]), "+"))
            D[i][j], B[i][j] = min(opts)
    j = min(range(m + 1), key=lambda k: D[n][k])
    cost, i, ops = D[n][j], n, []
    end = j
    while i > 0:
        op = B[i][j]
        if op == "s":
            c = _sub(t[i - 1], h[j - 1])
            mark = "=" if c == 0 else "~" if c <= 0.35 else "!"
            ops.append(f"{t[i - 1][0]}{mark}{h[j - 1][0]}")
            i, j = i - 1, j - 1
        elif op == "-":
            ops.append(f"-{t[i - 1][0]}")
            i -= 1
        else:
            ops.append(f"+{h[j - 1][0]}")
            j -= 1
    start = j
    full = sum(_indel(u) for u in t)
    return Comparison(score=round(max(0.0, 1 - cost / full), 3), target=" ".join(u[0] for u in t),
                      heard=" ".join(u[0] for u in h[start:end]), diff=" ".join(reversed(ops)))
