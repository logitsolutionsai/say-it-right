"""The lexicon: how the voice must say particular words, and where that came from.

One table merges voice.yaml `pronunciations` and any lexicon files (`lexicon:` in the settings, or --lexicon).
An entry is either a bare respelling or a mapping:

    ReLU: rel-you                               # the voice reads "rel-you"; captions keep "ReLU"
    Szegedy:
      say: Seg-eh-dee                           # what the TTS is given (a respelling it reads reliably)
      ipa: ˈsɛɡɛdi                              # the target sound: the phoneme check holds the take to it
      source: https://en.wikipedia.org/wiki/Christian_Szegedy
      note: Hungarian; English speakers say SEG-ə-dee

A key with a capital letter matches case-sensitively (AI is not "ai"); an all-lowercase key matches any case.
Multi-word keys ("Sergey Ioffe") match as phrases, longest first. A possessive keeps its 's (Szegedy's), an acronym's
plural its s (GPUs), and a key matches inside a hyphenated compound (Szegedy-style). An entry with only `ipa` is given
to a plain-text voice as that IPA's respelling.
Scripts can also carry {shown|spoken} inline, which wins over the table (OpenMontage's convention).
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class Entry:
    key: str
    say: str
    ipa: str | None = None
    source: str | None = None
    note: str | None = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}


@dataclass
class Term:
    """One place in a script where a lexicon entry (or an inline {shown|spoken}) applied."""
    shown: str                 # as written in the script
    said: str                  # what the voice is given
    ipa: str | None
    key: str | None            # the lexicon key, None for inline markup
    occurrence: int            # how many times `said`'s words occur in the spoken text before this term (see locate)
    at: int = -1               # where `shown` starts in Rendered.shown


@dataclass
class Rendered:
    spoken: str                # give this to a plain-text TTS
    shown: str                 # the on-screen wording (pass as --shown / alt when checking)
    ssml: str                  # <phoneme> for entries with IPA, <sub alias> for the rest
    terms: list[Term] = field(default_factory=list)


def _key(key) -> str:
    """A lexicon key as text. YAML reads an unquoted 1984 as a number (fine) and NO, Yes, On, null as a boolean or
    nothing (the word is lost)."""
    if isinstance(key, bool) or key is None:
        raise ValueError(f"lexicon key {key!r}: YAML reads an unquoted no/yes/on/off/null as {key!r}; put the word "
                         f"in quotes")
    return str(key)


def _said(key: str, what: str, value) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)) or not str(value).strip():
        raise ValueError(f"lexicon entry {key!r}: {what} must be text, got {value!r} (an empty value or an unquoted "
                         f"no/yes/on/off/null is not a respelling: put it in quotes)")
    return str(value)


def _entry(key: str, value) -> Entry:
    if isinstance(value, dict):
        say, ipa = value.get("say"), value.get("ipa")
        if not say and not ipa:
            raise ValueError(f"lexicon entry {key!r} needs `say` (a respelling) or `ipa`")
        if ipa:
            from .ipa import _VARIANTS, respell
            if not isinstance(ipa, str):
                raise ValueError(f"lexicon entry {key!r}: `ipa` must be text, got {ipa!r}")
            first = _VARIANTS.split(ipa.strip(), maxsplit=1)[0]        # "ˈniːtʃə, -tʃi": the first one is the target
            ipa = ipa if first == ipa.strip() else first.strip("/[] ")
        if not say:                                        # only the sound is known: a plain-text voice gets its
            say = respell(ipa, name=key[:1].isupper()) or key          # respelling, not a spelling it would misread
        return Entry(key=key, say=_said(key, "`say`", say), ipa=ipa or None, source=value.get("source"),
                     note=value.get("note"))
    return Entry(key=key, say=_said(key, "the respelling", value))


def load(paths=None, base: dict | None = None) -> dict[str, Entry]:
    """`pronunciations` from the settings, then each lexicon file (later files win)."""
    import yaml
    v = config.voice()
    own = (base if base is not None else v.get("pronunciations")) or {}
    if not isinstance(own, dict):
        raise ValueError("`pronunciations`: expected a mapping of word -> respelling or entry")
    table = {_key(k): _entry(_key(k), val) for k, val in own.items()}
    files = paths if paths is not None else v.get("lexicon")
    if isinstance(files, (str, Path)):
        files = [files]
    for p in files or []:
        with open(p, encoding="utf-8-sig") as f:           # -sig: Notepad's byte-order mark is not part of a key
            try:
                data = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"{p}: not valid YAML: {' '.join(str(e).split())}") from None
        data = data.get("pronunciations", data) if isinstance(data, dict) else data
        if not isinstance(data, dict):
            raise ValueError(f"{p}: expected a mapping of word -> respelling or entry")
        table.update({_key(k): _entry(_key(k), val) for k, val in data.items()})
    return table


def respellings() -> list[str]:
    """Every respelling in the settings' table (the listener skips the tails of these)."""
    try:
        return [e.say for e in load().values()]
    except FileNotFoundError:
        return [str(x) if not isinstance(x, dict) else str(x.get("say", "")) for x in
                (config.voice().get("pronunciations") or {}).values()]


_SIBILANT = {"s", "z", "ʃ", "ʒ", "tʃ", "dʒ", "ts", "dz", "ɕ", "ʑ", "tɕ", "dʑ", "ʂ", "ʐ", "tʂ", "ʈʂ", "ɖʐ"}
_VOICELESS = {"p", "t", "k", "f", "θ", "x", "χ", "ç", "h", "ħ", "ɸ", "q", "c", "ʈ", "ɬ", "pf", "ʔ"}


def possessive(ipa: str) -> str:
    """The IPA of a name's possessive (or plural), by the English rule: ɪz after a sibilant (Ruiz's, Leibniz's), s
    after another voiceless sound (Kraft's, Bach's), z otherwise (Nguyen's = wɪnz). An engine given
    <phoneme ph="wɪn">Nguyen's</phoneme> says "Win", so the 's has to be in the IPA."""
    from .ipa import Phone, phones
    base = ipa.strip().rstrip("/]").rstrip()
    toks = [t for w in phones(base) for t in w if isinstance(t, Phone)]
    last = toks[-1].sym if toks else ""
    suffix = "ɪz" if last in _SIBILANT else "s" if last in _VOICELESS else "z"
    lead = ipa[:len(ipa) - len(ipa.lstrip("/["))]
    tail = ipa[len(ipa.rstrip("/]")):]
    return lead + base.lstrip("/[") + suffix + tail


def _pattern(key: str) -> re.Pattern:
    """Where `key` stands in a script as a word of its own: also inside a hyphenated compound (Szegedy-style, pre-AI),
    with a possessive 's, or, for an acronym, its plural s (GPUs). Either apostrophe matches either (O'Brien)."""
    flags = 0 if any(c.isupper() for c in key) else re.IGNORECASE
    body = r"\s+".join(re.sub("['’]", "['’]", re.escape(p)) for p in key.split())
    letters = [c for c in key if c.isalpha()]
    tail = "['’]s|s" if len(letters) > 1 and all(c.isupper() for c in letters) else "['’]s"
    return re.compile(rf"(?<!\w){body}(?P<pos>{tail})?(?!\w)", flags)


_INLINE = re.compile(r"\{([^{}|]+)\|([^{}|]+)\}")


def _hits(words: list[str], want: list[str]) -> list[int]:
    return [i for i in range(len(words) - len(want) + 1) if words[i:i + len(want)] == want] if want else []


def render(script: str, table: dict[str, Entry] | None = None) -> Rendered:
    """The script three ways (spoken / shown / SSML) with every lexicon word replaced, and where each one went."""
    from .text import words as to_words
    table = load() if table is None else table
    spans = []                                             # (start, end, shown, said, ipa, key)
    taken = [False] * len(script)

    def claim(a, b):
        if any(taken[a:b]):
            return False
        taken[a:b] = [True] * (b - a)
        return True
    for m in _INLINE.finditer(script):
        if claim(m.start(), m.end()):
            spans.append((m.start(), m.end(), m.group(1), m.group(2), None, None))
    for key in sorted(table, key=lambda k: (-len(k.split()), -len(k))):
        e = table[key]
        for m in _pattern(key).finditer(script):
            if claim(m.start(), m.end()):
                pos = m.group("pos") or ""
                ipa = possessive(e.ipa) if (pos and e.ipa) else e.ipa
                if pos == "s" and len(e.say.split()[-1]) == 1:
                    pos = "'s"                             # "G P U" + s would be read "G P us": a voice wants G P U's
                spans.append((m.start(), m.end(), m.group(0), e.say + pos, ipa, key))
    spans.sort()
    spoken, shown, ssml, terms = [], [], [], []
    pos = at = 0
    for a, b, sh, said, ipa, key in spans:
        plain = script[pos:a]
        spoken.append(plain)
        shown.append(plain)
        ssml.append(html.escape(plain, quote=False))
        spoken.append(said)
        shown.append(sh)
        at += len(plain)
        if ipa:
            ssml.append(f'<phoneme alphabet="ipa" ph="{html.escape(ipa.strip("/[]"))}">{html.escape(sh)}</phoneme>')
        else:
            ssml.append(f'<sub alias="{html.escape(said)}">{html.escape(sh)}</sub>')
        # Count every earlier run of these words, an ordinary "win" before a name respelled "Win" included: locate()
        # looks the term up among the listener's words, where the two look alike. The last hit is the term itself.
        n = max(0, len(_hits(to_words("".join(spoken)), to_words(said))) - 1)
        terms.append(Term(shown=sh, said=said, ipa=ipa, key=key, occurrence=n, at=at))
        at += len(sh)
        pos = b
    tail = script[pos:]
    spoken.append(tail)
    shown.append(tail)
    ssml.append(html.escape(tail, quote=False))
    return Rendered(spoken="".join(spoken), shown="".join(shown), ssml="<speak>" + "".join(ssml) + "</speak>",
                    terms=terms)


def locate(words: list[str], term: Term) -> tuple[int, int] | None:
    """Where `term` sits in the listener's word list: (first index, end index), or None if it isn't there."""
    from .text import words as to_words
    want = to_words(term.said)
    hits = _hits(words, want)
    return (hits[term.occurrence], hits[term.occurrence] + len(want)) if term.occurrence < len(hits) else None
