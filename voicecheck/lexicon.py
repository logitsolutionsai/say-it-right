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
Multi-word keys ("Sergey Ioffe") match as phrases, longest first. A possessive keeps its 's (Szegedy's).
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
    occurrence: int            # 0 for the first time this `said` appears in the spoken text, 1 for the second...


@dataclass
class Rendered:
    spoken: str                # give this to a plain-text TTS
    shown: str                 # the on-screen wording (pass as --shown / alt when checking)
    ssml: str                  # <phoneme> for entries with IPA, <sub alias> for the rest
    terms: list[Term] = field(default_factory=list)


def _entry(key: str, value) -> Entry:
    if isinstance(value, dict):
        if not value.get("say") and not value.get("ipa"):
            raise ValueError(f"lexicon entry {key!r} needs `say` (a respelling) or `ipa`")
        return Entry(key=key, say=str(value.get("say") or key), ipa=value.get("ipa"), source=value.get("source"),
                     note=value.get("note"))
    return Entry(key=key, say=str(value))


def load(paths=None, base: dict | None = None) -> dict[str, Entry]:
    """`pronunciations` from the settings, then each lexicon file (later files win)."""
    import yaml
    v = config.voice()
    table = {k: _entry(k, val) for k, val in ((base if base is not None else v.get("pronunciations")) or {}).items()}
    files = paths if paths is not None else v.get("lexicon")
    if isinstance(files, (str, Path)):
        files = [files]
    for p in files or []:
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data = data.get("pronunciations", data) if isinstance(data, dict) else data
        if not isinstance(data, dict):
            raise ValueError(f"{p}: expected a mapping of word -> respelling or entry")
        table.update({str(k): _entry(str(k), val) for k, val in data.items()})
    return table


def respellings() -> list[str]:
    """Every respelling in the settings' table (the listener skips the tails of these)."""
    try:
        return [e.say for e in load().values()]
    except FileNotFoundError:
        return [str(x) if not isinstance(x, dict) else str(x.get("say", "")) for x in
                (config.voice().get("pronunciations") or {}).values()]


def possessive(ipa: str) -> str:
    """The IPA of a name's possessive, by the English rule: ɪz after a sibilant (Ruiz's), s after another voiceless
    sound (Kraft's), z otherwise (Nguyen's = wɪnz). An engine given <phoneme ph="wɪn">Nguyen's</phoneme> says "Win",
    so the 's has to be in the IPA."""
    from .ipa import Phone, phones
    base = ipa.strip().rstrip("/]").rstrip()
    toks = [t for w in phones(base) for t in w if isinstance(t, Phone)]
    last = toks[-1].sym if toks else ""
    suffix = "ɪz" if last in ("s", "z", "ʃ", "ʒ", "tʃ", "dʒ") else "s" if last in ("p", "t", "k", "f", "θ") else "z"
    lead = ipa[:len(ipa) - len(ipa.lstrip("/["))]
    tail = ipa[len(ipa.rstrip("/]")):]
    return lead + base.lstrip("/[") + suffix + tail


def _pattern(key: str) -> re.Pattern:
    flags = 0 if any(c.isupper() for c in key) else re.IGNORECASE
    body = r"\s+".join(re.escape(p) for p in key.split())
    return re.compile(rf"(?<![\w-]){body}(?P<pos>['’]s)?(?![\w-])", flags)


_INLINE = re.compile(r"\{([^{}|]+)\|([^{}|]+)\}")


def render(script: str, table: dict[str, Entry] | None = None) -> Rendered:
    """The script three ways (spoken / shown / SSML) with every lexicon word replaced, and where each one went."""
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
                spans.append((m.start(), m.end(), m.group(0), e.say + pos, ipa, key))
    spans.sort()
    spoken, shown, ssml, terms, seen = [], [], [], [], {}
    pos = 0
    for a, b, sh, said, ipa, key in spans:
        plain = script[pos:a]
        spoken.append(plain)
        shown.append(plain)
        ssml.append(html.escape(plain, quote=False))
        spoken.append(said)
        shown.append(sh)
        if ipa:
            ssml.append(f'<phoneme alphabet="ipa" ph="{html.escape(ipa.strip("/[]"))}">{html.escape(sh)}</phoneme>')
        else:
            ssml.append(f'<sub alias="{html.escape(said)}">{html.escape(sh)}</sub>')
        n = seen.get(said.lower(), 0)
        seen[said.lower()] = n + 1
        terms.append(Term(shown=sh, said=said, ipa=ipa, key=key, occurrence=n))
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
    if not want:
        return None
    hits = [i for i in range(len(words) - len(want) + 1) if words[i:i + len(want)] == want]
    return (hits[term.occurrence], hits[term.occurrence] + len(want)) if term.occurrence < len(hits) else None
