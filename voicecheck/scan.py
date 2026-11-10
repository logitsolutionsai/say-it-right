"""Which words of a script a voice is likely to get wrong, before anything is recorded.

Signals, strongest first: already in the lexicon (fine), a person or place name (capitalised mid-sentence), an
acronym or mixed-case term (ReLU, PyTorch, GPT), letters outside English (Søren, Łukasz), letters glued to digits
(GPT-4o, 70B), and rarity: a word that Whisper's tokenizer splits into several pieces is rare in the text the voice
models learned from, so they guess at it (this catches a name at the start of a sentence, too). Numbers are listed
with their readings, so an ambiguous one ("1959 people") can be spelled out in the script.
"""
from __future__ import annotations

import re
import unicodedata

from . import config
from . import text as T
from .lexicon import Entry, _pattern

_COMMON_CAPS = {"I", "OK", "TV", "US", "UK", "EU", "UN", "USA", "AM", "PM", "A"}
_SENT_END = re.compile(r"[.!?:;]['\"”’)\]]*\s*$")


def _tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(config.voice()["checker"])
    except Exception:                                      # no model cache and no network: rarity is skipped
        return None


def scan(script: str, table: dict[str, Entry] | None = None, rarity: bool = True) -> dict:
    table = table or {}
    covered = []
    from .lexicon import _INLINE
    for m in _INLINE.finditer(script):                     # {shown|spoken} spelled out in the script itself
        covered.append((m.start(), m.end(), m.group(1)))
    for key in table:
        for m in _pattern(key).finditer(script):
            covered.append((m.start(), m.end(), key))
    tok = _tokenizer() if rarity else None
    seen: dict[str, dict] = {}
    for m in re.finditer(r"[^\s\"“”()\[\]{},]+", script):
        raw = m.group(0)
        word = raw.strip(".,;:!?'’\"“”")
        word = re.sub(r"['’]s$", "", word)
        if not word or not re.search(r"[^\W\d_]", word):
            continue
        a = m.start() + raw.find(word)
        if any(s <= a < e for s, e, _ in covered):
            key = next(k for s, e, k in covered if s <= a < e)
            seen.setdefault(key, {"word": key, "count": 0, "reasons": ["in lexicon or spelled out inline"],
                                  "in_lexicon": True})
            seen[key]["count"] += 1
            continue
        first = a == 0 or bool(_SENT_END.search(script[:a])) or script[:a].strip() == ""
        reasons = []
        letters = re.sub(r"[^\w]", "", word)
        if re.search(r"\d", word) and re.search(r"[^\W\d_]", word):
            reasons.append("letters and digits")
        if letters.isupper() and len(letters) >= 2 and word not in _COMMON_CAPS:
            reasons.append("acronym: letters or a word?")
        elif re.search(r"[a-z][A-Z]|[A-Z]{2,}[a-z]", letters):
            reasons.append("mixed case")
        elif word[0].isupper() and not first:
            reasons.append("name (capitalised mid-sentence)")
        if any(ord(c) > 127 and unicodedata.category(c).startswith("L") for c in word):
            reasons.append("letters outside English")
        if tok is not None and word.isalpha():
            # Well-known names are one token as written (Christian, Microsoft, Tokyo); rare ones split both as written
            # and in lower case (Joaquin 2/2, Szegedy 3/4), while a common word starting a sentence is one token in
            # lower case (Batch 2/1). Lower-case words need 4+ pieces ("butchered" is 3, and fine).
            as_is, low = len(tok.tokenize(" " + word)), len(tok.tokenize(" " + word.lower()))
            if (as_is >= 2 and low >= 2) if word[0].isupper() else as_is >= 4:
                reasons.append(f"rare ({as_is} tokenizer pieces)")
        if not reasons:
            continue
        e = seen.setdefault(word, {"word": word, "count": 0, "reasons": [], "in_lexicon": False})
        e["count"] += 1
        e["reasons"] = list(dict.fromkeys(e["reasons"] + reasons))
    numbers = []
    for p in T.parts(script):
        if isinstance(p, tuple) and len(p) > 1:
            numbers.append({"readings": list(p[:4])})
    todo = [e for e in seen.values() if not e["in_lexicon"]]
    order = {"name (capitalised mid-sentence)": 0, "letters outside English": 1, "acronym: letters or a word?": 2,
             "mixed case": 3, "letters and digits": 4}
    todo.sort(key=lambda e: (min(order.get(r, 5) for r in e["reasons"]), -e["count"], e["word"].lower()))
    return {"research": todo, "in_lexicon": [e for e in seen.values() if e["in_lexicon"]], "numbers": numbers}
