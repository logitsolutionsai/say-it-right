"""Text as words, the way a voice says them.

A script and a transcript write the same speech differently: "1959" and "nineteen fifty-nine", "100" and "a hundred",
"$5" and "five dollars", "Dr." and "doctor". So every number (and common abbreviation) has several readings, and
`parts()` keeps all of them. Each caller picks the reading that fits its evidence: the listener picks the one that best
fits the audio, and the transcript comparison picks the one that best fits the other side. Nothing has to guess up
front how a voice reads "1959" (a year, or a count?).

    parts("In 1959 it cost $5.")  ->  ["In ", " ", ("nineteen fifty nine", "one thousand nine hundred fifty nine", ...),
                                       " ", " it cost ", " ", ("five dollars",), " ", "."]
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable, Sequence, Union

Part = Union[str, tuple]          # plain text, or the readings of one number (the first is the default)

# Letters that don't decompose under NFKD: Søren, Łukasz, Strauß, Æsir...
_FOLD = str.maketrans({"ø": "o", "ł": "l", "ß": "ss", "æ": "ae", "œ": "oe", "đ": "d", "þ": "th", "ð": "d", "ı": "i",
                       "’": "'", "‘": "'", "ʼ": "'", "`": "'"})
_SYMBOLS = [("&", " and "), ("+", " plus "), ("=", " equals "), ("@", " at ")]
_DIGIT = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def norm(text: str) -> list[str]:
    """Lower-case ASCII words: accents folded (Szegedy, Søren -> soren), hyphens and slashes split words, other
    punctuation dropped. Apostrophes stay inside words (it's, o'clock) but not around them ('quoted' -> quoted).
    Numbers are read by `parts()`; any digit left here is dropped."""
    t = unicodedata.normalize("NFKD", text.lower().translate(_FOLD))
    t = "".join(c for c in t if not unicodedata.combining(c))
    for sym, say in _SYMBOLS:
        t = t.replace(sym, say)
    t = re.sub(r"[^a-z' ]+", " ", t)
    return [w for w in (x.strip("'") for x in t.split()) if w]


# ---------------------------------------------------------------- number readings

def _n2w(n, **kw) -> str:
    from num2words import num2words
    return num2words(n, lang="en", **kw)


def _clean(s: str) -> str:
    return " ".join(re.sub(r"[-,]", " ", s).split())


def _no_and(s: str) -> str:
    return _clean(re.sub(r"\band\b", " ", s))


def _uniq(xs) -> tuple:
    out = []
    for x in xs:
        x = _clean(x)
        if x not in out:
            out.append(x)
    return tuple(out)


def _int_readings(digits: str, year_ok: bool = True) -> tuple:
    """'784' -> seven hundred eighty four | seven hundred and eighty four | seven eight four | ...
    num2words says "and" (British usage); many voices don't, so both are offered."""
    n = int(digits)
    cardinal = [_no_and(_n2w(n)), _clean(_n2w(n))]
    if (n < 200 or n in (1000, 10 ** 6, 10 ** 9, 10 ** 12)) and cardinal[0].startswith("one "):
        cardinal += ["a" + c[3:] for c in cardinal]       # "a hundred", which Whisper writes as "100"
    year = []
    if year_ok and len(digits) == 4 and digits[0] != "0" and 1100 <= n <= 2099:
        y = _clean(_n2w(n, to="year"))                     # nineteen fifty nine, nineteen hundred, twenty twenty
        year = [y, _no_and(y)]
        if 0 < n % 100 < 10:
            year.append(f"{_clean(_n2w(n // 100))} oh {_DIGIT[n % 10]}")   # nineteen oh five, twenty oh five
    spelled = []
    if len(digits) >= 3 or digits[0] == "0" and len(digits) > 1:
        spelled = [" ".join(_DIGIT[int(d)] for d in digits)]
        if "0" in digits:
            spelled.append(" ".join("oh" if d == "0" else _DIGIT[int(d)] for d in digits))
    if digits[0] == "0" and len(digits) > 1:               # 007, 0800: said digit by digit
        return _uniq(spelled + cardinal)
    if year and not 2000 <= n <= 2009:                     # 1959: a year is the likelier reading
        return _uniq(year + cardinal + spelled)
    return _uniq(cardinal + year + spelled)


def _decimal_readings(whole: str, frac: str) -> tuple:
    said = " ".join(_DIGIT[int(d)] for d in frac)
    if not whole or whole == "0":                          # .24, 0.24
        out = [f"point {said}", f"zero point {said}", f"oh point {said}", f"nought point {said}"]
        if whole == "0":
            out.insert(0, out.pop(1))
        return _uniq(out)
    heads = _int_readings(whole)
    out = [f"{h} point {said}" for h in heads]
    if len(frac) == 2 and frac[0] != "0":                  # 3.14 -> three point fourteen
        out.append(f"{heads[0]} point {_clean(_n2w(int(frac)))}")
    return _uniq(out)


def _number_readings(s: str) -> tuple:
    grouped = "," in s                                     # 1,234 is a count, never the year "twelve thirty four"
    s = s.replace(",", "")
    if "." in s:
        whole, frac = s.split(".", 1)
        return _decimal_readings(whole, frac)
    return _int_readings(s, year_ok=not grouped)


def _plural(reading: str) -> str:
    w = reading.split()
    w[-1] = w[-1][:-1] + "ies" if w[-1].endswith("y") else w[-1] + "s"
    return " ".join(w)


_SCALE = {"k": "thousand", "m": "million", "mn": "million", "b": "billion", "bn": "billion", "t": "trillion",
          "thousand": "thousand", "million": "million", "billion": "billion", "trillion": "trillion"}
_CURRENCY = {"$": ("dollar", "dollars"), "£": ("pound", "pounds"), "€": ("euro", "euros")}
_NUM = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"

# Abbreviations a voice expands and Whisper writes short ("Dr. Nguyen" for "Doctor Nguyen"): readings like a number's.
# Capitalised titles are case-sensitive (MS, ST stay acronyms); the period is optional (British "Dr Nguyen").
_ABBR = {"Dr": ("doctor", "drive"), "Mr": ("mister",), "Mrs": ("missus", "misses", "missis"), "Ms": ("miz", "ms"),
         "Prof": ("professor",), "St": ("saint", "street"), "Mt": ("mount",), "Jr": ("junior",), "Sr": ("senior",),
         "vs": ("versus", "vs"), "etc": ("et cetera", "etcetera"), "e.g": ("for example", "e g", "eg"),
         "i.e": ("that is", "i e", "ie")}
_ABBR_RE = "|".join(re.escape(k) for k in sorted(_ABBR, key=len, reverse=True))

_TOKEN = re.compile(
    rf"(?<![\w.'])(?P<abbr>{_ABBR_RE})(?:\.|(?![\w.']))"
    rf"|(?P<cur>[$£€])\s?(?P<amt>{_NUM})(?:\s?(?P<scale>thousand|million|billion|trillion|bn|mn|[kKmMbBtT])\b)?"
    rf"|(?P<time>\b\d{{1,2}}:\d{{2}}\b)"
    rf"|(?P<frac>\b\d+/\d+\b)"
    rf"|(?<![\w.])(?P<range>{_NUM})\s?[-–—]\s?(?P<range2>{_NUM})(?![\w.])"
    rf"|(?:(?:^|(?<=[\s(\[]))-|−)(?P<neg>{_NUM})"
    rf"|\b(?P<ord>\d+)(?:st|nd|rd|th)\b"
    rf"|'(?P<decade2>\d0)s\b"
    rf"|\b(?P<decade>\d*0)s\b"
    rf"|(?P<num>{_NUM})(?:\s?(?P<pct>%))?(?P<unit>[A-Za-z]+)?"
)


def _readings(m: re.Match) -> list[Part]:
    """The readings of one number-like match, as parts (a range, or a number with a unit, makes several)."""
    g = m.groupdict()
    if g["abbr"]:
        return [_ABBR[g["abbr"]]]
    if g["cur"]:
        one, many = _CURRENCY[g["cur"]]
        amt = g["amt"].replace(",", "")
        scale = _SCALE.get((g["scale"] or "").lower(), "")
        unit = one if amt in ("1", "1.0") and not scale else many
        out = [f"{r} {scale} {unit}" for r in _number_readings(amt)]
        if "." in amt and not scale and len(amt.split(".")[1]) == 2 and amt.split(".")[1] != "00":   # $4.99
            whole, cents = amt.split(".")
            w0, c = _int_readings(whole)[0], _clean(_n2w(int(cents)))
            out = [f"{w0} {c}", f"{w0} {one if whole == '1' else many} {c}",
                   f"{w0} {one if whole == '1' else many} and {c} cents"] + out
        return [_uniq(out)]
    if g["time"]:
        h, mm = g["time"].split(":")
        hw = _clean(_n2w(int(h)))
        if mm == "00":
            return [_uniq([f"{hw} o'clock", hw, f"{hw} hundred"])]
        mw = _clean(_n2w(int(mm)))
        return [_uniq([f"{hw} {mw}" if int(mm) >= 10 else f"{hw} oh {mw}", f"{hw} {mw}"])]
    if g["frac"]:
        a, b = (int(x) for x in g["frac"].split("/"))
        aw, bw = _clean(_n2w(a)), _clean(_n2w(b))
        if b == 2:
            one, many = "half", "halves"
        elif b == 4:
            one, many = "quarter", "quarters"
        else:
            one = _clean(_n2w(b, to="ordinal"))
            many = _plural(one)
        out = [f"{aw} {one if a == 1 else many}", f"{aw} over {bw}", f"{aw} {bw}"]
        if a == 1:
            out.insert(1, f"a {one}")
        return [_uniq(out)]
    if g["range"]:
        return [_number_readings(g["range"]), ("to", "", "through"), _number_readings(g["range2"])]
    if g["neg"]:
        rs = _number_readings(g["neg"])
        return [_uniq([f"minus {r}" for r in rs] + [f"negative {r}" for r in rs])]
    if g["ord"]:
        return [_uniq([_n2w(int(g["ord"]), to="ordinal")])]
    if g["decade2"]:                                       # '90s
        return [_uniq([_plural(_clean(_n2w(int(g["decade2"]))))])]
    if g["decade"]:                                        # 1990s, 80s
        d = g["decade"]
        rs = _int_readings(d)
        if len(d) == 4 and 1100 <= int(d) <= 2099:
            rs = rs[:2]                                    # the year readings: nineteen nineties
        return [_uniq(_plural(r) for r in rs)]
    rs = _number_readings(g["num"])
    if g["pct"]:
        rs = _uniq(f"{r} percent" for r in rs)
    out: list[Part] = [rs]
    unit = g["unit"]
    if unit:                                               # 3D, 70B, 4K, 5G, GPT-4o
        short = len(unit) == 1 or (len(unit) <= 3 and unit.isupper())
        options = [" ".join(unit.lower()) if short else unit.lower()]
        if unit.isupper() and unit.lower() in _SCALE:
            options.append(_SCALE[unit.lower()])           # 70B -> seventy billion
        out += [" ", _uniq(options)]
    return out


def parts(text: str) -> list[Part]:
    """Split text into plain strings and tuples of readings (one tuple per number-like token)."""
    text = unicodedata.normalize("NFKC", text).replace("⁄", "/")     # x² -> x2, ½ -> 1/2
    out: list[Part] = []
    pos = 0
    for m in _TOKEN.finditer(text):
        if m.start() > pos:
            out.append(text[pos:m.start()])
        out += [" ", *_readings(m), " "]
        pos = m.end()
    if pos < len(text):
        out.append(text[pos:])
    return out


def expand(ps: Sequence[Part], picks: dict | None = None) -> tuple[list[str], list[int]]:
    """Words for one choice of readings (picks: part index -> reading index), and for each word the part it came
    from (-1 for plain text)."""
    picks = picks or {}
    words, owner = [], []
    for i, p in enumerate(ps):
        ws = norm(p[picks.get(i, 0)] if isinstance(p, tuple) else p)
        words += ws
        owner += [i if isinstance(p, tuple) else -1] * len(ws)
    return words, owner


def choose(ps: Sequence[Part], score: Callable[[list[str]], float], rounds: int = 2) -> dict:
    """Pick the reading of each number that maximises score(words), one number at a time (coordinate ascent).
    Numbers rarely interact, so two rounds settle it."""
    picks: dict = {}
    options = [i for i, p in enumerate(ps) if isinstance(p, tuple) and len(p) > 1]
    if not options:
        return picks
    best = score(expand(ps, picks)[0])
    for _ in range(rounds):
        changed = False
        for i in options:
            for k in range(len(ps[i])):
                if k == picks.get(i, 0):
                    continue
                trial = {**picks, i: k}
                s = score(expand(ps, trial)[0])
                if s > best + 1e-9:
                    best, picks, changed = s, trial, True
        if not changed:
            break
    return picks


def words(text: str) -> list[str]:
    """The words of `text`, every number in its default reading."""
    return expand(parts(text))[0]
