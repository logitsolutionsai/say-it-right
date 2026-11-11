"""Proofread a transcript against its audio: `check` run in reverse.

`check` trusts the script and judges the audio. `proof` trusts the audio and judges a transcript that some model,
tool or typist wrote (Whisper, a meeting notetaker, auto-captions): which words aren't in the audio, what was said
that the transcript skips, which words were misheard, and which names it spelled wrong.

    doc = read("meeting.srt")
    report = proofread(Session(), y, sr, doc, table=lexicon.load(["names.yaml"]))
    fixed = doc.fix([i["edit"] for i in report["issues"] if i["confidence"] == "high" and i.get("edit")])

The evidence, strongest first:

- The letter ear (no language model) aligns the transcript to the audio. A word it can't find is not in the audio
  (`unheard`: a hallucinated sentence, a line from another take); speech it hears between two aligned words is
  missing from the transcript (`missing`).
- A second transcript (Whisper by default, or any other tool's with `second=`) is compared with the first. Where
  they differ, both wordings are aligned to the same stretch of audio and the letter ear says which fits better
  (`misheard`). Two models with language models guess; a model without one arbitrates.
- The lexicon is the glossary of names and terms that matter. A name's IPA is searched for in the phones of the
  whole recording, and the transcript's own near-spellings are checked by sound: where the audio says the name
  and the transcript wrote something else, the fix is the lexicon's spelling (`name`). Where the transcript wrote
  the name and the audio doesn't sound like it, that is reported too (`name-mismatch`).

Names can't be settled by letters ("Shivon" fits the sound of Siobhan better than "Siobhan" does), so a
disagreement about a capitalised word that isn't in the lexicon is reported as `spelling`, never auto-fixed.
"""
from __future__ import annotations

import bisect
import difflib
import itertools
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, ctc, verify
from . import text as T

LEVELS = {"high": 3, "medium": 2, "low": 1}
_FILLERS = {"uh", "um", "er", "ah", "eh", "mm", "hm", "hmm", "mhm", "uhm", "erm", "oh"}     # transcripts leave them out


# ---------------------------------------------------------------- the transcript as a document

@dataclass
class Region:
    """A stretch of the source that is spoken text (a caption's line, a JSON segment, a paragraph)."""
    start: int
    end: int
    cue: int | None = None          # caption or segment number, from 1
    t0: float | None = None         # the cue's own times, when the format has them
    t1: float | None = None


@dataclass
class Doc:
    kind: str                       # text, srt, vtt or json
    source: str                     # what the regions and every span index into
    regions: list
    data: object = None             # json: the parsed object, to write the fixed text back into

    def fix(self, edits: list) -> str:
        """The transcript with `edits` [(start, end, text)] applied, in its own format. An edit that crosses from
        one caption into the next puts its text in the first and only deletes from the others, so the timing lines
        between them survive. Overlapping edits: the earlier one wins."""
        cut = []
        for a, b, text in sorted((int(a), int(b), t) for a, b, t in edits):
            inside = [(max(a, r.start), min(b, r.end)) for r in self.regions if a <= r.end and b >= r.start
                      and max(a, r.start) <= min(b, r.end)]
            if a == b:
                inside = inside[:1] or [(a, b)]
            for k, (x, y) in enumerate(inside or [(a, b)]):
                if not cut or x >= cut[-1][1]:
                    cut.append((x, y, text if k == 0 else ""))
        out, pos = [], 0
        for a, b, text in cut:
            out += [self.source[pos:a], text]
            pos = b
        out.append(self.source[pos:])
        fixed = "".join(out)
        if self.kind != "json":
            return fixed

        def moved(p, end):                                 # where a region's edge went; text inserted exactly at
            return p + sum(len(t) - (b - a) for a, b, t in cut if b < p or (b == p and (end or a < b)))  # an end stays in
        marks = {p: moved(p, p == r.end) for r in self.regions for p in (r.start, r.end)}
        shape, obj = self.data
        obj = json.loads(json.dumps(obj))                      # a copy
        if shape == "text":
            obj["text"] = fixed
        else:
            segs = obj["segments"] if isinstance(obj, dict) else obj
            for seg, r in zip(segs, self.regions):
                seg["text"] = fixed[marks[r.start]:marks[r.end]]
                seg.pop("words", None)                         # word timings no longer match the text
            if isinstance(obj, dict) and isinstance(obj.get("text"), str):
                obj["text"] = "".join(s["text"] for s in segs)
        return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"

    def text(self) -> str:
        """Only the spoken text, one region per line."""
        return "\n".join(self.source[r.start:r.end].strip() for r in self.regions)


_TIMING = re.compile(r"^\s*((?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*((?:\d+:)?\d{1,2}:\d{2}[.,]\d{1,3})")
# Not speech: <i> and <v Speaker> tags, [Music], {\an8}, music notes, and (laughs)-style notes.
_MASK = re.compile(r"<[^>\n]{1,80}>|\[[^\]\n]{1,80}\]|\{\\[^}\n]{1,40}\}|[♪♫♬]+")
_PAREN = re.compile(r"\(([^()\n]{1,40})\)")
_NOISES = set("laughs laughter laughing applause music inaudible crosstalk silence sighs sigh coughs cough pause "
              "cheering chuckles chuckle unintelligible noise background playing clears throat gasps beep bleep "
              "no audio indistinct chatter phone ringing door closes opens groans screams whispers sobs".split())
_LABEL = re.compile(r"(?m)^[ \t]*(?:>>+[ \t]*|-[ \t]+)?([A-Z][^\s:]{0,20}(?:[ _][^\s:]{1,20}){0,2}):(?=[ \t]+\S)")
_STAMP = re.compile(r"(?m)^[ \t]*[\[(]?\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?[\])]?(?=[ \t])")


def _seconds(stamp: str) -> float:
    return sum(float(x) * 60 ** k for k, x in enumerate(reversed(stamp.replace(",", ".").split(":"))))


def _spoken(source: str, regions: list) -> list:
    """Split each region around what isn't speech: tags, [notes], (laughs), time stamps, and speaker labels. A
    label ("Maria:", "SPEAKER_01:") counts when at least two lines start with one (the transcript is written that
    way), or when it is in capitals or says speaker."""
    labels: dict = {}
    for r in regions:
        for m in _LABEL.finditer(source[r.start:r.end]):
            labels[m.group(1)] = labels.get(m.group(1), 0) + 1
    out = []
    for r in regions:
        text = source[r.start:r.end]
        holes = [m.span() for m in _MASK.finditer(text)] + [m.span() for m in _STAMP.finditer(text)]
        holes += [m.span() for m in _PAREN.finditer(text)
                  if m.group(1).isupper() or all(w in _NOISES for w in T.norm(m.group(1)))]
        holes += [m.span() for m in _LABEL.finditer(text)
                  if sum(labels.values()) >= 2 or m.group(1).isupper()
                  or re.fullmatch(r"(?i)(speaker|spk)[ _-]?\w{1,4}", m.group(1))]
        pos = 0
        for a, b in sorted(holes):
            if a > pos and text[pos:a].strip():
                out.append(Region(r.start + pos, r.start + a, r.cue, r.t0, r.t1))
            pos = max(pos, b)
        if text[pos:].strip():
            out.append(Region(r.start + pos, r.end, r.cue, r.t0, r.t1))
    return out


def parse(raw: str, kind: str = "text") -> Doc:
    """A transcript as a Doc. kind: text (also Markdown), srt, vtt, or json (Whisper's {"segments": [{"text",
    "start", "end"}]}, a bare list of such segments, or {"text": ...})."""
    raw = raw.lstrip("﻿")
    if kind == "json":
        obj = json.loads(raw)
        segs = obj.get("segments") if isinstance(obj, dict) else obj
        if isinstance(segs, list) and segs and all(isinstance(x, dict) and isinstance(x.get("text"), str)
                                                   for x in segs):
            regions, pos = [], 0
            for k, seg in enumerate(segs):
                def num(x):
                    return float(x) if isinstance(x, (int, float)) else None
                regions.append(Region(pos, pos + len(seg["text"]), k + 1, num(seg.get("start")), num(seg.get("end"))))
                pos += len(seg["text"]) + 1
            doc = Doc("json", "\n".join(s["text"] for s in segs), regions, ("segments", obj))
        elif isinstance(obj, dict) and isinstance(obj.get("text"), str):
            doc = Doc("json", obj["text"], [Region(0, len(obj["text"]))], ("text", obj))
        else:
            raise ValueError('unsupported JSON transcript: expected {"segments": [{"text": ...}]}, a list of such '
                             'segments, or {"text": ...}; pass the plain text instead')
    elif kind in ("srt", "vtt"):
        regions, pos, cue, on, t0, t1 = [], 0, 0, False, None, None
        for line in raw.splitlines(keepends=True):
            body = line.rstrip("\r\n")
            m = _TIMING.match(body)
            if m:
                cue, on, t0, t1 = cue + 1, True, _seconds(m.group(1)), _seconds(m.group(2))
            elif not body.strip():
                on = False
            elif on:
                regions.append(Region(pos, pos + len(body), cue, t0, t1))
            pos += len(line)
        doc = Doc(kind, raw, regions)
    else:
        doc = Doc("text", raw, [Region(0, len(raw))])
    doc.regions = _spoken(doc.source, doc.regions)
    return doc


def read(arg: str) -> Doc:
    """A transcript from a file (.txt, .md, .srt, .vtt, .json) or, if `arg` isn't one, from the text itself."""
    p = Path(arg) if len(arg) < 260 else None
    suffix = p.suffix.lower().lstrip(".") if p else ""
    try:
        is_file = bool(p) and suffix in ("txt", "md", "srt", "vtt", "json") and p.is_file()
    except OSError:
        is_file = False
    if not is_file:
        if suffix in ("srt", "vtt", "json") and not re.search(r"\s", arg):
            raise FileNotFoundError(f"no such transcript file: {arg}")
        return parse(arg)
    return parse(p.read_text(encoding="utf-8-sig"), suffix if suffix in ("srt", "vtt", "json") else "text")


@dataclass
class Unit:
    """One word-sized piece of the source: a plain word, or a number or abbreviation with its readings."""
    part: object                    # str, or a tuple of readings (text.parts)
    start: int
    end: int
    region: int


def units(doc: Doc) -> list:
    out = []
    for ri, r in enumerate(doc.regions):
        raw = doc.source[r.start:r.end]
        chars, back = [], []                                   # NFKC changes lengths (x² -> x2): keep a map back
        for i, c in enumerate(raw):
            n = unicodedata.normalize("NFKC", c).replace("⁄", "/")
            if "¼" <= c <= "¾" or "⅐" <= c <= "⅞":                  # 2½ is "2 and 1/2", as in text.parts
                n = (" and " if raw[:i].rstrip(" ")[-1:].isdigit() else "") + n
            chars.append(n)
            back += [i] * len(n)
        text = "".join(chars)

        def span(a, b, r=r, back=back):
            return r.start + back[a], r.start + back[b - 1] + 1

        def plain(a, b, text=text, span=span, ri=ri):
            for m in re.finditer(r"\S+", text[a:b]):
                if T.norm(m.group(0)):
                    out.append(Unit(m.group(0), *span(a + m.start(), a + m.end()), ri))
        pos = 0
        for m in T._TOKEN.finditer(text):
            plain(pos, m.start())
            for part in T._readings(m):
                if isinstance(part, tuple):
                    out.append(Unit(part, *span(m.start(), m.end()), ri))
            pos = m.end()
        plain(pos, len(text))
    return out


def _words(us: list) -> tuple[list[str], list[int]]:
    """Every unit in its default reading: the words, and for each word its unit."""
    words, owner = [], []
    for i, u in enumerate(us):
        ws = T.norm(u.part[0] if isinstance(u.part, tuple) else u.part)
        words += ws
        owner += [i] * len(ws)
    return words, owner


def _core(source: str, a: int, b: int) -> tuple[int, int]:
    """[a, b) without the punctuation around it ("Shivon," -> Shivon), so a fix keeps the comma."""
    while a < b and not (source[a].isalnum() or source[a] in "$£€#@&%+-−."):
        a += 1
    while a < b and source[a] in "-−." and not (a + 1 < b and source[a + 1].isdigit()):
        a += 1
    while b > a and not (source[b - 1].isalnum() or source[b - 1] in "%°+"):
        b -= 1
    return a, b


def _starts_sentence(source: str, a: int) -> bool:
    before = source[:a].rstrip(" \t\"'“‘([-–—>")
    return not before or before[-1] in ".!?:\n\r"


def _namelike(source: str, a: int, b: int, rare=None) -> bool:
    """A name, as far as spelling can tell: an acronym, a capitalised word that doesn't start a sentence, or one
    that does and that `rare(word)` says is no ordinary word (scan's test: the tokenizer splits it)."""
    a, b = _core(source, a, b)
    word = source[a:b]
    if not word or not word[0].isupper() or word in ("I", "I'm", "I'll", "I've", "I'd"):
        return False
    if (len(word) > 1 and word.isupper()) or not _starts_sentence(source, a):
        return True
    return bool(rare and word.isalpha() and rare(word))


def _rarity():
    """rare(word) for a capitalised word, by the test `scan` uses: well-known words are one tokenizer piece as
    written or in lower case; a rare name splits both ways. None when no tokenizer is cached."""
    from .scan import _tokenizer
    tok = _tokenizer()
    if tok is None:
        return None
    memo: dict = {}

    def rare(word: str, pieces: int = 2) -> bool:
        if (word, pieces) not in memo:
            memo[word, pieces] = (len(tok.tokenize(" " + word)) >= pieces and
                                  len(tok.tokenize(" " + word.lower())) >= pieces)
        return memo[word, pieces]
    return rare


def _letters(words) -> str:
    return "".join(words).replace("'", "")


def _sim(a: str, b: str) -> float:
    if not a or not b or min(len(a), len(b)) / max(len(a), len(b)) < 0.5:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _readings(us: list, limit: int = 64) -> list[list[str]]:
    """Every way to read a short run of units (their numbers in each reading), as word lists."""
    options = [list(u.part) if isinstance(u.part, tuple) else [u.part] for u in us]
    if not options or np.prod([len(o) for o in options]) > limit:
        return [_words(us)[0]]
    return [[w for piece in combo for w in T.norm(piece)] for combo in itertools.product(*options)]


def _clip(text: str, n: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


# ---------------------------------------------------------------- proofreading

def proofread(session, y: np.ndarray, sr: int, doc: Doc, table: dict | None = None, second="whisper") -> dict:
    """The report for one transcript (see references/proofreading.md). `second`: "whisper" (transcribe the audio
    again as a second opinion), another transcript as a Doc, or None (the letter ear and the lexicon alone)."""
    v = config.voice()
    near, gone, gain_min = v["slip_score"], v["gone_score"], float(v.get("proof_gain", 4.0))
    timing: dict = {}
    began = time.perf_counter()

    def lap(key, t0):
        timing[key] = round(timing.get(key, 0.0) + time.perf_counter() - t0, 2)

    us = units(doc)
    if not us:
        raise ValueError("the transcript has no words to check")
    listener = session._load("listener", timing)
    t0 = time.perf_counter()
    lp, sec = listener.log_probs(y, sr)
    parts = [u.part for u in us]
    r = listener.score_units(lp, parts, free_edges=True)
    W, S, F, O = r["words"], r["scores"], r["frames"], r["owner"]
    # Text that isn't in the audio has to be put somewhere, and the alignment pays for it with the words next to it
    # (after "...until we are gone. Please subscribe to the channel", "we are" went missing too). So the words it
    # couldn't find are taken out and the rest is aligned again; they stay unheard, their neighbours come back.
    named = set()                                          # a lexicon name written right scores low by its letters
    for e in (table or {}).values():                       # (that is why it is in the lexicon): it is not lost
        kw = T.words(e.key)
        for i in range(len(W) - len(kw) + 1 if kw else 0):
            if W[i:i + len(kw) - 1] == kw[:-1] and W[i + len(kw) - 1] in (kw[-1], kw[-1] + "'s"):
                named.update(range(i, i + len(kw)))
    lost, k = set(), 0
    while k < len(W):
        m = k
        while m < len(W) and m not in named and (F[m] is None or S[m] < gone):
            m += 1
        if m - k >= 2 or len(_letters(W[k:m])) >= 4:
            lost.update(range(k, m))
        k = max(m, k + 1)
    mine: dict = {}
    for i, o in enumerate(O):
        mine.setdefault(o, []).append(i)
    drop = {u for u, idx in mine.items() if all(i in lost for i in idx)} if lost else set()
    if drop and len(drop) < len(us):
        keep = [u for u in range(len(us)) if u not in drop]
        r2 = listener.score_units(lp, [parts[u] for u in keep], free_edges=True)
        again = {}
        for w, sc, fr, o in zip(r2["words"], r2["scores"], r2["frames"], r2["owner"]):
            again.setdefault(keep[o], []).append((w, sc, fr))
        merged = []
        for u in range(len(us)):
            merged += [(w, sc, fr, u) for w, sc, fr in again.get(u, [])] if u not in drop else \
                [(W[i], 0.0, None, u) for i in mine.get(u, [])]
        W, S, F, O = (list(x) for x in zip(*merged))
        r = {**r2, "measured": r["measured"] and r2["measured"]}
    free = listener.hear(lp)
    free_at = [a for _, a, _, _, _ in free]
    lap("letters", t0)
    n, total = len(W), lp.shape[0]
    # How much of the transcript the letter ear hears, word for word, with no script to lean on. A transcript of
    # this audio shares most of its words; a transcript of something else shares a handful by chance.
    shared = sum(b.size for b in difflib.SequenceMatcher(None, W, [h[0] for h in free],
                                                         autojunk=False).get_matching_blocks())
    agreement = shared / max(1, n)
    unrelated = agreement < float(v.get("proof_agree_min", 0.3))
    src = doc.source
    issues: list = []
    taken = [False] * n                                    # words an issue (or a lexicon name) already explains
    busy = np.zeros(total, dtype=bool)                     # frames an issue already explains

    # ---- helpers
    def found(k):
        return F[k] is not None and S[k] >= gone

    def bounds(i, j):
        """The audio around words i..j-1: from the start of the word before to the end of the word after, with
        those two words as context (they anchor the alternatives being compared)."""
        if i > 0 and found(i - 1):
            f0, left = F[i - 1][0], [W[i - 1]]
        else:
            f0, left = next((F[k][1] for k in range(i - 1, -1, -1) if found(k)), 0), []
        if j < n and found(j):
            f1, right = F[j][1], [W[j]]
        else:
            f1, right = next((F[k][0] for k in range(j, n) if found(k)), total), []
        return f0, max(f1, f0 + 1), left, right

    def inner(i, j):
        """The frames between the word before i and the word after j-1: where words i..j-1 must be."""
        f0 = next((F[k][1] for k in range(i - 1, -1, -1) if found(k)), 0)
        f1 = next((F[k][0] for k in range(j, n) if found(k)), total)
        return f0, max(f1, f0)

    def loglik(f0, f1, words):
        toks = listener._tokens(words)[0]
        if not toks:
            return float(lp[f0:f1, listener.blank].sum())
        if ctc.cells(f1 - f0, len(toks)) > verify.MAX_CELLS:
            return None
        ali = ctc.align(lp[f0:f1], toks, blank=listener.blank)
        return ali.total if ali else -np.inf

    def heard_in(f0, f1):
        k = bisect.bisect_left(free_at, f0)
        return [h for h in free[k:] if h[1] < f1 and (h[1] + h[2]) / 2 < f1][:60] if f1 > f0 else []

    def where(a):
        region = next((rg for rg in doc.regions if rg.start <= a <= rg.end), None)
        out = {"line": src.count("\n", 0, a) + 1}
        if region is not None and region.cue is not None:
            out["cue"] = region.cue
        return out

    def add(kind, i, j, replacement, confidence, why, evidence, frames=None):
        """Record an issue about words i..j-1 (i == j: something to insert before word i). `replacement`: the text
        that should stand there, "" to delete, None when the audio shows a problem but not the fix."""
        if i < j:
            a, b = _core(src, us[O[i]].start, us[O[j - 1]].end)
            f0, f1 = frames or (min((F[k][0] for k in range(i, j) if F[k]), default=None),
                                max((F[k][1] for k in range(i, j) if F[k]), default=None))
            if f0 is None or all(not found(k) for k in range(i, j)):
                f0, f1 = frames or inner(i, j)
            elif not frames:                               # whatever is said between their neighbours is theirs
                g0, g1 = inner(i, j)
                f0, f1 = min(f0, g0), max(f1, g1)
        else:
            a = b = us[O[i]].start if i < n else us[-1].end
            f0, f1 = frames or inner(i, i)
        text = src[a:b]
        edit = None
        if replacement is not None:
            if i == j:
                edit = [a, a, replacement + " "] if i < n else [a, a, " " + replacement]
            elif replacement == "":                        # delete the words, their punctuation and one space
                a2, b2 = us[O[i]].start, us[O[j - 1]].end
                while a2 > 0 and src[a2 - 1] in " \t":
                    a2 -= 1
                edit = [a2, b2, ""]
            else:
                if W[j - 1].endswith("'s") and not replacement.lower().endswith(("'s", "’s")):
                    replacement += src[b - 2:b] if src[b - 2:b - 1] in "'’" else "'s"
                if text[:1].isupper() and _starts_sentence(src, a):
                    replacement = replacement[:1].upper() + replacement[1:]
                edit = [a, b, replacement]
        lead, trail = src[max(0, a - 40):a], src[b:b + 40]
        issue = {"kind": kind, "at": [round(f0 * sec, 2), round(f1 * sec, 2)], **where(a),
                 "transcript": _clip(text), "suggest": replacement, "confidence": confidence, "why": why,
                 "context": _clip(f"{lead}[{text}]{trail}", 140), "evidence": evidence, "span": [a, b],
                 "edit": edit}
        issues.append(issue)
        for k in range(i, j):
            taken[k] = True
        busy[max(0, f0):max(f0, f1)] = True
        return issue

    # ---- 1. the lexicon: names written right, names written wrong, names the transcript missed
    glossary = []
    entries = [e for e in (table or {}).values() if T.words(e.key)]
    times = [(fr[0] * sec, fr[1] * sec) if fr else None for fr in F]
    ear = phones = psec = None
    if any(e.ipa for e in entries):
        from . import ipa as I
        from .lexicon import possessive
        from .phonemes import word_window
        ear = session._load("ear", timing)
        t0 = time.perf_counter()
        phones, psec = ear.timed(y, sr)
        lap("phones", t0)
    t0 = time.perf_counter()
    plan = []
    for e in entries:                                      # first claim every place the name is written right
        kw = T.words(e.key)
        m = len(kw)
        written = [i for i in range(n - m + 1) if W[i:i + m - 1] == kw[:-1] and W[i + m - 1] in (kw[-1], kw[-1] + "'s")
                   and not any(taken[i:i + m])]
        for i in written:
            taken[i:i + m] = [True] * m
            f0, f1 = inner(i, i + m)                       # its spelling may not fit its sound: the letters the
            busy[f0:f1] = True                             # ear hears there aren't "missing" from the transcript
        plan.append((e, kw, written))
    for e, kw, written in plan:
        m = len(kw)
        stat = {"word": e.key, "written": len(written), "confirmed": 0, "doubted": 0, "restored": 0}
        glossary.append(stat)
        for i in written:
            if not e.ipa:
                continue
            win = word_window(times, i, i + m, total * sec)
            if not win:
                continue
            target = possessive(e.ipa) if W[i + m - 1].endswith("'s") else e.ipa
            verdict = ear.judge(y, sr, target, win[0], win[1], word=e.key, pad=0.03)
            if verdict.ok:
                stat["confirmed"] += 1
                continue
            stat["doubted"] += 1
            add("name-mismatch", i, i + m, None, "medium" if verdict.score < 0.4 or verdict.hard > 1 else "low",
                f"the transcript writes {e.key}, but the audio there doesn't sound like its pronunciation",
                {"target": verdict.target, "heard": verdict.heard, "score": verdict.score, "diff": verdict.diff})
        # places the transcript may have written the name another way: near-spellings, and the name's sound
        sayw = T.words(e.say) if e.say and T.words(e.say) != kw else []
        targets = [t for t in (_letters(kw), _letters(sayw)) if len(t) >= 4]
        cands: dict = {}
        if targets:
            for size in sorted({m, m + 1, max(1, m - 1), len(sayw) or m}):
                for i in range(n - size + 1):
                    if any(taken[i:i + size]):
                        continue
                    s = _letters(W[i:i + size])
                    s = s[:-1] if W[i + size - 1].endswith("'s") else s
                    sim = max(_sim(s, t) for t in targets)
                    if sim >= 0.65:
                        cands[(i, i + size)] = {"sim": round(sim, 2)}
        if e.ipa and len(I.units(e.ipa)) >= 5:             # a short name sounds like too many ordinary words
            for hit in I.spot(e.ipa, [p for p, _, _ in phones], float(v.get("spot_min", 0.8))):
                h0, h1 = phones[hit["start"]][1] * psec, phones[hit["end"] - 1][2] * psec
                over = [k for k in range(n) if times[k] and (h0 - 0.02 <= (times[k][0] + times[k][1]) / 2 <= h1 + 0.02
                        or min(h1, times[k][1]) - max(h0, times[k][0]) >= 0.5 * (times[k][1] - times[k][0]))]
                if not over:                               # said where the transcript has nothing at all
                    k = next((k for k in range(n) if times[k] and times[k][0] >= h1), n)
                    if not busy[int(h0 / sec):int(h1 / sec) + 1].any():
                        stat["restored"] += 1
                        add("missing", k, k, e.key, "medium", f"the audio says {e.key} here and the transcript has "
                            "no word for it", {"target": hit["target"], "heard": hit["heard"], "score": hit["score"],
                                               "diff": hit["diff"]}, frames=(int(h0 / sec), int(h1 / sec) + 1))
                    continue
                key = (over[0], over[-1] + 1)
                if not any(taken[key[0]:key[1]]):
                    cands.setdefault(key, {}).update(hit=hit, h0=h0, h1=h1)
        for (i, j), c in sorted(cands.items(), key=lambda kv: (-kv[1].get("sim", 0), -kv[1].get("hit", {}).get(
                "score", 0), kv[0])):
            if any(taken[i:j]):
                continue
            wrote = _clip(src[slice(*_core(src, us[O[i]].start, us[O[j - 1]].end))], 60)
            if e.ipa:
                win = word_window(times, i, j, total * sec) or ((c["h0"], c["h1"]) if "hit" in c else None)
                if not win:
                    continue
                target = possessive(e.ipa) if W[j - 1].endswith("'s") else e.ipa
                verdict = ear.judge(y, sr, target, win[0], win[1], word=e.key, pad=0.03)
                if not verdict.ok:
                    continue
                weak = min(S[i:j]) < near
                level = "high" if "sim" in c else "medium" if weak else "low"
                why = (f"the audio says {e.key} (it matches the lexicon's pronunciation); the transcript wrote "
                       f"\"{wrote}\"" + ("" if "sim" in c or weak else ", which also fits the sound: check the "
                                         "meaning"))
                evidence = {"target": verdict.target, "heard": verdict.heard, "score": verdict.score,
                            "diff": verdict.diff, "spelling_similarity": c.get("sim")}
            else:                                          # no IPA: does the lexicon's respelling fit the audio
                f0, f1, left, right = bounds(i, j)         # as well as what the transcript wrote?
                a = loglik(f0, f1, left + W[i:j] + right)
                b = loglik(f0, f1, left + (sayw or kw) + right)
                if a is None or b is None or b - a < -gain_min:
                    continue
                level = "medium" if b >= a else "low"
                why = (f"\"{wrote}\" is spelled like {e.key}, and the lexicon's reading of {e.key} fits the audio "
                       f"here; add `ipa` to the entry for a check by sound")
                evidence = {"letters_gain": round(float(b - a), 1), "spelling_similarity": c.get("sim")}
            stat["restored"] += 1
            add("name", i, j, e.key, level, why, evidence)
    lap("names", t0)

    # ---- 2. a second transcript, and the letter ear to arbitrate
    opinion, rare = None, None
    if second is not None:
        if isinstance(second, Doc):
            bdoc, opinion = second, "the second transcript"
        else:
            checker = session._load("checker", timing)
            t0 = time.perf_counter()
            bdoc, opinion = parse(checker.transcribe(y, sr)), config.voice()["checker"]
            lap("transcript", t0)
        t0 = time.perf_counter()
        rare = _rarity()
        bus = units(bdoc)
        BW, BO = _words(bus)

        def btext(j1, j2, whole=False):
            """The second transcript's own writing of its words j1..j2-1. whole: for text to insert, keep a
            sentence's capital and full stop when the words are whole sentences there."""
            a, b = _core(bdoc.source, bus[BO[j1]].start, bus[BO[j2 - 1]].end)
            text = " ".join(bdoc.source[a:b].split())
            starts = _starts_sentence(bdoc.source, a)
            if whole and starts and bdoc.source[b:b + 1] in ".!?" and j2 - j1 >= 3:
                return text + bdoc.source[b]
            if starts and not _namelike(bdoc.source, a, b, rare) and text[:2] != "I ":
                text = text[:1].lower() + text[1:]         # capitalised only because it began a sentence there
            return text
        ops = difflib.SequenceMatcher(None, verify._same_sound(W), verify._same_sound(BW),
                                      autojunk=False).get_opcodes()
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                continue
            moved = True
            while moved:                                   # whole units on both sides ("0.01" against "0.001")
                moved = False
                if i1 > 0 and j1 > 0 and ((i1 < n and O[i1 - 1] == O[i1]) or (j1 < len(BW) and BO[j1 - 1] == BO[j1])):
                    i1, j1, moved = i1 - 1, j1 - 1, True
                if i2 < n and j2 < len(BW) and ((i2 > 0 and O[i2] == O[i2 - 1]) or (j2 > 0 and BO[j2] == BO[j2 - 1])):
                    i2, j2, moved = i2 + 1, j2 + 1, True
            tag = "insert" if i1 == i2 else "delete" if j1 == j2 else "replace"
            if any(taken[i1:i2]) or (i1 == i2 and i1 > 0 and i1 < n and taken[i1 - 1] and taken[i1]):
                continue
            a_units, b_units = us[O[i1]:O[i2 - 1] + 1] if i1 < i2 else [], bus[BO[j1]:BO[j2 - 1] + 1] if j1 < j2 else []
            a_ways = {_letters(verify._same_sound(ws)) for ws in _readings(a_units)}
            if a_ways & {_letters(verify._same_sound(ws)) for ws in _readings(b_units)}:
                continue                                   # the same words, written differently ("1959")
            f0, f1, left, right = bounds(i1, i2)
            la, lb = loglik(f0, f1, left + W[i1:i2] + right), loglik(f0, f1, left + BW[j1:j2] + right)
            if la is None or lb is None:
                continue
            gain = float(lb - la) if np.isfinite(lb) else -np.inf
            if not np.isfinite(la):
                gain = np.inf if np.isfinite(lb) else 0.0
            ev = {"letters_gain": round(min(gain, 999.0), 1), "second_transcript": btext(j1, j2) if j1 < j2 else "",
                  "letters_heard": " ".join(h[0] for h in heard_in(*inner(i1, i2)))}
            if i1 < i2:
                ev["word_scores"] = [[w, s] for w, s in zip(W[i1:i2], S[i1:i2])][:12]
            if tag == "insert" or i1 == i2:
                if gain >= gain_min:
                    add("missing", i1, i1, btext(j1, j2, whole=True), "high" if gain >= 2 * gain_min else "medium",
                        f"{opinion} heard \"{btext(j1, j2)}\" here and the audio has those letters", ev,
                        frames=inner(i1, i1))
                    for h in heard_in(f0, f1):             # (the word after may have been stretched over them)
                        if h[0] in BW[j1:j2]:
                            busy[h[1]:h[2]] = True
            elif tag == "delete" or j1 == j2:
                if gain >= gain_min:
                    sure = gain >= 2 * gain_min or max(S[i1:i2]) < gone
                    add("unheard", i1, i2, "", "high" if sure else "medium",
                        f"{opinion} didn't hear this, and the audio fits better without it", ev)
            elif gain >= gain_min:
                same = _sim(_letters(W[i1:i2]), _letters(BW[j1:j2]))
                ev["spelling_similarity"] = round(same, 2)
                namey = any(_namelike(src, u.start, u.end, rare) for u in a_units) or \
                    any(_namelike(bdoc.source, u.start, u.end, rare) for u in b_units)
                if namey:
                    add("spelling", i1, i2, None, "low", f"{opinion} spells this \"{btext(j1, j2)}\". A name's "
                        "spelling can't be settled by its sound: find out the right one and add it to the lexicon",
                        ev)
                else:
                    numbers = any(isinstance(u.part, tuple) for u in a_units + b_units)    # no two numbers are
                    short = i2 - i1 <= 5 and j2 - j1 <= 5                                # homophones; a long
                    tiny = len(_letters(W[i1:i2])) <= 3 and len(_letters(BW[j1:j2])) <= 3 and gain < 5 * gain_min
                    level = ("high" if min(S[i1:i2]) < near and (same < 0.6 or numbers) and short and not tiny
                             else "medium")                # swap is a muddle; a/the and in/of blur in fluent speech
                    add("misheard", i1, i2, btext(j1, j2), level, f"{opinion} heard \"{btext(j1, j2)}\", and the "
                        "audio's letters fit that better" + ("" if level == "high" else "; check the meaning before "
                                                              "taking it (they sound alike, or the stretch is long)"),
                        ev)
            elif gain > -gain_min and min(S[i1:i2]) < near:
                add("unsure", i1, i2, None, "low", f"{opinion} heard \"{btext(j1, j2)}\"; the audio doesn't clearly "
                    "favour either wording", ev)
        lap("compare", t0)

    # ---- 3. words that aren't in the audio at all
    t0 = time.perf_counter()
    rare = _rarity() if second is None else rare
    i = 0
    while i < n:
        if taken[i] or found(i):
            i += 1
            continue
        j = i
        while j < n and not taken[j] and not found(j):
            j += 1
        # Text that isn't in the audio drags its neighbours off their own sound ("...and peace. Thanks for
        # watching": the alignment gives up "peace" too). A word of the run that the free hearing has, in order,
        # between the run's neighbours was said after all; only the rest is unheard.
        f0, f1 = inner(i, j)
        hs = heard_in(f0, f1)
        kept: dict = {}
        for blk in difflib.SequenceMatcher(None, W[i:j], [h[0] for h in hs], autojunk=False).get_matching_blocks():
            for k in range(blk.size):
                kept[i + blk.a + k] = hs[blk.b + k]
        for h in kept.values():
            busy[h[1]:h[2]] = True
        k = i
        while k < j:
            if k in kept:
                k += 1
                continue
            m = k
            while m < j and m not in kept:
                m += 1
            if m - k >= 2 or len(_letters(W[k:m])) >= 4:
                g0 = max((kept[x][2] for x in kept if x < k), default=f0)
                g1 = min((kept[x][1] for x in kept if x >= m), default=f1)
                said = " ".join(h[0] for h in heard_in(g0, g1) if not busy[h[1]:h[2]].all())
                ev = {"letters_heard": said, "word_scores": [[w, sc] for w, sc in zip(W[k:m], S[k:m])][:12]}
                if len(_letters(said.split())) <= 1:
                    sure = m - k >= 3 or len(_letters(W[k:m])) >= 12
                    add("unheard", k, m, "", "high" if sure else "medium", "no speech in the audio matches these "
                        "words: there is nothing said in their place", ev, frames=(g0, max(g0, g1)))
                elif m - k == 1 and (_namelike(src, us[O[k]].start, us[O[k]].end, rare) or (rare and rare(W[k], 3))):
                    add("spelling", k, m, None, "low", f"the letter ear hears \"{_clip(said, 60)}\" where the "
                        f"transcript has this name or rare word. Letters say little about a name: put it in the "
                        "lexicon with its pronunciation to have it checked by sound", ev, frames=(g0, max(g0, g1)))
                else:
                    add("unheard", k, m, None, "medium" if m - k >= 2 else "low", "the audio doesn't say these "
                        f"words; in their place the letter ear hears \"{_clip(said, 80)}\"", ev,
                        frames=(g0, max(g0, g1)))
            k = m
        i = j
    # ---- 4. a word the letter ear hears as something else, whatever the transcripts wrote
    # With no script the letter model writes what it hears; when that differs from the transcript's word, both are
    # aligned to the same stretch and the difference in fit is the evidence. On LibriSpeech (60 utterances, one
    # substituted word each) every substitution gained 11.6 or more; of the correct words the model heard
    # differently, all but one rare name gained under 4.2. The word's own score separates the two far worse.
    heard_gain = float(v.get("proof_heard_gain", 10.0))
    for i in range(n):
        if taken[i] or not found(i) or len(W[i]) < 2:
            continue
        k0 = max(0, bisect.bisect_left(free_at, F[i][0]) - 1)
        hs = [h for h in free[k0:k0 + 6] if h[1] < F[i][1] and h[2] > F[i][0] and not busy[h[1]:h[2]].all()]
        said = [h[0] for h in hs]
        around = {_letters(W[x:y]) for x in range(max(0, i - 2), i + 1) for y in range(i + 1, min(n, i + 3) + 1)}
        if not said or _letters(said) in around:           # "passage way" heard as "passageway" is no error
            continue
        if W[i] in said:                                   # the word is there, and so is something the transcript
            at = said.index(W[i])                          # skips, which the alignment stretched this word over
            f0, f1, left, right = bounds(i, i + 1)
            la = loglik(f0, f1, left + [W[i]] + right)
            for extra, k in ((hs[:at], i), (hs[at + 1:], i + 1)):
                words = [h[0] for h in extra if h[0] not in _FILLERS]
                if not words or la is None or not np.isfinite(la):
                    continue
                lb = loglik(f0, f1, left + (words + [W[i]] if k == i else [W[i]] + words) + right)
                if lb is None or not np.isfinite(lb) or lb - la < heard_gain:
                    continue
                add("missing", k, k, " ".join(words), "medium" if lb - la >= 2.5 * heard_gain else "low",
                    f"the letter ear hears \"{' '.join(words)}\" here (spelled by ear), which the transcript skips",
                    {"letters_heard": " ".join(words), "letters_gain": round(float(lb - la), 1)},
                    frames=(extra[0][1], extra[-1][2]))
            continue
        f0, f1, left, right = bounds(i, i + 1)
        la, lb = loglik(f0, f1, left + [W[i]] + right), loglik(f0, f1, left + said + right)
        if la is None or lb is None or not np.isfinite(la) or not np.isfinite(lb) or lb - la < heard_gain:
            continue
        gain = float(lb - la)
        a0, b0 = us[O[i]].start, us[O[i]].end
        name = _namelike(src, a0, b0, rare) or bool(rare and rare(W[i], 3))
        ev = {"letters_heard": " ".join(said), "letters_gain": round(gain, 1), "word_scores": [[W[i], S[i]]]}
        if name:
            add("spelling", i, i + 1, None, "low", f"the letter ear hears \"{' '.join(said)}\" here. For a name "
                "that says little (a name's spelling doesn't follow its sound): put it in the lexicon with its "
                "pronunciation to have it checked by sound", ev)
        elif gain >= 2.5 * heard_gain and isinstance(us[O[i]].part, str) and O[i] not in (O[i - 1] if i else -1,
                                                                                           O[i + 1] if i + 1 < n else -1):
            add("misheard", i, i + 1, " ".join(said), "medium", f"the letter ear clearly hears \"{' '.join(said)}\" "
                f"here, not \"{W[i]}\"; its spelling is by ear, so check the word before taking it", ev)
        else:
            add("unsure", i, i + 1, None, "low", f"the letter ear hears \"{' '.join(said)}\" here (spelled by ear), "
                f"not \"{W[i]}\"; listen if this word matters", ev)
    # ---- 5. speech the transcript skips
    cover = busy.copy()
    for k in range(n):
        if found(k):
            cover[F[k][0]:F[k][1]] = True
    group: list = []

    def flush():
        if not group:
            return
        said = [h[0] for h in group]
        clear = min(h[3] for h in group) >= 0.8           # a short word counts only when every letter is clear
        if len(_letters(said)) >= 4 or (clear and len(_letters(said)) >= 2):
            f0, f1 = group[0][1], group[-1][2]
            k = next((k for k in range(n) if found(k) and F[k][0] >= f1 - 1), n)
            add("missing", k, k, " ".join(said) if clear else None, "medium" if len(_letters(said)) >= 8 else "low",
                f"the audio has speech here that the transcript skips; the letter ear hears \"{_clip(' '.join(said), 80)}\""
                " (spelled by ear)", {"letters_heard": " ".join(said)}, frames=(f0, f1))
        group.clear()
    for h in free:
        word, a, b, p, _ = h
        if len(word) >= 2 and p >= 0.5 and word not in _FILLERS and cover[a:b].mean() < 0.3:
            if group and a - group[-1][2] > 50:
                flush()
            group.append(h)
        elif cover[a:b].mean() >= 0.3:
            flush()
    flush()
    lap("gaps", t0)

    issues.sort(key=lambda x: (x["at"][0], x["line"]))
    note = r["note"]
    if unrelated:                                          # nothing here is worth applying: say so once, plainly
        for x in issues:
            x["confidence"] = "low"
        note = (f"only {agreement:.0%} of the transcript's words are heard in this audio: it looks like a transcript "
                "of a different recording (or another language); the issues below are not worth applying")
    counts: dict = {}
    for x in issues:
        counts[x["kind"]] = counts.get(x["kind"], 0) + 1
    timing["total"] = round(time.perf_counter() - began, 2)
    timing["audio_seconds"] = round(total * sec, 2)
    return {"clean": not issues, "issues": issues, "counts": counts, "words": n, "format": doc.kind,
            "second_opinion": opinion, "lexicon": glossary, "aligned": r["measured"] and not unrelated,
            "agreement": round(agreement, 3), "note": note,
            "weak_words": sum(1 for k in range(n) if found(k) and S[k] < near), "timing": timing}


def edits(report: dict, level: str = "high") -> list:
    """The edits of the issues at or above a confidence level ("high", "medium" or "all")."""
    floor = 0 if level == "all" else LEVELS[level]
    return [x["edit"] for x in report["issues"] if x.get("edit") and LEVELS[x["confidence"]] >= floor]
