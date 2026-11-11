"""Two ears on one take. Whisper (which has a language model) writes what it hears, and the words are compared with
the script. wav2vec2 (which has none) scores every scripted word by the letters it can actually hear, so it catches
the slips Whisper "corrects". Pace is syllables per second of speech; tone is pitch around the sentence's own drift.

Public names (also used by OpenMontage's narrate.py): Checker, Listener, match, slips, pace, syllables, prosody,
emotional, norm_words, PROSODY_V."""
from __future__ import annotations

import difflib
import sys

import numpy as np

from . import config, ctc
from . import text as T


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)          # stdout is for results (check prints JSON there)


def _guard_sklearn() -> None:
    """Windows App Control blocked one of scikit-learn's DLLs (2026-11-03: sklearn/utils/_heap, on every retry).
    transformers imports sklearn only as an option (generation's candidate_generator), but a blocked DLL there kills
    every Whisper/wav2vec2 load. If sklearn won't import, tell transformers it isn't installed. Called before the
    models load, not on import, so the text functions stay light."""
    try:
        import sklearn.metrics  # noqa: F401
    except (ImportError, OSError):
        import transformers.utils.import_utils as iu
        if hasattr(iu, "_sklearn_available"):
            iu._sklearn_available = False


def _local_first(load, *args, **kw):
    """`load(*args, **kw)` from the local Hugging Face cache, without asking the Hub whether the model changed (that
    check costs several seconds per model on every start, and fails offline). Only a model that isn't cached yet is
    downloaded."""
    try:
        return load(*args, local_files_only=True, **kw)
    except (OSError, ValueError):                          # not in the cache (or only partly): fetch it
        return load(*args, **kw)


def _speedup(model, role: str):
    """int8 dynamic quantisation of the model's Linear layers (CPU) when the `quantize` setting asks for it: true for
    all three models, or a list of roles ("checker", "listener", "phonemes")."""
    q = config.voice().get("quantize")
    if not (q is True or (isinstance(q, (list, tuple, set)) and role in q)):
        return model
    import warnings

    import torch
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                    # torch.ao.quantization is moving to torchao
        return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def norm_words(text: str) -> list[str]:
    """The script's words, numbers in their default reading (see voicecheck/text.py)."""
    return T.words(text)


# Words that sound the same: Whisper may write either, and neither is a slip (DL-12: "one pass per weight" was heard
# as "wait" on the good takes, so the checker kept a take that really said "wave"). Sets with a number word (to/two,
# for/four, won/one) are left out on purpose: in "zero [to] two hundred" they would hide the dropped word.
_HOMOPHONE_SETS = [("weight", "wait"), ("weights", "waits"), ("there", "their", "they're"), ("its", "it's"),
                   ("right", "write"), ("knew", "new"), ("whole", "hole"), ("here", "hear"), ("weak", "week"),
                   ("piece", "peace"), ("plain", "plane"), ("principal", "principle")]
_HOMOPHONES = {w: s[0] for s in _HOMOPHONE_SETS for w in s}


def _spellings() -> dict:
    """British spelling -> American, for the words Whisper may write either way whatever the voice's accent (a British
    script's "colour" comes back "color"). Built from stems and endings, so only real pairs fold: colour/color but
    not hour or four, centre/center but not more or are, realise/realize but not rise or wise."""
    out: dict = {}

    def pairs(stems: str, uk: str, us: str, ends: str = "", bare: bool = True) -> None:
        for stem in stems.split():
            for end in ([""] if bare else []) + ends.split():
                out[stem + uk + end] = stem + us + end
    pairs("arb arm behavi cand clam col endeav fav ferv flav glam harb hon hum lab neighb od parl ranc rig rum sav "
          "savi splend tum val vap vig", "our", "or",
          "s ed ing ite ites able ably al ally ful fully less er ers hood hoods ly y")
    for stem in ("cent fib theat calib sab somb spect lust meag lit millilit met kilomet centimet millimet nanomet "
                 "micromet").split():
        out.update({stem + "re": stem + "er", stem + "res": stem + "ers", stem + "red": stem + "ered",
                    stem + "ring": stem + "ering"})
    pairs("anonym apolog author capital categor central character civil colon computer critic custom digit emphas "
          "equal factor final general harmon initial item legal local maxim memor minim mobil modern monet neutral "
          "normal optim organ parallel penal personal popular priorit quant random rational real recogn regular "
          "serial special stabil standard summar symbol sympath synchron theor token util vector visual",
          "is", "iz", "e es ed ing er ers ation ations able", bare=False)          # not "emphasis"
    pairs("anal paral catal", "ys", "yz", "e ed ing er ers", bare=False)           # not "analyses", a noun too
    pairs("cancel channel counsel dial equal fuel funnel initial jewel label level libel marvel model panel pencil "
          "quarrel rival shovel signal spiral total travel tunnel", "l", "", "ed ing er ers", bare=False)
    pairs("catalog dialog analog", "ue", "", "s")
    pairs("defen offen licen preten", "ce", "se", "s")
    for line in ("okay=ok grey=gray greys=grays greyscale=grayscale tyre=tire tyres=tires plough=plow ploughs=plows "
                 "ploughed=plowed ploughing=plowing draught=draft draughts=drafts cheque=check cheques=checks "
                 "kerb=curb kerbs=curbs storey=story storeys=stories mould=mold moulds=molds moulded=molded "
                 "moulding=molding smoulder=smolder smouldering=smoldering cosy=cozy sceptic=skeptic "
                 "sceptics=skeptics sceptical=skeptical scepticism=skepticism judgement=judgment "
                 "judgements=judgments acknowledgement=acknowledgment acknowledgements=acknowledgments ageing=aging "
                 "programme=program programmes=programs pyjamas=pajamas moustache=mustache whisky=whiskey "
                 "doughnut=donut doughnuts=donuts yoghurt=yogurt yoghurts=yogurts sulphur=sulfur artefact=artifact "
                 "artefacts=artifacts enquiry=inquiry enquiries=inquiries enquire=inquire enquired=inquired "
                 "disc=disk discs=disks gaol=jail catalogued=cataloged practise=practice practises=practices "
                 "practised=practiced practising=practicing marvellous=marvelous counsellor=counselor "
                 "counsellors=counselors jewellery=jewelry woollen=woolen fulfil=fulfill fulfils=fulfills "
                 "fulfilment=fulfillment enrol=enroll enrols=enrolls enrolment=enrollment instalment=installment "
                 "instalments=installments skilful=skillful wilful=willful distil=distill instil=instill "
                 "anaemia=anemia anaemic=anemic anaesthesia=anesthesia anaesthetic=anesthetic "
                 "anaesthetics=anesthetics paediatric=pediatric paediatrics=pediatrics "
                 "paediatrician=pediatrician encyclopaedia=encyclopedia haemoglobin=hemoglobin "
                 "haemorrhage=hemorrhage leukaemia=leukemia orthopaedic=orthopedic oestrogen=estrogen foetus=fetus "
                 "foetal=fetal diarrhoea=diarrhea manoeuvre=maneuver manoeuvres=maneuvers manoeuvred=maneuvered "
                 "manoeuvring=maneuvering oesophagus=esophagus mediaeval=medieval aeon=eon aeons=eons").split():
        uk, us = line.split("=")
        out[uk] = us
    return out


_SPELLINGS = _spellings()


def _one_sound(w: str) -> str:
    if w.endswith("'s"):                                   # the neighbour's -> the neighbor's
        return _one_sound(w[:-2]) + "'s"
    w = _SPELLINGS.get(w, w)
    return _HOMOPHONES.get(w, w)


def _same_sound(words: list[str]) -> list[str]:
    """Each word in the one spelling that stands for its sound: homophones (weight/wait) and British/American
    spellings (colour/color, okay/ok) compare equal."""
    return [_one_sound(w) for w in words]


def _letters(words: list[str]) -> str:
    return "".join(_same_sound(words))


def _ratio(a: list[str], b: list[str]) -> float:
    return difflib.SequenceMatcher(None, _letters(a), _letters(b), autojunk=False).ratio()


def _fit(text: str, target: list[str]) -> list[str]:
    """The words of `text`, each number in the reading closest to `target` (Whisper's "100" against a script's
    "a hundred", a script's "1959" against a transcript's "nineteen fifty-nine")."""
    ps = T.parts(text)
    words, owner = T.expand(ps)
    options = {i for i, p in enumerate(ps) if isinstance(p, tuple) and len(p) > 1}
    if not options or _same_sound(words) == _same_sound(target):
        return words
    # Scoring the whole text for every reading of every number took seconds on a long script (36 s for 220 words
    # with 70 numbers, 2026-11-11). Words that already agree pin the text to the target, so each number is chosen
    # against the stretch of target between its nearest agreeing neighbours.
    to: list = [None] * len(words)                         # text word -> target word, where the two agree
    for a, b, n in difflib.SequenceMatcher(None, _same_sound(words), _same_sound(target),
                                           autojunk=False).get_matching_blocks():
        to[a:a + n] = range(b, b + n)
    begin, n = [], 0                                       # where each part's words start
    for p in ps:
        begin.append(n)
        n += len(T.norm(p[0] if isinstance(p, tuple) else p))
    begin.append(n)

    def settled(i: int) -> bool:
        """The default reading stands exactly where the target has it, with nothing extra beside it."""
        s, e = begin[i], begin[i + 1]
        before = to[s - 1] if s else -1
        after = to[e] if e < len(words) else len(target)
        return (before is not None and after == before + 1 + e - s
                and to[s:e] == list(range(before + 1, after)))
    open_ = [i for i in sorted(options) if not settled(i)]
    fixed = [t is not None and owner[k] not in set(open_) for k, t in enumerate(to)]
    picks: dict = {}
    k = 0
    while k < len(open_):
        first = last = open_[k]
        right = next((j for j in range(begin[last + 1], len(words)) if fixed[j]), None)
        while k + 1 < len(open_) and (right is None or begin[open_[k + 1]] <= right):    # no agreed word between them
            k += 1
            last = open_[k]
            right = next((j for j in range(begin[last + 1], len(words)) if fixed[j]), None)
        k += 1
        left = next((j for j in range(begin[first] - 1, -1, -1) if fixed[j]), None)
        lo, hi = left or 0, len(words) if right is None else right + 1
        t0, t1 = 0 if left is None else to[left], len(target) if right is None else to[right] + 1
        region = [" ".join(words[lo:begin[first]]), *ps[first:last + 1], " ".join(words[begin[last + 1]:hi])]
        there = _letters(target[t0:t1])
        # the whole text's ratio, 2 * matched / letters, with the rest of the text counted as it stands
        agreed = sum(len(w) for k, w in enumerate(_same_sound(words)) if to[k] is not None and not lo <= k < hi)
        rest = len(_letters(words[:lo] + words[hi:])) + len(_letters(target[:t0] + target[t1:]))

        def score(ws, there=there, agreed=agreed, rest=rest):
            here = _letters(ws)
            same = sum(n for _, _, n in difflib.SequenceMatcher(None, here, there, autojunk=False)
                       .get_matching_blocks())
            return 2 * (agreed + same) / max(1, rest + len(here) + len(there))
        for j, pick in T.choose(region, score).items():
            picks[first + j - 1] = pick
    return T.expand(ps, picks)[0]


def _find(words: list[str], phrase: str, nth: int | None = None) -> set[int]:
    """Indices of the words of `phrase` in `words`: every place it stands, or only its nth (0-based) place."""
    w = T.words(phrase)
    hits = [i for i in range(len(words) - len(w) + 1) if w and words[i:i + len(w)] == w]
    return {k for i in (hits if nth is None else hits[nth:nth + 1]) for k in range(i, i + len(w))}


def _hidden(words: list[str], text: str, ignore, shown: bool) -> set[int]:
    """The words of `text` that `ignore` names. A bare phrase hides every place it stands; a lexicon Term hides only
    its own place, so a name respelled "Win", or one written like a word (Said), leaves the ordinary win and said to
    be compared (DL: "We lose when Win wins" passed for "We win when Win wins")."""
    hide: set = set()
    for p in ignore:
        if isinstance(p, str):
            hide |= _find(words, p)
            continue
        phrase, nth = (p.shown, None) if shown else (p.said, p.occurrence)
        if shown and p.at >= 0 and text[p.at:p.at + len(p.shown)] == p.shown:    # how often it is written before
            w = T.words(phrase)
            before = T.words(text[:p.at + len(p.shown)])
            nth = max(0, sum(before[i:i + len(w)] == w for i in range(len(before) - len(w) + 1)) - 1) if w else 0
        hide |= _find(words, phrase, nth)
    return hide


def _mask(a: list[str], b: list[str], hide: set[int]) -> tuple[list[str], list[str]]:
    """Drop the script words in `hide` and whatever the transcript wrote in their place."""
    a2, b2 = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, _same_sound(a), _same_sound(b),
                                                      autojunk=False).get_opcodes():
        keep_a = [a[i] for i in range(i1, i2) if i not in hide]
        a2 += keep_a
        if op == "equal":
            b2 += [b[j] for j, i in zip(range(j1, j2), range(i1, i2)) if i not in hide]
        elif not any(i in hide for i in range(i1, i2)):
            b2 += b[j1:j2]
        else:                                  # a block mixing a hidden name with other words ("Andrej Karpathy"
            used = set()                       # -> "Andre Carpathi"): keep what Whisper wrote for the other words
            for w in keep_a:
                best = max(((difflib.SequenceMatcher(None, w, b[j]).ratio(), j) for j in range(j1, j2)
                            if j not in used), default=(0, None))
                if best[1] is not None and best[0] >= 0.5:
                    used.add(best[1])
            b2 += [b[j] for j in range(j1, j2) if j in used]
    return a2, b2


def match(expected: str, heard: str, alt: str | None = None, ignore=()) -> tuple[float, str]:
    """Similarity 0..1 of script and transcript, and a short diff for the log. `alt` (the on-screen wording) is tried
    too, so a word spelled phonetically for the voice (rel-you) still matches what Whisper writes (ReLU).
    `ignore`: words judged by another ear (names checked by sound): Whisper can't spell a name it has never seen, so
    they and whatever Whisper wrote in their place are left out of the comparison. Give the lexicon's Terms (from
    `render`) to leave out only the places those terms stand; a bare phrase leaves out every place it is written."""
    if alt:
        return max((_match(expected, heard, ignore, False), _match(alt, heard, ignore, True)), key=lambda r: r[0])
    return _match(expected, heard, ignore, False)


def _match(expected: str, heard: str, ignore, shown: bool) -> tuple[float, str]:
    b = _fit(heard, T.words(expected))
    a = _fit(expected, b)
    b = _fit(heard, a)
    hide = _hidden(a, expected, ignore, shown) if ignore else set()
    if hide:
        a, b = _mask(a, b, hide)
    sa, sb = _same_sound(a), _same_sound(b)
    diff = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, sa, sb, autojunk=False).get_opcodes():
        if op != "equal":
            diff.append(f"{' '.join(a[i1:i2]) or '-'} -> {' '.join(b[j1:j2]) or '-'}")
    # Score on letters, not words: "ex or" vs "XOR" or "give" vs "gives" (Whisper tidying) cost a letter or two,
    # while a real slip ("Ralu", "edit bias", "Titsian") changes several and still fails.
    return _ratio(a, b), "; ".join(diff)


# Syllable counting by spelling: vowel groups, minus the silent e of make / makes / asked / something, plus the two
# vowels that are two syllables (radio, idea, science, actual, being). An estimate (pace is syllables per second).
_SYLLABLES = {"every": 2, "everything": 3, "everyone": 3, "everybody": 4, "business": 2, "businesses": 3,
              "evening": 2, "different": 2, "difference": 2, "interesting": 3, "interest": 2, "several": 2,
              "camera": 2, "chocolate": 2, "average": 2, "favorite": 2, "favourite": 2, "basically": 3,
              "vegetable": 3, "comfortable": 3, "temperature": 3, "element": 3, "elements": 3, "w": 3,
              "maybe": 2, "recipe": 3, "recipes": 3, "cafe": 2, "acne": 2, "karate": 3, "simile": 3, "apostrophe": 4,
              "catastrophe": 4, "epitome": 4, "hyperbole": 4, "adobe": 3, "finale": 3, "coyote": 3, "sesame": 3,
              "lion": 2, "lions": 2, "quiet": 2, "quietly": 3, "rhythm": 2, "rhythms": 2}
_SILENT_E_INSIDE = r"(?<=[aeiouy][^aeiouy])e(?=(?:ly|ment|ments|ness|ful|fully|less|thing|things|times?|where|one" \
                   r"|body|fore|work|lines?|ty|teen)$)"
_TWO_VOWELS = (r"(?<![ctsxgnl])i[ao]|iu|(?<!g)eo(?![up])|geo(?!ns?$|rg)|(?<![ct])ie(?=n[ct])|scie|(?<![qg])u[ao]"
               r"|oe(?=[mt])|crea(?=t)|reac|(?<=[aeou])ing$|ism$|[dst]n't$")


def _word_syllables(w: str) -> int:
    import re
    if w in _SYLLABLES:
        return _SYLLABLES[w]
    v = re.sub(_SILENT_E_INSIDE, "", w)                    # some(e)thing, nin(e)ty
    extra = len(re.findall(_TWO_VOWELS, v))
    v = re.sub(r"(?<=[aeiou])y(?=[aeiou])|^y(?=[aeiou])|(?<=[^aeiou])y(?=[aeou])", "j", v)   # beyond, yes, canyon
    v = re.sub(r"y(?=ing$)", "y ", v)                      # trying: two syllables
    k = len(re.findall(r"[aeiouy]+", v))
    if k > 1:
        if re.search(r"ea$", v):
            extra += 1                                     # idea, area
        if re.search(r"[^aeiou]e$", v) and not re.search(r"[^aeiouy]le$", v):
            k -= 1                                         # make, whole; not table
        elif re.search(r"[^aeiousxzcgh]es$", v) and not re.search(r"[^aeiouy]les$", v):
            k -= 1                                         # makes, names; not boxes, pages, tables
        elif re.search(r"[^aeiouytd]ed$", v) and not re.search(r"[^aeiouy]red$|[^aeiouyl]led$", v):
            k -= 1                                         # asked, used, called; not wanted, hundred, settled
    return max(1, k + extra)


def syllables(text: str) -> int:
    import re
    n = sum(_word_syllables(w) for w in norm_words(text))
    for w in re.findall(r"\b[A-Z]{2,}\b", text):           # GPU, FBI: spelled out, a syllable a letter (NASA is a word)
        if len(w) <= 3 or not re.search("[AEIOUY]", w):
            n += sum(3 if c == "W" else 1 for c in w) - _word_syllables(w.lower())
    return n


def pace(text: str, seconds: float) -> float:
    """Syllables per second. Pass the seconds of speech (`speech_seconds`), not the file length: lead-in and tail
    silence would make every take look slow."""
    return syllables(text) / max(seconds, 1e-3)


def speech_seconds(spans, fallback: float) -> float:
    """From the first scripted word's start to the last one's end (the listener's spans), else `fallback`."""
    return (spans[-1][2] - spans[0][1]) if spans else fallback


def _respelling_tails(words: list[str]) -> set[int]:
    """Indices of the 2nd.. pieces of each `pronunciations` respelling found in `words`: the "you" of rel-you is
    spoken "yu", so the letter listener failed it on every take (0.20-0.29, 2026-11-04). Whisper still judges it."""
    from .lexicon import respellings
    tails = set()
    for said in respellings():
        w = T.words(said)
        for i in range(len(words) - len(w) + 1):
            if len(w) > 1 and words[i:i + len(w)] == w:
                tails.update(range(i + 1, i + len(w)))
    return tails


_NUMBER_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                    "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety "
                    "hundred thousand million billion trillion".split())


def slips(expected: str, heard: str, scores, near=0.65, gone=0.25, alt: str | None = None,
          skip: set | None = None) -> list[str]:
    """Words the voice got wrong. Whisper's language model writes what it expects to hear, so a letter-level
    score barely moves for one wrong short word ("zero for cap", "being sure the wrong answer" with "of"
    dropped, DL-08). A word is a slip when Whisper's transcript differs there AND the acoustic score is under
    `near`, or when the acoustic score alone is under `gone` (the word is not in the audio), or when it is a number
    word and the transcript has another number in its place.

    `scores` are the listener's [(word, score)] for `expected`; their words are the reading the listener heard
    (e.g. "nineteen fifty nine" for 1959), and the transcript is compared against those. `alt` is the on-screen
    wording: when Whisper heard exactly that, a respelled word's pieces ("million-th" for millionth, DL-19) are not
    slips unless they are missing from the audio. `skip`: word indices judged elsewhere (the phoneme check), never
    by their letters. The tail of a `pronunciations` respelling is never scored by its letters either."""
    if not scores:
        return []
    a = [w for w, _ in scores]
    b = _fit(heard or "", a)
    skip = set(skip or ()) | _respelling_tails(a)
    touched, swapped = set(), set()
    sa = _same_sound(a)
    if not (alt and _same_sound(_fit(alt, b)) == _same_sound(b)):
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, sa, _same_sound(b), autojunk=False).get_opcodes():
            if op != "equal":
                touched.update(range(i1, i2))
            # A number heard as another number ("fifteen" as "fifty") is a slip however well its letters fit the
            # audio: half of them do, and one wrong number in a long line barely moves the transcript match.
            if op == "replace" and {w for w in b[j1:j2] if w in _NUMBER_WORDS} - set(a[i1:i2]):
                swapped.update(i for i in range(i1, i2) if a[i] in _NUMBER_WORDS and a[i] not in b[j1:j2])
    for i in sorted(touched):                 # in "the the" a diff may blame either copy: a run of one word is
        for step in (-1, 1):                  # touched as a whole, so the copy the audio lacks is the one flagged
            j = i + step
            while 0 <= j < len(sa) and sa[j] == sa[i]:
                touched.add(j)
                j += step
    return [f"{w} ({s:.2f})" for i, (w, s) in enumerate(scores)
            if i not in skip and ((i in touched and s < near) or s < gone or i in swapped)]


PROSODY_V = 4          # bump when prosody() or the listener changes: cached takes are measured again


def _pitch_range(y: np.ndarray, sr: int) -> tuple[float, float]:
    """The pitch tracker's search range. "auto": from 60 Hz to 2.5x the voice's median pitch (a stressed word can
    rise ~16 st), never below 320 Hz, so low voices are measured exactly as before and high voices aren't clipped."""
    lim = config.voice().get("pitch_hz") or "auto"
    if lim != "auto":
        return float(lim[0]), float(lim[1])
    import librosa
    f0 = librosa.yin(y, fmin=60, fmax=700, sr=sr, frame_length=2048)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    n = min(len(f0), len(rms))
    loud = f0[:n][rms[:n] > np.median(rms[:n])]
    med = float(np.median(loud)) if len(loud) else 150.0
    return 60.0, float(min(1000.0, max(320.0, 2.5 * med)))


def prosody(y: np.ndarray, sr: int, spans) -> dict | None:
    """How 'emotional' a take sounds, in numbers (2026-10-27: the user heard emotion and stressed words).
    Pitch is measured against the sentence's own downward drift (a linear trend), so long sentences aren't
    flagged for declining. range: the pitch span around that trend (5th-95th percentile, semitones);
    peak_st / peak_word: how far the most raised word rises above the median word, and which word; peak_db /
    loud_word: the same for loudness (often a different word); tail: the final pitch movement (a sung question
    or an excited lift). spans: [(word, t0, t1)]."""
    import librosa
    hop = 256
    y = np.asarray(y, dtype=np.float32)
    fmin, fmax = _pitch_range(y, sr)
    f0, vf, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=1024, hop_length=hop)
    t = librosa.times_like(f0, sr=sr, hop_length=hop)
    ok = vf & ~np.isnan(f0)
    if ok.sum() < 5 or not spans:
        return None
    st = np.full_like(f0, np.nan)
    st[ok] = 12 * np.log2(f0[ok] / np.median(f0[ok]))
    if ok.sum() >= 8:                     # remove the sentence's natural downward drift: stress is a local rise
        k, c = np.polyfit(t[ok], st[ok], 1)
        st[ok] = st[ok] - (k * t[ok] + c)
    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0] + 1e-6)
    db = db[:len(t)]
    wst, wdb = [], []
    for _, a, b in spans:
        m = (t >= a) & (t < max(b, a + 0.05))
        v = m & ok
        wst.append(np.percentile(st[v], 90) if v.sum() >= 2 else np.nan)
        wdb.append(np.percentile(db[m], 90) if m.sum() >= 2 else np.nan)
    wst, wdb = np.array(wst), np.array(wdb)
    if np.all(np.isnan(wst)):
        return None
    k = int(np.nanargmax(wst))
    tail = st[ok][-max(3, int(ok.sum() * 0.15)):]
    out = {"range": round(float(np.percentile(st[ok], 95) - np.percentile(st[ok], 5)), 2),
           "peak_st": round(float(np.nanmax(wst) - np.nanmedian(wst)), 2), "peak_word": spans[k][0],
           "peak_db": None, "loud_word": None, "tail": round(float(np.mean(tail)), 2)}
    if not np.all(np.isnan(wdb)):
        j = int(np.nanargmax(wdb))
        out["peak_db"] = round(float(np.nanmax(wdb) - np.nanmedian(wdb)), 2)
        out["loud_word"] = spans[j][0]
    return out


def emotional(p: dict | None, limits: dict) -> list[str]:
    """Which emotion limits a take breaks (voice.yaml `flat`): [] when it is flat enough or unmeasurable."""
    if not p or not limits:
        return []
    out = []
    if p["range"] > limits.get("range_st", 99):
        out.append(f"pitch range {p['range']:.1f} st (limit {limits['range_st']})")
    if p["peak_st"] > limits.get("peak_st", 99):
        out.append(f"stressed '{p['peak_word']}' +{p['peak_st']:.1f} st (limit {limits['peak_st']})")
    if abs(p["tail"]) > limits.get("tail_st", 99):
        out.append(f"ending {'rises' if p['tail'] > 0 else 'drops'} {abs(p['tail']):.1f} st "
                   f"(limit {limits['tail_st']})")
    return out


def _to16k(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    y = np.asarray(y, dtype=np.float32)
    return y if sr == 16000 else librosa.resample(y, orig_sr=sr, target_sr=16000)


WINDOW_S, CONTEXT_S = 30.0, 5.0     # a long take is heard in windows, each with some audio either side for context
_HOP = 320                          # wav2vec2 makes one frame per 320 samples (20 ms at 16 kHz)
LONG_FRAMES = 2000                  # a take over 40 s is aligned in sections (ctc.sections)
MAX_CELLS = 300_000_000             # the largest alignment table worth building (bytes)


def ctc_log_probs(model, prep, y16: np.ndarray) -> np.ndarray:
    """[T, C] log-posteriors of a wav2vec2 CTC model for 16 kHz audio. A take of up to WINDOW_S + 2 x CONTEXT_S
    seconds goes through whole. A longer one is heard in windows of WINDOW_S, each with CONTEXT_S of audio either
    side whose frames are thrown away, and the windows' frames are joined: the model's memory grows with the square
    of the length, so a five-minute take heard whole needs more than 10 GB."""
    import torch

    def run(seg):
        x = prep(seg, sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            return torch.log_softmax(model(x).logits[0].float(), -1).numpy()
    n, win, ctx = len(y16), int(WINDOW_S * 16000), int(CONTEXT_S * 16000)
    if n <= win + 2 * ctx:
        return run(y16)
    out = []
    for start in range(0, n, win):
        a, b = max(0, start - ctx), min(n, start + win + ctx)
        if b - a < 400:                                    # a tail shorter than one frame
            break
        lp = run(y16[a:b])
        first = (start - a) // _HOP
        out.append(lp[first:first + win // _HOP] if start + win < n else lp[first:])
    return np.concatenate(out)


def _body(frames: list[int], gap: int = 25) -> tuple[int, int] | None:
    """Where a word is: the first and end frame of its letters, ignoring strays. When the audio holds speech the
    text skips, the alignment can park one of a word's letters half a second or more away, on a look-alike sound;
    the word is where most of its letter frames are."""
    if not frames:
        return None
    runs, start = [], 0
    for k in range(1, len(frames) + 1):
        if k == len(frames) or frames[k] - frames[k - 1] > gap:
            runs.append((start, k))
            start = k
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return frames[a], frames[b - 1] + 1


class Listener:
    """A second ear with no language model: wav2vec2 character CTC (CPU). The scripted words are force-aligned
    to its letter posteriors, and each word's score is the mean posterior of its aligned frames: low means the
    word is garbled or missing, whatever Whisper wrote. Each number in the script is read the way that best fits
    the audio ("1959" as "nineteen fifty nine" or "one thousand nine hundred fifty nine").

    After word_scores(): `spans` [(word, t0, t1)], `words` (the reading heard), `measured` (False when the audio
    is too short to hold the script; then every word scores 0.0, so the take fails as missing words) and `note`
    (why a stretch of a long take could not be aligned, if one could not)."""

    def __init__(self, log=_log):
        _guard_sklearn()
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        name = config.voice()["listener"]
        log(f"loading listener {name} (CPU)")
        self.proc = _local_first(Wav2Vec2Processor.from_pretrained, name)
        self.model = _speedup(_local_first(Wav2Vec2ForCTC.from_pretrained, name, dtype=torch.float32).eval(),
                              "listener")
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.vocab["<pad>"]
        self.spans: list = []
        self.words: list[str] = []
        self.measured = True
        self.note: str | None = None

    def log_probs(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
        """[T, C] letter log-posteriors of the take, and seconds per frame (~0.02)."""
        y16 = _to16k(y, sr)
        lp = ctc_log_probs(self.model, self.proc, y16)
        return lp, len(y16) / 16000 / max(1, lp.shape[0])

    def hear(self, lp: np.ndarray) -> list[tuple[str, int, int, float, int | None]]:
        """What the letter model hears with no script: [(word, first frame, end frame, mean probability, the frame
        where the word-boundary mark before it starts, or None)]. Its spelling is rough (no language model), which
        is the point: nothing is "corrected"."""
        chars = {i: ch for ch, i in self.vocab.items()}
        out, cur, sep = [], None, None
        for k, a, b, p in ctc.greedy(lp, self.blank):
            ch = chars.get(k, "")
            if len(ch) == 1 and (ch.isalpha() or ch == "'"):
                if cur is None:
                    cur = [ch, a, b, [p], sep]
                    sep = None
                else:
                    cur[0] += ch
                    cur[2] = b
                    cur[3].append(p)
            else:
                if cur is not None:
                    out.append(cur)
                    cur = None
                if ch == "|" and sep is None:
                    sep = a
        if cur is not None:
            out.append(cur)
        out = [(w.lower().strip("'"), a, b, float(np.mean(ps)), sp) for w, a, b, ps, sp in out]
        return [x for x in out if x[0]]

    def _tokens(self, words: list[str]) -> tuple[list[int], list[int]]:
        toks, owner = [], []
        for wi, w in enumerate(words):
            for ch in w.upper():
                if ch in self.vocab:
                    toks.append(self.vocab[ch])
                    owner.append(wi)
            if wi < len(words) - 1:
                toks.append(self.vocab["|"])
                owner.append(-1)
        return toks, owner

    def score_units(self, lp: np.ndarray, units: list, free_edges: bool = False) -> dict:
        """Align `units` (text.parts: plain strings and tuples of readings) to the posteriors `lp`. Returns
        {words, scores, frames, owner, measured, note}: the reading heard, each word's score 0..1, its (first, end)
        frame or None, the unit it came from, and whether every part of the take could be aligned.

        A take over LONG_FRAMES is cut where the script and the free hearing agree and aligned piece by piece, so an
        hour of audio costs no more memory than a minute; a piece with too little in common with the audio to cut
        (more than MAX_CELLS) is left unmeasured and named in `note`.

        `free_edges`: the audio may start before the text does and go on after it ends (a transcript of part of a
        recording). A wildcard label that fits any frame at no cost is put before the first word and after the
        last, so that audio is left over instead of the edge words being stretched across it."""
        n = lp.shape[0]
        star = lp.shape[1]
        if free_edges:
            lp = np.concatenate([lp, np.zeros((n, 1), dtype=lp.dtype)], axis=1)
        pieces = [(0, len(units), 0, n, False)]
        if n > LONG_FRAMES and len(units) > 1:
            default, own = [], []
            for i, u in enumerate(units):
                ws = T.norm(u[0] if isinstance(u, tuple) else u)
                default += ws
                own += [i] * len(ws)
            plain = [not isinstance(u, tuple) for u in units]
            heard = [(w, a, b, sep) for w, a, b, _, sep in self.hear(lp[:, :star])]
            secs = ctc.sections(default, heard, n, cuttable=lambda i: own[i] != own[i - 1] and plain[own[i]]
                                and plain[own[i - 1]])
            starts = [0] + [own[sec[0]] for sec in secs[1:]]
            pieces = [(u0, u1, f0, f1, lead) for u0, u1, (_, _, f0, f1, lead)
                      in zip(starts, starts[1:] + [len(units)], secs)]
        out = {"words": [], "scores": [], "frames": [], "owner": [], "measured": True, "note": None}
        for u0, u1, f0, f1, lead in pieces:
            sub, seg = units[u0:u1], lp[f0:f1]
            head = [self.vocab["|"]] if lead else []       # the piece was cut at a word-boundary mark
            tail = [star] if free_edges and f1 == n else []
            if free_edges and f0 == 0:
                head = [star]

            def expand(picks, sub=sub, u0=u0):
                words, owner = [], []
                for i, u in enumerate(sub):
                    ws = T.norm(u[picks.get(i, 0)] if isinstance(u, tuple) else u)
                    words += ws
                    owner += [u0 + i] * len(ws)
                return words, owner
            words, owner = expand({})
            if not words:
                continue
            ali = None
            if ctc.cells(len(seg), sum(len(w) + 1 for w in words)) > MAX_CELLS:
                out["note"] = (f"the text and the audio have too little in common to line up between "
                               f"{f0 * 0.02:.0f} s and {f1 * 0.02:.0f} s")
            else:
                def fit(ws, seg=seg, head=head, tail=tail):
                    a = ctc.align(seg, head + self._tokens(ws)[0] + tail, blank=self.blank)
                    return a.total if a else -np.inf
                words, owner = expand(T.choose(sub, fit))
                toks, tok_owner = self._tokens(words)
                toks, tok_owner = head + toks + tail, [-1] * len(head) + tok_owner + [-1] * len(tail)
                ali = ctc.align(seg, toks, blank=self.blank)
            out["words"] += words
            out["owner"] += owner
            if ali is None:                  # audio too short for the text: every word here is missing
                out["measured"] = False
                out["scores"] += [0.0] * len(words)
                out["frames"] += [None] * len(words)
                continue
            per, frames = [[] for _ in words], [[] for _ in words]
            for f, k in enumerate(ali.token.tolist()):
                if k >= 0 and tok_owner[k] >= 0:
                    per[tok_owner[k]].append(float(np.exp(ali.scores[f])))
                    frames[tok_owner[k]].append(f)
            out["scores"] += [round(float(np.mean(p)), 3) if p else 0.0 for p in per]
            out["frames"] += [(f0 + a, f0 + b) if fr else None for fr in frames for a, b in [_body(fr) or (0, 0)]]
        return out

    def word_scores(self, y: np.ndarray, sr: int, text: str) -> list[tuple[str, float]]:
        lp, sec = self.log_probs(y, sr)
        units = T.parts(text)
        if lp.shape[0] > LONG_FRAMES:        # one unit per word, so a long take can be cut between any two
            units = [q for p in units for q in ([p] if isinstance(p, tuple) else T.norm(p))]
        r = self.score_units(lp, units)
        self.words, self.measured, self.note = r["words"], r["measured"], r["note"]
        self.times = [(fr[0] * sec, fr[1] * sec) if fr else None for fr in r["frames"]]
        self.spans = [(w, *tm) for w, tm in zip(self.words, self.times) if tm]
        return list(zip(self.words, r["scores"]))


class Checker:
    """Whisper on the CPU, so it can sit next to the TTS model on a 4 GB GPU. Takes longer than 30 s are
    transcribed whole (transformers' long-form decoding)."""

    def __init__(self, log=_log):
        _guard_sklearn()
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        name = config.voice()["checker"]
        log(f"loading checker {name} (CPU)")
        model = _local_first(AutoModelForSpeechSeq2Seq.from_pretrained, name, dtype=torch.float32).eval()
        proc = _local_first(AutoProcessor.from_pretrained, name)
        self.asr = pipeline("automatic-speech-recognition", model=_speedup(model, "checker"), tokenizer=proc.tokenizer,
                            feature_extractor=proc.feature_extractor, device="cpu")

    def transcribe(self, y: np.ndarray, sr: int, beams: int | None = None) -> str:
        """The transcript. `beams`: beam width (default: the `beams` setting; 1 is greedy decoding)."""
        n = int(beams or config.voice().get("beams") or 1)
        return self.asr({"raw": _to16k(y, sr), "sampling_rate": 16000},
                        generate_kwargs={"num_beams": n})["text"].strip()
