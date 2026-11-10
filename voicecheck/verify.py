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


def _same_sound(words: list[str]) -> list[str]:
    return [_HOMOPHONES.get(w, w) for w in words]


def _letters(words: list[str]) -> str:
    return "".join(_same_sound(words))


def _ratio(a: list[str], b: list[str]) -> float:
    return difflib.SequenceMatcher(None, _letters(a), _letters(b), autojunk=False).ratio()


def _fit(text: str, target: list[str]) -> list[str]:
    """The words of `text`, each number in the reading closest to `target` (Whisper's "100" against a script's
    "a hundred", a script's "1959" against a transcript's "nineteen fifty-nine")."""
    ps = T.parts(text)
    return T.expand(ps, T.choose(ps, lambda ws: _ratio(ws, target)))[0]


def _find(words: list[str], phrase: str) -> set[int]:
    w = T.words(phrase)
    return {k for i in range(len(words) - len(w) + 1) if w and words[i:i + len(w)] == w for k in range(i, i + len(w))}


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
    they and whatever Whisper wrote in their place are left out of the comparison."""
    if alt:
        return max((match(expected, heard, ignore=ignore), match(alt, heard, ignore=ignore)), key=lambda r: r[0])
    b = _fit(heard, T.words(expected))
    a = _fit(expected, b)
    b = _fit(heard, a)
    hide = set().union(*(_find(a, p) for p in ignore)) if ignore else set()
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


def syllables(text: str) -> int:
    import re
    n = 0
    for w in norm_words(text):
        k = len(re.findall(r"[aeiouy]+", w))
        if w.endswith("e") and not w.endswith(("le", "ee")) and k > 1:
            k -= 1                                               # silent final e
        n += max(1, k)
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


def slips(expected: str, heard: str, scores, near=0.65, gone=0.25, alt: str | None = None,
          skip: set | None = None) -> list[str]:
    """Words the voice got wrong. Whisper's language model writes what it expects to hear, so a letter-level
    score barely moves for one wrong short word ("zero for cap", "being sure the wrong answer" with "of"
    dropped, DL-08). A word is a slip when Whisper's transcript differs there AND the acoustic score is under
    `near`, or when the acoustic score alone is under `gone` (the word is not in the audio).

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
    touched = set()
    sa = _same_sound(a)
    if not (alt and _same_sound(_fit(alt, b)) == _same_sound(b)):
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, sa, _same_sound(b), autojunk=False).get_opcodes():
            if op != "equal":
                touched.update(range(i1, i2))
    for i in sorted(touched):                 # in "the the" a diff may blame either copy: a run of one word is
        for step in (-1, 1):                  # touched as a whole, so the copy the audio lacks is the one flagged
            j = i + step
            while 0 <= j < len(sa) and sa[j] == sa[i]:
                touched.add(j)
                j += step
    return [f"{w} ({s:.2f})" for i, (w, s) in enumerate(scores)
            if i not in skip and ((i in touched and s < near) or s < gone)]


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


class Listener:
    """A second ear with no language model: wav2vec2 character CTC (CPU). The scripted words are force-aligned
    to its letter posteriors, and each word's score is the mean posterior of its aligned frames: low means the
    word is garbled or missing, whatever Whisper wrote. Each number in the script is read the way that best fits
    the audio ("1959" as "nineteen fifty nine" or "one thousand nine hundred fifty nine").

    After word_scores(): `spans` [(word, t0, t1)], `words` (the reading heard), `measured` (False when the audio
    is too short to hold the script; then every word scores 0.0, so the take fails as missing words)."""

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

    def log_probs(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
        """[T, C] letter log-posteriors of the take, and seconds per frame (~0.02)."""
        import torch
        y16 = _to16k(y, sr)
        x = self.proc(y16, sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            lp = torch.log_softmax(self.model(x).logits[0].float(), -1).numpy()
        return lp, len(y16) / 16000 / max(1, lp.shape[0])

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

    def word_scores(self, y: np.ndarray, sr: int, text: str) -> list[tuple[str, float]]:
        lp, sec = self.log_probs(y, sr)
        ps = T.parts(text)

        def fit(ws):
            ali = ctc.align(lp, self._tokens(ws)[0], blank=self.blank)
            return ali.total if ali else -np.inf
        words = T.expand(ps, T.choose(ps, fit))[0]
        toks, owner = self._tokens(words)
        ali = ctc.align(lp, toks, blank=self.blank)
        self.words, self.spans, self.times = words, [], [None] * len(words)
        self.measured = ali is not None
        if ali is None:                      # audio too short for the script: every word is missing
            return [(w, 0.0) for w in words]
        per, frames = [[] for _ in words], [[] for _ in words]
        for f, k in enumerate(ali.token.tolist()):
            if k >= 0 and owner[k] >= 0:
                per[owner[k]].append(float(np.exp(ali.scores[f])))
                frames[owner[k]].append(f)
        self.times = [(fr[0] * sec, (fr[-1] + 1) * sec) if fr else None for fr in frames]
        self.spans = [(w, *tm) for w, tm in zip(words, self.times) if tm]
        return [(w, round(float(np.mean(p)), 3) if p else 0.0) for w, p in zip(words, per)]


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
