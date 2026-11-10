"""A third ear, for the words that were researched: it hears sounds (IPA phones), not letters.

The letter listener can't judge a respelled name ("Seg-eh-dee" has no letters in common with Szegedy, and the
"you" of rel-you is spoken "yu"), and Whisper writes the spelling it knows. This ear transcribes the audio into
phones with wav2vec2 fine-tuned on multilingual phoneme recognition (facebook/wav2vec2-lv-60-espeak-cv-ft,
Apache-2.0) and compares them with the lexicon's IPA for that word (voicecheck.ipa.compare).

It also turns a reference recording into IPA (`hear`): a native speaker saying the name (Wikimedia Commons,
Lingua Libre, the person's own interview) becomes the target to check the voice against.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from . import config, ipa
from .verify import _guard_sklearn, _log, _speedup, _to16k

_SPECIAL = {"<pad>", "<s>", "</s>", "<unk>", "|", " "}


@dataclass
class Verdict:
    word: str                   # as shown
    said: str                   # what the voice was given
    target: str                 # the IPA it should sound like
    heard: str                  # the heard phones that line up with the target
    around: str                 # every phone heard in the window (the neighbours' edges included)
    score: float
    hard: int                   # sounds clearly different (!) or missing (-), not counting weak ones
    ok: bool
    diff: str
    t0: float
    t1: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def word_window(times: list, a: int, b: int, total: float) -> tuple[float, float] | None:
    """Where to listen for words a..b-1: from the end of the word before to the start of the word after. When a name
    is said wrong, the letter alignment of its respelling drifts (a misread "Siobhan" started 0.3 s before the
    aligned "Shiv-awn"); the neighbours, said right, pin the gap the name must be in. compare() ignores what bleeds
    in at the edges."""
    first, last = times[a], times[b - 1]
    if not first or not last:
        return None
    before = next((times[k][1] for k in range(a - 1, -1, -1) if times[k]), 0.0)
    after = next((times[k][0] for k in range(b, len(times)) if times[k]), total)
    return min(first[0], before), max(last[1], after)


def hard_errors(diff: str) -> int:
    """Clear substitutions (!) and missing non-weak sounds (-x) in a compare() diff."""
    n = 0
    for op in diff.split():
        if "!" in op:
            n += 1
        elif op.startswith("-") and op[1:] not in ipa._WEAK:
            n += 1
    return n


class PhonemeEar:
    def __init__(self, log=_log):
        _guard_sklearn()
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        name = config.voice()["phonemes"]
        log(f"loading phoneme ear {name} (CPU)")
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(name)
        self.model = _speedup(Wav2Vec2ForCTC.from_pretrained(name, dtype=torch.float32).eval(), "phonemes")
        with open(hf_hub_download(name, "vocab.json"), encoding="utf-8") as f:
            vocab = json.load(f)
        self.id2ph = {i: p for p, i in vocab.items()}
        self.blank = vocab.get("<pad>", 0)
        self._cache: tuple | None = None

    def log_probs(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
        """[T, C] phone log-posteriors and seconds per frame; the last take is cached (one pass serves every word)."""
        import hashlib

        import torch
        # Keyed by the audio's content: an id() can be reused by the next array, which would hand back the previous
        # take's phones (a batch, the server and the benchmark all check take after take).
        key = (hashlib.md5(np.ascontiguousarray(y).tobytes()).hexdigest(), len(y), sr)
        if self._cache and self._cache[0] == key:
            return self._cache[1], self._cache[2]
        y16 = _to16k(y, sr)
        x = self.fe(y16, sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            lp = torch.log_softmax(self.model(x).logits[0].float(), -1).numpy()
        sec = len(y16) / 16000 / max(1, lp.shape[0])
        self._cache = (key, lp, sec)
        return lp, sec

    def hear(self, y: np.ndarray, sr: int, t0: float | None = None, t1: float | None = None) -> list[str]:
        """The phones in [t0, t1] (the whole take by default), by greedy CTC decoding of the take's posteriors."""
        lp, sec = self.log_probs(y, sr)
        a = 0 if t0 is None else max(0, int(t0 / sec))
        b = lp.shape[0] if t1 is None else min(lp.shape[0], int(np.ceil(t1 / sec)))
        ids = lp[a:b].argmax(-1)
        out, prev = [], None
        for k in ids.tolist():
            if k != prev and k != self.blank:
                ph = self.id2ph.get(k, "")
                if ph and ph not in _SPECIAL:
                    out.append(ph)
            prev = k
        return out

    def judge(self, y: np.ndarray, sr: int, target_ipa: str, t0: float, t1: float, word: str = "", said: str = "",
              pad: float = 0.08) -> Verdict:
        """Does the voice say `target_ipa` between t0 and t1? The window is padded a little each side; compare()
        ignores what bleeds in from the neighbours."""
        heard = self.hear(y, sr, max(0.0, t0 - pad), t1 + pad)
        c = ipa.compare(target_ipa, heard)
        hard = hard_errors(c.diff)
        v = config.voice()
        ok = c.score >= v.get("phone_min", 0.6) and hard <= int(v.get("phone_hard_max", 0))
        return Verdict(word=word, said=said, target=c.target, heard=c.heard, around=" ".join(heard), score=c.score,
                       hard=hard, ok=ok,
                       diff=c.diff, t0=round(t0, 2), t1=round(t1, 2))
