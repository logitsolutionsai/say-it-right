"""One checking session: the models load once, on first use, and every take after that reuses them.

    s = Session()
    report = s.check(y, sr, "Ask Siobhan about it.", table=lexicon.load(["names.yaml"]))

`vc.py check` runs one take or a whole batch through a Session; a long-running caller (a narration pipeline) can keep
one around. Each report carries `timing`: seconds spent loading models and on each ear, so slow steps are visible.
"""
from __future__ import annotations

import time

import numpy as np

from . import config, verify
from .lexicon import locate, render


class Session:
    def __init__(self, log=verify._log):
        self.log = log
        self._checker = self._listener = self._ear = None

    def _load(self, name: str, timing: dict):
        t0 = time.perf_counter()
        if name == "checker" and self._checker is None:
            self._checker = verify.Checker(self.log)
        elif name == "listener" and self._listener is None:
            self._listener = verify.Listener(self.log)
        elif name == "ear" and self._ear is None:
            from .phonemes import PhonemeEar
            self._ear = PhonemeEar(self.log)
        timing["load"] = round(timing.get("load", 0.0) + time.perf_counter() - t0, 2)
        return {"checker": self._checker, "listener": self._listener, "ear": self._ear}[name]

    def check(self, y: np.ndarray, sr: int, script: str, shown: str | None = None, table: dict | None = None,
              pace: tuple | None = None, tone: bool | None = None) -> dict:
        """The full report for one take (see references/checks.md). `shown` given: `script` is already the spoken
        text (the legacy form). `tone`: measure pitch even without `flat` limits (it's the slowest step)."""
        v = config.voice()
        timing: dict = {}
        start = time.perf_counter()

        def lap(key, t0):
            timing[key] = round(time.perf_counter() - t0, 2)

        terms = []
        if shown is not None:
            spoken = script
        else:
            r = render(script, table if table is not None else {})
            spoken, shown, terms = r.spoken, r.shown, r.terms
        alt = shown if shown != spoken else None

        checker = self._load("checker", timing)
        t0 = time.perf_counter()
        heard = checker.transcribe(y, sr)
        by_sound = [p for t in terms if t.ipa for p in (t.said, t.shown)]      # judged by the phoneme ear instead
        score, diff = verify.match(spoken, heard, alt=alt, ignore=by_sound)
        beams = int(v.get("beams") or 1)
        retry = int(v.get("retry_beams") or 0)
        if score < v["match_min"] and retry > beams:       # a mismatch: the slower, surer decoder has the last word
            heard = checker.transcribe(y, sr, beams=retry)
            score, diff = verify.match(spoken, heard, alt=alt, ignore=by_sound)
            beams = retry
        lap("transcript", t0)

        listener = self._load("listener", timing)
        t0 = time.perf_counter()
        scores = listener.word_scores(y, sr, spoken)
        words, spans = listener.words, listener.spans
        skip, names, need_ear, unchecked = set(), [], [], []
        for t in terms:
            loc = locate(words, t)
            if loc is None:
                if t.ipa:
                    unchecked.append(t.shown)
                continue
            a, b = loc
            if t.ipa:
                skip.update(range(a, b))                   # judged by sound, not letters
                need_ear.append((t, a, b))
            else:
                skip.update(range(a + 1, b))               # a respelling's tail is never scored by its letters
        slipped = verify.slips(spoken, heard, scores, v["slip_score"], v["gone_score"], alt=alt, skip=skip)
        lap("letters", t0)

        if need_ear and listener.measured:
            from .phonemes import word_window
            ear = self._load("ear", timing)
            t0 = time.perf_counter()
            for t, a, b in need_ear:
                win = word_window(listener.times, a, b, len(y) / sr)
                if win:
                    verdict = ear.judge(y, sr, t.ipa, win[0], win[1], word=t.shown, said=t.said, pad=0.03)
                    names.append(verdict.as_dict())
                else:
                    unchecked.append(t.shown)
            lap("phones", t0)

        seconds = verify.speech_seconds(spans, len(y) / sr)
        rate = verify.pace(spoken, seconds)
        pace_lim = pace or v.get("pace_syl_s")
        limits = v.get("flat") or {}
        prosody = None
        if spans and (limits or tone):
            t0 = time.perf_counter()
            prosody = verify.prosody(y, sr, spans)
            lap("tone", t0)
        tone_flags = verify.emotional(prosody, limits)

        reasons = []
        if not listener.measured:
            reasons.append("the audio is too short to hold the script (the listener could not align it)")
        if score < v["match_min"]:
            reasons.append(f"transcript match {score:.3f} < {v['match_min']}")
        if slipped and listener.measured:
            listed = ", ".join(slipped[:8]) + (f" and {len(slipped) - 8} more" if len(slipped) > 8 else "")
            reasons.append(f"slipped words: {listed}")
        elif slipped:
            reasons.append(f"{len(slipped)} of {len(scores)} words missing")
        if unchecked and listener.measured:                # never pass a take whose name wasn't actually heard
            reasons.append(f"could not find {', '.join(unchecked)} in the take to check how it was said")
        bad_names = [n for n in names if not n["ok"]]
        if bad_names:
            reasons.append("mispronounced: " + ", ".join(f"{n['word']} (heard {n['heard'] or 'nothing'}, "
                                                          f"want {n['target']}, {n['score']:.2f})" for n in bad_names))
        if pace_lim and not (pace_lim[0] <= rate <= pace_lim[1]):
            reasons.append(f"pace {rate:.2f} syllables/s outside {pace_lim[0]}-{pace_lim[1]}")
        reasons += [f"tone: {f}" for f in tone_flags]
        timing["total"] = round(time.perf_counter() - start, 2)
        timing["audio_seconds"] = round(len(y) / sr, 2)
        return {
            "passed": not reasons, "reasons": reasons, "heard": heard, "spoken": spoken, "shown": shown,
            "match": round(score, 3), "diff": diff, "beams": beams, "slips": slipped, "names": names,
            "listener_measured": listener.measured, "word_scores": scores, "syllables_per_s": round(rate, 2),
            "speech_seconds": round(seconds, 2), "tone": prosody, "tone_flags": tone_flags, "timing": timing,
        }
