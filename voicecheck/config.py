"""Settings for the checker.

Defaults below; to change them, point VOICECHECK_CONFIG at a YAML file (unknown keys are ignored, so a full
OpenMontage `voice.yaml` works as is: `examples/openmontage_voice.yaml`). Keys read:

    checker          Whisper model that writes the transcript
    beams            Whisper's beam width: 1 (greedy) is about twice as fast as 5 (OpenAI's and transformers' default)
    retry_beams      a transcript that doesn't match the script is decoded again with this many beams before the take
                     fails on it (0: never); so a passing take costs one greedy pass, see README for the measurements
    listener         wav2vec2 letter model (no language model) that scores each scripted word
    phonemes         wav2vec2 phoneme model that checks how a researched word sounds (lexicon entries with `ipa`)
    match_min        transcript vs script letter similarity a take needs
    slip_score       a word Whisper changed AND the listener scored under this is a slip
    gone_score       a word the listener scored under this is missing, whatever Whisper wrote
    phone_min        a lexicon word whose sound scores under this against its `ipa` is mispronounced
    phone_hard_max   ...or that has more than this many clearly wrong or missing sounds (0: any one fails it)
    pace_syl_s       [low, high] syllables per second of speech; outside it the take fails. null = not checked
    flat             emotion limits {range_st, peak_st, tail_st}; a take over any of them fails. {} = not checked
    pitch_hz         "auto" (60 Hz up to 2.5x the voice's median, at least 320) or [low, high] for the pitch tracker
    pronunciations   word -> respelling, or word -> {say, ipa, source, note}; see voicecheck/lexicon.py
    lexicon          path(s) of lexicon YAML files merged over `pronunciations`
    proof_gain       `proof`: how much better (in log-likelihood, to the letter model) another wording must fit the
                     audio than the transcript's before it is suggested
    proof_heard_gain `proof`: how much better the letters the model hears by itself must fit than the transcript's word
                     before that word is reported (10: see references/proofreading.md for the calibration)
    proof_agree_min  `proof`: when the letter model hears a smaller share of the transcript's words than this, the
                     transcript is taken to be of another recording and nothing is suggested with confidence
    spot_min         `proof`: how closely a stretch of audio must match a lexicon name's IPA to count as that name
    quantize         int8 dynamic quantisation on the CPU: true (all three models) or a list of them ("checker",
                     "listener", "phonemes"). Faster, but it moves the calibrated scores: see README
"""
from __future__ import annotations

import os

DEFAULTS = {
    "checker": "openai/whisper-small.en",
    "beams": 1,
    "retry_beams": 5,
    "listener": "facebook/wav2vec2-large-960h-lv60-self",
    "phonemes": "facebook/wav2vec2-lv-60-espeak-cv-ft",
    "match_min": 0.95,
    "slip_score": 0.65,
    "gone_score": 0.25,
    "phone_min": 0.6,             # calibrated 2026-11-10 on 32 takes (8 names x 2 voices x right/naive)
    "phone_hard_max": 0,
    "proof_gain": 4.0,
    "proof_heard_gain": 10.0,
    "proof_agree_min": 0.3,
    "spot_min": 0.8,
    "pace_syl_s": None,
    "flat": {},
    "pitch_hz": "auto",
    "pronunciations": {},
    "lexicon": None,
    "quantize": False,
}

_cache: dict | None = None


def voice() -> dict:
    """The settings: DEFAULTS, overridden by the YAML file at $VOICECHECK_CONFIG (read once)."""
    global _cache
    if _cache is None:
        v = dict(DEFAULTS)
        path = os.environ.get("VOICECHECK_CONFIG")
        if path:
            import yaml
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError(f"{path}: expected a mapping of settings, got {type(data).__name__}")
            v.update(data)
        _cache = v
    return _cache


def reset(overrides: dict | None = None) -> dict:
    """Forget the cached settings (tests, or a long-running caller whose file changed); `overrides` win."""
    global _cache
    _cache = None
    v = voice()
    if overrides:
        v.update(overrides)
    return v
