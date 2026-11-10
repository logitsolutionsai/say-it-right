import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voicecheck import config  # noqa: E402


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    """Every test starts from the built-in defaults, whatever VOICECHECK_CONFIG says."""
    monkeypatch.delenv("VOICECHECK_CONFIG", raising=False)
    config.reset()
    yield
    config.reset()


# ---- a fake letter model, so the listener's alignment and reading choice run without downloading anything

LETTERS = ["<pad>", "|", "'"] + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
VOCAB = {ch: i for i, ch in enumerate(LETTERS)}


def fake_posteriors(spoken: str, frames_per_letter: int = 3, gap: int = 2, garble: dict | None = None) -> np.ndarray:
    """Log-posteriors of a take that clearly says `spoken`: each letter peaks for a few frames, blanks between.
    garble: {word_index: replacement_word} to make the "audio" say something else for that word."""
    words = spoken.upper().split()
    if garble:
        words = [garble.get(i, w).upper() for i, w in enumerate(words)]
    seq = []
    for wi, w in enumerate(words):
        for ch in w:
            seq += [VOCAB[ch]] * frames_per_letter + [0] * gap
        if wi < len(words) - 1:
            seq += [VOCAB["|"]] * 2 + [0] * gap
    seq = [0] * 5 + seq + [0] * 5
    lp = np.full((len(seq), len(LETTERS)), np.log(0.002 / (len(LETTERS) - 1)))
    for t, k in enumerate(seq):
        lp[t, k] = np.log(0.998)
    return lp


def fake_listener(lp: np.ndarray):
    from voicecheck import verify
    lis = object.__new__(verify.Listener)
    lis.vocab, lis.blank = VOCAB, 0
    lis.spans, lis.words, lis.measured, lis.times = [], [], True, []
    lis.log_probs = lambda y, sr: (lp, 0.02)
    return lis


# ---- the reference scripts import OpenMontage modules: stub them so their pure functions can be tested

def stub_openmontage():
    for name in ["logit", "logit.lecture", "logit.lecture.audio_util", "logit.lecture.paths", "logit.lecture.render",
                 "logit.engine"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["logit.lecture.audio_util"].tool_env = lambda: None
    sys.modules["logit.lecture.paths"].BUILD = sys.modules["logit.lecture.paths"].CONTENT = Path(".")
    sys.modules["logit.lecture.paths"].meta = types.SimpleNamespace(TERMS=[], PHRASES=[])
    sys.modules["logit.lecture.render"].Slot = object
    sys.modules["logit.engine"].verify = None
    if "PIL" not in sys.modules:                       # number_audit draws contact sheets with Pillow
        try:
            import PIL  # noqa: F401
        except ImportError:
            pil = types.ModuleType("PIL")
            pil.Image = pil.ImageDraw = pil.ImageFont = types.SimpleNamespace()
            pil.voicecheck_stub = True                     # removed again by unstub_pil(): transformers probes PIL
            sys.modules["PIL"] = pil
    ref = str(ROOT / "reference")
    if ref not in sys.path:
        sys.path.insert(0, ref)


def unstub_pil():
    if getattr(sys.modules.get("PIL"), "voicecheck_stub", False):
        del sys.modules["PIL"]
