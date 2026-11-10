import numpy as np
import pytest

from voicecheck import ctc


def test_feasible():
    assert ctc.feasible(3, [1, 2, 3])
    assert not ctc.feasible(2, [1, 2, 3])
    assert not ctc.feasible(3, [1, 1, 2])            # the repeated label needs a blank between
    assert not ctc.feasible(10, [])


def test_align_recovers_a_clear_path():
    seq = [0, 1, 1, 0, 2, 0, 0, 2, 3, 0]             # letters 1, 2, 2, 3 (the repeat separated by blanks)
    lp = np.full((len(seq), 4), np.log(0.01))
    for t, k in enumerate(seq):
        lp[t, k] = np.log(0.97)
    ali = ctc.align(lp, [1, 2, 2, 3], blank=0)
    assert ali.labels.tolist() == seq
    assert ali.token.tolist() == [-1, 0, 0, -1, 1, -1, -1, 2, 3, -1]
    assert ali.total == pytest.approx(float(ali.scores.sum()))


def test_align_returns_none_when_audio_is_too_short():
    assert ctc.align(np.zeros((2, 4)), [1, 2, 3]) is None


def test_matches_torchaudio_when_available():
    torch = pytest.importorskip("torch")
    AF = pytest.importorskip("torchaudio.functional")
    if not hasattr(AF, "forced_align"):
        pytest.skip("this torchaudio has no forced_align")
    rng = np.random.default_rng(1)
    for _ in range(60):
        C, T = int(rng.integers(5, 30)), int(rng.integers(10, 200))
        tg = rng.integers(1, C, size=int(rng.integers(1, T // 3)))
        if not ctc.feasible(T, list(tg)):
            continue
        lp = torch.log_softmax(torch.randn(T, C) * 3, -1)
        a, s = AF.forced_align(lp[None], torch.tensor(tg[None], dtype=torch.int32), blank=0)
        mine = ctc.align(lp.numpy(), tg, blank=0)
        assert mine.total == pytest.approx(float(s[0].double().sum()), abs=1e-4)
        assert (a[0].numpy() == mine.labels).all()
