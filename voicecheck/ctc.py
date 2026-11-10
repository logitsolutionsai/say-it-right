"""CTC forced alignment (Viterbi) in numpy: which frames of the audio carry which letter of the script.

It gives the same best path as `torchaudio.functional.forced_align` (tests/test_ctc.py compares the two), without
depending on torchaudio, whose alignment API has been deprecated and moved before. It is also cheap enough to run once
per candidate reading of the script, which is how the listener picks the reading that best fits the audio."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Alignment:
    labels: np.ndarray        # [T] the label each frame is aligned to (blank included)
    scores: np.ndarray        # [T] log-probability of that label at that frame
    token: np.ndarray         # [T] index into `targets` of the frame's label, -1 on blank frames
    total: float              # log-probability of the whole path (compares readings of the same audio)


def feasible(n_frames: int, targets: Sequence[int]) -> bool:
    """A CTC path needs a frame per label, plus a blank between two equal labels in a row."""
    repeats = sum(1 for i in range(1, len(targets)) if targets[i] == targets[i - 1])
    return 0 < len(targets) and len(targets) + repeats <= n_frames


def align(log_probs: np.ndarray, targets: Sequence[int], blank: int = 0) -> Alignment | None:
    """Best CTC path for `targets` through `log_probs` ([T, C], log-softmax). None when the audio is too short to hold
    the targets (or there are none): the caller decides what that means, it is never silently "no problem"."""
    lp = np.asarray(log_probs, dtype=np.float64)
    n_frames = lp.shape[0]
    tg = np.asarray(targets, dtype=np.int64)
    if not feasible(n_frames, list(tg)):
        return None
    n_states = 2 * len(tg) + 1                     # blank, t0, blank, t1, ..., blank
    ext = np.full(n_states, blank, dtype=np.int64)
    ext[1::2] = tg
    emit = lp[:, ext]                              # [T, S]
    skip = np.zeros(n_states, dtype=bool)          # a label may follow the previous label directly (no blank)
    skip[3::2] = ext[3::2] != ext[1:-2:2]
    neg = -np.inf
    alpha = np.full(n_states, neg)
    alpha[:2] = emit[0, :2]
    back = np.zeros((n_frames, n_states), dtype=np.int8)
    p1, p2, best = np.full(n_states, neg), np.full(n_states, neg), np.empty(n_states)
    no_skip = ~skip
    for t in range(1, n_frames):
        p1[1:] = alpha[:-1]                        # from the previous state
        p2[2:] = alpha[:-2]                        # skipping a blank
        p2[no_skip] = neg
        np.copyto(best, alpha)                     # staying (ties keep the earlier choice, as torchaudio does)
        row = back[t]
        m = p1 > best
        best[m] = p1[m]
        row[m] = 1
        m = p2 > best
        best[m] = p2[m]
        row[m] = 2
        alpha = best + emit[t]
    s = n_states - 1 if alpha[-1] >= alpha[-2] else n_states - 2
    total = float(alpha[s])
    states = np.empty(n_frames, dtype=np.int64)
    for t in range(n_frames - 1, -1, -1):
        states[t] = s
        s -= int(back[t, s])
    return Alignment(labels=ext[states], scores=emit[np.arange(n_frames), states],
                     token=np.where(states % 2 == 1, (states - 1) // 2, -1), total=total)
