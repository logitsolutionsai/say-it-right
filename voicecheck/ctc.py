"""CTC forced alignment (Viterbi) in numpy: which frames of the audio carry which letter of the script.

It gives the same best path as `torchaudio.functional.forced_align` (tests/test_ctc.py compares the two), without
depending on torchaudio, whose alignment API has been deprecated and moved before. It is also cheap enough to run once
per candidate reading of the script, which is how the listener picks the reading that best fits the audio.

`greedy()` is the other direction: what the model hears with no script at all (the best label per frame). The
transcript proofreader compares the two, and long takes are cut into sections where both agree (`sections()`)."""
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
    skip = np.zeros(n_states, dtype=bool)          # a label may follow the previous label directly (no blank)
    skip[3::2] = ext[3::2] != ext[1:-2:2]
    neg = -np.inf
    alpha = np.full(n_states, neg)                 # emissions are read a frame at a time (lp[t, ext]): a [T, S]
    alpha[:2] = lp[0, ext[:2]]                     # float table would be 8x the size of `back`
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
        alpha = best + lp[t, ext]
    s = n_states - 1 if alpha[-1] >= alpha[-2] else n_states - 2
    total = float(alpha[s])
    states = np.empty(n_frames, dtype=np.int64)
    for t in range(n_frames - 1, -1, -1):
        states[t] = s
        s -= int(back[t, s])
    return Alignment(labels=ext[states], scores=lp[np.arange(n_frames), ext[states]],
                     token=np.where(states % 2 == 1, (states - 1) // 2, -1), total=total)


def cells(n_frames: int, n_targets: int) -> int:
    """The size of align()'s backpointer table in bytes: frames x (2 x labels + 1)."""
    return n_frames * (2 * n_targets + 1)


def greedy(log_probs: np.ndarray, blank: int = 0) -> list[tuple[int, int, int, float]]:
    """What the model hears without a script: [(label, first frame, end frame, mean probability)] for each run of
    one non-blank label on the best path (repeats collapsed, blanks dropped)."""
    lp = np.asarray(log_probs)
    if not len(lp):
        return []
    ids = lp.argmax(-1)
    top = np.exp(lp[np.arange(len(ids)), ids].astype(np.float64))
    edges = np.flatnonzero(np.diff(ids)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(ids)]))
    return [(int(ids[a]), int(a), int(b), float(top[a:b].mean())) for a, b in zip(starts, ends) if ids[a] != blank]


def sections(script: Sequence[str], heard: Sequence[tuple], n_frames: int, cuttable=None,
             every: int = 1000) -> list[tuple[int, int, int, int, bool]]:
    """Cut a long take where the script and the free hearing agree, so each piece can be aligned on its own:
    [(first word, end word, first frame, end frame, starts with a separator)]. `heard` is [(word, first frame, end
    frame, separator frame or None)]: the words of greedy() and, when the model marked the word boundary before a
    word, the frame where that mark starts. A cut goes between two script words that both sit inside a run the
    hearing matched exactly, no sooner than `every` frames after the last cut: at the boundary mark when there is
    one (the piece then starts with a separator label, so the mark's frames aren't charged to a word), else in the
    middle of the pause. `cuttable(i)` may veto a cut before script word i. With nothing in common there are no
    cuts: one section, and the caller decides whether it's too big."""
    import difflib
    hw = [h[0] for h in heard]
    cuts, last = [], 0
    for i, j, n in difflib.SequenceMatcher(None, list(script), hw, autojunk=False).get_matching_blocks():
        for k in range(1, n):
            sep = heard[j + k][3]
            frame = sep if sep is not None else (heard[j + k - 1][2] + heard[j + k][1]) // 2
            if frame - last >= every and n_frames - frame >= every // 4 and (cuttable is None or cuttable(i + k)):
                cuts.append((i + k, frame, sep is not None))
                last = frame
    out, w0, f0, lead = [], 0, 0, False
    for w, f, sep in cuts:
        out.append((w0, w, f0, f, lead))
        w0, f0, lead = w, f, sep
    out.append((w0, len(script), f0, n_frames, lead))
    return out
