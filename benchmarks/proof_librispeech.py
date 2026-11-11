"""How well does `vc proof` find and fix errors in a transcript? Measured on real human speech with known text.

    python benchmarks/proof_librispeech.py --data path/to/LibriSpeech/test-clean --n 150

LibriSpeech test-clean (https://www.openslr.org/12, CC BY 4.0, 346 MB) is read English with exact reference text.
For each of `--n` utterances (10-20 s, taken evenly across the set) the reference is turned into a "transcript from
some tool" by injecting one known error, in rotation:

    far         a word replaced by an unrelated word of similar length
    near        a word replaced by the most similarly spelled other word (walked -> walks): what ASR errors look like
    delete      one or two words removed
    insert      one or two unrelated words added
    hallucinate a sentence appended that nobody said ("thank you for watching")
    clean       nothing changed (measures false alarms)

and `proof` is run on it twice: with Whisper as the second opinion, and with the letter ear alone. Reported:
detection (an issue at the injected error), repair (applying the fixes at a confidence level gives back the
reference at that spot), false alarms per 1,000 words, and the word error rate before and after applying fixes.

Read the numbers for what they are: the errors are injected, not made by a real recogniser, and the letter model was
trained on LibriSpeech's training sets, so this is its home ground. Out-of-domain speech will score lower. The one
threshold tuned on this dataset (`proof_heard_gain`) was set on 60 other utterances, which are left out here.
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KINDS = ["far", "near", "delete", "insert", "hallucinate", "clean"]
HALLUCINATIONS = ["thank you for watching", "please subscribe to the channel", "thanks for listening everyone",
                  "see you in the next video"]


def utterances(data: Path, n: int, skip_tuning: bool = False) -> list[tuple[Path, str]]:
    """`n` utterances of 10-20 s spread evenly over the set, as (flac path, reference words in lower case).
    skip_tuning: leave out the 60 utterances `proof_heard_gain` was set on (every k-th, as `utterances(data, 60)`)."""
    import soundfile as sf
    rows = []
    for trans in sorted(data.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt, _, text = line.partition(" ")
            path = trans.parent / f"{utt}.flac"
            info = sf.info(str(path))
            if 10.0 <= info.frames / info.samplerate <= 20.0:
                rows.append((path, text.lower()))
    tuned = set(range(0, len(rows), max(1, len(rows) // 60))) if skip_tuning else set()
    rows = [r for i, r in enumerate(rows) if i not in tuned]   # the 60 the one threshold was set on stay out
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def inject(words: list[str], kind: str, pool: list[str], rng: random.Random) -> tuple[list[str], int, int]:
    """One error of `kind` in `words`: (the transcript's words, first changed index, end index). For a deletion the
    range is empty and sits where the words were."""
    inner = list(range(1, len(words) - 1))                 # never the first or last word: keep context both sides
    content = [i for i in inner if len(words[i]) >= 4]
    if kind == "clean":
        return list(words), 0, 0
    if kind == "hallucinate":
        extra = rng.choice(HALLUCINATIONS).split()
        return words + extra, len(words), len(words) + len(extra)
    if kind in ("far", "near"):
        i = rng.choice(content or inner)
        old = words[i]
        if kind == "far":
            options = [w for w in pool if abs(len(w) - len(old)) <= 2 and
                       difflib.SequenceMatcher(None, w, old).ratio() < 0.4]
            new = rng.choice(options)
        else:
            options = sorted({w for w in pool if w != old and w[0] == old[0] and abs(len(w) - len(old)) <= 2},
                             key=lambda w: (-difflib.SequenceMatcher(None, w, old).ratio(), w))
            new = options[0]
        return words[:i] + [new] + words[i + 1:], i, i + 1
    k = rng.choice([1, 2])
    if kind == "delete":
        i = rng.choice(inner[:-k] or inner)
        return words[:i] + words[i + k:], i, i
    i = rng.choice(inner)
    extra = [rng.choice([w for w in pool if len(w) >= 4]) for _ in range(k)]
    return words[:i] + extra + words[i:], i, i + k


def errors(ref: list[str], hyp: list[str]) -> int:
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="LibriSpeech/test-clean (the folder with the speaker folders)")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "proof_librispeech_summary.json"))
    a = ap.parse_args()

    from voicecheck import proof
    from voicecheck import text as T
    from voicecheck.cli import _audio
    from voicecheck.session import Session

    rows = utterances(Path(a.data), a.n, skip_tuning=True)
    pool = sorted({w for _, text in rows for w in text.split() if w.isalpha()})
    rng = random.Random(a.seed)
    session = Session(log=lambda m: print(m, file=sys.stderr, flush=True))
    listener = session._load("listener", {})
    checker = session._load("checker", {})
    cache: dict = {}
    hear, transcribe = listener.log_probs, checker.transcribe

    def log_probs(y, sr):                                  # both modes share the model passes of an utterance
        if "lp" not in cache:
            cache["lp"] = hear(y, sr)
        return cache["lp"]

    def asr(y, sr, beams=None):
        if "asr" not in cache:
            cache["asr"] = transcribe(y, sr)
        return cache["asr"]
    listener.log_probs, checker.transcribe = log_probs, asr

    modes = {"with_whisper": "whisper", "letter_ear_only": None}
    levels = ["high", "medium", "all"]
    stats = {m: {k: {"n": 0, "detected": 0, **{f"repaired_{lv}": 0 for lv in levels}} for k in KINDS} for m in modes}
    totals = {m: {"words": 0, "false_alarms": {lv: 0 for lv in levels}, "errors_before": 0,
                  **{f"errors_after_{lv}": 0 for lv in levels}, "clean_reported_clean": 0}
              for m in modes}
    audio_seconds, log = 0.0, []
    began = time.time()
    for n, (path, ref_text) in enumerate(rows):
        kind = KINDS[n % len(KINDS)]
        ref = ref_text.split()
        hyp, e0, e1 = inject(ref, kind, pool, rng)
        text = " ".join(hyp)
        starts, pos = [], 0
        for w in hyp:
            starts.append(pos)
            pos += len(w) + 1
        lo = starts[e0] - 1 if e0 < len(hyp) else len(text)   # the injected error, as a stretch of characters
        hi = (starts[e1 - 1] + len(hyp[e1 - 1]) if e1 > e0 else lo + 1) + 1
        y, sr = _audio(str(path))
        audio_seconds += len(y) / sr
        cache.clear()
        for mode, second in modes.items():
            doc = proof.parse(text)
            report = session.proof(y, sr, doc, table={}, second=second)
            tot, st = totals[mode], stats[mode][kind]
            tot["words"] += len(hyp)
            tot["errors_before"] += errors(ref, hyp)
            st["n"] += 1
            at = [x for x in report["issues"] if kind != "clean" and x["span"][0] <= hi and x["span"][1] >= lo]
            st["detected"] += bool(at)
            for lv in levels:
                floor = 0 if lv == "all" else proof.LEVELS[lv]
                tot["false_alarms"][lv] += sum(1 for x in report["issues"] if x not in at
                                               and proof.LEVELS[x["confidence"]] >= floor)
                fixed = T.words(doc.fix(proof.edits(report, lv)))
                left = errors(ref, fixed)
                tot[f"errors_after_{lv}"] += left
                st[f"repaired_{lv}"] += kind != "clean" and left == 0
            if kind == "clean":
                tot["clean_reported_clean"] += not any(proof.LEVELS[x["confidence"]] >= 2 for x in report["issues"])
            log.append({"utt": path.stem, "mode": mode, "kind": kind, "transcript": text, "error": [e0, e1],
                        "issues": [{k: x[k] for k in ("kind", "confidence", "transcript", "suggest", "span")}
                                   for x in report["issues"]]})
        if (n + 1) % 10 == 0:
            print(f"[{n + 1}/{len(rows)}] {time.time() - began:.0f} s", file=sys.stderr, flush=True)

    summary = {"dataset": "LibriSpeech test-clean", "utterances": len(rows), "audio_minutes": round(audio_seconds / 60, 1),
               "seed": a.seed, "wall_minutes": round((time.time() - began) / 60, 1), "modes": {}}
    for mode in modes:
        tot = totals[mode]
        per_k = 1000 / max(1, tot["words"])
        summary["modes"][mode] = {
            "by_error": {k: {"n": s["n"], "detected": round(s["detected"] / max(1, s["n"]), 3),
                             **{f"repaired_{lv}": round(s[f"repaired_{lv}"] / max(1, s["n"]), 3) for lv in levels}}
                         for k, s in stats[mode].items() if k != "clean"},
            "clean_transcripts": stats[mode]["clean"]["n"],
            "clean_reported_clean": tot["clean_reported_clean"],
            "false_alarms_per_1000_words": {lv: round(tot["false_alarms"][lv] * per_k, 2) for lv in levels},
            "wer_before": round(tot["errors_before"] / tot["words"], 4),
            **{f"wer_after_{lv}": round(tot[f"errors_after_{lv}"] / tot["words"], 4) for lv in levels}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    out.with_name(out.stem.replace("_summary", "") + ".jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in log), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
