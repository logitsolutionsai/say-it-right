"""Pronunciation audit of the final narration: for every word that is a name, acronym, number, respelling or capitalised
mid-sentence, cut a narrow window around it from narration.wav (forced-aligner times) and decode its letters with the
wav2vec2 listener (no language model, so it can't 'fill in' a word the way Whisper does). Also decodes every sentence whole.
Usage: pron_audit.py TS-06 [TS-07 ...]  -> prints and writes build/<ID>/pron_audit.txt"""
import json
import os
import re
import sys
from pathlib import Path

import librosa
import soundfile as sf
import torch

ROOT = Path(os.environ.get("OPENMONTAGE", Path(__file__).resolve().parents[2] / "OpenMontage"))   # a sibling folder
sys.path.insert(0, str(ROOT))
from logit.engine import verify  # noqa: E402

verify._guard_sklearn()
lis = verify.Listener(log=lambda *a: None)
COMMON = {"the", "a", "an", "and", "but", "so", "then", "in", "it's", "its", "his", "he", "she", "this", "that", "on", "by",
          "for", "at", "of", "to", "with", "now", "and", "after", "about", "days", "months", "doctors", "police", "his",
          "march", "june", "may", "january", "february", "september", "november", "december", "august", "july", "april",
          "october", "new", "last", "today", "workers", "chairman", "prosecutors", "reporters", "shares", "six", "ten",
          "british", "german", "us", "nineteen", "twenty", "two", "one", "it", "they", "we", "you", "i", "not", "no",
          "psychiatrists", "summer", "months", "defects", "ex-engineers", "rival", "during", "money", "the", "his"}


def letters(y, sr):
    if len(y) < int(0.05 * sr):
        return ""
    x = lis.proc(librosa.resample(y, orig_sr=sr, target_sr=16000), sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        ids = lis.model(x).logits[0].argmax(-1)
    return lis.proc.decode(ids)


def interesting(w, first):
    raw = w["shown"].strip(".,?!:;'\"“”")
    sp = w.get("spoken", w["shown"])
    if not raw:
        return False
    if sp.strip(".,?!") != w["shown"].strip(".,?!"):
        return True                                  # a respelling or a number written as words
    if re.search(r"\d", raw) or raw.isupper() and len(raw) >= 2:
        return True
    if raw[0].isupper() and not first and raw.lower() not in COMMON:
        return True
    return raw.lower() in ("ai", "gta", "bt", "gpt", "dax", "ceo", "coo") or raw == "IT"   # not the pronoun "it"


for ep in sys.argv[1:]:
    b = ROOT / "logit" / "build" / ep
    t = json.loads((b / "timing.json").read_text(encoding="utf-8"))
    y, sr = sf.read(b / "narration.wav", dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    out = [f"# {ep} pronunciation audit (wav2vec2 letters, no language model)"]
    for seg in t["segments"]:
        words = seg["words"]
        for i, w in enumerate(words):
            prev = words[i - 1]["shown"] if i else "."
            first = i == 0 or prev.endswith((".", "?", "!", ":"))
            if not interesting(w, first):
                continue
            a = max(0, int((w["start"] - 0.04) * sr))
            e = min(len(y), int((w["end"] + 0.04) * sr))
            ctx_a = max(0, int((words[max(0, i - 1)]["start"]) * sr))
            ctx_e = min(len(y), int((words[min(len(words) - 1, i + 1)]["end"]) * sr))
            out.append(f"{seg['id']:10s} {w['start']:6.2f}  {w['shown']:22s} spoken={w.get('spoken', w['shown']):22s} "
                       f"word=[{letters(y[a:e], sr)}]  ctx=[{letters(y[ctx_a:ctx_e], sr)}]")
    txt = "\n".join(out)
    (b / "pron_audit.txt").write_text(txt, encoding="utf-8")
    print(txt)
