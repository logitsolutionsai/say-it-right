"""Whole-sentence audit: decode each sentence of narration.wav with the wav2vec2 listener (no language model) and align the
letters to the scripted spoken words; list words that are missing or garbled. Usage: word_audit.py TS-06 [TS-07 ...]"""
import difflib
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


def letters(y, sr):
    x = lis.proc(librosa.resample(y, orig_sr=sr, target_sr=16000), sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        ids = lis.model(x).logits[0].argmax(-1)
    return lis.proc.decode(ids)


def norm(s):
    return re.sub(r"[^A-Z' ]", " ", s.upper().replace("-", " ")).split()


for ep in sys.argv[1:]:
    b = ROOT / "logit" / "build" / ep
    t = json.loads((b / "timing.json").read_text(encoding="utf-8"))
    y, sr = sf.read(b / "narration.wav", dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    rep = [f"# {ep} word audit"]
    for seg in t["segments"]:
        sent = []
        for w in seg["words"]:
            sent.append(w)
            if w.get("sent_end") or w is seg["words"][-1]:
                a, e = int(max(0, sent[0]["start"] - 0.05) * sr), int(min(len(y) / sr, sent[-1]["end"] + 0.08) * sr)
                heard = norm(letters(y[a:e], sr))
                want = norm(" ".join(x.get("spoken", x["shown"]) for x in sent))
                sm = difflib.SequenceMatcher(a=want, b=heard, autojunk=False)
                bad = []
                for op, i1, i2, j1, j2 in sm.get_opcodes():
                    if op == "equal":
                        continue
                    ws, hs = " ".join(want[i1:i2]), " ".join(heard[j1:j2])
                    if ws and difflib.SequenceMatcher(a=ws.replace(" ", ""), b=hs.replace(" ", "")).ratio() >= 0.75:
                        continue                         # split/joined words ("START UP"), near spellings
                    bad.append(f"{ws or '∅'} -> {hs or '∅'}")
                if bad:
                    rep.append(f"{seg['id']:10s} {sent[0]['start']:6.2f} | " + " ; ".join(bad) + "   || " + " ".join(heard))
                sent = []
    txt = "\n".join(rep)
    (b / "word_audit.txt").write_text(txt, encoding="utf-8")
    print(txt)
