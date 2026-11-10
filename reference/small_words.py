"""Small words a take can swallow without any model noticing: decode each one with its two neighbours, in a narrow window, and report where it isn't heard.
Whole-sentence decoding missed 1.1's "zero [to] two hundred"; a three-word window catches it.

    python -m logit.lecture02.runs.small_words [ids]
"""
from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys

import numpy as np

from logit.lecture.audio_util import tool_env
from logit.lecture.paths import BUILD, CONTENT
from logit.lecture.render import Slot

def heard_ok(core, left, right, heard):
    """Is the small word in the decoded window? Yes when it is decoded as a word of its own, or when the window's
    letters fit the three words better than the two neighbours alone. (Fixed 2026-11-10: the old test, similarity
    > 0.9 with the three words, passed "parameters [to] gradients" at 0.95: a dropped two-letter word costs too
    few letters next to long neighbours for any fixed threshold.)"""
    if core in heard.split():
        return True
    h = heard.replace(" ", "")

    def fits(s):
        return difflib.SequenceMatcher(None, s, h, autojunk=False).ratio()
    return fits("".join(left + [core] + right)) > fits("".join(left + right))


WATCH = {"to", "by", "of", "not", "no", "than", "per", "from", "into", "or", "minus", "plus", "times", "with", "without", "half", "each", "only", "never"}
SR = 16000


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    scenes = json.loads((CONTENT / "data" / "scenes.json").read_text(encoding="utf-8"))
    name = "facebook/wav2vec2-large-960h-lv60-self"
    n, bad = 0, []
    with Slot(print):
        proc = Wav2Vec2Processor.from_pretrained(name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Wav2Vec2ForCTC.from_pretrained(name).eval().to(device)

        def decode(y):
            x = proc(y, sampling_rate=SR, return_tensors="pt").input_values.to(device)
            with torch.no_grad():
                ids = model(x).logits.argmax(-1)
            return proc.batch_decode(ids)[0].lower()
        for s in scenes:
            if only and s["id"] not in only:
                continue
            t = json.loads((CONTENT / "voice" / f"{s['id']}.json").read_text(encoding="utf-8"))
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(CONTENT / "voice" / t["audio"]), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True, env=tool_env()).stdout
            y = np.frombuffer(raw, dtype=np.float32)
            W = t["words"]
            for k, w in enumerate(W):
                core = re.sub(r"\W", "", w["w"].lower())
                if core not in WATCH or k == 0 or k == len(W) - 1:
                    continue
                if W[k - 1]["w"][-1] in ".?!:;" or w["w"][-1] in ".?!:;":     # a sentence break: the neighbour isn't a neighbour
                    continue
                n += 1
                i, j = int((W[k - 1]["start"] - .05) * SR), int((W[k + 1]["end"] + .08) * SR)
                heard = decode(y[max(0, i):j])
                left = re.sub(r"[^a-z ]", " ", W[k - 1]["w"].lower().replace("-", " ")).split()
                right = re.sub(r"[^a-z ]", " ", W[k + 1]["w"].lower().replace("-", " ")).split()
                want = " ".join(left + [core] + right)
                if not heard_ok(core, left, right, heard):
                    bad.append((s["id"], want, heard))
    (BUILD / "small_words.json").write_text(json.dumps(bad, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{n} small words checked; not heard in {len(bad)}:")
    for b in bad:
        print(f"{b[0]:>5} want [{b[1]}] heard [{b[2]}]")


if __name__ == "__main__":
    main()
