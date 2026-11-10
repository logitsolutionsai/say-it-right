"""What did the voice actually say? Whisper corrects toward real words; wav2vec2 has no language model and writes the letters it
hears. This cuts every occurrence of each technical word and name out of the scene audio (by ElevenLabs' own word timings) and
decodes it, so a wrong pronunciation shows up as wrong letters.

    python -m logit.lecture.voice_listen      # writes <build>/voice_listen.json and prints term -> heard variants
"""
from __future__ import annotations

import collections
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from logit.lecture.audio_util import tool_env  # noqa: E402
from logit.lecture.paths import BUILD, CONTENT, meta  # noqa: E402
from logit.lecture.render import Slot  # noqa: E402

TERMS = meta.TERMS
PHRASES = meta.PHRASES
# Terms that are also ordinary words in lower case: only their exact spelling counts (He = Kaiming He, Ba = Jimmy Ba).
ORDINARY = ("he", "ba", "eta")


def is_term(base, exact, lower):
    """Audit a scripted word when it is a term as written, or a term in another case unless that lower-case form is
    an ordinary word. (Fixed 2026-11-10: `terms` had lower-cased keys, so `base in terms` matched the pronoun "he"
    and never the name "He".)"""
    return base in exact or (base.lower() in lower and base.lower() not in ORDINARY)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    scenes = json.loads((CONTENT / "data" / "scenes.json").read_text(encoding="utf-8"))
    name = "facebook/wav2vec2-large-960h-lv60-self"
    heard = collections.defaultdict(list)
    with Slot(print):
        proc = Wav2Vec2Processor.from_pretrained(name)
        model = Wav2Vec2ForCTC.from_pretrained(name).eval()
        torch.set_num_threads(6)

        def decode(y):
            x = proc(y, sampling_rate=16000, return_tensors="pt").input_values
            with torch.no_grad():
                ids = model(x).logits.argmax(-1)
            return proc.batch_decode(ids)[0].lower()
        exact = {t.strip() for t in TERMS}
        terms = {t.lower().strip(): t for t in TERMS}
        for s in scenes:
            t = json.loads((CONTENT / "voice" / f"{s['id']}.json").read_text(encoding="utf-8"))
            if t.get("source") != "elevenlabs":
                continue
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(CONTENT / "voice" / t["audio"]), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"], capture_output=True, env=tool_env()).stdout
            y = np.frombuffer(raw, dtype=np.float32)
            words = t["words"]
            clean = [re.sub(r"^\W+|\W+$", "", w["w"]) for w in words]

            def cut(a, b):
                i, j = int(max(0, words[a]["start"] - 0.05) * 16000), int((words[b]["end"] + 0.07) * 16000)
                return decode(y[i:j]) if j - i > 1600 else ""
            for k, w in enumerate(clean):
                base = re.sub(r"'s$", "", w)
                if is_term(base, exact, terms):
                    heard[base].append((s["id"], cut(k, k)))
            for ph in PHRASES:
                pw = ph.split()
                for k in range(len(clean) - len(pw) + 1):
                    if clean[k:k + len(pw)] == pw:
                        heard["«" + ph + "»"].append((s["id"], cut(k, k + len(pw) - 1)))
            print(s["id"], flush=True)
    out = {k: v for k, v in sorted(heard.items())}
    (BUILD / "voice_listen.json").write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print("\n=== term -> what was heard (count) ===")
    for k, v in out.items():
        c = collections.Counter(h for _, h in v)
        print(f"{k}: " + "; ".join(f"{h or '∅'} x{n}" for h, n in c.most_common(7)))


if __name__ == "__main__":
    main()
