"""Letter-level listening check of narration windows (wav2vec2, no language model: it cannot "fill in" a word the way Whisper
does). Usage: listen.py TS-06:24.7:27.5 TS-07:79.3:83.7 ...  (episode:start_s:end_s on build/<ID>/narration.wav).
Used 2026-11-08 to catch TS-06 "from Microsoft" spoken as "for Microsoft" and to confirm every hook and loop line."""
import os
import sys
from pathlib import Path
import librosa, soundfile as sf, torch
ROOT = Path(os.environ.get("OPENMONTAGE", Path(__file__).resolve().parents[2] / "OpenMontage"))   # a sibling folder
sys.path.insert(0, str(ROOT))
from logit.engine import verify
verify._guard_sklearn()
lis = verify.Listener(log=lambda *a: None)
def letters(y, sr):
    x = lis.proc(librosa.resample(y, orig_sr=sr, target_sr=16000), sampling_rate=16000, return_tensors="pt").input_values
    with torch.no_grad():
        ids = lis.model(x).logits[0].argmax(-1)
    return lis.proc.decode(ids)
for spec in sys.argv[1:]:
    ep, a, b = spec.split(":")
    y, sr = sf.read(ROOT / "logit" / "build" / ep / "narration.wav")
    if y.ndim > 1:
        y = y.mean(1)
    print(ep, a, b, "|", letters(y[int(float(a) * sr):int(float(b) * sr)], sr))
