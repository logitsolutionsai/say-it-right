"""Listen to every voiced scene with Whisper and report the ones that don't match the script (word slips, drops, respellings gone wrong).

    python -m logit.lecture.voice_qa            # writes <build>/voice_qa.json, prints the worst

Takes one slot from the render queue, so it never competes with renders for RAM. Long scenes are transcribed whole
(transformers >= 4.57 decodes past 30 s itself; the old hard 28 s windows could cut a word in two, 2026-11-10)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from logit.engine import verify  # noqa: E402
from logit.lecture.audio_util import tool_env  # noqa: E402
from logit.lecture.paths import BUILD, CONTENT  # noqa: E402
from logit.lecture.render import Slot  # noqa: E402


def main(threshold=0.93):
    sys.stdout.reconfigure(encoding="utf-8")
    scenes = json.loads((CONTENT / "data" / "scenes.json").read_text(encoding="utf-8"))
    res = {}
    path = BUILD / "voice_qa.json"
    if path.exists():
        res = json.loads(path.read_text(encoding="utf-8"))
    with Slot(print):
        checker = None
        for s in scenes:
            t = json.loads((CONTENT / "voice" / f"{s['id']}.json").read_text(encoding="utf-8"))
            if t.get("source") != "elevenlabs" or res.get(s["id"], {}).get("key") == t.get("key"):
                continue
            checker = checker or verify.Checker()
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(CONTENT / "voice" / t["audio"]), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"], capture_output=True, env=tool_env()).stdout
            y = np.frombuffer(raw, dtype=np.float32)
            heard = checker.transcribe(y, 16000)
            score, diff = verify.match(s["vo"], heard, alt=s["spoken"])
            res[s["id"]] = {"key": t.get("key"), "score": round(score, 3), "diff": diff, "heard": heard}
            print(f"{s['id']}: {score:.3f} {diff[:160]}", flush=True)
            path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = sorted(((v["score"], k) for k, v in res.items() if v["score"] < threshold))
    print(f"{len(res)} scenes heard; under {threshold}: {bad}")


if __name__ == "__main__":
    main()
