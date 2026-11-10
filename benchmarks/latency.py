"""How long a check takes: one cold call, a batch, and repeated calls against `vc.py serve`.

    python benchmarks/latency.py takes.json [--quantize [checker,listener,phonemes]]

takes.json: a list of {"audio": ..., "script": ...} (paths relative to the file), optionally with "lexicon". Prints a
JSON summary: wall seconds per scenario, and the per-step `timing` the reports carry. Run it on an idle machine.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VC = [sys.executable, str(ROOT / "vc.py")]


def call(args, env) -> tuple[float, dict]:
    t0 = time.perf_counter()
    r = subprocess.run(VC + args, capture_output=True, text=True, encoding="utf-8", env=env)
    wall = time.perf_counter() - t0
    if r.returncode not in (0, 1):
        raise RuntimeError(r.stderr[-2000:])
    return wall, json.loads(r.stdout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("takes")
    ap.add_argument("--quantize", nargs="?", const="all", help="int8: all models (no value) or a comma list of roles")
    ap.add_argument("--port", type=int, default=8799)
    a = ap.parse_args()
    base = Path(a.takes).resolve().parent
    takes = json.loads(Path(a.takes).read_text(encoding="utf-8"))
    for t in takes:
        t["audio"] = str((base / t["audio"]).resolve())
    lex = sorted({str((base / t["lexicon"]).resolve()) for t in takes if t.get("lexicon")})
    lexargs = [x for p in lex for x in ("--lexicon", p)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", VOICECHECK_PORT=str(a.port))
    if a.quantize:
        cfg = Path(tempfile.mkdtemp()) / "q.yaml"
        roles = "true" if a.quantize == "all" else "[" + ", ".join(a.quantize.split(",")) + "]"
        cfg.write_text(f"quantize: {roles}\n", encoding="utf-8")
        env["VOICECHECK_CONFIG"] = str(cfg)
    out = {"takes": len(takes), "quantize": a.quantize, "audio_seconds": []}

    cold = []                                              # 1. one process per take (the old way)
    for t in takes:
        wall, rep = call(["check", t["audio"], t["script"], "--local", *lexargs], env)
        cold.append({"wall": round(wall, 2), "timing": rep["timing"], "passed": rep["passed"], "beams": rep["beams"]})
        out["audio_seconds"].append(rep["timing"]["audio_seconds"])
    out["cold_single"] = {"mean_wall": round(statistics.mean(c["wall"] for c in cold), 2), "runs": cold}

    batch = Path(tempfile.mkdtemp()) / "batch.json"        # 2. one process for all takes
    batch.write_text(json.dumps([{"audio": t["audio"], "script": t["script"]} for t in takes]), encoding="utf-8")
    wall, rep = call(["check", "--batch", str(batch), "--local", *lexargs], env)
    out["batch"] = {"wall": round(wall, 2), "per_take": round(wall / len(takes), 2),
                    "steps": [r["timing"] for r in rep["reports"]]}

    srv = subprocess.Popen(VC + ["serve", "--port", str(a.port)], env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)              # 3. a warm server
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=0.5)
                break
            except Exception:
                time.sleep(0.5)
        warm = []
        for i, t in enumerate(takes + takes[:1]):          # the first call loads the models; the rest are warm
            wall, rep = call(["check", t["audio"], t["script"], *lexargs], env)
            warm.append({"wall": round(wall, 2), "timing": rep["timing"]})
        out["server"] = {"first_call": warm[0]["wall"], "warm_mean_wall": round(statistics.mean(w["wall"] for w in
                                                                                                 warm[1:]), 2),
                         "runs": warm}
    finally:
        srv.terminate()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
