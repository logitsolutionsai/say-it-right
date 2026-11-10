"""Set up say-it-right in this folder: a virtual environment (.venv) with everything it needs. Standard library only.

    python install.py              # create .venv and install the requirements
    python install.py --models     # ...and download the three models now (~3.5 GB) instead of on first use
    python install.py --check      # only test an existing install

Afterwards `python vc.py <command>` works with any Python: vc.py runs itself inside .venv.
- Needs Python 3.10-3.13 (PyTorch wheels); a newer default is fine if `py -3.13` (Windows) or python3.1x exists.
- Linux gets PyTorch's CPU build (the checker runs on the CPU; the default CUDA build is ~2 GB more).
- Windows: if Smart App Control blocks a freshly released wheel's DLL, the tested versions in requirements-lock.txt
  are installed instead.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv"
MODELS = {"openai/whisper-small.en": ["*.json", "*.txt", "model.safetensors"],
          "facebook/wav2vec2-large-960h-lv60-self": ["*.json", "model.safetensors"],
          "facebook/wav2vec2-lv-60-espeak-cv-ft": ["*.json", "pytorch_model.bin"]}
# Import everything and run librosa once: librosa loads numba (a DLL Windows may block) only on first use.
PROBE = ("import numpy, torch, transformers, librosa, soundfile, num2words, yaml; "
         "librosa.resample(numpy.zeros(2205, dtype='float32'), orig_sr=22050, target_sr=16000); print(torch.__version__)")


def say(msg: str) -> None:
    print(f"[install] {msg}", flush=True)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def pick_python() -> list[str]:
    if (3, 10) <= sys.version_info[:2] <= (3, 13):
        return [sys.executable]
    candidates = [["py", f"-3.{m}"] for m in (13, 12, 11, 10)] if os.name == "nt" else \
        [[f"python3.{m}"] for m in (13, 12, 11, 10)]
    for c in candidates:
        if shutil.which(c[0]) and subprocess.run(c + ["-c", "pass"], capture_output=True).returncode == 0:
            return c
    sys.exit(f"Python {sys.version.split()[0]} is outside 3.10-3.13 and no other version was found; "
             "install Python 3.13 and run this again with it.")


def run(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    say(" ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=check)


def pip(*args) -> None:
    uv = shutil.which("uv")
    if uv:
        run([uv, "pip", "install", "--python", venv_python(), *args])
    else:
        run([venv_python(), "-m", "pip", "install", *args])


def probe() -> tuple[bool, str]:
    r = subprocess.run([str(venv_python()), "-c", PROBE], capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def install() -> None:
    if not venv_python().exists():
        py = pick_python()
        run(py + ["-m", "venv", VENV])
        if not shutil.which("uv"):
            run([venv_python(), "-m", "pip", "install", "--upgrade", "pip"], check=False)
    if sys.platform.startswith("linux"):
        pip("torch", "--index-url", "https://download.pytorch.org/whl/cpu")
    pip("-r", HERE / "requirements.txt")
    ok, out = probe()
    if not ok and ("Application Control" in out or "DLL load failed" in out or "WinError" in out):
        say("a package's DLL was blocked (Windows Smart App Control?): installing the tested versions instead")
        pip("-r", HERE / "requirements-lock.txt")
        ok, out = probe()
    if not ok:
        sys.exit(f"the environment doesn't import cleanly:\n{out}")
    say(f"dependencies ready (torch {out.splitlines()[-1]})")


def download_models() -> None:
    code = ("from huggingface_hub import snapshot_download\n"
            f"for name, pats in {MODELS!r}.items():\n"
            "    print('downloading', name, flush=True)\n"
            "    snapshot_download(name, allow_patterns=pats)\n")
    run([venv_python(), "-c", code])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="store_true", help="download the models now (~3.5 GB)")
    ap.add_argument("--check", action="store_true", help="only test an existing install")
    a = ap.parse_args()
    if a.check:
        ok, out = probe() if venv_python().exists() else (False, "no .venv yet: run python install.py")
        print(("ok: " if ok else "not ready: ") + out.splitlines()[-1] if out else "")
        sys.exit(0 if ok else 1)
    install()
    if a.models:
        download_models()
    r = subprocess.run([str(venv_python()), str(HERE / "vc.py"), "respell", "ʃɪˈvɔːn", "--name"],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0 or "Shiv-awn" not in r.stdout:
        sys.exit(f"self-test failed:\n{r.stdout}{r.stderr}")
    say(f"ready. Try:  python {HERE / 'vc.py'} --help")


if __name__ == "__main__":
    main()
