"""Set up say-it-right: a virtual environment with everything it needs. Standard library only.

    python install.py              # create the environment and install the requirements
    python install.py --models     # ...and download the three models now (~3.5 GB) instead of on first use
    python install.py --check      # only test an existing install, and say where it is

Afterwards `python vc.py <command>` works with any Python: vc.py runs itself inside the environment.
- The environment is `.venv` in this folder. In Claude Code's plugin cache, where every plugin update brings a new
  folder, it is ~/.say-it-right/venv instead, and VOICECHECK_VENV puts it anywhere (the rule is vc.venv_dir).
- With uv installed, uv makes the environment and fetches Python 3.13 itself when it has to. Without uv this needs
  Python 3.10-3.13 (the tested versions); a newer default is fine if `py -3.13` (Windows) or python3.1x exists.
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
sys.path.insert(0, str(HERE))
import vc  # noqa: E402  (where the environment is, and its stamp: vc.py has to find the same one)

VENV = vc.venv_dir(HERE)
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


def pick_python() -> str:
    """What to build the environment with: this interpreter when it is a tested version; otherwise, with uv, a bare
    version (uv finds an installed one or downloads it); otherwise another interpreter installed here."""
    if (3, 10) <= sys.version_info[:2] <= (3, 13):
        return sys.executable
    if shutil.which("uv"):
        return "3.13"
    candidates = [["py", f"-3.{m}"] for m in (13, 12, 11, 10)] if os.name == "nt" else \
        [[f"python3.{m}"] for m in (13, 12, 11, 10)]
    for c in candidates:
        if shutil.which(c[0]):
            r = subprocess.run(c + ["-c", "import sys; print(sys.executable)"], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
    sys.exit(f"Python {sys.version.split()[0]} is outside 3.10-3.13 and no other version was found. Install uv "
             "(https://docs.astral.sh/uv/, it fetches Python itself) or Python 3.13, then run this again.")


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


def usable() -> bool:
    """The environment has a Python that runs, and something to install into it with: uv, or its own pip."""
    def works(*args) -> bool:
        try:
            return subprocess.run([str(venv_python()), *args], capture_output=True).returncode == 0
        except OSError:
            return False
    return venv_python().exists() and works("-c", "pass") and (
        bool(shutil.which("uv")) or works("-m", "pip", "--version"))


def discard(ours: bool = False) -> None:
    """Delete the environment's folder, but only one this run made (`ours`) or one that is a virtual environment
    or empty: VOICECHECK_VENV could point at anything."""
    is_venv = (VENV / "pyvenv.cfg").exists() or (VENV.is_dir() and not any(VENV.iterdir()))
    if VENV.exists() and not (ours or is_venv):
        sys.exit(f"{VENV} exists and is not a virtual environment: move it away, or point VOICECHECK_VENV elsewhere.")
    shutil.rmtree(VENV, ignore_errors=True)


def create_venv() -> None:
    python, uv, ours = pick_python(), shutil.which("uv"), not VENV.exists()
    VENV.parent.mkdir(parents=True, exist_ok=True)
    made = run([uv, "venv", "--python", python, VENV] if uv else [python, "-m", "venv", VENV], check=False)
    if made.returncode != 0 or not usable():
        discard(ours)       # Debian without python3-venv leaves bin/python and no pip: the next run would trust it
        sys.exit("couldn't create the virtual environment (the reason is above). Install uv "
                 "(https://docs.astral.sh/uv/), which needs nothing else; or on Debian/Ubuntu run "
                 "`sudo apt install python3-venv`. Then run this again.")
    if not uv:
        run([venv_python(), "-m", "pip", "install", "--upgrade", "pip"], check=False)


def install() -> None:
    if VENV.exists() and not usable():
        say(f"{VENV} has no working Python, or nothing to install with: making it again")
        discard()
    if not VENV.exists():
        create_venv()
    elif (VENV / vc.STAMP).exists() and not vc.stale(VENV, HERE) and probe()[0]:
        say(f"already set up in {VENV}")
        return
    if sys.platform.startswith("linux"):
        pip("torch", "--index-url", "https://download.pytorch.org/whl/cpu")
    pip("-r", HERE / "requirements.txt")       # into an existing environment this installs only what changed
    ok, out = probe()
    if not ok and ("Application Control" in out or "DLL load failed" in out or "WinError" in out):
        say("a package's DLL was blocked (Windows Smart App Control?): installing the tested versions instead")
        pip("-r", HERE / "requirements-lock.txt")
        ok, out = probe()
    if not ok:
        sys.exit(f"the environment doesn't import cleanly:\n{out}")
    (VENV / vc.STAMP).write_text(vc.requirements_hash(HERE) + "\n", encoding="utf-8")
    say(f"dependencies ready in {VENV} (torch {out.splitlines()[-1]})")


def download_models() -> None:
    code = ("from huggingface_hub import snapshot_download\n"
            f"for name, pats in {MODELS!r}.items():\n"
            "    print('downloading', name, flush=True)\n"
            "    snapshot_download(name, allow_patterns=pats)\n")
    run([venv_python(), "-c", code])


def check() -> bool:
    if not venv_python().exists():
        ok, out = False, "no environment yet: run python install.py"
    elif vc.stale(VENV, HERE):
        ok, out = False, "the skill was updated since it was set up: run python install.py again"
    else:
        ok, out = probe()
        out = f"torch {out.splitlines()[-1]}" if ok else out
    print(f"{'ok' if ok else 'not ready'}: {out.splitlines()[-1] if out else ''} (environment: {VENV})")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="store_true", help="download the models now (~3.5 GB)")
    ap.add_argument("--check", action="store_true", help="only test an existing install, and say where it is")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if check() else 1)
    install()
    if a.models:
        download_models()
    r = subprocess.run([str(venv_python()), str(HERE / "vc.py"), "respell", "ʃɪˈvɔːn", "--name"],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0 or "Shiv-awn" not in r.stdout:
        sys.exit(f"self-test failed:\n{r.stdout}{r.stderr}")
    say(f"ready. Try:  python \"{HERE / 'vc.py'}\" --help")


if __name__ == "__main__":
    main()
