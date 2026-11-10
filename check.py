"""Check one voice take against its script (shortcut for `python vc.py check`).

    python check.py take.wav "The line the voice was given."
    python check.py take.wav "Spoken form, like rel-you." --shown "On-screen form, like ReLU."
    python check.py take.wav script.txt --lexicon names.yaml

Prints one JSON report on stdout (progress goes to stderr) and exits 0 when the take passes, 1 when it fails a check,
2 when the check itself could not run. The models run on the CPU and download on first use (about 2.2 GB, plus
1.3 GB for the phoneme model when a lexicon entry has IPA)."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, str(Path(__file__).resolve().parent / "vc.py"), "check", *sys.argv[1:]]))
