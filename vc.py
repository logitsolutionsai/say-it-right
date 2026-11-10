"""say-it-right: make a voice-over say every word right, names included. `python vc.py --help` for the commands.

Runs itself inside this folder's .venv when there is one (made by `python install.py`), so any Python can call it."""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _venv_python():
    for p in (HERE / ".venv" / "Scripts" / "python.exe", HERE / ".venv" / "bin" / "python"):
        if p.exists():
            return p
    return None


if __name__ == "__main__":
    venv = _venv_python()
    inside = os.environ.get("VOICECHECK_IN_VENV") == "1" or (
        venv is not None and Path(sys.prefix).resolve() == venv.parent.parent.resolve())
    if venv is not None and not inside:
        env = dict(os.environ, VOICECHECK_IN_VENV="1")
        sys.exit(subprocess.call([str(venv), str(HERE / "vc.py"), *sys.argv[1:]], env=env))
    try:
        import numpy  # noqa: F401
        import yaml  # noqa: F401
    except ImportError:
        sys.exit(f"say-it-right isn't installed yet: run  python {HERE / 'install.py'}")
    sys.path.insert(0, str(HERE))
    from voicecheck.cli import main
    sys.exit(main())
