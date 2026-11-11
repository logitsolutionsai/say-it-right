"""say-it-right: make a voice-over say every word right, names included. `python vc.py --help` for the commands.

Runs itself inside the skill's own environment when there is one (made by `python install.py`), so any Python can
call it. Standard library only; install.py imports `venv_dir` and the stamp from here, so both find the same place."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAMP = "say-it-right.stamp"        # a file in the environment: the hash of the requirements it was installed from


def venv_python(venv: Path) -> Path | None:
    for p in (venv / "Scripts" / "python.exe", venv / "bin" / "python"):
        if p.exists():
            return p
    return None


def venv_dir(here: Path = HERE, env=None) -> Path:
    """Where the skill's environment lives. A cloned skill folder keeps it inside, as `.venv`. Claude Code's plugin
    cache gets a new folder with every plugin version, so there (or when VOICECHECK_VENV is set) the environment is
    per user and survives updates: $VOICECHECK_VENV, else $CLAUDE_PLUGIN_DATA/venv, else ~/.say-it-right/venv."""
    env = os.environ if env is None else env
    if env.get("VOICECHECK_VENV"):
        return Path(env["VOICECHECK_VENV"]).expanduser()
    if "/plugins/cache/" not in here.as_posix():
        return here / ".venv"
    places = [Path(env["CLAUDE_PLUGIN_DATA"]) / "venv"] if env.get("CLAUDE_PLUGIN_DATA") else []
    places.append(Path.home() / ".say-it-right" / "venv")
    # Claude Code sets CLAUDE_PLUGIN_DATA for hooks but not for the Bash tool: an environment that exists wins.
    return next((p for p in places if venv_python(p)), places[0])


def requirements_hash(here: Path = HERE) -> str:
    """requirements.txt without its comments and blank lines, hashed: what an environment was installed from."""
    try:
        lines = (here / "requirements.txt").read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    wanted = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    return hashlib.sha256("\n".join(wanted).encode("utf-8")).hexdigest()


def stale(venv: Path, here: Path = HERE) -> str | None:
    """What to tell the user when the environment was installed from other requirements than this folder's (the
    skill was updated since), else None. An environment without a stamp predates stamps and counts as current."""
    try:
        stamp = (venv / STAMP).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if stamp == requirements_hash(here):
        return None
    return (f"say-it-right was updated, and its environment ({venv}) is from the version before: run  "
            f"python \"{here / 'install.py'}\"  again (it installs only what changed)")


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    venv = venv_dir()
    python = venv_python(venv)
    if python is not None and os.environ.get("VOICECHECK_IN_VENV") != "1":
        if stale(venv):
            _fail(stale(venv))
        if Path(sys.prefix).resolve() != venv.resolve():
            env = dict(os.environ, VOICECHECK_IN_VENV="1")
            sys.exit(subprocess.call([str(python), str(HERE / "vc.py"), *sys.argv[1:]], env=env))
    try:
        import numpy  # noqa: F401
        import yaml  # noqa: F401
    except ImportError:
        _fail(f"say-it-right isn't installed yet: run  python \"{HERE / 'install.py'}\"")
    sys.path.insert(0, str(HERE))
    from voicecheck.cli import main
    sys.exit(main())
