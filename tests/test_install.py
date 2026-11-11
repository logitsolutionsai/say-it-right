"""Where the skill's environment lives, when it counts as out of date, and which Python builds it.
No network and no real environment: the file system is tmp_path, and uv and other interpreters are faked."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

import install
import vc

ROOT = Path(__file__).resolve().parents[1]


def fake_venv(folder: Path, stamp: str | None = None) -> Path:
    for exe in ("Scripts/python.exe", "bin/python"):
        (folder / exe).parent.mkdir(parents=True, exist_ok=True)
        (folder / exe).write_text("", encoding="utf-8")
    (folder / "pyvenv.cfg").write_text("", encoding="utf-8")
    if stamp is not None:
        (folder / vc.STAMP).write_text(stamp, encoding="utf-8")
    return folder


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    return tmp_path / "home"


# ---- where the environment lives

def test_a_cloned_skill_folder_keeps_its_environment_inside(tmp_path, home):
    here = tmp_path / ".claude" / "skills" / "say-it-right"
    assert vc.venv_dir(here, {}) == here / ".venv"
    assert vc.venv_dir(here, {"CLAUDE_PLUGIN_DATA": str(tmp_path / "data")}) == here / ".venv"


def test_the_plugin_cache_gets_a_per_user_environment(tmp_path, home):
    here = tmp_path / ".claude" / "plugins" / "cache" / "say-it-right" / "say-it-right" / "0.2.0"
    assert vc.venv_dir(here, {}) == home / ".say-it-right" / "venv"
    newer = here.with_name("0.3.0")
    assert vc.venv_dir(newer, {}) == vc.venv_dir(here, {})          # the point: it survives a plugin update


def test_voicecheck_venv_wins_everywhere(tmp_path, home):
    env = {"VOICECHECK_VENV": str(tmp_path / "mine"), "CLAUDE_PLUGIN_DATA": str(tmp_path / "data")}
    assert vc.venv_dir(tmp_path / "skills" / "say-it-right", env) == tmp_path / "mine"
    assert vc.venv_dir(tmp_path / "plugins" / "cache" / "m" / "p" / "1", env) == tmp_path / "mine"


def test_claude_plugin_data_is_used_in_the_plugin_cache(tmp_path, home):
    here = tmp_path / "plugins" / "cache" / "m" / "p" / "1"
    env = {"CLAUDE_PLUGIN_DATA": str(tmp_path / "data")}
    assert vc.venv_dir(here, env) == tmp_path / "data" / "venv"
    # Claude Code gives hooks that variable and the Bash tool none: both must reach the environment that exists
    fake_venv(home / ".say-it-right" / "venv")
    assert vc.venv_dir(here, env) == vc.venv_dir(here, {}) == home / ".say-it-right" / "venv"


# ---- the stamp

def test_stamp_follows_the_requirements_not_the_comments(tmp_path):
    (tmp_path / "requirements.txt").write_text("# tested 2026\nnumpy>=1.26\n\ntorch>=2.4\n", encoding="utf-8")
    before = vc.requirements_hash(tmp_path)
    (tmp_path / "requirements.txt").write_text("# tested 2027\r\nnumpy>=1.26\r\ntorch>=2.4\r\n", encoding="utf-8")
    assert vc.requirements_hash(tmp_path) == before
    (tmp_path / "requirements.txt").write_text("numpy>=1.26\ntorch>=2.9\n", encoding="utf-8")
    assert vc.requirements_hash(tmp_path) != before


def test_stale_environment_says_to_run_install_again(tmp_path):
    (tmp_path / "requirements.txt").write_text("numpy>=1.26\n", encoding="utf-8")
    venv = fake_venv(tmp_path / "venv", stamp=vc.requirements_hash(tmp_path))
    assert vc.stale(venv, tmp_path) is None
    (tmp_path / "requirements.txt").write_text("numpy>=2\n", encoding="utf-8")
    msg = vc.stale(venv, tmp_path)
    assert "was updated" in msg and str(tmp_path / "install.py") in msg and str(venv) in msg
    (venv / vc.STAMP).unlink()                       # an environment from before stamps existed still works
    assert vc.stale(venv, tmp_path) is None


def test_vc_stops_on_a_stale_environment_before_running_anything(tmp_path):
    venv = fake_venv(tmp_path / "venv", stamp="from an older version")
    env = {k: v for k, v in os.environ.items() if k != "VOICECHECK_IN_VENV"}
    r = subprocess.run([sys.executable, str(ROOT / "vc.py"), "--help"], capture_output=True, text=True,
                       encoding="utf-8", env=dict(env, VOICECHECK_VENV=str(venv)))
    assert r.returncode == 2 and "was updated" in r.stderr and "install.py" in r.stderr and not r.stdout


# ---- which Python builds the environment

def too_new(monkeypatch, which, run=None):
    def no_run(*a, **k):
        raise AssertionError("no interpreter should be started")
    monkeypatch.setattr(sys, "version_info", (3, 15, 0, "final", 0))
    monkeypatch.setattr(install.shutil, "which", which)
    monkeypatch.setattr(install.subprocess, "run", run or no_run)


def test_a_tested_python_builds_the_environment_itself(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 12, 4, "final", 0))
    monkeypatch.setattr(install.shutil, "which", lambda name: "/bin/uv" if name == "uv" else None)
    assert install.pick_python() == sys.executable


def test_too_new_python_with_uv_lets_uv_find_or_fetch_one(monkeypatch):
    too_new(monkeypatch, which=lambda name: "/bin/uv" if name == "uv" else None)
    assert install.pick_python() == "3.13"


def test_too_new_python_without_uv_uses_another_installed_one(monkeypatch):
    asked = []

    def run(cmd, **kw):
        asked.append(cmd[:2] if cmd[0] == "py" else cmd[:1])
        newest = cmd[:2] == ["py", "-3.13"] or cmd[0] == "python3.13"
        return subprocess.CompletedProcess(cmd, 1 if newest else 0, stdout="" if newest else "/opt/py312/python\n")

    too_new(monkeypatch, which=lambda name: None if name == "uv" else f"/bin/{name}", run=run)
    assert install.pick_python() == "/opt/py312/python"
    assert len(asked) == 2                           # 3.13 isn't there, 3.12 is: it stops looking


def test_too_new_python_alone_says_what_to_install(monkeypatch):
    too_new(monkeypatch, which=lambda name: None)
    with pytest.raises(SystemExit) as e:
        install.pick_python()
    assert "uv" in str(e.value) and "3.13" in str(e.value)


# ---- a failed or broken environment doesn't stay in the way

def test_a_failed_venv_is_removed_and_says_what_to_install(tmp_path, monkeypatch):
    venv = tmp_path / "venv"

    def debian_without_python3_venv(cmd, check=True):
        fake_venv(venv)                              # what `python -m venv` leaves there when ensurepip is missing
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(install, "VENV", venv)
    monkeypatch.setattr(install, "run", debian_without_python3_venv)
    monkeypatch.setattr(install.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as e:
        install.create_venv()
    assert "python3-venv" in str(e.value) and "uv" in str(e.value)
    assert not venv.exists()


def test_only_a_virtual_environment_is_ever_deleted(tmp_path, monkeypatch):
    precious = tmp_path / "documents"
    precious.mkdir()
    (precious / "thesis.txt").write_text("years of work", encoding="utf-8")
    monkeypatch.setattr(install, "VENV", precious)   # VOICECHECK_VENV pointed at the wrong folder
    with pytest.raises(SystemExit) as e:
        install.discard()
    assert "not a virtual environment" in str(e.value) and (precious / "thesis.txt").exists()
    monkeypatch.setattr(install, "VENV", fake_venv(tmp_path / "venv"))
    install.discard()
    assert not (tmp_path / "venv").exists()
