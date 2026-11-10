"""End to end with the real models, on takes synthesised by Windows' built-in voices (System.Speech).

Off by default (they load ~3.5 GB of models and take a few minutes):

    VOICECHECK_MODEL_TESTS=1 python -m pytest tests/test_models.py -v
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [
    pytest.mark.skipif(os.environ.get("VOICECHECK_MODEL_TESTS") != "1", reason="set VOICECHECK_MODEL_TESTS=1"),
    pytest.mark.skipif(sys.platform != "win32" or not shutil.which("powershell"), reason="needs Windows speech"),
]


def say(text: str, path: Path, voice: str = "Microsoft Zira Desktop") -> Path:
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"try {{ $s.SelectVoice('{voice}') }} catch {{}}; $s.SetOutputToWaveFile('{path}'); "
          f"$s.Speak('{text.replace(chr(39), chr(39) * 2)}'); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    return path


def check(audio: Path, script: str, *extra) -> tuple[int, dict]:
    r = subprocess.run([sys.executable, str(ROOT / "vc.py"), "check", str(audio), script, *extra],
                       capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    return r.returncode, json.loads(r.stdout)                 # stdout must be pure JSON


@pytest.fixture(scope="module")
def lexicon(tmp_path_factory):
    f = tmp_path_factory.mktemp("lex") / "names.yaml"
    f.write_text("Siobhan: {say: Shiv-awn, ipa: ʃɪˈvɔːn}\n", encoding="utf-8")
    return str(f)


def test_a_correct_read_of_numbers_passes(tmp_path):
    code, r = check(say("In 1959 the network had 784 weights.", tmp_path / "n.wav"),
                    "In 1959 the network had 784 weights.")
    assert code == 0 and r["passed"], r["reasons"]


def test_a_dropped_small_word_is_caught(tmp_path):
    code, r = check(say("The model goes from zero two hundred quickly.", tmp_path / "s.wav"),
                    "The model goes from zero to two hundred quickly.")
    assert code == 1 and any(s.startswith("to ") for s in r["slips"])


def test_a_name_said_right_passes_and_said_wrong_fails(tmp_path, lexicon):
    code, r = check(say("Ask Shiv-awn about it.", tmp_path / "good.wav"), "Ask Siobhan about it.", "--lexicon", lexicon)
    assert code == 0 and r["names"][0]["ok"], r["reasons"]
    code, r = check(say("Ask Siobhan about it.", tmp_path / "bad.wav"), "Ask Siobhan about it.", "--lexicon", lexicon)
    assert code == 1 and not r["names"][0]["ok"]


def test_audio_too_short_for_the_script_fails_closed(tmp_path):
    code, r = check(say("Hello.", tmp_path / "short.wav"),
                    "Every take is checked before the video ships, with two different models listening.")
    assert code == 1 and r["listener_measured"] is False
