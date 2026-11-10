"""The command line, for the commands that need no model (check itself runs in tests/test_models.py)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def vc(*args, **kw):
    r = subprocess.run([sys.executable, str(ROOT / "vc.py"), *args], capture_output=True, text=True,
                       encoding="utf-8", cwd=kw.get("cwd", ROOT), env=kw.get("env"))
    return r.returncode, r.stdout, r.stderr


def lexicon_file(tmp_path):
    f = tmp_path / "names.yaml"
    f.write_text("Szegedy:\n  say: Seg-eh-dee\n  ipa: ˈsɛɡɛdi\nReLU: rel-you\n", encoding="utf-8")
    return str(f)


def test_render_formats(tmp_path):
    lex = lexicon_file(tmp_path)
    code, out, _ = vc("render", "Szegedy used ReLU.", "--lexicon", lex)
    assert code == 0 and out.strip() == "Seg-eh-dee used rel-you."
    code, out, _ = vc("render", "Szegedy used ReLU.", "--lexicon", lex, "-f", "json")
    data = json.loads(out)
    assert data["shown"] == "Szegedy used ReLU." and data["terms"][0]["ipa"] == "ˈsɛɡɛdi"


def test_render_reads_a_script_file(tmp_path):
    lex = lexicon_file(tmp_path)
    script = tmp_path / "line.txt"
    script.write_text("Ask Szegedy.", encoding="utf-8")
    code, out, _ = vc("render", str(script), "--lexicon", lex, "-f", "ssml")
    assert code == 0 and '<phoneme alphabet="ipa" ph="ˈsɛgɛdi">Szegedy</phoneme>' in out


def test_respell_and_pls(tmp_path):
    code, out, _ = vc("respell", "ʃɪˈvɔːn", "--name")
    assert code == 0 and json.loads(out)["candidates"][0] == "Shiv-awn"
    code, out, _ = vc("export-pls", "--lexicon", lexicon_file(tmp_path))
    assert code == 0 and out.startswith("<?xml") and "<alias>rel-you</alias>" in out


def test_scan_is_json_on_stdout(tmp_path):
    code, out, _ = vc("scan", "Then Siobhan met Szegedy in 1959.", "--fast", "--lexicon", lexicon_file(tmp_path))
    data = json.loads(out)
    assert code == 0 and [e["word"] for e in data["research"]] == ["Siobhan"]
    assert data["numbers"][0]["readings"][0] == "nineteen fifty nine"


def test_bad_input_exits_2(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n", encoding="utf-8")
    code, out, err = vc("render", "x", "--lexicon", str(bad))
    assert code == 2 and "error:" in err and out == ""
