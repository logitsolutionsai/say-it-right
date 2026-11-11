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
    assert code == 0 and '<phoneme alphabet="ipa" ph="ˈsɛ.gɛ.di">Szegedy</phoneme>' in out


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


# ---- bug hunt 2026-11-11: exit 1 means "the take failed", so bad input must never exit 1

def one_line_error(err):
    return err.strip().startswith("error:") and "Traceback" not in err and len(err.strip().splitlines()) == 1


def test_bad_yaml_and_bad_lexicon_values_exit_2_with_one_line(tmp_path):
    import os
    for name, text in [("syntax.yaml", "A: [unclosed\n"), ("number.yaml", "A: {ipa: 5}\n"), ("empty.yaml", "ReLU:\n"),
                       ("bool.yaml", "NO: no\n")]:
        f = tmp_path / name
        f.write_text(text, encoding="utf-8")
        for cmd in (["render", "x"], ["scan", "x", "--fast"], ["export-pls"]):
            code, out, err = vc(*cmd, "--lexicon", str(f))
            assert code == 2 and out == "" and one_line_error(err), (name, cmd, code, err)
    cfg = tmp_path / "voice.yaml"                           # the settings file: a year as a key, then broken YAML
    cfg.write_text("pronunciations:\n  1984: nineteen eighty-four\n", encoding="utf-8")
    env = dict(os.environ, VOICECHECK_CONFIG=str(cfg))
    code, out, err = vc("render", "In 1984", env=env)
    assert code == 0 and out.strip() == "In nineteen eighty-four", err              # was an AttributeError, exit 1
    cfg.write_text("pronunciations: [unclosed\n", encoding="utf-8")
    code, out, err = vc("render", "x", env=env)
    assert code == 2 and one_line_error(err) and "YAML" in err, err


def test_a_script_file_that_is_not_there_is_an_error_not_the_script(tmp_path):
    code, out, err = vc("render", str(tmp_path / "nope.txt"))
    assert code == 2 and out == "" and one_line_error(err) and "nope.txt" in err   # was rendered as the script
    code, out, _ = vc("render", "Save it as notes.txt")                             # a sentence is still a script
    assert code == 0 and out.strip() == "Save it as notes.txt"
    code, out, err = vc("scan", "It costs $.50 and " + "1" * 400 + " more.", "--fast")
    assert code == 0 and json.loads(out)["research"] == [], err                     # were a ValueError, OverflowError


def test_a_script_file_with_a_byte_order_mark_or_a_long_path(tmp_path):
    import os
    lex = lexicon_file(tmp_path)
    bom = tmp_path / "bom.txt"
    bom.write_bytes(b"\xef\xbb\xbfSzegedy used ReLU.")
    code, out, _ = vc("render", str(bom), "--lexicon", lex)
    assert code == 0 and out == "Seg-eh-dee used rel-you.\n"                        # was "\ufeffSeg-eh-dee..."
    deep = tmp_path
    for _ in range(6):
        deep = deep / ("d" * 45)
    long = deep / "script.txt"
    wide = ("\\\\?\\" if os.name == "nt" else "") + str(long)                       # Windows opens any path this way
    try:
        os.makedirs(os.path.dirname(wide), exist_ok=True)
        with open(wide, "w", encoding="utf-8") as f:
            f.write("Ask Szegedy.")
    except OSError:
        return                                                                      # this file system has no long paths
    assert len(str(long)) > 260
    code, out, err = vc("render", str(long), "--lexicon", lex)
    assert code == 0 and out.strip() == "Ask Seg-eh-dee.", err                      # was the path, read as the script


def test_batch_files(tmp_path):
    import os

    import pytest

    from voicecheck import cli

    def batch(name, text, raw=False):
        f = tmp_path / name
        f.write_bytes(text if raw else text.encode("utf-8"))
        return cli._batch(str(f))
    (tmp_path / "line.txt").write_text("FROM THE FILE", encoding="utf-8")
    a, b = batch("crlf.tsv", b"\xef\xbb\xbfa.wav\tHello there. \r\nb.wav\tline.txt\r\n", raw=True)
    assert a == {"audio": str(tmp_path / "a.wav"), "script": "Hello there."}        # no BOM in the path, no CR
    assert b["script"] == str(tmp_path / "line.txt")                                # relative to the batch file
    cwd = os.getcwd()
    os.chdir(tmp_path.parent)
    try:
        assert cli._text(b["script"]) == "FROM THE FILE"                            # was the text "line.txt"
    finally:
        os.chdir(cwd)
    assert batch("one.json", '{"audio": "C:\\\\takes\\\\a.wav", "script": "Hi"}')[0]["script"] == "Hi"   # a bare dict
    assert batch("list.yaml", "- {audio: a.wav, script: 'Hi', shown: Hi}\n")[0]["shown"] == "Hi"
    for name, text, why in [("num.yaml", "- {audio: a.wav, script: 1959}\n", "script must be text"),
                            ("bool.yaml", "- {audio: a.wav, script: yes}\n", "script must be text"),
                            ("audio.json", '[{"audio": 5, "script": "x"}]', "audio must be text"),
                            ("scalar.json", "5", "no takes"), ("empty.json", "", "no takes"),
                            ("spaces.tsv", "a.wav    Hello there\n", "expected audio<TAB>script"),
                            ("words.json", '["a.wav"]', "needs audio and script"),
                            ("broken.json", '[{"audio": ', "not valid JSON/YAML")]:
        with pytest.raises(ValueError, match=why):
            batch(name, text)


def test_an_empty_batch_does_not_pass(tmp_path):
    f = tmp_path / "takes.tsv"
    f.write_text("a.wav    Hello there\n", encoding="utf-8")                        # spaces, not a tab: no takes read
    code, out, err = vc("check", "--batch", str(f), "--local")                      # fails before any model loads
    assert code == 2 and out == "" and one_line_error(err), err                     # was {"passed": true}, exit 0
