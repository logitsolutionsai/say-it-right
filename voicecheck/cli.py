"""say-it-right command line. Results go to stdout (JSON unless noted), progress to stderr.

    python vc.py scan script.txt                      which words need researching
    python vc.py lookup "Christian Szegedy"           how sources say a name (Wikipedia, Wiktionary, Wikidata audio)
    python vc.py respell "ˈsɛɡɛdi" --name             IPA -> respellings a plain-text TTS reads
    python vc.py hear reference.ogg                   the IPA a recording actually says (a native speaker, say)
    python vc.py render script.txt -f ssml            the script with the lexicon applied, in an engine's syntax
    python vc.py export-pls --lang en-US              the lexicon as a PLS file (ElevenLabs, Azure, Polly)
    python vc.py check take.wav script.txt            the take against the script: words, names, pace, tone
    python vc.py check --batch takes.json             many takes, models loaded once
    python vc.py serve                                keep the models loaded; later checks use it automatically

Exit codes: 0 passed, 1 the take failed a check, 2 the command itself failed (bad input, missing model).
A script argument can be the text itself or a path to a .txt file. --lexicon FILE (repeatable) adds lexicon files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _out(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=lambda o: o.item() if isinstance(o, np.generic)
                     else str(o)))


def _text(arg: str) -> str:
    p = Path(arg)
    if len(arg) < 260 and p.suffix.lower() in (".txt", ".md") and p.exists():
        return p.read_text(encoding="utf-8").strip()
    return arg


def _table(args):
    from . import lexicon
    return lexicon.load(args.lexicon) if args.lexicon else lexicon.load()


def _audio(path: str):
    import soundfile as sf
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        import librosa                                     # mp3/m4a and friends via audioread
        y, sr = librosa.load(path, sr=None, mono=True)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.ascontiguousarray(y, dtype=np.float32), int(sr)


# ---------------------------------------------------------------- commands

def cmd_scan(args) -> int:
    from .scan import scan
    _out(scan(_text(args.script), _table(args), rarity=not args.fast))
    return 0


def cmd_lookup(args) -> int:
    from .lookup import lookup
    _out(lookup(args.name, lang=args.lang, context=args.context))
    return 0


def cmd_respell(args) -> int:
    from . import ipa
    _out({"ipa": args.ipa, "candidates": ipa.candidates(args.ipa, name=args.name)})
    return 0


def cmd_hear(args) -> int:
    from .phonemes import PhonemeEar
    path = args.audio
    if path.startswith(("http://", "https://")):
        from .lookup import download
        path = download(path)
    y, sr = _audio(path)
    ear = PhonemeEar()
    phones = ear.hear(y, sr, args.start, args.end)
    _out({"audio": args.audio, "ipa": "".join(phones), "phones": phones})
    return 0


def cmd_render(args) -> int:
    from .engines import to_engine
    from .lexicon import render
    script, table = _text(args.script), _table(args)
    if args.format == "json":
        r = render(script, table)
        _out({"spoken": r.spoken, "shown": r.shown, "ssml": r.ssml,
              "terms": [t.__dict__ for t in r.terms]})
    else:
        opts = {"lang": args.lang, "voice": args.voice, "bare": args.bare} if args.format == "ssml" else None
        if args.format == "ssml" and not args.voice and not args.bare:
            print("note: Azure needs a voice: add --voice en-US-AvaNeural (or yours); Google/Polly: --bare",
                  file=sys.stderr)
        print(to_engine(script, table, args.format, opts))
    return 0


def cmd_export_pls(args) -> int:
    from .engines import pls
    sys.stdout.write(pls(_table(args), lang=args.lang))
    return 0


def _batch(path: str) -> list[dict]:
    """A batch file: JSON or YAML list of {audio, script, shown?}, or lines of `audio<TAB>script`. Paths are relative
    to the batch file."""
    base = Path(path).resolve().parent
    text = Path(path).read_text(encoding="utf-8")
    if path.lower().endswith((".json", ".yaml", ".yml")):
        import yaml
        items = yaml.safe_load(text) or []
    else:
        items = [dict(zip(("audio", "script"), line.split("	", 1))) for line in text.splitlines()
                 if line.strip() and "	" in line]
    for it in items:
        if not isinstance(it, dict) or "audio" not in it or "script" not in it:
            raise ValueError(f"{path}: each entry needs audio and script, got {it!r}")
        if not Path(it["audio"]).is_absolute():
            it["audio"] = str(base / it["audio"])
    return items


def cmd_check(args) -> int:
    from . import server
    if args.batch:
        items = _batch(args.batch)
    else:
        if not args.audio or not args.script:
            raise ValueError("check needs AUDIO and SCRIPT, or --batch FILE")
        items = [{"audio": args.audio, "script": args.script, "shown": args.shown}]
    for it in items:                                       # absolute paths: a server may run elsewhere
        it["audio"] = str(Path(it["audio"]).resolve())
        if Path(str(it["script"])).suffix.lower() in (".txt", ".md") and Path(it["script"]).exists():
            it["script"] = str(Path(it["script"]).resolve())
    if not args.local and server.alive():
        lex = [str(Path(p).resolve()) for p in (args.lexicon or [])]
        print(f"using the running server at {server.url()}", file=sys.stderr)
        reports = server.request(items, lex, args.pace, args.tone)
    else:
        from .session import Session
        table, session, reports = _table(args), Session(), []
        for it in items:
            y, sr = _audio(it["audio"])
            shown = _text(it["shown"]) if it.get("shown") else None
            rep = session.check(y, sr, _text(it["script"]), shown=shown, table=table, pace=args.pace,
                                tone=args.tone)
            rep["audio"] = it["audio"]
            reports.append(rep)
    if args.batch:
        _out({"passed": all(r["passed"] for r in reports), "takes": len(reports),
              "failed": [r["audio"] for r in reports if not r["passed"]], "reports": reports})
    else:
        _out(reports[0])
    return 0 if all(r["passed"] for r in reports) else 1


def cmd_serve(args) -> int:
    from . import server
    server.serve(args.port or server.port())
    return 0


# ---------------------------------------------------------------- entry point

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vc.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def lex(p):
        p.add_argument("--lexicon", action="append", help="a lexicon YAML file (repeatable)")
        return p
    p = lex(sub.add_parser("scan", help="list the words a voice may get wrong"))
    p.add_argument("script")
    p.add_argument("--fast", action="store_true", help="skip the tokenizer rarity check")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("lookup", help="how public sources pronounce a name or word")
    p.add_argument("name")
    p.add_argument("--lang", default="en", help="Wikipedia/Wiktionary language edition (default en)")
    p.add_argument("--context", help="words to disambiguate the search, e.g. 'computer scientist'")
    p.set_defaults(fn=cmd_lookup)

    p = sub.add_parser("respell", help="IPA -> respellings for plain-text TTS")
    p.add_argument("ipa")
    p.add_argument("--name", action="store_true", help="capitalise (a proper name)")
    p.set_defaults(fn=cmd_respell)

    p = sub.add_parser("hear", help="the IPA a recording says (file or URL)")
    p.add_argument("audio")
    p.add_argument("--start", type=float)
    p.add_argument("--end", type=float)
    p.set_defaults(fn=cmd_hear)

    p = lex(sub.add_parser("render", help="the script with the lexicon applied"))
    p.add_argument("script")
    p.add_argument("-f", "--format", default="text",
                   choices=["text", "ssml", "elevenlabs", "cartesia", "kokoro", "piper", "json"])
    p.add_argument("--lang", default="en-US", help="SSML xml:lang (English voices get US-English phones)")
    p.add_argument("--voice", help="SSML: wrap in <voice name=...> (Azure requires one)")
    p.add_argument("--bare", action="store_true", help="SSML: a plain <speak> (Google, Polly)")
    p.set_defaults(fn=cmd_render)

    p = lex(sub.add_parser("export-pls", help="the lexicon as a PLS pronunciation lexicon"))
    p.add_argument("--lang", default="en-US")
    p.set_defaults(fn=cmd_export_pls)

    p = lex(sub.add_parser("check", help="check a take (or a batch of takes) against its script"))
    p.add_argument("audio", nargs="?", help="the take (wav, flac, ogg, mp3...)")
    p.add_argument("script", nargs="?", help="the script as written (the lexicon is applied), or a .txt path")
    p.add_argument("--batch", help="check many takes with the models loaded once: a JSON/YAML list of "
                                   "{audio, script} or lines of audio<TAB>script")
    p.add_argument("--shown", help="legacy: SCRIPT is the spoken text and this is the on-screen wording")
    p.add_argument("--pace", type=float, nargs=2, metavar=("LOW", "HIGH"), help="syllables/s window")
    p.add_argument("--tone", action="store_true", help="measure pitch even with no `flat` limits set (slower)")
    p.add_argument("--local", action="store_true", help="don't use a running `vc.py serve`")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("serve", help="keep the models loaded for fast repeated checks (127.0.0.1 only)")
    p.add_argument("--port", type=int, help="default 8767, or VOICECHECK_PORT")
    p.set_defaults(fn=cmd_serve)
    return ap


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")           # IPA and names on a Windows console
        except Exception:
            pass
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
