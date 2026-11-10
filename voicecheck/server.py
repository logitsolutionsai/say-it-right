"""`vc.py serve`: keep the models loaded between checks, so a re-record loop doesn't pay ~20 s of loading per take.

    python vc.py serve &                 # once; listens on 127.0.0.1:8767 (or --port / VOICECHECK_PORT)
    python vc.py check take.wav "..."    # finds the server and uses it; without one it loads the models itself

Local only: it binds to 127.0.0.1 and reads the audio and lexicon files named in each request from this machine.
The server's settings come from VOICECHECK_CONFIG as it was when the server started.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 8767


def port() -> int:
    return int(os.environ.get("VOICECHECK_PORT", DEFAULT_PORT))


def url(p: int | None = None) -> str:
    return f"http://127.0.0.1:{p or port()}"


def alive(timeout: float = 0.3) -> bool:
    try:
        with urllib.request.urlopen(url() + "/health", timeout=timeout) as r:
            return json.loads(r.read()).get("ok") is True
    except Exception:
        return False


def request(items: list[dict], lexicon: list[str] | None, pace, tone: bool) -> list[dict]:
    body = json.dumps({"items": items, "lexicon": lexicon or [], "pace": pace, "tone": tone}).encode("utf-8")
    req = urllib.request.Request(url() + "/check", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.loads(r.read().decode("utf-8"))
    if "error" in out:
        raise ValueError(f"server: {out['error']}")
    return out["reports"]


def serve(p: int) -> None:
    from . import lexicon as L
    from .cli import _audio, _text
    from .session import Session
    session = Session()
    lock = threading.Lock()                                # one check at a time: the models aren't re-entrant

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[serve] {fmt % args}", file=sys.stderr, flush=True)

        def _send(self, obj, code=200):
            data = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send({"ok": True, "loaded": [k for k in ("_checker", "_listener", "_ear")
                                                   if getattr(session, k) is not None]})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/check":
                return self._send({"error": "not found"}, 404)
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
                table = L.load(req.get("lexicon") or None) if req.get("lexicon") else L.load()
                reports = []
                with lock:
                    for it in req["items"]:
                        y, sr = _audio(it["audio"])
                        shown = _text(it["shown"]) if it.get("shown") else None
                        rep = session.check(y, sr, _text(it["script"]), shown=shown, table=table,
                                            pace=req.get("pace"), tone=bool(req.get("tone")))
                        rep["audio"] = it["audio"]
                        reports.append(rep)
                self._send({"reports": reports})
            except Exception as e:                         # report it to the client; keep serving
                self._send({"error": f"{type(e).__name__}: {e}"}, 400)

    httpd = ThreadingHTTPServer(("127.0.0.1", p), Handler)
    print(f"[serve] say-it-right listening on {url(p)} (models load on the first check)", file=sys.stderr,
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
