"""`vc.py serve`: keep the models loaded between checks, so a re-record loop doesn't pay ~20 s of loading per take.

    python vc.py serve &                 # once; listens on 127.0.0.1:8767 (or --port / VOICECHECK_PORT)
    python vc.py check take.wav "..."    # finds the server and uses it; without one it loads the models itself

Local only: it binds to 127.0.0.1 and reads the audio and lexicon files named in each request from this machine.
Because it reads local files, it also refuses anything a web page could send it: a request must be addressed to
127.0.0.1 or localhost (a page that points its own domain at 127.0.0.1 is turned away) and must be JSON (which a
page can't post across origins without asking first, and the server never grants that).
The server's settings come from VOICECHECK_CONFIG as it was when the server started. A client only uses a server
of its own version: one left running from before an update is ignored.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_PORT = 8767


def port() -> int:
    return int(os.environ.get("VOICECHECK_PORT", DEFAULT_PORT))


def url(p: int | None = None) -> str:
    return f"http://127.0.0.1:{p or port()}"


def alive(timeout: float = 0.3) -> bool:
    """Is a server of this version listening? An older one (left running across an update) doesn't count."""
    from . import __version__
    try:
        with urllib.request.urlopen(url() + "/health", timeout=timeout) as r:
            health = json.loads(r.read())
    except Exception:
        return False
    if health.get("ok") is True and health.get("version") != __version__:
        print(f"note: the server at {url()} is version {health.get('version', 'older')}, this is {__version__}: "
              "not using it (stop it and run `vc.py serve` again)", file=sys.stderr)
        return False
    return health.get("ok") is True


def _post(path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url() + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            out = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:                     # the server answers errors as JSON with a 4xx status
        out = json.loads(e.read().decode("utf-8") or "{}")
    if "error" in out:
        raise ValueError(f"server: {out['error']}")
    return out


def request(items: list[dict], lexicon: list[str] | None, pace, tone: bool) -> list[dict]:
    return _post("/check", {"items": items, "lexicon": lexicon or [], "pace": pace, "tone": tone})["reports"]


def request_proof(audio: str, transcript: str, lexicon: list[str] | None, second) -> dict:
    """`second`: "whisper", None, or the path of another transcript."""
    return _post("/proof", {"audio": audio, "transcript": transcript, "lexicon": lexicon or [],
                            "second": second})["report"]


def serve(p: int) -> None:
    from . import __version__, proof
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

        def _local(self) -> bool:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
            return host in ("127.0.0.1", "localhost", "::1")

        def do_GET(self):
            if not self._local():
                self._send({"error": "this server only answers requests addressed to 127.0.0.1"}, 403)
            elif self.path == "/health":
                self._send({"ok": True, "version": __version__,
                            "loaded": [k for k in ("_checker", "_listener", "_ear")
                                       if getattr(session, k) is not None]})
            else:
                self._send({"error": "not found"}, 404)

        def do_POST(self):
            if not self._local():
                return self._send({"error": "this server only answers requests addressed to 127.0.0.1"}, 403)
            if self.headers.get_content_type() != "application/json":
                return self._send({"error": "send JSON (Content-Type: application/json)"}, 415)
            if self.path not in ("/check", "/proof"):
                return self._send({"error": "not found"}, 404)
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
                table = L.load(req.get("lexicon") or None) if req.get("lexicon") else L.load()
                with lock:
                    if self.path == "/proof":
                        y, sr = _audio(req["audio"])
                        second = req.get("second")
                        second = proof.read(second) if second not in (None, "whisper") else second
                        report = session.proof(y, sr, proof.read(req["transcript"]), table=table, second=second)
                        return self._send({"report": report})
                    reports = []
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
