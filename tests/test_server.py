"""`vc serve` reads local files for whoever asks, so it must only answer this machine's own tools. No models load
here: /health and the refusals come before any work."""
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from voicecheck import __version__, server


@pytest.fixture(scope="module")
def port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
    threading.Thread(target=server.serve, args=(p,), daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{p}/health", timeout=0.5).read()
            return p
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("the server did not start")


def ask(port, path, body=None, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_says_its_version_and_a_client_of_another_version_stays_away(port, monkeypatch, capsys):
    assert ask(port, "/health") == (200, {"ok": True, "version": __version__, "loaded": []})
    monkeypatch.setenv("VOICECHECK_PORT", str(port))
    assert server.alive()
    monkeypatch.setattr("voicecheck.__version__", "9.9.9")
    assert not server.alive() and "not using it" in capsys.readouterr().err


def test_a_request_a_web_page_could_send_is_refused(port):
    body = json.dumps({"items": [{"audio": "C:/secret.wav", "script": "C:/secret.txt"}]}).encode()
    # a page can post text/plain or a form to another origin without asking; it can't post JSON
    for kind in ("text/plain", "application/x-www-form-urlencoded", None):
        code, out = ask(port, "/check", body, {"Content-Type": kind} if kind else {})
        assert code == 415 and "JSON" in out["error"]
    # a page whose own domain points at 127.0.0.1 (DNS rebinding) still names that domain as the host
    code, out = ask(port, "/check", body, {"Content-Type": "application/json", "Host": "evil.example"})
    assert code == 403
    assert ask(port, "/health", headers={"Host": "evil.example:80"})[0] == 403
    assert ask(port, "/health", headers={"Host": f"localhost:{port}"})[0] == 200


def test_a_bad_request_is_an_error_the_client_can_read(port, monkeypatch):
    code, out = ask(port, "/proof", json.dumps({"audio": "no/such/file.wav", "transcript": "Hello."}).encode(),
                    {"Content-Type": "application/json"})
    assert code == 400 and "error" in out
    assert ask(port, "/nowhere", b"{}", {"Content-Type": "application/json"})[0] == 404
    monkeypatch.setenv("VOICECHECK_PORT", str(port))
    with pytest.raises(ValueError, match="server:"):
        server.request_proof("no/such/file.wav", "Hello.", [], None)
