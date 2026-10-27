"""C2 (part): scripts/smoke_hosted.py passes against the real server and catches a server that accepts anything."""

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from arrival.web.app import make_server

from conftest import REPO, SCANS

_spec = importlib.util.spec_from_file_location("smoke_hosted", REPO / "scripts" / "smoke_hosted.py")
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


def _serve(srv):
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture
def arrival_url(tmp_path):
    srv = make_server(port=0, out_dir=tmp_path / "out", scans_dir=SCANS)
    try:
        yield _serve(srv), tmp_path / "out"
    finally:
        srv.shutdown()
        srv.server_close()


class _AcceptsEverything(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _ok(self, code, body=b"<html>ok</html>", ctype="text/html"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/jobs/"):
            return self._ok(200, json.dumps({"id": "job_x", "status": "queued"}).encode(), "application/json")
        self._ok(200)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._ok(202, json.dumps({"id": "job_x", "status": "queued", "status_url": "/jobs/job_x"}).encode(),
                 "application/json")


def test_passes_against_the_arrival_server(arrival_url, capsys):
    url, out = arrival_url
    assert smoke.main([url, "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["ok"] and {r["check"] for r in res["results"] if r["required"]} >= {
        "good upload queued", "job status readable", "upload without a scan field refused"}
    assert len(list((out / "jobs").iterdir())) == 1  # only the good upload became a job


def test_passes_against_a_public_server(tmp_path, capsys):
    """--public shows a visitor only the jobs their cookie owns, as DEPLOY.md hosts it."""
    srv = make_server(port=0, out_dir=tmp_path / "out", scans_dir=SCANS, public=True)
    try:
        assert smoke.main([_serve(srv), "--json"]) == 0, capsys.readouterr().out
    finally:
        srv.shutdown()
        srv.server_close()


def test_fails_a_server_that_accepts_bad_uploads_and_crafted_paths(capsys):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _AcceptsEverything)
    try:
        assert smoke.main([_serve(srv)]) == 1
    finally:
        srv.shutdown()
        srv.server_close()
    out = capsys.readouterr().out
    assert "FAIL upload without a scan field refused" in out and "FAIL crafted path refused" in out


def test_unreachable_server_fails_fast(capsys):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _AcceptsEverything)
    port = srv.server_address[1]
    srv.server_close()  # nothing listens on this port now
    assert smoke.main([f"http://127.0.0.1:{port}", "--timeout", "2"]) == 1
    assert len(capsys.readouterr().out.strip().splitlines()) == 2  # the index check, then FAIL


def test_wait_reports_a_job_the_worker_never_finished(arrival_url, capsys):
    url, _ = arrival_url  # no worker is running
    assert smoke.main([url, "--wait", "0.5"]) == 1
    assert "FAIL worker finished the job: status=queued" in capsys.readouterr().out


@pytest.mark.parametrize("args, why", [(["--scan", "no_such_scan.json"], "no_such_scan.json"),
                                       (["--timeout", "0"], "--timeout"), (["--timeout", "nan"], "--timeout"),
                                       (["--wait", "-1"], "--wait"), (["--wait", "nan"], "--wait")])
def test_bad_arguments_exit_2_before_any_request(args, why, capsys):
    with pytest.raises(SystemExit) as e:
        smoke.main(["http://127.0.0.1:9", *args])
    assert e.value.code == 2
    assert why in capsys.readouterr().err
