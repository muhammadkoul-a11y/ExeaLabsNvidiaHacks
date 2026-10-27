"""Smoke-test a running Arrival web server, local or hosted: a good upload must be queued and readable, bad ones
must be refused, and nothing must leak through a crafted job path. Standard library only.

    python scripts/smoke_hosted.py http://127.0.0.1:8000
    python scripts/smoke_hosted.py https://arrival.example.org --wait 600    # also wait for the worker's dataset.zip

Exits 0 when every check passes, 1 otherwise, 2 on bad arguments. Each upload creates a real job on the server
(named smoke_<unix time>); point it at a server whose queue you may write to.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from http.cookiejar import CookieJar
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SCAN = REPO / "scans" / "hero_kitchen" / "roomplan.json"
JOB_STATES = ("queued", "running", "done", "failed")


class Smoke:
    def __init__(self, base: str, timeout: float = 30.0):
        self.base, self.timeout = base.rstrip("/"), timeout
        self.results: list[dict] = []
        # one visitor: a --public server shows a job only to the cookie it set on the upload
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None):
        """(status, headers, body bytes); HTTP errors are returned, not raised. A network failure gives status 0."""
        req = urllib.request.Request(self.base + path, data=body, method=method, headers=headers or {})
        try:
            with self.opener.open(req, timeout=self.timeout) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers or {}), e.read()
        except (urllib.error.URLError, OSError) as e:
            return 0, {}, str(e).encode()

    def check(self, name: str, ok: bool, detail: str = "", required: bool = True) -> bool:
        self.results.append({"check": name, "ok": bool(ok), "required": required, "detail": detail})
        return ok

    @staticmethod
    def multipart(field: str, filename: str, data: bytes, ctype: str = "application/json") -> tuple[bytes, str]:
        boundary = "smoke-" + uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n"
                f"Content-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
        return body, f"multipart/form-data; boundary={boundary}"

    def run(self, scan: bytes, wait: float = 0.0, require_healthz: bool = False, poll: float = 2.0) -> bool:
        code, hdrs, _ = self.request("GET", "/")
        self.check("index page", code == 200 and "html" in hdrs.get("Content-Type", ""), f"GET / -> {code}")
        if code == 0:
            return False  # unreachable: the rest would only repeat it
        code, _, _ = self.request("GET", "/healthz")
        self.check("healthz", code == 200, f"GET /healthz -> {code}" + ("" if require_healthz else " (optional)"),
                   required=require_healthz)

        body, ctype = self.multipart("scan", f"smoke_{int(time.time())}.json", scan)
        code, _, raw = self.request("POST", "/jobs", body, {"Content-Type": ctype, "Accept": "application/json"})
        job = _json(raw)
        good = self.check("good upload queued", code == 202 and isinstance(job.get("id"), str) and job.get("status_url"),
                          f"POST /jobs -> {code} {_short(raw)}")
        if good:
            code, _, raw = self.request("GET", job["status_url"], headers={"Accept": "application/json"})
            st = _json(raw)
            self.check("job status readable", code == 200 and st.get("id") == job["id"] and st.get("status") in JOB_STATES,
                       f"GET {job['status_url']} -> {code} status={st.get('status')}")

        bad = [
            ("upload without a scan field refused", *self.multipart("file", "x.json", scan)),
            ("non-multipart upload refused", scan, "application/json"),
            ("empty scan refused", *self.multipart("scan", "empty.json", b"")),
        ]
        for name, b, ct in bad:
            code, _, raw = self.request("POST", "/jobs", b, {"Content-Type": ct, "Accept": "application/json"})
            self.check(name, 400 <= code < 500, f"POST /jobs -> {code} {_short(raw)}")
        for path in ("/jobs/job_nosuchjob", "/jobs/..%2F..%2Fetc%2Fpasswd", "/jobs/../../etc/passwd",
                     "/download/..%2F..%2Fpyproject.toml"):
            code, _, raw = self.request("GET", path)
            leaked = b"root:" in raw or b"[project]" in raw
            self.check(f"crafted path refused: {path}", code in (400, 403, 404) and not leaked, f"GET -> {code}")

        if good and wait > 0:
            deadline, st = time.monotonic() + wait, {}
            while time.monotonic() < deadline:
                st = _json(self.request("GET", job["status_url"], headers={"Accept": "application/json"})[2])
                if st.get("status") in ("done", "failed"):
                    break
                time.sleep(poll)
            done = self.check("worker finished the job", st.get("status") == "done",
                              f"status={st.get('status')} stage={st.get('stage')} error={st.get('error')}")
            if done:
                code, hdrs, raw = self.request("GET", f"/jobs/{job['id']}/download")
                self.check("dataset.zip downloads", code == 200 and raw[:2] == b"PK", f"GET download -> {code}, {len(raw)} bytes")
        return all(r["ok"] for r in self.results if r["required"])


def _json(raw: bytes) -> dict:
    try:
        v = json.loads(raw)
    except ValueError:
        return {}
    return v if isinstance(v, dict) else {}


def _short(raw: bytes, n: int = 120) -> str:
    return raw[:n].decode("utf-8", "replace").replace("\n", " ")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("url", help="base URL of the server, e.g. http://127.0.0.1:8000")
    ap.add_argument("--scan", default=str(DEFAULT_SCAN), help="the good upload (default: the hero roomplan.json)")
    ap.add_argument("--wait", type=float, default=0.0, help="seconds to wait for the worker to finish the job (0: don't)")
    ap.add_argument("--require-healthz", action="store_true", help="fail when GET /healthz is not 200")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-request timeout in seconds")
    ap.add_argument("--json", action="store_true", help="print the results as JSON")
    a = ap.parse_args(argv)
    if not a.url.startswith(("http://", "https://")):
        ap.error("url must start with http:// or https://")
    if not a.timeout > 0:
        ap.error("--timeout must be above 0")
    if not a.wait >= 0:
        ap.error("--wait must be 0 or more")
    if not Path(a.scan).is_file():
        ap.error(f"--scan {a.scan} is not a file")
    s = Smoke(a.url, a.timeout)
    ok = s.run(Path(a.scan).read_bytes(), wait=a.wait, require_healthz=a.require_healthz)
    if a.json:
        print(json.dumps({"url": a.url, "ok": ok, "results": s.results}, indent=2))
    else:
        for r in s.results:
            mark = "ok  " if r["ok"] else ("FAIL" if r["required"] else "warn")
            print(f"{mark} {r['check']}: {r['detail']}")
        print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
