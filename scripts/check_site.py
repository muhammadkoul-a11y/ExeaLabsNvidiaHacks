#!/usr/bin/env python3
"""Check that the static demo site loads and every asset path it references resolves.

Usage:  python scripts/check_site.py [--base http://localhost:8765] [--offline]

Default: fetches over HTTP from a running `make site` and requires a 200 for the page and every referenced file.
--offline: resolves the same paths on disk under site/ instead (no server needed).
With no server at --base it says so in one line and exits 1.
Referenced paths come from index.html (src, href, importmap), css/*.css (url(...)), js/*.js (static and dynamic import
specifiers, new URL('...', import.meta.url), and fetched data paths), plus the files the data README promises
(6 episodes, 24 previews). Standard library only.
"""
import argparse, re, sys, urllib.request, urllib.error
from pathlib import Path
from urllib.parse import urljoin, urlparse

SITE = Path(__file__).resolve().parent.parent / "site"
FAMILIES = ["counter_to_sink", "open_drawer", "open_door", "open_fridge", "put_in_drawer", "wipe_counter"]


def refs_html(text):
    out = set(re.findall(r'(?:src|href)="([^"#]+)"', text))
    out |= set(re.findall(r'"three":\s*"([^"]+)"', text))          # importmap
    return {r for r in out if not r.startswith(("http:", "https:", "data:", "mailto:", "#"))}


def refs_css(text):
    return {r for r in re.findall(r'url\(\s*["\']?([^"\')]+)["\']?\s*\)', text) if not r.startswith(("data:", "#", "http", "%23"))}


def refs_js(text):
    out = set(re.findall(r'(?:from|import)\s*\(?\s*[\'"](\.{1,2}/[^\'"]+)[\'"]', text))
    out |= {r for r in re.findall(r"new URL\(\s*[`'\"]([^`'\"$]+)[`'\"]\s*,\s*import\.meta\.url", text) if not r.endswith('/')}
    out |= set(re.findall(r"\bj\(\s*[`'\"]([^`'\"$]+\.json)[`'\"]", text))
    out |= {"scan_points.bin"} if "scan_points.bin" in text else set()
    return out


def collect():
    """Return a list of (referencing file relative to site/, absolute path under site/)."""
    found = []
    def add(src, ref, base):
        p = (base / ref.split("?")[0]).resolve()
        found.append((src, p))
    idx = SITE / "index.html"
    for r in refs_html(idx.read_text()):
        add("index.html", r, SITE)
    for css in (SITE / "css").glob("*.css"):
        for r in refs_css(css.read_text()):
            add(f"css/{css.name}", r, css.parent)
    data_base = SITE / "data"
    for js in (SITE / "js").glob("*.js"):
        txt = js.read_text()
        for r in refs_js(txt):
            if r.endswith(".json") or r.endswith(".bin") or r.startswith("frames/") or r.startswith("episodes/"):
                add(f"js/{js.name}", r, data_base)
            elif r.startswith("."):
                add(f"js/{js.name}", r, js.parent)
    # data files the site promises
    for f in FAMILIES:
        add("data README", f"episodes/{f}.json", data_base)
        for cam in ("ego_view", "wrist"):
            for w in ("mid", "end"):
                add("data README", f"frames/{f}_{cam}_{w}.png", data_base)
    for n in ("rooms.json", "geoms.json", "stats.json", "scan_points.json", "scan_points.bin"):
        add("data README", n, data_base)
    for n in ("three/three.module.js", "three/OrbitControls.js", "gsap/gsap.min.js", "gsap/ScrollTrigger.min.js", "lenis/lenis.min.js"):
        add("vendor", n, SITE / "vendor")
    for n in ("fraunces-opsz.woff2", "fraunces-opsz-italic.woff2", "dm-mono-400.woff2", "dm-mono-500.woff2"):
        add("fonts", n, SITE / "fonts")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8765")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    found = collect()
    bad, seen = [], set()
    if not a.offline:
        try:
            with urllib.request.urlopen(a.base + "/", timeout=10) as r:
                html = r.read().decode()
        except (urllib.error.URLError, OSError) as e:
            # checked first: with no server each path would wait on its own refused connect (about 4 s on Windows)
            print(f"no site at {a.base}/ ({e}); start one with `make site`, or pass --offline")
            sys.exit(1)
        if "<canvas" in html and "js/main.js" in html:
            print(f"page ok: {a.base}/ ({len(html)} bytes)")
        else:
            bad.append(("page", a.base, "page loaded but looks wrong"))
    for src, p in found:
        if p in seen:
            continue
        seen.add(p)
        try:
            rel = p.relative_to(SITE.resolve()).as_posix()
        except ValueError:
            bad.append((src, str(p), "escapes site/")); continue
        if a.offline:
            if not p.is_file():
                bad.append((src, rel, "missing on disk"))
        else:
            url = urljoin(a.base + "/", rel)
            try:
                with urllib.request.urlopen(url, timeout=10) as r:
                    if r.status != 200:
                        bad.append((src, rel, f"HTTP {r.status}"))
            except (urllib.error.URLError, OSError) as e:
                bad.append((src, rel, f"unreachable: {e}"))
    print(f"checked {len(seen)} distinct referenced paths")
    for src, rel, why in bad:
        print(f"  FAIL {rel}  (from {src}): {why}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
