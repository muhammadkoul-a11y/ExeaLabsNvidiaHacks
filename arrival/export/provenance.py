"""Where a twin and a dataset came from: the input scan's hash, the code's git sha, the seeds and the generators.

    stamp_twin(twin, input_path=...)                  # twin["provenance"], written into twin.json by write_twin
    run_record(twin, seed=3, generator="mujoco_scripted_ik", ...)   # goes into meta/info.json as arrival_provenance

Nothing here reaches the network or reads anything but the files it is pointed at. A field that cannot be known is
None, never a guess: a wheel install has no git checkout (set ARRIVAL_GIT_SHA at build time), and a twin compiled
from an in-memory bundle has no input file, only the hash of the normalized bundle.

Object provenance is a separate, per-object label (OBJECT_PROVENANCE): `detected` for something
`arrival.twin.detect` found in uploaded RGB-D frames, `authored` for a person-written entry in scan.json,
`default_synthetic` for the catalog objects `place_default_objects` puts on a counter because RoomPlan sees no small
objects and no detection found any.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
OBJECT_PROVENANCE = ("detected", "authored", "default_synthetic")
GIT_TIMEOUT_S = 5


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(obj) -> str:
    """sha256 of the JSON form with sorted keys and no whitespace, so equal data hashes equal."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@lru_cache(maxsize=4)
def _git(repo: str) -> tuple[str | None, bool | None]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, timeout=GIT_TIMEOUT_S,
                                 check=True)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    # git looks upward for a repository, so an installed wheel in a venv under some other project would get that
    # project's HEAD: only a checkout whose top level is `repo` itself says anything about this code
    top = run("rev-parse", "--show-toplevel")
    try:
        own = bool(top) and os.path.samefile(top, repo)
    except OSError:
        own = False
    sha = run("rev-parse", "HEAD") if own else None
    if not sha:
        return None, None
    status = run("status", "--porcelain", "--untracked-files=no")
    return sha, (None if status is None else bool(status))


def git_state(repo: str | Path = ROOT) -> dict:
    """{"sha", "dirty"} of the checkout the code runs from. ARRIVAL_GIT_SHA overrides (a built image has no .git);
    sha and dirty are None when neither is available."""
    env = os.environ.get("ARRIVAL_GIT_SHA", "").strip()
    if env:
        return {"sha": env, "dirty": None, "source": "env"}
    sha, dirty = _git(str(repo))
    return {"sha": sha, "dirty": dirty, "source": "git" if sha else None}


def input_record(bundle: dict | None = None, input_path: str | Path | None = None) -> dict:
    """The scan that went in: sha256 of the file as received (when there is one) and of the normalized bundle."""
    rec: dict = {"file": None, "file_sha256": None, "bundle_sha256": None}
    if input_path is not None:
        p = Path(input_path)
        rec["file"] = p.name
        rec["file_sha256"] = sha256_file(p) if p.is_file() else None
    elif bundle is not None and bundle.get("input_sha256"):
        rec["file_sha256"] = bundle["input_sha256"]
    if bundle is not None:
        rec["bundle_sha256"] = canonical_sha256({k: v for k, v in bundle.items() if k != "input_sha256"})
    return rec


def object_counts(twin: dict) -> dict:
    """How many twin objects carry each provenance label; an object with none is counted as `authored`."""
    counts = {k: 0 for k in OBJECT_PROVENANCE}
    for o in twin.get("objects", []):
        counts[o.get("provenance") or "authored"] = counts.get(o.get("provenance") or "authored", 0) + 1
    return counts


def stamp_twin(twin: dict, bundle: dict | None = None, input_path: str | Path | None = None,
               generators: dict | None = None) -> dict:
    """Add twin["provenance"]. `bundle` is the normalized scan the twin was compiled from (its hash and, when the
    loader recorded it, its input file's hash); `input_path` hashes a file directly. Returns the twin."""
    twin["provenance"] = {
        "schema_version": SCHEMA_VERSION,
        "input": input_record(bundle, input_path),
        "code": git_state(),
        "generators": {"geometry": twin.get("geometry_source"), "twin_compiler": "arrival.twin.compile",
                       "objects": object_counts(twin), **(generators or {})},
        "seeds": dict(twin.get("provenance", {}).get("seeds", {})),
    }
    return twin


def note_seed(twin: dict, name: str, seed: int) -> None:
    """Record a seed that shaped the twin (the default-object placement) in twin["provenance"]["seeds"]."""
    twin.setdefault("provenance", {}).setdefault("seeds", {})[name] = seed


def run_record(twin: dict, seed: int | None = None, generator: str | None = None, **params) -> dict:
    """The record a dataset keeps: the twin's own provenance plus this run's seed, generator and parameters."""
    base = dict(twin.get("provenance") or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "input": base.get("input") or input_record(None),
        "twin_code": base.get("code"),
        "twin_generators": base.get("generators"),
        "twin_seeds": base.get("seeds", {}),
        "run_code": git_state(),
        "run": {"seed": seed, "generator": generator, **params},
    }
