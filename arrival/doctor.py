"""L22: ``python -m arrival doctor`` checks this machine before a run, offline.

Checks the Python version, the core dependencies, MuJoCo, an ffmpeg binary, which optional credentials and URLs
are configured, and free disk for the output folder. A key is reported only as set or not set, with where it was
found; its value is never read into the output, never printed and never sent anywhere.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

MIN_PYTHON = (3, 11)
# import name -> distribution name, for the dependencies in pyproject.toml
CORE = {"numpy": "numpy", "pyarrow": "pyarrow", "yaml": "pyyaml", "PIL": "pillow", "imageio": "imageio",
        "imageio_ffmpeg": "imageio-ffmpeg"}
from arrival.jobs.nebius import JOB_SETTINGS, STORE_SETTINGS

NEBIUS_PURPOSES = {
    "NEBIUS_IAM_TOKEN": "Nebius Serverless AI Jobs access token (fanout --backend nebius)",
    "NEBIUS_PROJECT_ID": "Nebius project for fanout jobs",
    "NEBIUS_JOB_IMAGE": "Nebius job container image",
    "NEBIUS_JOB_S3_SECRET_ID": "Nebius MysteryBox secret ID for fanout storage",
    "NEBIUS_BUCKET": "Nebius Object Storage bucket for fanout artifacts",
    "NEBIUS_ACCESS_KEY_ID": "Nebius Object Storage access key ID",
    "NEBIUS_SECRET_ACCESS_KEY": "Nebius Object Storage secret access key",
}

# (variable, what it unlocks). Every one is optional: the demo, the sim and the tests need none of them.
OPTIONAL_ENV = (
    ("TOKEN_FACTORY_API_KEY", "Nemotron planner and instruction wordings (sim --planner vlm, --instructions nemotron)"),
    ("TAVILY_API_KEY", "appliance research (research, sim --tavily)"),
    ("GR00T_POLICY_URL", "a GR00T policy server (--policy vla)"),
    ("COSMOS_URL", "Cosmos appearance restyle"),
    *((k, NEBIUS_PURPOSES.get(k, "Nebius Cloud fanout")) for k in (*JOB_SETTINGS, *STORE_SETTINGS)),
)
# Model ids the key needs, with what each one drives. The ultra role is never called, so it is not checked.
TOKEN_FACTORY_MODELS = (
    ("NEMOTRON_SUPER_MODEL", "the planner (sim --planner vlm)"),
    ("NEMOTRON_LIGHTNING_MODEL", "the instruction writer (--instructions nemotron)"),
    ("NEMOTRON_OMNI_MODEL", "small-object detection in scan frames"),
)
FREE_WARN_GIB, FREE_FAIL_GIB = 2.0, 0.25
GIB = 1024 ** 3
REPO = Path(__file__).resolve().parents[1]


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail | info
    detail: str


def _version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "unknown version"


def check_python(version_info: tuple[int, ...] | None = None) -> Check:
    v = sys.version_info if version_info is None else version_info
    major, minor = v[0], v[1]
    micro = v[2] if len(v) > 2 else 0
    ok = (major, minor) >= MIN_PYTHON
    text = f"{major}.{minor}.{micro} on {platform.system()} {platform.machine()}"
    return Check("python", "ok" if ok else "fail", text if ok else f"{text}; needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer")


def check_core(core_deps: dict[str, str] | None = None) -> list[Check]:
    deps = CORE if core_deps is None else core_deps
    out = []
    for mod, dist in deps.items():
        try:
            found = importlib.util.find_spec(mod) is not None
        except (ValueError, ModuleNotFoundError):
            found = False
        if not found:
            out.append(Check(dist, "fail", "not installed: pip install -e ."))
        else:
            out.append(Check(dist, "ok", _version(dist)))
    return out


def check_mujoco() -> Check:
    try:
        import mujoco
    except ImportError:
        return Check("mujoco", "warn", "not installed; sim, eval and work need it: pip install -e '.[sim]' (demo and serve do not)")
    except Exception as e:  # noqa: BLE001 - a broken native library raises more than ImportError
        return Check("mujoco", "fail", f"installed but does not import: {type(e).__name__}")
    return Check("mujoco", "ok", getattr(mujoco, "__version__", _version("mujoco")))


def check_ffmpeg(exe: str | None = None) -> Check:
    source = "custom"
    if exe is None:
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
            source = "imageio-ffmpeg"
        except Exception:  # noqa: BLE001 - missing package or no bundled binary: try the system one
            pass
        if not exe or not Path(exe).exists():
            exe, source = shutil.which("ffmpeg"), "PATH"
        if not exe:
            return Check("ffmpeg", "fail", "no ffmpeg found (imageio-ffmpeg ships one: pip install -e .); videos need it")
    elif not Path(exe).exists():
        return Check("ffmpeg", "fail", f"no ffmpeg found at {exe}")

    try:
        r = subprocess.run([exe, "-version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
        first = (r.stdout or "").splitlines()[0] if r.returncode == 0 and r.stdout else ""
    except (OSError, subprocess.SubprocessError):
        first = ""
    if not first:
        return Check("ffmpeg", "fail", f"{source} binary does not run")
    return Check("ffmpeg", "ok", f"{first} ({source})")


def _env_file_names(paths: list[Path]) -> dict[str, Path]:
    """Names of the non-empty KEY=value lines in each .env file, first file winning. Values are dropped on the spot."""
    found: dict[str, Path] = {}
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.removeprefix("export ").strip().strip("'\"")
            if key and value.strip().strip("'\"") and key not in found:
                found[key] = path
    return found


def check_env(environ: dict[str, str] | None = None, env_files: list[Path] | None = None) -> list[Check]:
    environ = os.environ if environ is None else environ
    files = [Path.cwd() / ".env", REPO / ".env"] if env_files is None else [Path(p) for p in env_files]
    in_file = _env_file_names(files)
    out = []
    for name, purpose in OPTIONAL_ENV:
        if environ.get(name):
            out.append(Check(name, "ok", "set in the environment"))
        elif name in in_file:
            out.append(Check(name, "ok", f"set in {in_file[name].name} (read by the module that needs it)"))
        else:
            out.append(Check(name, "info", f"not set (optional: {purpose})"))
    if environ.get("TOKEN_FACTORY_API_KEY") or "TOKEN_FACTORY_API_KEY" in in_file:
        for name, purpose in TOKEN_FACTORY_MODELS:
            if not (environ.get(name) or name in in_file):
                out.append(Check(name, "warn", f"not set, so {purpose} falls back offline even with the key (see .env.example)"))
    return out


def check_disk(out_dir: str | Path = "out") -> Check:
    try:
        p = Path(out_dir).resolve()
    except (OSError, ValueError, TypeError) as e:
        return Check("disk", "fail", f"invalid output path {out_dir!r}: {e}")
    if p.is_file():
        return Check("disk", "fail", f"{out_dir} is an existing file, not a directory")

    probe = p
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists():
        return Check("disk", "fail", f"path does not exist and has no existing parent: {out_dir}")

    try:
        free = shutil.disk_usage(probe).free / GIB
    except OSError as e:
        return Check("disk", "fail", f"cannot read disk usage for {probe}: {type(e).__name__}")
    writable = os.access(probe, os.W_OK)
    text = f"{free:.1f} GiB free where {out_dir} will be written"
    if not writable:
        return Check("disk", "fail", f"{text}, but {probe} is not writable")
    if free < FREE_FAIL_GIB:
        return Check("disk", "fail", f"{text}; need at least {FREE_FAIL_GIB} GiB")
    if free < FREE_WARN_GIB:
        return Check("disk", "warn", f"{text}; low (under {FREE_WARN_GIB:g} GiB)")
    return Check("disk", "ok", text)


def run_checks(out_dir: str | Path = "out", environ: dict[str, str] | None = None, env_files: list[Path] | None = None) -> list[Check]:
    return ([check_python()] + check_core() + [check_mujoco(), check_ffmpeg()]
            + check_env(environ, env_files) + [check_disk(Path(out_dir))])


def exit_code(checks: list[Check]) -> int:
    return 1 if any(c.status == "fail" for c in checks) else 0


def render(checks: list[Check]) -> str:
    tag = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL", "info": "info"}
    lines = [f"[{tag.get(c.status, c.status)}] {c.name}: {c.detail}" for c in checks]
    bad = [c for c in checks if c.status == "fail"]
    warn = [c for c in checks if c.status == "warn"]
    lines.append(f"{len(bad)} problem(s), {len(warn)} warning(s)" if bad or warn else "everything needed is in place")
    return "\n".join(lines)


def render_json(checks: list[Check]) -> str:
    return json.dumps({"ok": exit_code(checks) == 0, "checks": [asdict(c) for c in checks]}, indent=2)
