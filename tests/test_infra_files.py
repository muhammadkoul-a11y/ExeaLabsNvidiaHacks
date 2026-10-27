import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_TARGETS = {"install", "test", "sim", "validate", "serve", "clean", "demo"}


def _ci():
    return yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))


def _steps(ci):
    return [s for job in ci["jobs"].values() for s in job["steps"]]


def _make_targets():
    return set(re.findall(r"^([A-Za-z_][\w-]*):", (ROOT / "Makefile").read_text(encoding="utf-8"), re.M))


def test_ci_runs_pytest_demo_and_validate():
    runs = "\n".join(s.get("run", "") for s in _steps(_ci()))
    assert "pytest -q -p no:cacheprovider" in runs
    assert "python -m arrival demo --episodes 2" in runs
    assert "python -m arrival validate" in runs
    assert "scripts/demo_all.py" in runs


def test_ci_is_offline_and_headless():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "secrets." not in text
    ci = _ci()
    job = ci["jobs"]["test"]
    assert "MUJOCO_GL" in text and "PYOPENGL_PLATFORM" in text and "preflight()" in text
    assert "libosmesa6" in text and "libegl1" in text and "libgl1" in text
    assert job["timeout-minutes"] == 20
    assert job["strategy"]["matrix"]["python-version"] == ["3.11", "3.12"]
    assert ci["concurrency"]["cancel-in-progress"] is True


def test_makefile_has_required_and_readme_targets():
    targets = _make_targets()
    assert REQUIRED_TARGETS <= targets
    section = (ROOT / "README.md").read_text(encoding="utf-8").split("## Get started", 1)[1].split("\n## ", 1)[0]
    named = set(re.findall(r"\bmake ([A-Za-z_][\w-]*)", section))
    assert named <= targets


def test_makefile_clean_only_removes_out():
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "OUT ?= out" in text
    assert "rm -rf $(OUT)" in text.split("\nclean:", 1)[1]


def test_make_install_installs_what_make_lint_and_contributing_need():
    # make lint runs ruff, which only the lint extra installs; CONTRIBUTING offers `make install` as the same step
    install = (ROOT / "Makefile").read_text(encoding="utf-8").split("\ninstall:", 1)[1].split("\n\n", 1)[0]
    make_extras = set(re.search(r'-e "\.\[([\w,]+)\]"', install).group(1).split(","))
    guide = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    guide_extras = set(re.search(r'pip install -e "\.\[([\w,]+)\]"\s+# or: make install', guide).group(1).split(","))
    assert "lint" in make_extras
    assert make_extras == guide_extras


@pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")
def test_make_dry_run_parses():
    r = subprocess.run(["make", "-n", "test", "sim", "validate", "demo"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "scripts/demo_all.py" in r.stdout
    assert "--episodes 2 --people 1 --res 64" in r.stdout


def test_dockerfile_and_ignore():
    d = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in d
    assert "MUJOCO_GL=egl" in d and 'pip install -e ".[sim]"' in d
    assert 'CMD ["python", "-m", "arrival", "serve", "--host", "0.0.0.0"]' in d
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    for p in ("out/", ".venv/", ".git/", "ios/", "*.egg-info"):
        assert p in ignored


def _docker_up():
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(["docker", "info", "--format", "{{.OSType}}"], capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return False
    return probe.returncode == 0 and probe.stdout.strip() == "linux"


@pytest.mark.skipif(not _docker_up(), reason="requires a Docker daemon running Linux containers")
def test_docker_image_builds_and_runs():
    b = subprocess.run(["docker", "build", "-q", "-t", "arrival-test", "."], cwd=ROOT, capture_output=True, text=True)
    assert b.returncode == 0, b.stderr
    r = subprocess.run(["docker", "run", "--rm", "arrival-test", "python", "-m", "arrival", "validate", "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("script", ["fetch_menagerie", "make_demo_video", "make_detection_fixture", "render_contributors"])
def test_generator_scripts_answer_help_instead_of_running(script):
    """These take no options and rewrite tracked files (docs/media, test fixtures, README) or clone 34 MB when run."""
    if script in ("make_demo_video", "make_detection_fixture"):
        pytest.importorskip("mujoco")
    path = ROOT / "scripts" / f"{script}.py"
    h = subprocess.run([sys.executable, str(path), "--help"], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert h.returncode == 0 and h.stdout.startswith("usage:")
    bad = subprocess.run([sys.executable, str(path), "--bogus"], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert bad.returncode == 2 and "unrecognized arguments" in bad.stderr
