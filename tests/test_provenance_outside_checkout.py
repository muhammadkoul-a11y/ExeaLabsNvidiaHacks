"""The code sha in twin and dataset provenance comes from Arrival's own checkout, never from a repository the code only
sits inside: a wheel installed into a venv under some other project recorded that project's HEAD as Arrival's code."""

import shutil
import subprocess

import pytest

from arrival.export import provenance

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@example.com",
                           "-c", "commit.gpgsign=false", *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Another project's checkout with a venv's site-packages inside it, where `pip install` puts arrival/."""
    monkeypatch.delenv("ARRIVAL_GIT_SHA", raising=False)
    root = tmp_path / "project"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "README").write_text("not arrival\n", encoding="utf-8")
    _git(root, "add", "README")
    _git(root, "commit", "-q", "-m", "unrelated")
    site = root / ".venv" / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    return root, site


def test_code_inside_another_checkout_has_no_sha(project):
    _, site = project
    assert provenance.git_state(site) == {"sha": None, "dirty": None, "source": None}


def test_a_checkout_of_its_own_still_reports_its_head(project):
    root, _ = project
    assert provenance.git_state(root) == {"sha": _git(root, "rev-parse", "HEAD"), "dirty": False, "source": "git"}
