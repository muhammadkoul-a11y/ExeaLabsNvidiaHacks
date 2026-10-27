"""`arrival fanout` run from outside the repo with a relative --scans or --out. Each local shard runs with its cwd at the
repo root, so the shard used to look for the scans under the repo (every attempt failed with "no scan for room") and
wrote its dataset into the repo's out/ while fanout waited for it under the caller's (exit 2, "wrote no expected shard
folder"). The shard now gets both paths as the caller meant them."""

import shutil
from pathlib import Path

from arrival.jobs import fanout
from arrival.jobs.fanout import FanoutConfig, LocalBackend
from conftest import SCANS


class _Proc:
    def poll(self):
        return 0


def _shard_command(monkeypatch, scans):
    seen = []
    monkeypatch.setattr(fanout.subprocess, "Popen", lambda argv, cwd, **kw: seen.append((argv, Path(cwd))) or _Proc())
    be = LocalBackend(Path("out") / "fanout_store", Path("out") / "fanout_scratch")
    be.poll(be.submit(FanoutConfig("r1", 1, "hero_kitchen", "close_drawer", 1, 0, 32, 0, scans), 0))  # closes the log
    argv, cwd = seen[0]
    return cwd / argv[argv.index("--scans") + 1], cwd / argv[argv.index("--out") + 1]


def test_relative_scans_and_out_reach_the_shard_as_the_caller_meant(tmp_path, monkeypatch):
    shutil.copytree(SCANS / "hero_kitchen", tmp_path / "myscans" / "hero_kitchen")
    monkeypatch.chdir(tmp_path)
    scans, out = _shard_command(monkeypatch, "myscans")
    assert (scans / "hero_kitchen" / "scan.json").is_file()
    assert out.resolve() == (tmp_path / "out" / "fanout_scratch" / "r1" / "0of1").resolve()


def test_default_scans_still_falls_back_to_the_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no scans/ here, so the shard's own resolve_scans finds <repo>/scans
    scans, _ = _shard_command(monkeypatch, "scans")
    assert (scans / "hero_kitchen" / "scan.json").is_file()
