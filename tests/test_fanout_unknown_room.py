"""fanout with the local backend refuses a --room that has no scan before any shard runs. Every shard's sim used to
fail to load it, be retried until max_attempts, and the run said "rerun to resume" while run.json kept the misspelt
room, so a rerun with the right one under the same --run-id was refused as "started with different settings"."""
import shutil

import pytest

from arrival import cli
from conftest import SCANS

pytest.importorskip("mujoco")  # cmd_fanout checks --families against arrival.sim.tasks


@pytest.fixture
def submitted(monkeypatch):
    """LocalBackend.submit raises instead of starting a shard, so a test sees how far a run got without a sim."""
    def submit(self, cfg, i):
        raise RuntimeError(f"submitted shard {i} of room {cfg.room}")
    monkeypatch.setattr("arrival.jobs.fanout.LocalBackend.submit", submit)


def test_unknown_room_is_a_usage_error_before_any_shard(tmp_path, capsys, submitted):
    rc = cli.main(["fanout", "--room", "hero_kitchn", "--shards", "2", "--episodes", "2", "--out", str(tmp_path)])
    assert rc == cli.EXIT_USAGE
    assert capsys.readouterr().err.startswith("fanout: no scan for room 'hero_kitchn' under ")
    assert not (tmp_path / "fanout_store").exists() and not (tmp_path / "fanout_scratch").exists()


def test_the_corrected_room_then_runs_under_the_same_run_id(tmp_path, submitted):
    argv = ["fanout", "--shards", "2", "--episodes", "2", "--out", str(tmp_path)]
    assert cli.main(argv + ["--room", "hero_kitchn"]) == cli.EXIT_USAGE
    with pytest.raises(RuntimeError, match="submitted shard 0 of room hero_kitchen"):
        cli.main(argv + ["--room", "hero_kitchen"])


def test_a_room_under_the_given_scans_folder_is_accepted(tmp_path, submitted):
    scans = tmp_path / "my_scans"
    shutil.copytree(SCANS / "hero_kitchen", scans / "my_kitchen")
    with pytest.raises(RuntimeError, match="submitted shard 0 of room my_kitchen"):
        cli.main(["fanout", "--room", "my_kitchen", "--scans", str(scans), "--out", str(tmp_path / "out")])


def test_nebius_shards_are_not_checked_against_local_scans(tmp_path, capsys, monkeypatch):
    """A Nebius job reads the scans baked into its image, which this machine cannot see."""
    for key in ("NEBIUS_IAM_TOKEN", "NEBIUS_PROJECT_ID", "NEBIUS_JOB_IMAGE"):
        monkeypatch.delenv(key, raising=False)
    rc = cli.main(["fanout", "--backend", "nebius", "--room", "only_in_the_image", "--out", str(tmp_path)])
    assert rc == cli.EXIT_USAGE
    assert "no scan for room" not in capsys.readouterr().err
