import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.export.validate import validate  # noqa: E402
from arrival.generate.retarget import main, retarget  # noqa: E402
from arrival.scan.bundle import load_room  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402


@pytest.fixture(scope="module")
def twin():
    return compile_twin(load_room("hero_kitchen", "scans"))


@pytest.fixture(scope="module")
def run(twin, tmp_path_factory):
    out = tmp_path_factory.mktemp("rt")
    return out, retarget(twin, out, attempts=6, res=64, log=lambda *_: None)


def test_small_run_exports_valid_labelled_dataset(run):
    out, r = run
    rep, info = r["report_data"], json.loads((r["dataset"] / "meta" / "info.json").read_text())
    assert rep["attempts"] == 6 and rep["successes"] >= 3
    assert validate(r["dataset"]) == []
    assert info["arrival_generator"] == "seed_replay_retarget" and info["arrival_physically_valid"] is True
    assert info["arrival_seed_episode_id"] and (out / info["arrival_seed_ref"]).exists()
    assert info["arrival_attempts"] == 6 and info["arrival_dropped"] == 6 - info["arrival_successes"]
    assert "mimicgen" not in json.dumps(info).lower()


def test_small_start_poses_differ(run):
    lines = (run[1]["dataset"] / "meta" / "episodes.jsonl").read_text().splitlines()
    xy = np.array([json.loads(ln)["arrival_start_xy"] for ln in lines])
    assert len(xy) >= 3 and np.ptp(xy, axis=0).max() > 0.01


def test_small_out_of_reach_is_dropped_not_exported(twin, tmp_path):
    r = retarget(twin, tmp_path, offsets=[(0.0, 1.2, 0.0, 0.0)], res=64, log=lambda *_: None)
    assert r["report_data"]["successes"] == 0 and r["report_data"]["drop_reasons"]["unreachable"] == 1
    assert r["info"]["total_episodes"] == 0


@pytest.mark.parametrize(
    "args, message",
    [
        (["--room", "nope_room"], "no scan for room 'nope_room'"),
        (["--attempts", "0"], "--attempts must be >= 1"),
        (["--attempts", "-2"], "--attempts must be >= 1"),
        (["--seed", "-1"], "--seed must be >= 0"),
        (["--res", "0"], "--res must be a positive multiple of 16"),
        (["--res", "40"], "--res must be a positive multiple of 16"),
    ],
)
def test_bad_arguments_are_usage_errors_before_any_run(tmp_path, capsys, args, message):
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as e:
        main([*args, "--out", str(out)])
    assert e.value.code == 2
    assert message in capsys.readouterr().err
    assert not out.exists()
