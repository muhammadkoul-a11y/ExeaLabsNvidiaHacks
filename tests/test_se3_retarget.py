"""G7: executed-source SE(3) retargeting; no external framework, service or GPU required."""
import copy
import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.export.provenance import canonical_sha256  # noqa: E402
from arrival.export.validate import validate  # noqa: E402
from arrival.generate.se3_retarget import (  # noqa: E402
    BRIDGE_STEPS, GENERATOR, RetargetPolicy, generate, interpolate, inverse, main, pose, record_source,
    retarget_trajectory, rigid,
)
from arrival.sim.collect import SimEnv  # noqa: E402
from arrival.sim.policy import ScriptedExpert  # noqa: E402
from arrival.sim.tasks import finish_task, make_tasks  # noqa: E402


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def test_full_se3_roundtrip_and_minimum_jerk_bridge():
    # Nonplanar reference rotations: yaw-only and translation-only implementations cannot pass.
    q = np.array([0.7, 0.2, -0.3, 0.5])
    q /= np.linalg.norm(q)
    r = np.empty(9)
    mujoco.mju_quat2Mat(r, q)
    a = pose([1, -2, .5], r.reshape(3, 3))
    b = pose([-.2, .3, 1.2], rz(np.pi))
    eef = pose([.4, .5, .8], rz(-.4))
    relative = inverse(a) @ eef
    np.testing.assert_allclose(inverse(b) @ (b @ relative), relative, atol=1e-12)
    assert not np.allclose((b @ relative)[:3, :3], eef[:3, :3])
    samples = [{"segment": "object", "eef_relative": relative, "gripper": 0.0},
               {"segment": "sink", "eef_relative": inverse(b) @ eef, "gripper": .05}]
    trajectory = retarget_trajectory(samples, {"object": b, "sink": a}, eef)
    np.testing.assert_allclose(inverse(b) @ trajectory[BRIDGE_STEPS][0], relative, atol=1e-12)
    np.testing.assert_allclose(inverse(a) @ trajectory[-1][0], inverse(b) @ eef, atol=1e-12)
    assert all(g == 0.0 for _, g in trajectory[BRIDGE_STEPS + 1:-1])  # bridge holds the grasp
    assert trajectory[-1][1] == .05
    frames = interpolate(a, b, steps=20)
    for t in frames:
        rigid(t)
    np.testing.assert_allclose(frames[-1], b, atol=1e-12)
    np.testing.assert_allclose(frames[9][:3, 3], (a[:3, 3] + b[:3, 3]) / 2)
    # Quintic timing starts and ends more slowly than it crosses the middle.
    assert np.linalg.norm(frames[0][:3, 3] - a[:3, 3]) < np.linalg.norm(frames[10][:3, 3] - frames[9][:3, 3]) / 10
    np.testing.assert_allclose(interpolate(a, a)[-1], a, atol=1e-12)


@pytest.mark.parametrize("bad", [np.zeros((4, 4)), np.full((4, 4), np.nan), np.diag([-1, 1, 1, 1])])
def test_non_rigid_frames_are_rejected(bad):
    with pytest.raises(ValueError, match="SE\\(3\\)"):
        inverse(bad)


@pytest.fixture(scope="module")
def twin():
    from arrival.scan.bundle import load_room
    from arrival.twin.compile import compile_twin

    return compile_twin(load_room("hero_kitchen", "scans"))


@pytest.fixture(scope="module")
def source(twin):
    task = make_tasks(twin, ["counter_to_sink"], 1, seed=0, people=0, log=lambda *_: None)[0]
    finish_task(task)
    with SimEnv(twin, task, res=32, cameras=()) as env:
        return record_source(env, task)


@pytest.fixture(scope="module")
def result(twin, source, tmp_path_factory):
    # Two physics replays; the source is recorded once, separately. Also move the physical sink basin.
    twin = copy.deepcopy(twin)
    next(f for f in twin["fixtures"] if f["id"] == source["task"]["target"])["position"][0] += .025
    def no_fresh_expert(*args):
        pytest.fail("retargeting must consume the recording, not plan a fresh expert episode")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ScriptedExpert, "reset", no_fresh_expert)
        return generate(twin, tmp_path_factory.mktemp("se3"), source=source, res=32,
                        offsets=[(.04, -.015, .1), (-.04, .015, -.1)], log=lambda *_: None)


def test_shifted_rotated_starts_and_shifted_sink_succeed_in_physics(result, source):
    report = result["report_data"]
    assert report["successes"] == report["attempts"] == 2, report
    assert report["dropped"] == 0
    assert validate(result["dataset"]) == []
    rows = report["rows"]
    assert all(r["human_contact"] is False and r["detail"]["object_speed"] < .05 for r in rows)
    assert np.ptp([r["object_frame"][0][3] for r in rows]) > .05
    assert abs(rows[0]["object_frame"][1][0] - rows[1]["object_frame"][1][0]) > .1
    assert rows[0]["sink_frame"][0][3] == pytest.approx(source["sink_frame"][0][3] + .025)
    # The oracle requires the object below the counter top and inside the basin, not hovering over it.
    assert all(r["detail"]["object_pos"][2] < .92 for r in rows)


def test_export_records_source_and_frames_without_mimicgen_claim(result, source):
    root = result["dataset"]
    info = json.loads((root / "meta/info.json").read_text())
    assert info["arrival_generator"] == GENERATOR
    assert info["arrival_physically_valid"] is True
    assert info["arrival_provenance"]["run"]["generator"] == GENERATOR
    assert info["arrival_source_seed"] == source["source_seed"] == 0
    assert json.loads((root / info["arrival_source_ref"]).read_text()) == source
    assert json.loads((root / "meta/embodiment.json").read_text())["joints_are_placeholder"] is False
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    assert all(e["arrival_source_id"] == source["source_id"] and not e["arrival_human_contact"] for e in episodes)
    assert all(set(e["arrival_retarget_frames"]) == {"object", "sink"} for e in episodes)
    # Both segments come from actual recorded poses, with joint motion, rather than expert waypoint metadata.
    assert {s["segment"] for s in source["samples"]} == {"object", "sink"}
    assert np.ptp(np.asarray(source["states"])[:, :7], axis=0).max() > .1
    assert source["source_id"] == canonical_sha256({k: v for k, v in source.items() if k != "source_id"})


def test_unreachable_attempt_is_not_exported(twin, source, tmp_path):
    r = generate(twin, tmp_path, source=source, res=32, offsets=[(3, 0, 0)], log=lambda *_: None)
    assert r["report_data"]["rows"][0]["outcome"] == "drop_unreachable"
    assert r["info"]["total_episodes"] == 0 and r["info"]["arrival_dropped"] == 1
    assert not list(r["dataset"].glob("data/**/*.parquet"))


def test_lost_grasp_is_not_exported(twin, source, tmp_path, monkeypatch):
    original = RetargetPolicy.act

    def open_fingers(self, obs):
        action = original(self, obs).copy()
        action[7] = .05  # sabotage the grasp; the physics oracle must drop the replay
        return action

    monkeypatch.setattr(RetargetPolicy, "act", open_fingers)
    r = generate(twin, tmp_path, source=source, res=32, offsets=[(.02, 0, .05)], log=lambda *_: None)
    assert r["report_data"]["rows"][0]["outcome"] == "drop_physics_fail"
    assert r["info"]["total_episodes"] == 0
    assert not list(r["dataset"].glob("data/**/*.parquet"))


def test_source_tampering_is_rejected_before_replay(source):
    bad = copy.deepcopy(source)
    bad["samples"][0]["eef_relative"][0][3] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        RetargetPolicy(bad)


def test_human_contact_result_is_never_exported(twin, source, tmp_path, monkeypatch):
    from arrival.generate import se3_retarget

    run = se3_retarget.run_episode

    def mark_contact(*args):
        episode = run(*args)
        assert episode["success"]  # isolate the contact gate from the basin success gate
        episode["safety"]["human_contact"] = True
        return episode

    monkeypatch.setattr(se3_retarget, "run_episode", mark_contact)
    r = generate(twin, tmp_path, source=source, res=32, offsets=[(.02, 0, .05)], log=lambda *_: None)
    assert r["report_data"]["rows"][0]["outcome"] == "drop_physics_fail"
    assert r["info"]["total_episodes"] == 0
    assert not list(r["dataset"].glob("data/**/*.parquet"))


def test_cli_rejects_nonobject_source_without_outputs(tmp_path):
    source_path = tmp_path / "bad.json"
    source_path.write_text("[]")
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as error:
        main(["--source", str(source_path), "--out", str(out)])
    assert error.value.code == 2
    assert not out.exists()


@pytest.mark.parametrize("args", [["--attempts", "0"], ["--seed", "-1"], ["--res", "40"], ["--room", "missing"]])
def test_cli_rejects_bad_arguments_without_outputs(args, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as error:
        main([*args, "--out", str(out)])
    assert error.value.code == 2
    assert not out.exists()
