"""A real scan is not the hero layout (#42): tasks are built in each fixture's own frame, a layout the base cannot stand
in is rejected before any sim time is spent, a task that still crashes is skipped with a reason and never written,
and the walkers plan in the work wall's frame, so rotating the whole room rotates their paths.

Rooms: `rotated_kitchen` (runs on the W, S and E walls), `u_kitchen_42` (a U with a corner cabinet), the hero
kitchen (north wall)."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.export.validate import validate  # noqa: E402
from arrival.jobs.queue import create_job  # noqa: E402
from arrival.jobs.worker import run_once  # noqa: E402
from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim.collect import SimEnv, collect, run_episode  # noqa: E402
from arrival.sim.people import Walkers, frame  # noqa: E402
from arrival.sim.policy import ScriptedExpert  # noqa: E402
from arrival.sim.preflight import base_pose, check_task  # noqa: E402
from arrival.sim.tasks import SIM_FAMILIES, finish_task, make_tasks  # noqa: E402
from arrival.twin.compile import compile_twin, to_local  # noqa: E402

SCANS = "tests/fixtures/scans"
QUIET = lambda *_: None  # noqa: E731
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rotated():
    return compile_twin(load_room("rotated_kitchen", SCANS))


def _hinge_left_scan() -> dict:
    """rotated_kitchen's scan with its fridge said to be hinged on the left, the corner side: the scan's hinge wins
    over the one the compiler infers from the walls, so the stand spot beside the free edge is outside the room."""
    scan = json.loads((REPO / SCANS / "rotated_kitchen" / "scan.json").read_text())
    next(f for f in scan["fixtures"] if f["id"] == "fridge_w")["hinge"] = "left"
    return scan


@pytest.fixture(scope="module")
def rotated_hinged_left():
    from arrival.scan.bundle import validate

    return compile_twin(validate(_hinge_left_scan()))


@pytest.fixture(scope="module")
def ushape():
    return compile_twin(load_room("u_kitchen_42", SCANS))


# ---- pre-flight ----

def test_preflight_keeps_every_hero_variation(hero_twin):
    pre = {}
    tasks = make_tasks(hero_twin, list(SIM_FAMILIES), 40, seed=1, log=QUIET, preflight=pre)
    # counter_to_counter has no second run to carry to; move_on_table needs a scanned table, a kitchen has none; and
    # the hero scan has no dishwasher, so its three families have nothing to work on
    skipped = {"counter_to_counter", "move_on_table", "open_dishwasher", "load_dishwasher", "unload_dishwasher"}
    assert pre["rejected"] == 0 and set(pre["families"]) == skipped
    assert {t["family"] for t in tasks} == set(SIM_FAMILIES) - skipped
    assert all(check_task(hero_twin, t) is None for t in tasks)


def test_mug_beyond_the_hands_reach_gets_a_base_that_reaches_it(rotated):
    # over mug_2 on the east run the IK solve from the rule's stand spot falls 1.3 cm short at the 0 offset (0.76 m
    # out) and 2.6 cm at +0.08 m (0.77 m out), and those episodes miss (L17 sweep), so the reach planner moves the
    # base 0.1 m along the run instead of dropping the variation; from there it succeeds at all four offsets
    pre, log = {}, []
    tasks = make_tasks(rotated, ["put_in_drawer"], 8, seed=1, log=log.append, preflight=pre)
    assert {t["object_id"] for t in tasks} == {"mug_1", "mug_2"} and pre["rejected"] == 0 and pre["shifted"] == 1
    assert all(t.get("base_shift") == [0.1, 0.0] for t in tasks if t["object_id"] == "mug_2")
    assert all("base_shift" not in t for t in tasks if t["object_id"] == "mug_1")
    assert any("mug_2 stands +0.10 m along" in line and "counter_e" in line for line in log)


def test_fridge_in_the_corner_is_rejected_because_the_base_would_stand_outside(rotated_hinged_left):
    pre, log = {}, []
    tasks = make_tasks(rotated_hinged_left, ["open_fridge", "open_drawer"], 6, seed=2, log=log.append, preflight=pre)
    assert {t["family"] for t in tasks} == {"open_drawer"}  # the fridge's free edge is at the corner of the room
    assert pre["families"] == ["open_fridge"] and pre["rejected"] == 1
    assert "outside the room" in pre["reasons"][0]["reason"] and "fridge_w" in pre["reasons"][0]["reason"]
    assert any("rejected by pre-flight" in line for line in log)


def test_stand_spot_must_clear_the_walls(hero_twin):
    task = make_tasks(hero_twin, ["open_drawer"], 1, seed=0, log=QUIET)[0]
    assert check_task(hero_twin, task) is None
    base, _, anchor = base_pose(hero_twin, task)
    tight = copy.deepcopy(hero_twin)  # a wall 8 cm behind the base column
    tight["walls"].append({"id": "wall_extra", "start": [base[0] - 0.5, base[1] + 0.08],
                           "end": [base[0] + 0.5, base[1] + 0.08], "height": 2.4, "thickness": 0.02})
    why = check_task(tight, task)
    assert why and "wall_extra" in why and "needs 0.15 m" in why


def test_pick_and_place_points_must_be_within_reach(hero_twin):
    task = make_tasks(hero_twin, ["counter_to_sink"], 1, seed=0, log=QUIET)[0]
    assert check_task(hero_twin, task) is None
    far = copy.deepcopy(hero_twin)
    sink = next(f for f in far["fixtures"] if f["id"] == task["target"])
    sink["position"][0] += 1.2  # the base parks between the mug and the sink, so a far sink pulls it out of reach
    why = check_task(far, task)
    assert why and "beyond the arm's reach" in why and "IK solve misses" in why


def test_preflight_rejection_is_counted_in_the_dataset_stats(ushape, tmp_path):
    res = collect(ushape, ["open_drawer", "open_door"], 2, tmp_path, people=0, res=32, log=QUIET)
    r, info = res["report_data"], res["info"]
    assert r["skipped_unreachable"] == 1 and r["skipped_error"] == 0 and r["skipped"] == 1  # storage_b's drawer; its door stands along the run
    assert {x["stage"] for x in r["skip_reasons"]} == {"preflight"}
    assert "storage_b" in r["skip_reasons"][0]["reason"] and "sink" in r["skip_reasons"][0]["reason"]
    assert info["arrival_skipped_unreachable"] == 1 and info["arrival_attempts"] == r["attempts"] == 2


# ---- tasks in the fixture's frame ----

def test_start_offsets_run_along_the_counter_whatever_wall_it_is_on(rotated):
    tasks = make_tasks(rotated, ["wipe_counter", "put_in_drawer", "counter_to_sink"], 30, seed=4, log=QUIET)
    fx = {f["id"]: f for f in rotated["fixtures"]}
    objs = {o["id"]: o for o in rotated["objects"]}
    offsets = {(0.0, 0.0), (0.08, 0.0), (-0.08, 0.0), (0.0, 0.04)}  # POSE_OFFSETS in the counter's own x and y
    seen = set()
    for t in tasks:
        src = fx[t["source"]]
        seen.add(src["id"])
        a, b = to_local(objs[t["object_id"]]["position"][:2], src), to_local(t["start"][:2], src)
        got = (round(b[0] - a[0], 4), round(b[1] - a[1], 4))
        # along the same axis and way as a POSE_OFFSETS entry; shorter only where it would cross the counter's edge
        assert any(all((g == 0) == (c == 0) and g * c >= 0 and abs(g) <= abs(c) for g, c in zip(got, o))
                   for o in offsets), (t["task_id"], src["id"], got)
        assert abs(b[0]) < src["size"][0] / 2 and abs(b[1]) < src["size"][1] / 2  # still on the counter top
    assert {"counter_w", "counter_s"} <= seen  # a west wall run (yaw -90) and a south wall run (yaw 180)


@pytest.mark.parametrize("fixture", ["counter_w", "counter_s", "counter_e"])
def test_every_stand_spot_is_in_front_of_its_fixture_in_the_room(rotated, fixture):
    for fam in ("open_drawer", "open_door", "close_drawer", "close_door"):
        t = {"family": fam, "fixture": fixture}
        base, yaw, anchor = base_pose(rotated, t)
        assert check_task(rotated, t) is None, (fam, fixture)
        lx, ly = to_local(base[:2], anchor)
        assert ly > anchor["size"][1] / 2 + 0.5 and abs(lx) < anchor["size"][0] / 2  # out in front, along the run
        n = np.array([-math.sin(anchor["yaw"]), math.cos(anchor["yaw"])])
        assert np.allclose([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))], -n, atol=1e-9)  # facing it


# ---- running end to end ----

def test_u_shape_runs_end_to_end_with_people(ushape, tmp_path):
    res = collect(ushape, ["counter_to_sink", "open_drawer", "wipe_counter"], 3, tmp_path, people=2, res=32, seed=2,
                  log=QUIET)
    r = res["report_data"]
    assert r["attempts"] == 3 and r["successes"] == 3 and r["human_contacts"] == 0, r["families"]
    assert r["skipped_unreachable"] == 1 and r["skipped_error"] == 0  # storage_b, in the corner of the U
    assert validate(res["dataset"]) == []
    assert res["info"]["arrival_physically_valid"] is True and res["info"]["total_episodes"] == 3


def test_u_shape_sink_cabinet_door_closes(ushape):
    """storage_sink's door opens toward storage_b in the corner of the U. On main the push from in front meets the
    open panel with the hand on the way in and throws the door to 1.887 rad, by its 1.9 rad stop, on every seed tried.
    The push from the side to 40 deg first (#152) shuts it."""
    from arrival.sim.collect import run_episode
    from arrival.sim.tasks import finish_task

    task = finish_task({"family": "close_door", "fixture": "storage_sink", "lighting": "neutral", "people": 0,
                        "seed": 0})
    ep = run_episode(SimEnv(ushape, task, res=32), ScriptedExpert(), task)
    assert ep["success"] and ep["detail"]["qpos"] < np.radians(5), ep["detail"]


def test_rotated_kitchen_runs_end_to_end_on_every_wall(rotated, tmp_path):
    fams = ["open_drawer", "open_door", "put_in_drawer", "counter_to_sink", "wipe_counter"]
    res = collect(rotated, fams, 5, tmp_path, people=1, res=32, seed=3, log=QUIET)
    r = res["report_data"]
    assert r["attempts"] == 5 and r["successes"] == 5 and r["human_contacts"] == 0, r["families"]
    assert validate(res["dataset"]) == []


# ---- skip, never crash ----

def _episodes(res) -> int:
    return res["info"]["total_episodes"]


def test_a_task_that_crashes_in_the_rollout_is_skipped_and_never_written(ushape, tmp_path, monkeypatch):
    real = ScriptedExpert.reset
    calls = []

    def reset(self, env, task):
        calls.append(task["task_id"])
        if len(calls) == 1:
            raise RuntimeError("IK failed for the grasp")
        return real(self, env, task)

    monkeypatch.setattr(ScriptedExpert, "reset", reset)
    logs = []
    res = collect(ushape, ["open_drawer"], 2, tmp_path, people=0, res=32, log=logs.append)
    r = res["report_data"]
    assert (r["attempts"], r["successes"], r["skipped_error"], r["skipped"]) == (1, 1, 1, 1 + r["skipped_unreachable"])
    bad = next(x for x in r["skip_reasons"] if x["stage"] == "rollout")
    assert bad["task_id"] == calls[0] and "IK failed for the grasp" in bad["reason"]
    assert _episodes(res) == 1 and validate(res["dataset"]) == []  # the failed task wrote no episode
    assert any("SKIP at rollout" in line and "IK failed" in line for line in logs)


def test_the_walker_crash_from_42_skips_the_task_at_build(rotated, tmp_path, monkeypatch):
    real = Walkers.__init__
    state = {"n": 0}

    def boom(self, *a, **kw):
        state["n"] += 1
        if state["n"] == 1:
            raise ValueError("high - low < 0")  # numpy's message when the floor strip is inverted
        real(self, *a, **kw)

    monkeypatch.setattr(Walkers, "__init__", boom)
    res = collect(rotated, ["open_drawer"], 2, tmp_path, people=1, res=32, log=QUIET)
    r = res["report_data"]
    assert (r["skipped_error"], r["attempts"], r["successes"]) == (1, 1, 1)
    assert next(x for x in r["skip_reasons"] if x["stage"] == "build")["reason"].startswith("ValueError: high - low < 0")
    assert _episodes(res) == 1


def test_an_unreachable_task_is_not_an_attempt_and_the_run_still_finishes(rotated_hinged_left, tmp_path):
    res = collect(rotated_hinged_left, ["open_fridge"], 2, tmp_path, people=0, res=32, log=QUIET)
    r = res["report_data"]
    assert (r["attempts"], r["successes"], r["skipped_unreachable"]) == (0, 0, 1)
    assert _episodes(res) == 0 and r["families"] == {}


def test_policy_server_errors_are_not_swallowed_as_skips(ushape, tmp_path, monkeypatch):
    from arrival.sim.policy import VLAClient

    def down(self, env, task):
        raise ConnectionRefusedError("policy server down")

    monkeypatch.setenv("GR00T_POLICY_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(VLAClient, "reset", down)
    with pytest.raises(OSError):
        collect(ushape, ["open_drawer"], 1, tmp_path, people=0, res=32, policy_name="vla", log=QUIET)


# ---- worker ----

def _upload(out: Path, room: str) -> dict:
    data = (REPO / SCANS / room / "scan.json").read_bytes()
    return create_job(out / "jobs", f"{room}.json", data=data)


def test_worker_reports_an_unsupported_layout_not_a_traceback(tmp_path):
    out = tmp_path / "out"
    job = create_job(out / "jobs", "rotated_kitchen.json", data=json.dumps(_hinge_left_scan()).encode())
    st = run_once(out, families=("open_fridge",), episodes=2, people=0, res=32, log=QUIET)
    assert st["id"] == job["id"] and st["status"] == "failed" and st["stage"] == "sim"
    assert st["error_kind"] == "unsupported_layout" and st["error"].startswith("unsupported layout: ")
    assert "outside the room" in st["error"] and st["error_reasons"]
    assert st["attempts"] == 0 and st["skipped"] == 1 and st["zip"] is None
    on_disk = json.loads((out / "jobs" / job["id"] / "status.json").read_text())
    assert on_disk["error_kind"] == "unsupported_layout"
    assert not (out / "jobs" / job["id"] / "error.txt").exists()


def test_worker_finishes_a_partial_dataset_and_records_the_skips(tmp_path):
    out = tmp_path / "out"
    _upload(out, "u_kitchen_42")
    st = run_once(out, families=("open_drawer", "open_door"), episodes=2, people=0, res=32, log=QUIET)
    assert st["status"] == "done" and st["successes"] == 2 and st["skipped"] == 1 and st["error_kind"] is None
    assert Path(st["zip"]).is_file()


# ---- the walkers plan in the work wall's frame ----

def _rotate(twin: dict, theta: float) -> list[dict]:
    c, s = math.cos(theta), math.sin(theta)
    return [{**f, "position": [c * f["position"][0] - s * f["position"][1], s * f["position"][0] + c * f["position"][1],
                               f["position"][2]], "yaw": f.get("yaw", 0.0) + theta} for f in twin["fixtures"]]


@pytest.mark.parametrize("theta", [-math.pi / 2, math.pi / 2, math.pi])
def test_rotating_the_room_rotates_the_walkers_paths_and_the_fridge_trip(hero_twin, theta):
    floor, base, front = (0.4, 1.65, 3.6, 2.7), (2.6, 1.0), 0.75
    R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    assert np.allclose(frame(theta) @ R, np.eye(2))
    visits = 0
    for seed in range(8):
        a = Walkers(2, floor, seed=seed, base_xy=base, front_y=front, fixtures=hero_twin["fixtures"])
        b = Walkers(2, floor, seed=seed, base_xy=base, front_y=front, fixtures=_rotate(hero_twin, theta), yaw=theta)
        assert [len(p) for p in a.paths] == [len(p) for p in b.paths]
        for pa, pb in zip(a.paths, b.paths):
            assert np.allclose([R @ p for p in pa], pb, atol=1e-9)
        assert (a.visit is None) == (b.visit is None)
        if a.visit:
            visits += 1
            for k in ("hinge", "centre", "s1", "s2"):
                assert np.allclose(R @ a.visit[k], b.visit[k], atol=1e-9), k
            assert math.isclose(math.cos(b.visit["yaw"] - a.visit["yaw"] - theta), 1.0)
            assert a.visit["q_open"] == b.visit["q_open"] and a.visit["person"] == b.visit["person"]
    assert visits >= 2  # the comparison above covers the fridge trip on a rotated wall


def test_no_strip_in_front_of_the_run_falls_back_to_a_floor_walk_with_a_note(rotated):
    env = SimEnv(rotated, _drawer_task(rotated, "counter_w"), res=32)
    try:
        floor, base, front, note = env._free_floor(env.anchor, env.data.body("robot_base").xpos)
        assert note is None and front is not None
        tiny = {**rotated, "walls": [{**w, "start": [w["start"][0] * 0.3, w["start"][1] * 0.3],
                                       "end": [w["end"][0] * 0.3, w["end"][1] * 0.3]} for w in rotated["walls"]]}
        fake = type("E", (), {"twin": tiny})()
        floor, base, front, note = SimEnv._free_floor(fake, env.anchor, env.data.body("robot_base").xpos)
        assert front is None and base is not None and "no free strip" in note
    finally:
        env.close()


@pytest.mark.parametrize("room", ["rotated_kitchen", "u_kitchen_42"])
def test_start_offsets_keep_the_object_on_its_counter(room, rotated, ushape):
    """The -0.08 m offset put u_kitchen_42's mug 0.04 m past storage_sink's end and rotated_kitchen's bowl 0.025 m
    past counter_w's; the offset now stops short of the edge."""
    twin = {"rotated_kitchen": rotated, "u_kitchen_42": ushape}[room]
    fx = {f["id"]: f for f in twin["fixtures"]}
    half = {o["id"]: max(o["size"][:2]) / 2 for o in twin["objects"]}
    starts = 0
    for holdout in (False, True):
        for mobile in (False, True):
            for t in make_tasks(twin, list(SIM_FAMILIES), 200, seed=0, holdout=holdout, mobile=mobile, log=QUIET):
                if "start" not in t:
                    continue
                starts += 1
                src, h = fx[t["source"]], half[t["object_id"]]
                lx, ly = to_local(t["start"], src)
                assert abs(lx) + h <= src["size"][0] / 2 + 1e-9 and abs(ly) + h <= src["size"][1] / 2 + 1e-9,                     (t["family"], t["object_id"], t["start"])
    assert starts


def test_unload_dishwasher_puts_the_mug_back_on_the_counter_from_the_offset_nearest_the_dishwasher(ushape):
    """unload_dishwasher sets the object down at its offset start. At storage_sink's end, next to the dishwasher,
    the mug went half over the edge and fell onto the dishwasher top (2 of 8 eval episodes at seed 1)."""
    tasks = make_tasks(ushape, ["unload_dishwasher"], 12, seed=1, log=QUIET)
    task = finish_task(min(tasks, key=lambda t: t["start"][1]))
    sink_counter = next(f for f in ushape["fixtures"] if f["id"] == task["source"])
    ep = run_episode(SimEnv(ushape, task, res=32), ScriptedExpert(), task)
    assert ep["success"], ep["detail"]
    assert ep["detail"]["object_pos"][2] > sink_counter["size"][2]  # on the counter top, not the dishwasher (0.885 m)


def _drawer_task(twin: dict, fixture: str) -> dict:
    return {"family": "open_drawer", "fixture": fixture, "handle": f"{fixture}_drawer0_handle",
            "joint": f"{fixture}_drawer0", "lighting": "neutral", "people": 1, "seed": 1}
