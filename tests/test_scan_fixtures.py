"""Every fixture a scan can produce gets a MuJoCo body, a sink with nothing under it gets its own basin, a
scanned kitchen gets tasks on its cabinets, and the drawer sits in the counter's own frame on any wall (#42)."""

import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.catalog.fixtures import DEFAULT_SIZES, FIXTURE_MAP  # noqa: E402
from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim.scene import build_mjcf, drawer_x, front_stand, part_names  # noqa: E402
from arrival.sim.preflight import BASE_CLEAR  # noqa: E402
from arrival.sim.tasks import make_tasks  # noqa: E402
from arrival.twin.compile import _rect_distance, compile_twin, is_counter_host, to_world  # noqa: E402

SCANS = "tests/fixtures/scans"
WALLS = [{"id": "wall_n", "start": [0.0, 0.0], "end": [4.0, 0.0], "height": 2.4, "thickness": 0.02},
         {"id": "wall_e", "start": [4.0, 0.0], "end": [4.0, 3.0], "height": 2.4, "thickness": 0.02},
         {"id": "wall_s", "start": [4.0, 3.0], "end": [0.0, 3.0], "height": 2.4, "thickness": 0.02},
         {"id": "wall_w", "start": [0.0, 3.0], "end": [0.0, 0.0], "height": 2.4, "thickness": 0.02}]


def _bundle(fixtures: list[dict]) -> dict:
    return {"room_id": "one_fixture", "geometry_source": "roomplan", "walls": WALLS, "objects": [], "fixtures": fixtures}


def _fixture(fid: str, category: str, size, position, yaw=0.0) -> dict:
    return {"id": fid, "category": category, "size": list(size), "position": list(position), "yaw": yaw}


def _model(twin: dict):
    m = mujoco.MjModel.from_xml_string(build_mjcf(twin, (2.0, 2.0, 0.75), 90.0))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def _solid_geoms(m, prefix: str) -> list[str]:
    return [m.geom(i).name for i in range(m.ngeom)
            if m.geom(i).name.startswith(prefix) and (m.geom_contype[i] or m.geom_conaffinity[i])]


@pytest.mark.parametrize("category", sorted(FIXTURE_MAP))
def test_every_scanned_category_gets_a_solid_body(category):
    size = DEFAULT_SIZES.get(category, [0.5, 0.5, 0.5])
    twin = compile_twin(_bundle([_fixture("it", category, size, [2.0, 0.4, size[2] / 2])]))
    m, _ = _model(twin)
    bodies = {m.body(i).name for i in range(m.nbody)}
    assert "it" in bodies or category == "sink", category  # a sink is cut into the counter round it
    assert _solid_geoms(m, "it"), category  # the arm, the base and the walkers cannot pass through it


def test_sink_with_nothing_under_it_gets_a_basin_unit():
    # the compiler synthesizes a counter round a lone sink; a twin authored by hand, or one with that counter
    # removed, has only the sink
    twin = compile_twin(_bundle([_fixture("sink", "sink", [0.6, 0.5, 0.25], [2.0, 0.4, 0.85])]))
    twin["fixtures"] = [f for f in twin["fixtures"] if f["type"] == "sink"]
    assert [f["id"] for f in twin["fixtures"]] == ["sink"]
    m, d = _model(twin)
    assert {"sink", "sink_faucet_lever"} <= {m.body(i).name for i in range(m.nbody)}
    assert _solid_geoms(m, "sink_carcass") and _solid_geoms(m, "sink_wall")
    assert np.allclose(d.geom("sink_bottom").xpos[:2], [2.0, 0.4], atol=0.01)
    assert [p["joint"] for p in part_names(twin)] == ["sink_faucet"]


def test_u_kitchen_42_is_all_there():
    twin = compile_twin(load_room("u_kitchen_42", SCANS))
    m, _ = _model(twin)
    bodies = {m.body(i).name for i in range(m.nbody)}
    for f in twin["fixtures"]:
        assert f["id"] in bodies or f["type"] == "sink", f["id"]
        assert _solid_geoms(m, f["id"]), f["id"]
    assert m.body(int(m.geom("sink_bottom").bodyid[0])).name == "storage_sink"
    joints = [p["joint"] for p in part_names(twin)]
    assert joints.count("sink_faucet") == 1
    assert {"storage_a_door", "storage_b_drawer0", "dishwasher_door", "oven_oven_door", "fridge_door"} <= set(joints)


def test_drawer_and_sink_hole_follow_the_run_on_a_west_wall():
    twin = compile_twin(load_room("rotated_kitchen", SCANS))
    c = next(f for f in twin["fixtures"] if f["id"] == "counter_w")
    assert math.isclose(drawer_x(c, twin["fixtures"]), -c["size"][0] / 4)  # the fridge is beyond the far end
    m, d = _model(twin)
    expect = to_world((-c["size"][0] / 4, c["size"][1] / 2 - 0.02), c)
    assert np.allclose(d.body("counter_w_drawer0").xpos[:2], expect, atol=0.01)
    sink = next(f for f in twin["fixtures"] if f["id"] == "sink_w")
    assert np.allclose(d.geom("sink_w_bottom").xpos[:2], sink["position"][:2], atol=0.01)


def test_hero_kitchen_builds_as_before(hero_twin):
    m, d = _model(hero_twin)
    names = {m.geom(i).name for i in range(m.ngeom)}
    assert not any(n.endswith("_box") and not n.startswith(("fridge", "dishwasher", "microwave", "stove")) for n in names)
    assert not any(n.endswith("_carcass") and n.startswith("sink") for n in names)
    assert m.body(int(m.geom("sink_1_bottom").bodyid[0])).name == "counter_a"
    assert np.allclose(d.geom("sink_1_bottom").xpos[:2], [1.6, 0.4], atol=1e-6)
    assert [f["id"] for f in hero_twin["fixtures"] if is_counter_host(f)] == ["counter_a", "counter_b"]


def test_no_drawer_or_door_task_where_the_base_cannot_stand():
    twin = compile_twin(load_room("u_kitchen_42", SCANS))
    fams = ["open_drawer", "open_door", "close_drawer", "close_door"]
    tasks = make_tasks(twin, fams, 40, seed=0)
    used = {t["fixture"] for t in tasks}
    assert {"storage_sink", "storage_a"} <= used  # storage_b is in the corner of the U: its doors stand along the run
    assert {t["family"] for t in tasks if t["fixture"] == "storage_b"} <= {"open_door", "close_door"}
    fx = {f["id"]: f for f in twin["fixtures"]}
    for t in tasks:
        if t.get("base_shift"):
            continue  # stands off the rule's spot; test_corner_cabinet_... checks that one
        c = fx[t["fixture"]]
        along, out = front_stand(c, t["family"], twin["fixtures"])
        spot = to_world((along, c["size"][1] / 2 + out), c)
        assert min(_rect_distance(list(spot), f) for f in twin["fixtures"]) >= BASE_CLEAR, t["task_id"]


def test_corner_cabinet_door_stands_along_its_run_but_the_blocked_drawer_and_outside_fridge_stay_rejected():
    """u_kitchen_42's storage_b has the sink run on its stand spot: its door takes a spot 0.25 m along the run (the
    wide fallback grid) and opens in the sim; its drawer is pulled from in front of the sink, so it stays
    rejected. rotated_kitchen's fridge is hinged on its corner side (inferred from the walls), so its free edge and
    the stand spot beside it are in the room."""
    from arrival.sim.collect import SimEnv, run_episode
    from arrival.sim.policy import ScriptedExpert
    from arrival.sim.tasks import finish_task

    twin = compile_twin(load_room("u_kitchen_42", SCANS))
    pre: dict = {}
    corner = []
    for fam in ("open_door", "close_door", "open_drawer", "close_drawer"):
        pre_f: dict = {}
        tasks = make_tasks(twin, [fam], 24, seed=5, log=lambda *_: None, preflight=pre_f)
        corner += [t for t in tasks if t.get("fixture") == "storage_b"]
        pre["rejected"] = pre.get("rejected", 0) + pre_f["rejected"]
        pre.setdefault("reasons", []).extend(pre_f["reasons"])
    assert {t["family"] for t in corner} == {"open_door"}  # close_door is not offered at the corner: see reach.WIDE_FAMILIES
    assert all(t["base_shift"] == [0.25, 0.0] for t in corner)
    assert pre["rejected"] == 3  # open_drawer, close_drawer (the sink run is on the stand spot) and close_door (not offered at the corner)
    for fam in ("open_door",):
        t = finish_task(next(t for t in corner if t["family"] == fam))
        ep = run_episode(SimEnv(twin, t, res=32), ScriptedExpert(), t)
        assert ep["success"], fam
    room = compile_twin(load_room("rotated_kitchen", SCANS))
    pre = {}
    assert make_tasks(room, ["open_fridge"], 4, seed=5, log=lambda *_: None, preflight=pre)
    assert pre["families"] == [] and pre["rejected"] == 0
