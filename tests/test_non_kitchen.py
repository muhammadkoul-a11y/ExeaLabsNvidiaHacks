"""SIM-5 and L9: an authored non-kitchen RoomPlan fixture (tests/fixtures/roomplan/office.json, a table and two chairs) becomes
a room the arm can work in: the table is a table, furniture is solid, the room holds office things (a laptop, a book,
a cup) instead of a mug and plate, and move_on_table episodes pass physics."""

import random
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.catalog.objects import GRASPABLE, ROOM_OBJECTS, TABLE_MOVABLE, catalog_id, room_kind, sample_size  # noqa: E402
from arrival.scan import roomplan  # noqa: E402
from arrival.sim.collect import SimEnv, run_episode  # noqa: E402
from arrival.sim.policy import ScriptedExpert  # noqa: E402
from arrival.sim.scene import build_mjcf, is_table, part_names  # noqa: E402
from arrival.sim.tasks import SIM_FAMILIES, make_tasks  # noqa: E402
from arrival.twin.compile import compile_twin, place_default_objects, to_local  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "roomplan"


def _room(name: str) -> dict:
    twin = compile_twin(roomplan.load(FIXTURES / f"{name}.json", name))
    place_default_objects(twin, seed=0)
    return twin


@pytest.fixture(scope="module")
def office():
    return _room("office")


def test_table_is_built_as_a_table(office):
    table = next(f for f in office["fixtures"] if is_table(f))
    m = mujoco.MjModel.from_xml_string(build_mjcf(office, (2.0, 0.6, 0.75)))
    names = {m.geom(i).name for i in range(m.ngeom)}
    assert f'{table["id"]}_top' in names and all(f'{table["id"]}_leg{k}' in names for k in range(4))
    joints = {m.joint(i).name for i in range(m.njnt)}
    assert not any(j.startswith(table["id"]) for j in joints)  # no drawers or doors under a dining table
    assert not any(p["fixture"] == table["id"] for p in part_names(office))


def test_furniture_is_solid(office):
    chairs = [f for f in office["fixtures"] if f["source_category"] == "chair"]
    m = mujoco.MjModel.from_xml_string(build_mjcf(office, (2.0, 0.6, 0.75)))
    for c in chairs:
        g = m.geom(f'{c["id"]}_box')
        assert m.geom_contype[g.id] and m.geom_conaffinity[g.id]


def test_office_tasks_are_table_tasks_only(office):
    fams = {t["family"] for t in make_tasks(office, list(SIM_FAMILIES), 40, log=lambda *_: None)}
    assert fams == {"move_on_table"}  # no drawer, door or sink tasks invented for an office


def test_move_on_table_passes_physics(office):
    task = make_tasks(office, ["move_on_table"], 1, seed=0, log=lambda *_: None)[0]
    env = SimEnv(office, task, res=32)
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"], ep["detail"]
    assert ep["detail"]["target_error_m"] < 0.02 and ep["ik_error"] < 0.02


def test_people_walk_the_office_without_crashing(office):
    # the walker crash on a room with no kitchen-style run (`high - low < 0`) is gone since L1 (the walkers use the
    # whole floor when there is no strip), so this room no longer drops its people
    task = make_tasks(office, ["move_on_table"], 1, seed=0, people=2, log=lambda *_: None)[0]
    env = SimEnv(office, task, res=32)
    assert env.walkers is not None and task["people"] == 2 and "people_skipped" not in task
    ep = run_episode(env, ScriptedExpert(), task)
    assert ep["success"], ep["detail"]


def test_desk_behind_a_chair_gets_no_task():
    # study: the desk's only free long side is where its chair stands, so a fixed base has nowhere to stand
    assert make_tasks(_room("study"), ["move_on_table"], 4, log=lambda *_: None) == []


# ---- L9: room-appropriate objects ----

def _fx(category: str) -> dict:
    return {"id": category, "source_category": category}


def test_room_kind_from_scanned_categories(office):
    assert room_kind(office["fixtures"]) == "office"  # a table, chairs and storage
    assert room_kind(_room("study")["fixtures"]) == "bedroom"  # a bed wins over the television
    assert room_kind([_fx("sofa"), _fx("television"), _fx("table")]) == "living_room"
    assert room_kind([_fx("table"), _fx("chair"), _fx("sink")]) == "kitchen"  # one kitchen fixture is enough
    assert room_kind([]) == "kitchen"  # nothing scanned: the pipeline's original assumption
    assert set(ROOM_OBJECTS) == {"kitchen", "office", "living_room", "bedroom"}


def test_office_holds_office_things_not_a_mug_and_plate(office):
    labels = {o["label"] for o in office["objects"]}
    assert labels == {"laptop", "book", "cup"}
    table = next(f for f in office["fixtures"] if is_table(f))
    for o in office["objects"]:  # on the table, inside the band along a long edge a fixed arm can reach
        lx, ly = to_local(o["position"], table)
        assert abs(lx) < table["size"][0] / 2 and 0.0 < table["size"][1] / 2 - abs(ly) <= 0.40
        assert abs(o["position"][2] - (table["position"][2] + table["size"][2] / 2 + o["size"][2] / 2)) < 1e-4


def test_living_room_gets_a_remote_and_a_kitchen_keeps_its_five():
    living = compile_twin(roomplan.load(FIXTURES / "office.json", "office"))
    chair = next(f for f in living["fixtures"] if f["source_category"] == "chair")
    living["fixtures"].append({**chair, "id": "sofa_x", "source_category": "sofa"})
    assert room_kind(living["fixtures"]) == "living_room"
    assert {o["label"] for o in place_default_objects(living, seed=0)} == {"remote", "book", "cup"}
    kitchen = {"walls": [], "objects": [], "fixtures": [
        {"id": "s", "type": "sink", "source_category": "sink", "size": [0.6, 0.5, 0.25], "position": [1.6, 0.4, 0.85],
         "yaw": 0.0},
        {"id": "c", "type": "counter", "source_category": "counter", "size": [3.0, 0.65, 0.92],
         "position": [1.5, 0.35, 0.46], "yaw": 0.0}]}
    assert {o["label"] for o in place_default_objects(kitchen, seed=0)} == set(ROOM_OBJECTS["kitchen"])


def test_book_and_remote_are_graspable_and_laptop_is_scenery():
    assert {"book", "remote"} <= set(GRASPABLE) and {"book", "remote"} <= set(TABLE_MOVABLE)
    assert "laptop" not in GRASPABLE  # 30 x 21 cm: wider than the 10 cm gripper opening in both directions


@pytest.mark.parametrize("label", ["book", "remote", "cup"])
def test_objects_move_on_the_table_in_physics(office, label):
    # the office's own objects are a laptop, a book and a cup; put the label under test where the book sits
    base = next(o for o in office["objects"] if o["label"] == "book")
    size = sample_size(label, random.Random(0))
    z = base["position"][2] - base["size"][2] / 2 + size[2] / 2
    obj = {**base, "id": f"{label}_1", "label": label, "catalog_id": catalog_id(label), "size": size,
           "position": [*base["position"][:2], z]}
    twin = {**office, "objects": [obj]}
    tasks = make_tasks(twin, ["move_on_table"], 1, seed=0, log=lambda *_: None)
    assert tasks, f"a {label} lying on the table should be reachable"
    ep = run_episode(SimEnv(twin, tasks[0], res=32), ScriptedExpert(), tasks[0])
    assert ep["success"], ep["detail"]
    assert ep["detail"]["target_error_m"] < 0.02


def test_target_spot_taken_by_another_object_is_not_offered(office):
    # a book set down on the cup that already sits at its target lands on it and misses (seen as 2 of 6 episodes
    # in an office sim): the other way along the edge is tried, and with both spots taken nothing is offered
    import math

    from arrival.sim.tasks import table_move

    table = next(f for f in office["fixtures"] if is_table(f))
    book = next(o for o in office["objects"] if o["label"] == "book")
    cup = next(o for o in office["objects"] if o["label"] == "cup")
    laptop = next(o for o in office["objects"] if o["label"] == "laptop")
    args = (table, book, office["fixtures"], office["walls"])
    first = table_move(*args, [book])
    assert first is not None
    block1 = {**cup, "position": [*first["target_xy"], cup["position"][2]]}
    second = table_move(*args, [book, block1])
    assert second is not None and math.dist(second["target_xy"], first["target_xy"]) > 0.3
    block2 = {**laptop, "position": [*second["target_xy"], laptop["position"][2]]}
    assert table_move(*args, [book, block1, block2]) is None
