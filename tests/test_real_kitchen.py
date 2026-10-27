"""A real iPhone RoomPlan scan of a kitchen (tests/fixtures/roomplan/kitchen_real_01.json, geometry only: the
coreModel blob and every frame were left out). It was the first real kitchen through the pipeline, and it showed
that RoomPlan objects face their local +Z: before the fix every cabinet, appliance and fridge faced its wall, the
robot's stand spots fell outside the room and the job ended 'unsupported layout'."""
from pathlib import Path

import numpy as np
import pytest

from arrival.scan import roomplan
from arrival.twin.compile import compile_twin

FIXTURE = Path(__file__).parent / "fixtures" / "roomplan" / "kitchen_real_01.json"


@pytest.fixture(scope="module")
def twin():
    return compile_twin(roomplan.load(FIXTURE, "kitchen_real_01"))


def _bounds(twin):
    pts = np.array([p for w in twin["walls"] for p in (w["start"], w["end"])])
    return pts.min(axis=0), pts.max(axis=0)


def test_every_fixture_faces_into_the_room(twin):
    from arrival.sim.preflight import front

    lo, hi = _bounds(twin)
    fixtures = [f for f in twin["fixtures"] if f["type"] != "sink"]
    assert len(fixtures) >= 5
    for f in fixtures:
        c = np.array(f["position"][:2])
        ahead = c + front(f)[:2] * (f["size"][1] / 2 + 0.5)  # half a metre in front of the face
        behind = c - front(f)[:2] * (f["size"][1] / 2 + 0.5)
        assert np.all(ahead > lo) and np.all(ahead < hi), (f["id"], f["type"], ahead.round(2))
        assert not (np.all(behind > lo) and np.all(behind < hi)), (f["id"], "its back is in the room too")


def _cabinet(twin, prefix):
    return next(f for f in twin["fixtures"] if f["id"].startswith(prefix))


def test_corner_cabinet_door_stays_out_of_the_blind_corner(twin):
    # F9337652 runs along the east wall into the corner; 11793D80, on the north wall, stands against the last 0.64 m
    # of its front. The door used to be drawn there, hinged beside the neighbour, and it stopped at 0.8 deg
    from arrival.sim.scene import DRAWER_W, door_hinge_x, door_w, drawer_x, drawers_at_right, front_span, has_drawers

    c, fx = _cabinet(twin, "F9337652"), twin["fixtures"]
    x0, x1 = front_span(c, fx)
    assert x0 == pytest.approx(-c["size"][0] / 2) and x1 == pytest.approx(0.135, abs=0.005)
    assert has_drawers(c, fx) and drawers_at_right(c, fx)
    assert drawer_x(c, fx) + DRAWER_W / 2 == pytest.approx(x1)  # the stack fills the end beside the corner
    hinge, w = door_hinge_x(c, fx), door_w(c, fx)
    assert hinge <= drawer_x(c, fx) - DRAWER_W / 2  # the door is left of the stack, on the open front
    assert hinge - w >= x0 and w >= 0.3


def test_narrow_cabinet_is_one_door(twin):
    # CC64C90B is 0.43 m wide: the 0.5 m drawer stack stuck into the dishwasher and the door was -0.07 m wide
    from arrival.sim.scene import door_w, has_door, has_drawers, part_names

    c, fx = _cabinet(twin, "CC64C90B"), twin["fixtures"]
    assert not has_drawers(c, fx) and has_door(c, fx)
    assert door_w(c, fx) == pytest.approx(c["size"][0] - 0.04)
    joints = {p["joint"] for p in part_names(twin) if p["fixture"] == c["id"] and "upper_door" not in p["joint"]}
    assert joints == {f'{c["id"]}_door'}  # the base cabinet's own parts; the wall cabinet above keeps its door


def test_cabinets_away_from_corners_keep_their_layout(twin):
    from arrival.sim.scene import door_hinge_x, door_w, drawers_at_right, front_span, has_drawers

    c, fx = _cabinet(twin, "11793D80"), twin["fixtures"]
    assert front_span(c, fx) == pytest.approx((-c["size"][0] / 2, c["size"][0] / 2))
    assert has_drawers(c, fx) and not drawers_at_right(c, fx)
    assert door_hinge_x(c, fx) == pytest.approx(c["size"][0] / 2 - 0.02)
    assert door_w(c, fx) == pytest.approx(c["size"][0] / 2 - 0.04)


@pytest.mark.parametrize("people,seed", [(0, 5), (2, 3)])
def test_expert_works_every_cabinet_door_and_drawer(twin, tmp_path, people, seed):
    # before the layout fix: open_door 0 of 3, close_door 0 of 3, open_drawer and close_drawer 2 of 3 each
    pytest.importorskip("mujoco")
    from arrival.sim.collect import collect

    for fam in ("open_door", "close_door", "open_drawer", "close_drawer"):
        r = collect(twin, [fam], 4, tmp_path / fam, people=people, res=32, seed=seed,
                    log=lambda *a, **k: None)["report_data"]
        assert (r["attempts"], r["successes"], r["skipped_unreachable"]) == (4, 4, 0), (fam, r)
        assert r["human_contacts"] == 0, (fam, r)


def test_tasks_get_stand_spots_inside_the_room(twin):
    pytest.importorskip("mujoco")
    from arrival.sim.tasks import make_tasks

    log, pre = [], {}
    tasks = make_tasks(twin, ["open_drawer", "open_door", "close_drawer", "close_door"], 8, seed=0,
                       log=log.append, preflight=pre)
    assert tasks, log
    assert not any("outside the room" in str(r) for r in pre.get("reasons", [])), pre.get("reasons")
