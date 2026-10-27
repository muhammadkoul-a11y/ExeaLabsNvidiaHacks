"""TWN-3 leftover: the doors, windows and openings a RoomPlan export carries are drawn in the sim, on their wall, as
visual-only geoms. A closed door is a frame, a wood leaf and a knob; an open door and an opening are a frame around a
dark recess; a window is a frame, a sill, a pane and a mullion. They never collide, so physics does not change."""

import copy
import json
import math
import re

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.scan import roomplan  # noqa: E402
from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim.scene import build_mjcf, portals_xml  # noqa: E402
from arrival.twin.compile import compile_twin, place_default_objects  # noqa: E402

from conftest import REPO, SCANS  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures" / "roomplan"
BASE = (2.0, 0.6, 0.75)  # a stand spot clear of the office's table and chairs (test_non_kitchen)
HERO_BASE = (1.3, 1.0, 0.75)
PORTAL = re.compile(r"(door|window|opening)(\d+)_(\w+)$")
PORTAL_TAG = re.compile(r'<geom name="(?:door|window|opening)\d+_\w+"[^>]*/>')
PARTS = {"closed": {"jamb0", "jamb1", "head", "leaf", "knob"},
         "open": {"jamb0", "jamb1", "head", "recess"},
         "window": {"jamb0", "jamb1", "head", "sill", "pane", "mullion"},
         "opening": {"jamb0", "jamb1", "head", "recess"},
         "raised": {"jamb0", "jamb1", "head", "sill", "recess"}}  # an opening above the floor (a pass-through)
MAX_PROUD, KNOB_PROUD = 0.03, 0.06  # how far past each wall face a box (the sill at most) and the knob reach
TOL = 1e-3

FLOOR_Y, W, D, H = -1.4, 6.0, 5.0, 2.5
# Synthetic export: after normalization the north wall is y = 5 and the south wall y = 0 (both run along +x), the
# east wall x = 6 and the west wall x = 0 (both along +y). (wall, parts, extent along the wall, z extent)
EXPECTED = {
    ("door", 0): ("north", "closed", (0.55, 1.45), (0.0, 2.13)),  # scanned 3 cm above the floor: starts on it
    ("door", 1): ("north", "open", (2.1, 2.9), (0.0, 2.0)),
    ("window", 0): ("north", "window", (3.7, 5.3), (0.9, 2.1)),
    ("opening", 0): ("east", "opening", (2.0, 3.0), (0.0, 2.0)),  # parent is not a wall: the nearest wall
    ("opening", 1): ("west", "raised", (1.4, 2.6), (0.9, 1.5)),
    ("opening", 2): ("south", "opening", (0.0, 6.0), (0.0, 2.2)),  # 9 m wide on a 6 m wall: clipped to it
}  # window 1 (2.5 m from every wall) and window 2 (zero size) are not drawn


def _t(x, y, z, yaw=0.0):
    """Column-major ARKit transform: rotation about +Y by `yaw`, translation (x, y, z)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, x, y, z, 1]


def _surface(kind, ident, parent, x, z, w, h, sill=0.0, yaw=0.0, is_open=None):
    """A door, window or opening `w` wide and `h` tall with its bottom `sill` above the floor, centred at ARKit x, z."""
    cat = {kind: {"isOpen": is_open}} if kind == "door" else {kind: {}}
    return {"identifier": ident, "parentIdentifier": parent, "category": cat, "confidence": {"high": {}},
            "dimensions": [w, h, 0], "transform": _t(x, FLOOR_Y + sill + h / 2, z, yaw)}


def _export():
    """A 6 by 5 m room in Apple's encoding (floor at ARKit y = -1.4, walls centred on the origin)."""
    def wall(ident, x, z, length, yaw):
        return {"identifier": ident, "category": {"wall": {}}, "confidence": {"high": {}},
                "dimensions": [length, H, 0], "transform": _t(x, FLOOR_Y + H / 2, z, yaw)}

    walls = [wall("north", 0, -D / 2, W, 0), wall("south", 0, D / 2, W, 0),
             wall("east", W / 2, 0, D, math.pi / 2), wall("west", -W / 2, 0, D, math.pi / 2)]
    doors = [_surface("door", "front", "north", -2.0, -D / 2, 0.9, 2.1, sill=0.03, is_open=False),
             _surface("door", "hall", "north", -0.5, -D / 2, 0.8, 2.0, is_open=True)]
    windows = [_surface("window", "view", "north", 1.5, -D / 2, 1.6, 1.2, sill=0.9),
               _surface("window", "nowhere", None, 0.0, 0.0, 1.0, 1.0, sill=1.0),
               _surface("window", "speck", "south", 0.0, D / 2, 0.0, 0.0, sill=1.0)]
    openings = [_surface("opening", "arch", "gone", W / 2, 0.0, 1.0, 2.0, yaw=math.pi / 2),
                _surface("opening", "hatch", "west", -W / 2, 0.5, 1.2, 0.6, sill=0.9, yaw=math.pi / 2),
                _surface("opening", "wide", "south", 0.0, D / 2, 9.0, 2.2)]
    return {"walls": walls, "floors": [], "doors": doors, "windows": windows, "openings": openings, "objects": []}


def _fixture(name: str) -> dict:
    twin = compile_twin(roomplan.load(FIXTURES / f"{name}.json", name))
    place_default_objects(twin, seed=0)
    return twin


def _bare(twin: dict) -> dict:
    return {**twin, "doors": [], "windows": [], "openings": []}


def _model(twin: dict, base=BASE, **kw):
    return mujoco.MjModel.from_xml_string(build_mjcf(twin, base, **kw))


@pytest.fixture(scope="module")
def synthetic():
    return compile_twin(roomplan.to_bundle(_export(), "portals"))


@pytest.fixture(scope="module")
def office():
    return _fixture("office")


def _portal_geoms(m) -> dict:
    """{(kind, k): {part: geom id}} for every door, window and opening geom in the model."""
    out: dict = {}
    for g in range(m.ngeom):
        hit = PORTAL.match(m.geom(g).name)
        if hit:
            out.setdefault((hit[1], int(hit[2])), {})[hit[3]] = g
    return out


def _in_wall(m, d, g, wall) -> tuple[np.ndarray, np.ndarray]:
    """Geom g's centre in its wall's frame (u along the wall from its start, v across it, z up) and its half extents
    along those axes. A box runs along the wall; the knob's axis runs across it."""
    start = np.array(wall["start"], float)
    along = np.array(wall["end"], float) - start
    along /= np.linalg.norm(along)
    across = np.array([-along[1], along[0]])
    p, R, s = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
    centre = np.array([(p[:2] - start) @ along, (p[:2] - start) @ across, p[2]])
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        assert abs(R[:2, 2] @ across) == pytest.approx(1.0, abs=1e-6)  # the axis goes through the wall
        return centre, np.array([s[0], s[1], s[0]])
    assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
    assert abs(R[:2, 0] @ along) == pytest.approx(1.0, abs=1e-6) and R[2, 2] == pytest.approx(1.0, abs=1e-6)
    return centre, np.array(s[:3])


def _check_on_wall(m, d, geoms: dict, wall: dict, extent, zs):
    """Every geom is centred on the wall's line, runs through it and stands at most a few cm proud of either face,
    and lies inside the expected rectangle, which the geoms together fill."""
    half_t = max(wall["thickness"], 0.05) / 2
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for part, g in geoms.items():
        c, h = _in_wall(m, d, g, wall)
        assert abs(c[1]) < TOL, part  # on the wall's centre line
        proud = h[1] - half_t  # past each face of the drawn wall
        if part == "knob":
            assert proud == pytest.approx(KNOB_PROUD, abs=TOL)
        else:
            assert 0.001 <= proud <= MAX_PROUD + 1e-6, part
        if part != "knob":
            lo, hi = np.minimum(lo, [c[0] - h[0], c[2] - h[2]]), np.maximum(hi, [c[0] + h[0], c[2] + h[2]])
        assert extent[0] - TOL <= c[0] - h[0] and c[0] + h[0] <= extent[1] + TOL, part
        assert zs[0] - TOL <= c[2] - h[2] and c[2] + h[2] <= zs[1] + TOL, part
    assert lo == pytest.approx([extent[0], zs[0]], abs=TOL) and hi == pytest.approx([extent[1], zs[1]], abs=TOL)


def _world_extent(wall: dict, a: float, b: float) -> tuple[float, float]:
    """A world x (or y) range on an axis-aligned wall as a range of u, the distance from the wall's start."""
    (x0, y0), (x1, y1) = wall["start"], wall["end"]
    axis = 0 if abs(x1 - x0) > abs(y1 - y0) else 1
    s, sign = wall["start"][axis], math.copysign(1.0, wall["end"][axis] - wall["start"][axis])
    return tuple(sorted(((a - s) * sign, (b - s) * sign)))


def test_each_portal_gets_its_parts(synthetic):
    m = _model(synthetic)
    got = {key: set(parts) for key, parts in _portal_geoms(m).items()}
    assert got == {key: PARTS[e[1]] for key, e in EXPECTED.items()}
    assert sum(len(p) for p in got.values()) == 5 + 4 + 6 + 4 + 5 + 4


def test_portals_lie_on_their_wall(synthetic):
    m = _model(synthetic)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    walls = {w["id"]: w for w in synthetic["walls"]}
    for key, geoms in _portal_geoms(m).items():
        wall_id, _, along, zs = EXPECTED[key]
        _check_on_wall(m, d, geoms, walls[wall_id], _world_extent(walls[wall_id], *along), zs)


@pytest.mark.parametrize("name,counts", [
    ("office", {("door", 0): "closed", ("door", 1): "open", ("window", 0): "window"}),
    ("study", {("door", 0): "closed"}),
])
def test_authored_fixtures_draw_every_portal_on_its_wall(name, counts):
    twin = _fixture(name)
    m = _model(twin)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    geoms = _portal_geoms(m)
    assert {key: set(parts) for key, parts in geoms.items()} == {key: PARTS[v] for key, v in counts.items()}
    walls = {w["id"]: w for w in twin["walls"]}
    for (kind, k), parts in geoms.items():
        p = twin[f"{kind}s"][k]
        wall = walls[p["wall_id"]]
        if kind == "door":
            assert ("leaf" in parts) == (not p["is_open"])
        start, end = np.array(wall["start"]), np.array(wall["end"])
        u = (np.array(p["position"][:2]) - start) @ (end - start) / np.linalg.norm(end - start)
        (w, _, h), zc = p["size"], p["position"][2]
        z0 = max(zc - h / 2, 0.0) if kind == "window" else 0.0  # these doors are scanned within 1 um of the floor
        _check_on_wall(m, d, parts, wall, (u - w / 2, u + w / 2), (z0, zc + h / 2))


def test_export_without_parent_ids_lands_on_the_same_walls():
    """Exports from before iOS 17 carry no parentIdentifier: the nearest wall is the same one."""
    raw = json.loads((FIXTURES / "office.json").read_text())
    for key in ("doors", "windows", "openings"):
        for s in raw[key]:
            del s["parentIdentifier"]
    orphan = compile_twin(roomplan.to_bundle(raw, "office"))
    assert all(p["wall_id"] is None for k in ("doors", "windows") for p in orphan[k])
    assert portals_xml(orphan) == portals_xml(compile_twin(roomplan.load(FIXTURES / "office.json", "office")))


def test_portal_geoms_never_collide(synthetic, office):
    for twin in (synthetic, office):
        m = _model(twin)
        gs = [g for parts in _portal_geoms(m).values() for g in parts.values()]
        assert gs
        assert all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 and m.geom_group[g] == 2 for g in gs)
        assert all(m.geom_bodyid[g] == 0 for g in gs)  # on the world body: no mass, no new body or joint


def test_rest_of_the_scene_is_unchanged(office):
    """Strip the portal geoms from the MJCF and what is left is the scene built without doors, windows or openings."""
    for kw in ({}, {"n_people": 2, "seed": 3}):
        full, bare = build_mjcf(office, BASE, **kw), build_mjcf(_bare(office), BASE, **kw)
        assert len(PORTAL_TAG.findall(full)) == 15 and not PORTAL_TAG.findall(bare)
        assert PORTAL_TAG.sub("", full) == bare


def test_rooms_without_portals_get_none():
    hero = compile_twin(load_room("hero_kitchen", SCANS))
    assert not hero.get("doors") and portals_xml(hero) == []
    assert not _portal_geoms(_model(hero, HERO_BASE))


def test_physics_is_unchanged(office):
    """A cup dropped through the closed door's knob falls to the floor, no contact ever involves a portal geom, and
    with the arm driven at random every joint and object follows the same trajectory, to the bit, as in the room
    without doors and windows."""
    knob = _model(office).geom("door0_knob").pos
    twin = copy.deepcopy(office)
    twin["objects"].append({"id": "cup_drop", "catalog_id": "cup", "label": "cup", "size": [0.07, 0.07, 0.09],
                            "position": [float(knob[0]), 0.09, float(knob[2])], "yaw": 0.0})  # the room is y > 0
    runs = []
    for t in (twin, _bare(twin)):
        m = _model(t)
        d = mujoco.MjData(m)
        rng = np.random.default_rng(0)
        lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
        portal = {g for parts in _portal_geoms(m).values() for g in parts.values()}
        trace = []
        for _ in range(8):
            d.ctrl[:] = rng.uniform(lo, hi)
            for _ in range(50):
                mujoco.mj_step(m, d)
                assert not {int(g) for g in (*d.contact.geom1, *d.contact.geom2)} & portal
            trace.append(np.concatenate([d.qpos, d.qvel]))
        runs.append((np.array(trace), d.body("cup_drop").xpos.copy()))
    (with_portals, cup), (without, _) = runs
    assert np.array_equal(with_portals, without)
    assert cup[2] == pytest.approx(0.045, abs=0.003)  # resting on the floor, its 9 cm height halved


def test_authored_door_without_ids_goes_on_the_nearest_wall():
    """An authored twin's door may carry only a size and a position: it is a closed door on the nearest wall."""
    hero = compile_twin(load_room("hero_kitchen", SCANS))
    hero["doors"] = [{"size": [0.9, 0.0, 2.1], "position": [2.0, 3.01, 1.05]}]
    m = _model(hero, HERO_BASE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    geoms = _portal_geoms(m)
    assert {key: set(parts) for key, parts in geoms.items()} == {("door", 0): PARTS["closed"]}
    wall = next(w for w in hero["walls"] if w["id"] == "wall_south")
    _check_on_wall(m, d, geoms[("door", 0)], wall, _world_extent(wall, 1.55, 2.45), (0.0, 2.1))


def test_portals_show_in_renders(office):
    """Seen from inside the room and from behind the wall, the doors and the window change part of the picture."""
    cam = '<camera name="behind" pos="1.6 -2.2 1.4" xyaxes="1 0 0 0 0.2 1" fovy="60"/></worldbody>'
    frames = []
    for t in (office, _bare(office)):
        m = mujoco.MjModel.from_xml_string(build_mjcf(t, BASE).replace("</worldbody>", cam))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, 96, 128)
        try:
            shots = []
            for c in ("overview", "behind"):
                r.update_scene(d, c)
                shots.append(r.render().astype(int))
        finally:
            r.close()
        frames.append(shots)
    for a, b in zip(*frames):
        changed = (np.abs(a - b).max(axis=2) > 25).mean()
        assert 0.01 < changed < 0.5, changed
