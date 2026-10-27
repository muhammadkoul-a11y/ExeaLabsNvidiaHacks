"""TWN-3: every Apple object category, doors/windows/openings, polygonCorners floors, islands, stories, curved walls."""

import copy
import json
import math

import pytest

from arrival.catalog.fixtures import DEFAULT_SIZES, FIXTURE_MAP, default_size
from arrival.scan import roomplan
from arrival.twin.compile import compile_twin, layout_dict

from conftest import REPO

# CapturedRoom.Object.Category cases, as Apple spells them (developer.apple.com/documentation/roomplan).
APPLE_CATEGORIES = ("bathtub", "bed", "chair", "dishwasher", "fireplace", "oven", "refrigerator", "sink",
                    "sofa", "stairs", "storage", "stove", "table", "television", "toilet", "washerDryer")
FLOOR_Y = -1.4
W, D, H = 6.0, 5.0, 2.5
FIXTURES = REPO / "tests" / "fixtures" / "roomplan"


def _t(x, y, z, yaw=0.0):
    """Column-major ARKit transform: rotation about +Y by `yaw`, translation (x, y, z)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, x, y, z, 1]


def _wall(ident, x, z, length, yaw):
    return {"identifier": ident, "category": {"wall": {}}, "confidence": {"high": {}},
            "dimensions": [length, H, 0], "transform": _t(x, FLOOR_Y + H / 2, z, yaw)}


def _room(objects=(), floors=None, **extra):
    """W by D room, floor at ARKit y = FLOOR_Y, walls centred on the origin."""
    walls = [_wall("north", 0, -D / 2, W, 0), _wall("south", 0, D / 2, W, 0),
             _wall("east", W / 2, 0, D, math.pi / 2), _wall("west", -W / 2, 0, D, math.pi / 2)]
    if floors is None:
        floors = [{"identifier": "floor", "category": {"floor": {}}, "dimensions": [W, D, 0],
                   "transform": _t(0, FLOOR_Y, 0)}]
    return {"walls": walls, "floors": floors, "doors": [], "windows": [], "openings": [],
            "objects": list(objects), **extra}


def _object(cat, x=0.0, z=0.0, size=None, confidence="high", ident=None):
    sx, sy, sz = size or default_size(roomplan._snake(cat))
    return {"identifier": ident or cat, "category": {cat: {}}, "confidence": {confidence: {}},
            "dimensions": [sx, sz, sy], "transform": _t(x, FLOOR_Y + sz / 2, z)}


def _surface(kind, ident, parent, x, y, z, w, h, yaw=0.0, is_open=None):
    cat = {kind: {"isOpen": is_open}} if kind == "door" else {kind: {}}
    return {"identifier": ident, "parentIdentifier": parent, "category": cat, "confidence": {"high": {}},
            "dimensions": [w, h, 0], "transform": _t(x, y, z, yaw)}


def test_catalog_covers_every_apple_category():
    snake = {roomplan._snake(c) for c in APPLE_CATEGORIES}
    assert snake <= set(FIXTURE_MAP) and snake <= set(DEFAULT_SIZES)
    assert "washer_dryer" in snake


@pytest.mark.parametrize("cat", APPLE_CATEGORIES)
def test_category_normalizes_and_compiles(cat, tmp_path):
    b = roomplan.to_bundle(_room([_object(cat, x=1.0, z=-0.5)]), "r")
    assert [f["category"] for f in b["fixtures"]] == [roomplan._snake(cat)]
    f = b["fixtures"][0]
    assert f["size"] == pytest.approx(default_size(f["category"]))
    assert f["position"][2] - f["size"][2] / 2 == pytest.approx(0.0, abs=1e-6)
    twin = compile_twin(b)
    fx = next(t for t in twin["fixtures"] if t["id"] == cat)
    assert fx["type"] == FIXTURE_MAP[f["category"]]
    assert fx["position"][2] - fx["size"][2] / 2 == pytest.approx(0.0, abs=1e-6)
    row = layout_dict(twin)["main_group"]["bottom_row"]
    assert any(e["name"] == cat and e["type"] == fx["type"] for e in row)


def test_missing_dimensions_use_default_size():
    o = _object("sofa")
    del o["dimensions"]
    b = roomplan.to_bundle(_room([o]), "r")
    assert b["fixtures"][0]["size"] == default_size("sofa")


def test_doors_windows_openings():
    rp = _room()
    # On the north wall (ARKit z = -D/2), which normalizes to y = D along +X after the min-corner shift.
    rp["doors"] = [_surface("door", "d1", "north", -1.0, FLOOR_Y + 1.05, -D / 2, 0.9, 2.1, is_open=True)]
    rp["windows"] = [_surface("window", "w1", "north", 1.5, FLOOR_Y + 1.5, -D / 2, 1.2, 1.0)]
    rp["openings"] = [_surface("opening", "o1", "nowhere", W / 2, FLOOR_Y + 1.0, 0.0, 1.0, 2.0, yaw=math.pi / 2)]
    b = roomplan.to_bundle(rp, "r")
    d, w, o = b["doors"][0], b["windows"][0], b["openings"][0]
    assert d["wall_id"] == "north" and d["is_open"] is True
    assert d["position"] == pytest.approx([2.0, D, 1.05]) and d["size"] == pytest.approx([0.9, 0.0, 2.1])
    assert w["wall_id"] == "north" and "is_open" not in w
    assert w["position"] == pytest.approx([4.5, D, 1.5]) and w["yaw"] == 0.0
    assert o["wall_id"] is None  # parent not among the walls
    assert o["position"] == pytest.approx([6.0, 2.5, 1.0]) and abs(o["yaw"]) == pytest.approx(math.pi / 2)
    twin = compile_twin(b)
    assert twin["doors"][0]["id"] == "d1" and len(twin["windows"]) == 1 and len(twin["openings"]) == 1


def _seg_dist(p, w):
    (x0, y0), (x1, y1) = w["start"], w["end"]
    dx, dy = x1 - x0, y1 - y0
    t = max(0.0, min(1.0, ((p[0] - x0) * dx + (p[1] - y0) * dy) / (dx * dx + dy * dy)))
    return math.dist(p, [x0 + t * dx, y0 + t * dy])


@pytest.mark.parametrize("name", ["office", "study"])
def test_authored_fixture_portals_sit_on_their_walls(name):
    raw = json.loads((FIXTURES / f"{name}.json").read_text())
    b = roomplan.load(FIXTURES / f"{name}.json", name)
    walls = {w["id"]: w for w in b["walls"]}
    for key in ("doors", "windows", "openings"):
        assert len(b[key]) == len(raw[key])
        for p in b[key]:
            assert p["wall_id"] in walls
            assert _seg_dist(p["position"][:2], walls[p["wall_id"]]) <= 0.02
            assert 0 < p["position"][2] - p["size"][2] / 2 + 0.05  # sill at or above the floor
    if name == "office":
        assert [d["is_open"] for d in b["doors"]] == [False, True]


def test_polygon_floor_sets_extent():
    # L-shaped floor polygon in local plane coordinates, wider than the walls on +X.
    corners = [[-W / 2, -D / 2, 0], [W / 2 + 1, -D / 2, 0], [W / 2 + 1, 0, 0], [W / 2, 0, 0],
               [W / 2, D / 2, 0], [-W / 2, D / 2, 0]]
    # Floor transform: local x -> ARKit x, local y -> ARKit z, local z -> ARKit y.
    m = [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, FLOOR_Y, 0, 1]
    floor = {"identifier": "fl", "category": {"floor": {}}, "dimensions": [W + 1, D, 0], "transform": m,
             "polygonCorners": corners}
    b = roomplan.to_bundle(_room(floors=[floor]), "r")
    assert len(b["floors"]) == 1 and b["floors"][0]["z"] == 0.0
    xs = [c[0] for c in b["floors"][0]["corners"]]
    ys = [c[1] for c in b["floors"][0]["corners"]]
    assert len(xs) == 6 and max(xs) == pytest.approx(W + 1) and min(xs) == pytest.approx(0.0)
    assert min(ys) == pytest.approx(0.0) and max(ys) == pytest.approx(D)
    doc = layout_dict(compile_twin(b))
    assert doc["room"]["floor"][0]["size"] == pytest.approx([W + 1, D, 0.02])
    assert doc["room"]["floor"][0]["pos"] == pytest.approx([(W + 1) / 2, D / 2, 0.0])


@pytest.mark.parametrize("name", ["office", "study"])
def test_authored_fixture_polygon_floor_matches_walls(name):
    b = roomplan.load(FIXTURES / f"{name}.json", name)
    assert len(b["floors"]) == 1 and len(b["floors"][0]["corners"]) == 4
    xs = [c[0] for c in b["floors"][0]["corners"]]
    ys = [c[1] for c in b["floors"][0]["corners"]]
    wx = [p[0] for w in b["walls"] for p in (w["start"], w["end"])]
    wy = [p[1] for w in b["walls"] for p in (w["start"], w["end"])]
    assert [min(xs), max(xs), min(ys), max(ys)] == pytest.approx([min(wx), max(wx), min(wy), max(wy)], abs=0.05)


def test_floor_without_polygon_keeps_wall_extent():
    b = roomplan.to_bundle(_room(), "r")
    assert b["floors"] == []
    assert layout_dict(compile_twin(b))["room"]["floor"][0]["size"] == pytest.approx([W, D, 0.02])


def test_island():
    island = _object("table", x=0.0, z=0.0, size=[2.0, 0.9, 0.92], ident="island")
    dining = _object("table", x=0.0, z=1.5, size=[1.4, 0.8, 0.75], ident="dining")
    against_wall = _object("storage", x=0.0, z=-D / 2 + 0.3, size=[1.0, 0.6, 0.9], ident="cab")
    twin = compile_twin(roomplan.to_bundle(_room([island, dining, against_wall]), "r"))
    fx = {f["id"]: f for f in twin["fixtures"]}
    assert fx["island"]["type"] == "counter" and fx["island"]["island"] is True
    assert "default_config_name" not in fx["island"]
    assert fx["dining"]["type"] == "counter" and fx["dining"].get("default_config_name") == "dining"
    assert "island" not in fx["dining"]
    assert fx["cab"]["type"] == "hinge_cabinet" and "island" not in fx["cab"]
    row = {e["name"]: e for e in layout_dict(twin)["main_group"]["bottom_row"]}
    assert row["island"].get("island") is True and "island" not in row["dining"]


def test_freestanding_storage_is_not_island():
    sideboard = _object("storage", x=0.0, z=0.0, size=[1.6, 0.5, 1.1], ident="sideboard")
    table = _object("table", x=0.0, z=1.5, size=[1.6, 0.9, 1.1], ident="island")
    twin = compile_twin(roomplan.to_bundle(_room([sideboard, table]), "r"))
    fx = {f["id"]: f for f in twin["fixtures"]}
    assert fx["sideboard"]["type"] == "hinge_cabinet" and "island" not in fx["sideboard"]
    assert fx["island"]["type"] == "counter" and fx["island"]["island"] is True


def test_island_needs_walls():
    rp = _room([_object("table", size=[2.0, 0.9, 0.92], ident="island")])
    rp["walls"] = []
    twin = compile_twin(roomplan.to_bundle(rp, "r"))
    assert "island" not in twin["fixtures"][0]


def _two_story_structure():
    ground = _room([_object("sofa", x=1.0)])
    upper = copy.deepcopy(_room([_object("bed", x=0.5, ident="bed"), _object("chair", x=-1.5, ident="c1"),
                                 _object("chair", x=-2.0, z=1.0, ident="c2")]))
    for item in upper["walls"] + upper["floors"] + upper["objects"]:  # second story 3 m up
        item["transform"] = list(item["transform"])
        item["transform"][13] += 3.0
    ground["story"], upper["story"] = 0, 1
    return ground, upper


def test_multi_story_structure_rooms():
    ground, upper = _two_story_structure()
    b = roomplan.to_bundle({"rooms": [ground, upper], "version": 2}, "r")
    assert b["story"] == 1 and b["story_count"] == 2  # most objects
    assert sorted(f["id"] for f in b["fixtures"]) == ["bed", "c1", "c2"]
    for f in b["fixtures"]:
        assert f["position"][2] - f["size"][2] / 2 == pytest.approx(0.0, abs=1e-6)  # measured from story 1 floor
    b0 = roomplan.to_bundle({"rooms": [ground, upper]}, "r", story=0)
    assert b0["story"] == 0 and [f["category"] for f in b0["fixtures"]] == ["sofa"]
    assert b0["fixtures"][0]["position"][2] == pytest.approx(default_size("sofa")[2] / 2)
    with pytest.raises(ValueError):
        roomplan.to_bundle({"rooms": [ground, upper]}, "r", story=2)


def test_multi_story_flat_export():
    ground, upper = _two_story_structure()
    flat = {k: [] for k in ("walls", "floors", "doors", "windows", "openings", "objects")}
    for room in (ground, upper):
        for k in flat:
            flat[k] += [{**item, "story": room["story"]} for item in room[k]]
    b = roomplan.to_bundle(flat, "r")
    assert b["story"] == 1 and b["story_count"] == 2 and len(b["fixtures"]) == 3 and len(b["walls"]) == 4
    assert compile_twin(b)["story_count"] == 2


def test_single_story_defaults():
    b = roomplan.load(REPO / "scans" / "hero_kitchen" / "roomplan.json", "hero_kitchen")
    assert b["story"] == 0 and b["story_count"] == 1
    assert b["doors"] == [] and b["windows"] == [] and b["openings"] == [] and b["floors"] == []


def _circumcenter(a, b, c):
    (ax, ay), (bx, by), (cx, cy) = a, b, c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return ux, uy


@pytest.mark.parametrize("sweep,n_seg", [(math.pi / 2, 6), (math.radians(10), 2)])
def test_curved_wall_is_approximated_by_chords(sweep, n_seg):
    """A wall with curve {startAngle, endAngle, radius} becomes equal chords every 15°, at least two."""
    radius = 4.0
    rp = _room()
    rp["walls"][0]["curve"] = {"startAngle": 0.0, "endAngle": sweep, "radius": radius}
    b = roomplan.to_bundle(rp, "r")
    segs = [w for w in b["walls"] if w["id"] == "north" or w["id"].startswith("north_")]
    assert len(segs) == n_seg
    for i, w in enumerate(segs[:-1]):
        assert w["end"] == pytest.approx(segs[i + 1]["start"], abs=1e-6)
    chord = 2 * radius * math.sin(sweep / (2 * n_seg))
    for w in segs:
        assert math.dist(w["start"], w["end"]) == pytest.approx(chord, abs=1e-6)
        assert w["height"] == pytest.approx(H)
        assert w["thickness"] > 0
    pts = [segs[0]["start"]] + [w["end"] for w in segs]
    cx, cy = _circumcenter(pts[0], pts[1], pts[2])
    assert all(math.hypot(p[0] - cx, p[1] - cy) == pytest.approx(radius, abs=1e-6) for p in pts)
    assert math.dist(pts[0], pts[-1]) == pytest.approx(2 * radius * math.sin(sweep / 2), abs=1e-6)
    assert compile_twin(b)["walls"]
