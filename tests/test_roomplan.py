import copy
import json
import math
import struct
import zlib
from pathlib import Path

import pytest

from arrival.scan import bundle, roomplan
from arrival.scan.four_photo import bundle_from_photos
from arrival.twin.compile import compile_twin

from conftest import SCANS


def test_sink_coordinate_conversion():
    b = roomplan.load(SCANS / "hero_kitchen" / "roomplan.json", "hero_kitchen")
    sink = next(f for f in b["fixtures"] if f["id"] == "sink_1")
    assert sink["category"] == "sink"
    assert sink["position"] == pytest.approx([1.6, 0.40, 0.85], abs=1e-6)
    assert sink["size"] == pytest.approx([0.6, 0.5, 0.25], abs=1e-6)
    assert sink["yaw"] == 0.0
    assert b["geometry_source"] == "roomplan"


def test_roomplan_walls_project_to_floor():
    b = roomplan.load(SCANS / "hero_kitchen" / "roomplan.json", "hero_kitchen")
    north = next(w for w in b["walls"] if w["id"] == "wall_north")
    assert sorted([north["start"][0], north["end"][0]]) == pytest.approx([0.0, 4.0])
    assert north["start"][1] == pytest.approx(0.0) and north["end"][1] == pytest.approx(0.0)
    assert north["height"] == pytest.approx(2.4)
    assert north["thickness"] == pytest.approx(0.02)
    east = next(w for w in b["walls"] if w["id"] == "wall_east")
    assert sorted([east["start"][1], east["end"][1]]) == pytest.approx([0.0, 3.0])


def test_yaw_from_rotation():
    # 90 degrees about ARKit +Y: local +X maps to ARKit -Z, i.e. Z-up +Y, and local +Z (the object's front, measured on
    # the real scan kitchen_real_01) maps to Z-up +X; the twin's yaw puts the front on +y at 0, so the yaw is -pi/2.
    m = [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    rp = {"walls": [], "objects": [{"identifier": "t", "category": "table", "dimensions": [1, 1, 1], "transform": m}]}
    b = roomplan.to_bundle(rp, "r")
    assert b["fixtures"][0]["yaw"] == pytest.approx(-1.5707963, abs=1e-6)


def test_hero_scan_loads(hero_bundle):
    assert hero_bundle["room_id"] == "hero_kitchen"
    assert hero_bundle["geometry_source"] == "authored_sample"
    assert {f["id"] for f in hero_bundle["fixtures"]} == {"counter_a", "sink_1", "counter_b", "fridge_1", "stove_1"}


@pytest.mark.parametrize("key,value", [("frame", "y_up"), ("units", "feet"), ("schema_version", "2")])
def test_rejects_bad_bundle(tmp_path, hero_bundle, key, value):
    bad = copy.deepcopy(hero_bundle)
    bad[key] = value
    p = tmp_path / "scan.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        bundle.load(p)


def _tiny_png(path):
    raw = b"\x00\xff\x00\x00"
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return path


def test_four_photo_slot(tmp_path):
    scans = tmp_path / "scans"
    (scans / "hero_kitchen").mkdir(parents=True)
    (scans / "hero_kitchen" / "scan.json").write_text((SCANS / "hero_kitchen" / "scan.json").read_text())
    photos = [_tiny_png(tmp_path / f"p{i}.png") for i in range(4)]
    b = bundle_from_photos(photos, "my_room", scans_dir=scans)
    assert b["geometry_source"] == "four_photo_unmeasured"
    assert b["blocks_claims"] == ["geometry", "realism"]
    assert {"counter_a", "sink_1", "counter_b", "fridge_1", "stove_1"} <= {f["id"] for f in b["fixtures"]}
    assert len(list((scans / "my_room" / "photos").iterdir())) == 4
    with pytest.raises(ValueError):
        bundle_from_photos(photos[:3], "my_room", scans_dir=scans)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "roomplan"
FIXTURE_PATHS = sorted(FIXTURES.glob("*.json"))


def _closed(walls, tol=0.05):
    ends = [tuple(p) for w in walls for p in (w["start"], w["end"])]
    return all(sum(math.dist(e, o) <= tol for o in ends) >= 2 for e in ends)


@pytest.mark.parametrize("path", FIXTURE_PATHS, ids=[p.stem for p in FIXTURE_PATHS])
def test_roomplan_fixture_normalizes_and_compiles(path):
    raw = json.loads(path.read_text())
    b = roomplan.load(path, path.stem)
    assert all(isinstance(f["category"], str) for f in b["fixtures"])
    n_low = sum(1 for o in raw["objects"] if "low" in o.get("confidence", {}))
    assert len(b["fixtures"]) == len(raw["objects"]) - n_low
    for f in b["fixtures"]:
        assert f["position"][2] - f["size"][2] / 2 >= -0.05, f  # resting on/above the floor
        assert f["position"][2] >= 0
    xs = [p[0] for w in b["walls"] for p in (w["start"], w["end"])]
    ys = [p[1] for w in b["walls"] for p in (w["start"], w["end"])]
    assert min(xs) == pytest.approx(0, abs=1e-6) and min(ys) == pytest.approx(0, abs=1e-6)
    longest = max(b["walls"], key=lambda w: math.dist(w["start"], w["end"]))
    assert longest["start"][1] == pytest.approx(longest["end"][1], abs=1e-6)  # runs along X
    assert _closed(b["walls"])
    twin = compile_twin(b)
    assert all(f["position"][2] >= 0 for f in twin["fixtures"])


def test_keyed_enum_camelcase_and_low_confidence():
    eye = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, -1.0, -1, 1]
    rp = {"floors": [{"transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1.5, 0, 1]}],
          "walls": [{"identifier": "w", "dimensions": [4, 2.5, 0], "transform":
                     [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, -0.25, 0, 1]}],
          "objects": [{"identifier": "a", "category": {"washerDryer": {}}, "confidence": {"high": {}},
                       "dimensions": [0.6, 0.9, 0.6], "transform": eye},
                      {"identifier": "b", "category": {"sink": {}}, "confidence": {"low": {}},
                       "dimensions": [0.6, 0.2, 0.5], "transform": eye}]}
    b = roomplan.to_bundle(rp, "r")
    assert [f["category"] for f in b["fixtures"]] == ["washer_dryer"]
    assert b["fixtures"][0]["position"][2] == pytest.approx(0.5)
    assert b["walls"][0]["thickness"] > 0


def _footprint(f):
    (cx, cy, _), (sx, sy, _) = f["position"], f["size"]
    return cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2


def _overlap(a, b, tol=1e-6):
    ax0, ax1, ay0, ay1 = _footprint(a)
    bx0, bx1, by0, by1 = _footprint(b)
    return ax0 < bx1 - tol and bx0 < ax1 - tol and ay0 < by1 - tol and by0 < ay1 - tol


def test_hero_fixtures_tile_without_overlap(hero_bundle):
    fx = {f["id"]: f for f in hero_bundle["fixtures"]}
    sink = fx.pop("sink_1")  # the sink is inset into counter_a by design
    pairs = [(a, b) for i, a in enumerate(fx.values()) for b in list(fx.values())[i + 1:]]
    clashes = [(a["id"], b["id"]) for a, b in pairs if _overlap(a, b)]
    assert not clashes, clashes
    cx0, cx1, _, _ = _footprint(fx["counter_a"])
    sx0, sx1, _, _ = _footprint(sink)
    assert cx0 <= sx0 and sx1 <= cx1
    for f in fx.values():
        x0, x1, _, _ = _footprint(f)
        assert 0.0 <= x0 and x1 <= 4.0


def test_hero_objects_rest_on_a_counter_outside_appliances(hero_bundle):
    fixtures = hero_bundle["fixtures"]
    counters = [f for f in fixtures if f["category"] == "counter"]
    appliances = [f for f in fixtures if f["category"] in ("refrigerator", "stove")]
    for o in hero_bundle["objects"]:
        x0, x1, y0, y1 = _footprint(o)
        inside = [c for c in counters if _footprint(c)[0] <= x0 and x1 <= _footprint(c)[1]
                  and _footprint(c)[2] <= y0 and y1 <= _footprint(c)[3]]
        assert inside, (o["id"], "not fully on a counter top")
        assert not [a for a in appliances if _overlap(o, a)], (o["id"], "inside an appliance")


def test_four_photo_rerun_clears_stale_and_rejects_bad_room_id(tmp_path):
    scans = tmp_path / "scans"
    (scans / "hero_kitchen").mkdir(parents=True)
    (scans / "hero_kitchen" / "scan.json").write_text((SCANS / "hero_kitchen" / "scan.json").read_text())
    pngs = [_tiny_png(tmp_path / f"p{i}.png") for i in range(4)]
    jpgs = []
    for i, p in enumerate(pngs):
        j = tmp_path / f"q{i}.jpg"
        j.write_bytes(p.read_bytes())
        jpgs.append(j)
    bundle_from_photos(jpgs, "r", scans_dir=scans)
    bundle_from_photos(pngs, "r", scans_dir=scans)
    assert sorted(f.suffix for f in (scans / "r" / "photos").iterdir()) == [".png"] * 4
    with pytest.raises(ValueError):
        bundle_from_photos(pngs, "../x", scans_dir=scans)


def test_select_story_ignores_low_confidence_objects():
    def obj(i, story, conf):
        return {"identifier": i, "story": story, "confidence": conf, "category": {"table": {}},
                "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], "dimensions": [1, 1, 1]}
    rp = {"walls": [], "objects": [obj("a", 0, "high"), obj("b", 0, "high"),
                                   obj("c", 1, "low"), obj("d", 1, "low"), obj("e", 1, "low")]}
    assert roomplan._select_story(rp, None)[1] == 0
