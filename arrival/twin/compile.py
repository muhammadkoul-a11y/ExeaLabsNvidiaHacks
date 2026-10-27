"""ScanBundle -> twin dict (twin.json) + RoboCasa-subset layout (layout.yaml)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import yaml

from arrival.catalog.fixtures import default_config_name, fixture_type
from arrival.catalog.objects import ROOM_OBJECTS, catalog_id, room_kind, sample_size
from arrival.export.provenance import OBJECT_PROVENANCE, note_seed, object_counts, stamp_twin

COUNTER_DEPTH = 0.65
COUNTER_HEIGHT = 0.92
SYNTH_RADIUS = 0.3
SYNTH_MARGIN = 0.3  # a synthesized counter reaches at least this far past each end of the sink or stove it serves
CLIP_TOL = 0.02  # a counter clipped at a fridge still covers the fixture it serves to within this
ADJACENT_GAP = 0.05  # an appliance this close to a counter's footprint is built in beside it
DEFAULT_OBJECTS = ROOM_OBJECTS["kitchen"]  # sizes come from the catalog, varied per seed; other rooms: room_kind()
PLACE_MARGIN = 0.08
TABLE_EDGE_BAND = 0.40  # outside a kitchen, objects on a table deeper than twice this lie within it of a long edge
# A kitchen island: counter-height block (RoomPlan calls it a table) clear of every wall.
# Dining tables measure about 0.75 m, so the band starts above them. Storage stays a cabinet
# wherever it stands, so a freestanding sideboard is not retyped.
ISLAND_CATEGORIES = ("counter", "table")
ISLAND_HEIGHT = (0.85, 1.15)
ISLAND_CLEARANCE = 0.3
PASS_THROUGH = ("doors", "windows", "openings", "floors")


def to_local(xy, fixture: dict) -> tuple[float, float]:
    """World (x, y) -> fixture-local (x, y): origin at the fixture centre, +X along the fixture's yaw."""
    yaw = float(fixture.get("yaw", 0.0))
    dx, dy = xy[0] - fixture["position"][0], xy[1] - fixture["position"][1]
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def to_world(local, fixture: dict) -> tuple[float, float]:
    yaw = float(fixture.get("yaw", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    return (fixture["position"][0] + c * local[0] - s * local[1],
            fixture["position"][1] + s * local[0] + c * local[1])


def _rect_distance(xy: list[float], fixture: dict) -> float:
    """Horizontal distance from a point to a fixture's footprint rectangle (in the fixture's local frame)."""
    lx, ly = to_local(xy, fixture)
    sx, sy, _ = fixture["size"]
    return math.hypot(max(abs(lx) - sx / 2, 0.0), max(abs(ly) - sy / 2, 0.0))


def _footprint_gap(f: dict, counter: dict) -> float:
    """Horizontal gap between a fixture's footprint (centre and corners) and a counter's footprint."""
    hx, hy = f["size"][0] / 2, f["size"][1] / 2
    pts = [f["position"][:2]] + [to_world((dx, dy), f) for dx in (-hx, hx) for dy in (-hy, hy)]
    return min(_rect_distance(list(p), counter) for p in pts)


def is_counter_host(fixture: dict) -> bool:
    """A worktop that can carry a sink, including RoomPlan's floor-standing storage cabinets."""
    return fixture["type"] == "counter" or (
        fixture.get("source_category") == "storage"
        and abs(fixture["size"][2] - COUNTER_HEIGHT) <= 0.15
        and abs(fixture["position"][2] - fixture["size"][2] / 2) <= 0.1
    )


def _local_x_range(f: dict, frame: dict, half_depth: float | None = None) -> tuple[float, float] | None:
    """Local-X extent of f's footprint in `frame`'s coordinates; None when f lies outside the band |y| < half_depth."""
    hx, hy = f["size"][0] / 2, f["size"][1] / 2
    pts = [to_local(to_world((dx, dy), f), frame) for dx in (-hx, hx) for dy in (-hy, hy)]
    if half_depth is not None and (max(p[1] for p in pts) <= -half_depth + 1e-6
                                   or min(p[1] for p in pts) >= half_depth - 1e-6):
        return None
    return min(p[0] for p in pts), max(p[0] for p in pts)


def _clip_span(lo: float, hi: float, frame: dict, obstacles: list[dict], keep: tuple[float, float]) -> tuple[float, float]:
    """Shorten a synthesized counter's local-X span [lo, hi] so it stops at a fridge, dishwasher or tall cabinet
    standing in its row, never cutting into `keep`, the span of the fixture it was synthesized for."""
    for o in obstacles:
        if o["position"][2] - o["size"][2] / 2 >= COUNTER_HEIGHT:
            continue  # hangs above the counter top
        r = _local_x_range(o, frame, COUNTER_DEPTH / 2)
        if r is None or r[1] <= lo or r[0] >= hi:
            continue
        if r[1] <= keep[0] + CLIP_TOL:
            lo = max(lo, min(r[1], keep[0] - CLIP_TOL))
        elif r[0] >= keep[1] - CLIP_TOL:
            hi = min(hi, max(r[0], keep[1] + CLIP_TOL))
    return lo, hi


def _synth_counters(fixtures: list[dict]) -> list[dict]:
    """A counter under each sink or stove that no counter holds. Neighbouring sinks share one counter (extended, not
    stacked, so counters never overlap each other) and a counter stops at the fridge or dishwasher beside it."""
    hosts = [f for f in fixtures if is_counter_host(f)]
    obstacles = [f for f in fixtures if f["source_category"] not in ("sink", "stove")]  # incl. the real counters
    out: list[dict] = []
    for f in fixtures:
        if f["source_category"] not in ("sink", "stove"):
            continue
        if any(_rect_distance(f["position"], c) <= SYNTH_RADIUS or _footprint_gap(f, c) <= ADJACENT_GAP
               for c in hosts):
            continue  # set into a counter, or built in beside one (a stove between two counters)
        # a sink within 0.3 m of a synthesized counter's end joins its run; a stove must sit on the counter itself
        held = next((c for c in out if (_footprint_gap(f, c) if f["source_category"] == "sink"
                                        else _rect_distance(f["position"], c)) <= SYNTH_RADIUS), None)
        if held is not None:
            _extend(held, f, obstacles)  # same synthesized run: widen it if f hangs off an end
            continue
        if any(_footprint_gap(f, c) <= ADJACENT_GAP for c in out):
            continue  # built in beside a synthesized counter
        half = max(f["size"][0] + 1.2, 1.8) / 2
        keep = _local_x_range(f, f)
        lo, hi = _clip_span(-half, half, f, obstacles, keep)
        out.append({
            "id": f"counter_synth_{f['id']}",
            "type": "counter",
            "size": [hi - lo, COUNTER_DEPTH, COUNTER_HEIGHT],
            "position": [*to_world(((lo + hi) / 2, 0.0), f), COUNTER_HEIGHT / 2],
            "yaw": float(f.get("yaw", 0.0)),  # long axis along the fixture's local X
            "source_category": "synthesized",
        })
    return out


def _extend(counter: dict, f: dict, obstacles: list[dict]) -> None:
    """Widen a synthesized counter so the sink or stove f (same heading) lies wholly on it, with a 0.3 m margin."""
    if abs(math.remainder(float(f.get("yaw", 0.0)) - counter["yaw"], 2 * math.pi)) > 0.02:
        return
    r = _local_x_range(f, counter)
    sx = counter["size"][0]
    lo, hi = -sx / 2, sx / 2
    if r[0] >= lo + SYNTH_MARGIN and r[1] <= hi - SYNTH_MARGIN:
        return
    new_lo, new_hi = _clip_span(min(lo, r[0] - SYNTH_MARGIN), max(hi, r[1] + SYNTH_MARGIN), counter, obstacles, r)
    counter["position"][:2] = list(to_world(((new_lo + new_hi) / 2, 0.0), counter))
    counter["size"][0] = new_hi - new_lo


def wall_distance(fixture: dict, walls: list[dict], step: float = 0.05) -> float:
    """Horizontal distance from a fixture's footprint to the nearest wall segment (points sampled every `step`)."""
    best = math.inf
    for w in walls:
        (x0, y0), (x1, y1) = w["start"], w["end"]
        n = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / step))
        for i in range(n + 1):
            t = i / n
            best = min(best, _rect_distance([x0 + t * (x1 - x0), y0 + t * (y1 - y0)], fixture))
    return best


def is_island(fixture: dict, walls: list[dict]) -> bool:
    lo, hi = ISLAND_HEIGHT
    return (fixture["source_category"] in ISLAND_CATEGORIES and lo <= fixture["size"][2] <= hi
            and bool(walls) and wall_distance(fixture, walls) > ISLAND_CLEARANCE)


FRIDGE_FRONT_BAND = 0.9  # m in front of a fridge where its door swings and the robot stands beside the free edge
FRIDGE_HINGE_MARGIN = 0.05  # m: the hinge goes right only when that side has this much less room than the left


def _front_side_room(f: dict, fixtures: list[dict], walls: list[dict], step: float = 0.02) -> tuple[float, float]:
    """(left, right): free floor beside a fixture's left and right edges (local -x and +x) in the band
    FRIDGE_FRONT_BAND deep in front of its face, up to the nearest wall face or other fixture there. A neighbour
    flush with the face (the same run) is behind the band and does not count; a wall at the room's corner does."""
    sx, sy = f["size"][0], f["size"][1]
    room = [math.inf, math.inf]

    def hit(lx: float, ly: float, pad: float = 0.0) -> None:
        if not sy / 2 <= ly <= sy / 2 + FRIDGE_FRONT_BAND:
            return
        if lx <= -sx / 2:
            room[0] = min(room[0], max(0.0, -sx / 2 - lx - pad))
        elif lx >= sx / 2:
            room[1] = min(room[1], max(0.0, lx - sx / 2 - pad))

    for w in walls:
        (x0, y0), (x1, y1) = w["start"], w["end"]
        n = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / step))
        for i in range(n + 1):
            hit(*to_local((x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n), f), w.get("thickness", 0.0) / 2)
    for g in fixtures:
        if g["id"] == f["id"] or g["type"] == "sink":
            continue
        hx, hy = g["size"][0] / 2, g["size"][1] / 2
        co, si = math.cos(g.get("yaw", 0.0)), math.sin(g.get("yaw", 0.0))
        pts = [to_local((g["position"][0] + co * dx - si * dy, g["position"][1] + si * dx + co * dy), f)
               for dx in (-hx, hx) for dy in (-hy, hy)]
        ax0, ax1 = min(p[0] for p in pts), max(p[0] for p in pts)
        ay0, ay1 = min(p[1] for p in pts), max(p[1] for p in pts)
        if ay1 <= sy / 2 or ay0 >= sy / 2 + FRIDGE_FRONT_BAND:
            continue  # flush with the fridge's face or behind it, or beyond the band
        if ax1 <= -sx / 2:
            hit(ax1, sy / 2)
        elif ax0 >= sx / 2:
            hit(ax0, sy / 2)
    return room[0], room[1]


def fridge_hinge(f: dict, fixtures: list[dict], walls: list[dict]) -> str:
    """Hinge side for a fridge whose scan does not give one: the side with less room in front of it, so the door
    opens away from the wall or cabinet beside it and its free edge, where the robot stands, faces the open room.
    rotated_kitchen's fridge stands with its right side in the room's corner: hinged left (the old default for
    every scan) the stand spot beside its free edge fell outside the room and open_fridge was never run."""
    left, right = _front_side_room(f, fixtures, walls)
    return "right" if right + FRIDGE_HINGE_MARGIN < left else "left"


def compile_twin(bundle: dict) -> dict:
    walls = [dict(w) for w in bundle.get("walls", [])]
    fixtures = []
    for f in bundle.get("fixtures", []):
        fx = {
            "id": f["id"],
            "type": fixture_type(f["category"]),
            "size": [float(v) for v in f["size"]],
            "position": [float(v) for v in f["position"]],
            "yaw": float(f.get("yaw", 0.0)),
            "source_category": f["category"],
        }
        if is_island(fx, walls):
            fx["type"] = "counter"
            fx["island"] = True
        else:
            cfg = default_config_name(f["category"])
            if cfg and fx["type"] == "counter":
                fx["default_config_name"] = cfg
        fixtures.append(fx)
    fixtures += _synth_counters(fixtures)
    given = {f["id"]: f.get("hinge") for f in bundle.get("fixtures", [])}
    for fx in fixtures:
        if not fx["type"].startswith("fridge"):
            continue
        if given.get(fx["id"]) in ("left", "right"):
            fx["hinge"] = given[fx["id"]]
        elif fridge_hinge(fx, fixtures, walls) == "right":  # left is the default and stays unwritten
            fx["hinge"], fx["hinge_source"] = "right", "inferred_from_walls"
    objects = [
        {
            "id": o["id"],
            "catalog_id": catalog_id(o["label"]),
            "label": o["label"],
            "size": [float(v) for v in o["size"]],
            "position": [float(v) for v in o["position"]],
            "yaw": float(o.get("yaw", 0.0)),
            # a scan.json object is somebody's entry; place_default_objects' own label survives a recompile
            "provenance": o.get("provenance") if o.get("provenance") in OBJECT_PROVENANCE else "authored",
        }
        for o in bundle.get("objects", [])
    ]
    twin = {
        "room_id": bundle["room_id"],
        "geometry_source": bundle["geometry_source"],
        "blocks_claims": list(bundle.get("blocks_claims", [])),
        "frame": "z_up",
        "units": "meters",
        "walls": walls,
        "fixtures": fixtures,
        "objects": objects,
    }
    for key in PASS_THROUGH:
        if key in bundle:
            twin[key] = [dict(item) for item in bundle[key]]
    for key in ("story", "story_count"):
        if key in bundle:
            twin[key] = bundle[key]
    return stamp_twin(twin, bundle=bundle)


def place_default_objects(twin: dict, seed: int = 0) -> list[dict]:
    """RoomPlan captures no small objects: put the default objects for the kind of room (`room_kind`: DEFAULT_OBJECTS
    in a kitchen, a book, laptop or remote in an office or living room) on a counter or table top (the one holding or touching a
    sink if any), away from sinks/stoves, deterministically from `seed`, each at a seeded size. An object that
    does not fit the counter is skipped. Mutates twin."""
    counters = [f for f in twin["fixtures"] if is_counter_host(f)]
    blockers = [f for f in twin["fixtures"] if f["type"] in ("sink", "stove")]
    sinks = [f for f in blockers if f["type"] == "sink"]
    if not counters:
        return []
    counter = min(counters, key=lambda c: (min((_rect_distance(s["position"], c) for s in sinks), default=0.0),
                                           c.get("default_config_name") == "dining", c["id"]))
    rng = random.Random(seed)
    sx, sy, _ = counter["size"]
    top = counter["position"][2] + counter["size"][2] / 2
    placed: list[dict] = []
    kind = room_kind(twin["fixtures"])
    # a big table outside a kitchen: things lie along its long edges, where someone sits (and a fixed arm reaches),
    # not in the middle of a 1.6 m top
    edges = kind != "kitchen" and sy > 2 * TABLE_EDGE_BAND
    for label in ROOM_OBJECTS[kind]:
        size = sample_size(label, rng)
        half = max(size[0], size[1]) / 2 + PLACE_MARGIN
        if sx / 2 <= half or sy / 2 <= half:
            continue
        for _ in range(200):
            local = (rng.uniform(-sx / 2 + half, sx / 2 - half), rng.uniform(-sy / 2 + half, sy / 2 - half))
            if edges:
                local = (local[0], rng.choice((-1, 1)) * rng.uniform(sy / 2 - TABLE_EDGE_BAND, sy / 2 - half))
            x, y = to_world(local, counter)
            if any(_rect_distance([x, y], b) < half for b in blockers if b is not counter):
                continue
            if any(math.dist([x, y], o["position"][:2]) < half + max(o["size"][:2]) for o in placed):
                continue
            placed.append({"id": f"{label}_1", "catalog_id": catalog_id(label), "label": label, "size": list(size),
                           "position": [round(x, 6), round(y, 6), round(top + size[2] / 2, 6)], "yaw": 0.0,
                           "placed_by": "place_default_objects", "provenance": "default_synthetic"})
            break
    twin["objects"] += placed
    if placed:  # these were put there by us, not measured: say so in the twin's own record
        note_seed(twin, "default_objects", seed)
        twin["provenance"].setdefault("generators", {})["objects"] = object_counts(twin)
    return placed


def fill_objects(twin: dict, detection: dict | None = None, seed: int = 0) -> list[dict]:
    """Objects for a twin whose scan lists none: the ones `arrival.twin.detect.detect_objects` found (labelled
    `detected`) when there are any, else `place_default_objects` (`default_synthetic`). The detection run, less its
    object list, is kept in twin["object_detection"] so a fallback says why. Mutates twin."""
    run = dict(detection or {"state": "not_run", "reason": "no RGB-D frames"})
    found = run.pop("objects", None) or []
    twin["object_detection"] = run
    if not found:
        return place_default_objects(twin, seed=seed)
    twin["objects"] += found
    twin["provenance"].setdefault("generators", {})["objects"] = object_counts(twin)
    return found


def _wall_entry(w: dict) -> dict:
    (x0, y0), (x1, y1) = w["start"], w["end"]
    length, angle = math.hypot(x1 - x0, y1 - y0), math.atan2(y1 - y0, x1 - x0)
    pos = _r([(x0 + x1) / 2, (y0 + y1) / 2, w["height"] / 2])
    a = abs(math.sin(angle))
    if a < 1e-6:
        size = [length, w["thickness"], w["height"]]
    elif a > 1 - 1e-6:
        size = [w["thickness"], length, w["height"]]
    else:  # oblique wall: length along local X, rotated into place
        return {"name": w["id"], "type": "wall", "size": _r([length, w["thickness"], w["height"]]), "pos": pos,
                "z_rot": round(angle, 6)}
    return {"name": w["id"], "type": "wall", "size": _r(size), "pos": pos}


def layout_dict(twin: dict) -> dict:
    walls, xs, ys = [], [], []
    for w in twin.get("walls", []):
        (x0, y0), (x1, y1) = w["start"], w["end"]
        xs += [x0, x1]
        ys += [y0, y1]
        walls.append(_wall_entry(w))
    room: dict = {"walls": walls}
    corners = [c for fl in twin.get("floors", []) for c in fl["corners"]]
    if corners:  # a scanned floor polygon sets the floor extent; otherwise the wall bounding box does
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    if xs:
        room["floor"] = [{"name": "floor", "type": "floor",
                          "size": _r([max(xs) - min(xs), max(ys) - min(ys), 0.02]),
                          "pos": _r([(max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, 0.0])}]
    bottom_row = []
    for f in twin["fixtures"]:
        entry = {"name": f["id"], "type": f["type"], "size": _r(f["size"]), "pos": _r(f["position"])}
        if f.get("yaw"):
            entry["z_rot"] = round(f["yaw"], 6)
        if f.get("default_config_name"):
            entry["default_config_name"] = f["default_config_name"]
        if f.get("island"):
            entry["island"] = True
        bottom_row.append(entry)
    return {
        "room": room,
        "main_group": {"group_origin": [0, 0], "group_pos": [0.0, 0.0], "group_z_rot": 0.0,
                       "bottom_row": bottom_row},
    }


def _r(v: list[float]) -> list[float]:
    return [round(float(x), 6) for x in v]


def write_twin(twin: dict, twin_dir: str | Path) -> tuple[Path, Path]:
    twin_dir = Path(twin_dir)
    twin_dir.mkdir(parents=True, exist_ok=True)
    twin_path, layout_path = twin_dir / "twin.json", twin_dir / "layout.yaml"
    twin_path.write_text(json.dumps(twin, indent=2))
    layout_path.write_text(yaml.safe_dump(layout_dict(twin), sort_keys=False, default_flow_style=None))
    return twin_path, layout_path
