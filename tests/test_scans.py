"""TWN-2: every scan under scans/ normalizes, compiles, writes a layout, has no overlapping fixture
footprints, nothing below the floor, a closed wall loop, and builds in MuJoCo."""

import importlib.util
import math

import pytest
import yaml

from arrival.scan import bundle, roomplan
from arrival.twin.compile import compile_twin, write_twin

from conftest import REPO, SCANS

WALL_TOL = 0.01
FLOOR_TOL = 1e-6
OVERLAP_TOL = 1e-3  # metres; flush fixtures in a real RoomPlan scan cross by float noise (0.04 mm in kitchen_real_01)
HERO_KNOWN_OVERLAPS = {("counter_a", "fridge_1"), ("counter_b", "stove_1")}  # the TWN-6 bug, nothing else


def _cases():
    out = []
    for d in sorted(p for p in SCANS.iterdir() if p.is_dir()):
        for source in ("scan", "roomplan"):
            if (d / f"{source}.json").is_file():
                out.append(pytest.param(d.name, source, id=f"{d.name}-{source}"))
    return out


CASES = _cases()


def _twin(room: str, source: str) -> dict:
    if source == "scan":
        return compile_twin(bundle.load_room(room, SCANS))
    return compile_twin(roomplan.load(SCANS / room / "roomplan.json", room))


def _corners(f: dict) -> list[tuple[float, float]]:
    sx, sy, _ = f["size"]
    x, y, _ = f["position"]
    c, s = math.cos(f.get("yaw", 0.0)), math.sin(f.get("yaw", 0.0))
    return [(x + c * dx - s * dy, y + s * dx + c * dy)
            for dx, dy in ((sx / 2, sy / 2), (-sx / 2, sy / 2), (-sx / 2, -sy / 2), (sx / 2, -sy / 2))]


def _overlap(a: dict, b: dict, tol: float = OVERLAP_TOL) -> bool:
    """Separating-axis test on the two footprint rectangles. Touching edges, or crossing by tol or less, do not count."""
    ca, cb = _corners(a), _corners(b)
    for poly in (ca, cb):
        for i in range(4):
            ex, ey = poly[(i + 1) % 4][0] - poly[i][0], poly[(i + 1) % 4][1] - poly[i][1]
            n = math.hypot(ex, ey) or 1.0
            ex, ey = ex / n, ey / n
            pa = [-ey * x + ex * y for x, y in ca]
            pb = [-ey * x + ex * y for x, y in cb]
            if max(pa) <= min(pb) + tol or max(pb) <= min(pa) + tol:
                return False
    return True


def _contains(outer: dict, inner: dict, tol: float = 1e-6) -> bool:
    from arrival.twin.compile import to_local

    sx, sy, _ = outer["size"]
    return all(abs(lx) <= sx / 2 + tol and abs(ly) <= sy / 2 + tol
               for lx, ly in (to_local(c, outer) for c in _corners(inner)))


def _allowed(a: dict, b: dict) -> bool:
    """A sink or stove inset in a counter or counter-height cabinet, or the counter synthesized around that fixture."""
    from arrival.twin.compile import is_counter_host

    for counter, fixture in ((a, b), (b, a)):
        if not is_counter_host(counter) or fixture["type"] not in ("sink", "stove"):
            continue
        if counter["id"] == f"counter_synth_{fixture['id']}" or _contains(counter, fixture):
            return True
    return False


@pytest.mark.parametrize("room,source", CASES)
def test_no_overlapping_fixtures(room, source):
    fx = _twin(room, source)["fixtures"]
    bad = {tuple(sorted((a["id"], b["id"]))) for i, a in enumerate(fx) for b in fx[i + 1:]
           if _overlap(a, b) and not _allowed(a, b)}
    if (room, source) == ("hero_kitchen", "scan") and bad == HERO_KNOWN_OVERLAPS:
        pytest.xfail(f"TWN-6: authored hero kitchen overlaps {sorted(bad)}")
    assert not bad


@pytest.mark.parametrize("room,source", CASES)
def test_nothing_below_floor(room, source):
    twin = _twin(room, source)
    for item in twin["fixtures"] + twin["objects"]:
        assert item["position"][2] - item["size"][2] / 2 >= -FLOOR_TOL, item["id"]


@pytest.mark.parametrize("room,source", CASES)
def test_walls_closed(room, source):
    walls = _twin(room, source)["walls"]
    assert len(walls) >= 3
    for i, w in enumerate(walls):
        others = [p for j, o in enumerate(walls) if j != i for p in (o["start"], o["end"])]
        for end in (w["start"], w["end"]):
            assert min(math.dist(end, p) for p in others) <= WALL_TOL, (w["id"], end)


@pytest.mark.parametrize("room,source", CASES)
def test_layout_writes(room, source, tmp_path):
    twin = _twin(room, source)
    _, layout_path = write_twin(twin, tmp_path)
    doc = yaml.safe_load(layout_path.read_text())
    assert len(doc["main_group"]["bottom_row"]) == len(twin["fixtures"])
    assert len(doc["room"]["walls"]) == len(twin["walls"])


@pytest.mark.parametrize("room,source", CASES)
def test_scene_builds_in_mujoco(room, source):
    mujoco = pytest.importorskip("mujoco")
    from arrival.sim.scene import build_mjcf

    twin = _twin(room, source)
    xs = [p for w in twin["walls"] for p in (w["start"][0], w["end"][0])]
    ys = [p for w in twin["walls"] for p in (w["start"][1], w["end"][1])]
    base = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 0.75)
    model = mujoco.MjModel.from_xml_string(build_mjcf(twin, base))
    assert model.ngeom > len(twin["walls"])


def test_overlap_ignores_float_noise_between_flush_fixtures():
    """kitchen_real_01's dishwasher and the cabinet beside it share an edge to 0.04 mm (RoomPlan float noise)."""
    a = {"position": [1.0, 0.0, 0.0], "size": [1.0, 0.6, 0.9], "yaw": 3.141592654}
    flush = {"position": [2.0 - 0.00004, 0.0, 0.0], "size": [1.0, 0.6, 0.9], "yaw": 3.141592654}
    into = {"position": [2.0 - 0.005, 0.0, 0.0], "size": [1.0, 0.6, 0.9], "yaw": 0.0}
    assert not _overlap(a, flush)
    assert _overlap(a, into)


def test_scan_files_are_read_as_utf8_whatever_the_locale(tmp_path, monkeypatch):
    """Windows defaults to cp1252, which garbles "Schüssel" and cannot decode "Á" (UTF-8 byte 0x81) at all."""
    import builtins
    import io
    import json

    from arrival.twin.accuracy import load_measurements

    real = io.open

    def cp1252(file, mode="r", buffering=-1, encoding=None, *a, **kw):
        if "b" not in mode and encoding in (None, "locale"):
            encoding = "cp1252"
        return real(file, mode, buffering, encoding, *a, **kw)

    scan = json.loads((SCANS / "hero_kitchen" / "scan.json").read_text(encoding="utf-8"))
    scan["objects"][0]["label"] = "Ásztal tál"
    (tmp_path / "k").mkdir()
    (tmp_path / "k" / "scan.json").write_text(json.dumps(scan, ensure_ascii=False), encoding="utf-8")
    rp = json.loads((SCANS / "hero_kitchen" / "roomplan.json").read_text(encoding="utf-8"))
    (tmp_path / "rp.json").write_text(json.dumps({**rp, "note": "Ásztal"}, ensure_ascii=False), encoding="utf-8")
    m = json.loads((REPO / "tests" / "fixtures" / "measurements" / "hero_kitchen.json").read_text(encoding="utf-8"))
    (tmp_path / "m.json").write_text(json.dumps({**m, "measured_by": "Ágnes"}, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(builtins, "open", cp1252)
    monkeypatch.setattr(io, "open", cp1252)
    assert bundle.load_room("k", tmp_path)["objects"][0]["label"] == "Ásztal tál"
    assert roomplan.load(tmp_path / "rp.json", "rp")["fixtures"]
    assert load_measurements(tmp_path / "m.json")["measured_by"] == "Ágnes"


@pytest.mark.parametrize("arg", ["hero_kitchn", "nope.json"])
def test_render_rooms_refuses_a_path_with_no_scan(tmp_path, capsys, arg):
    spec = importlib.util.spec_from_file_location("render_rooms", REPO / "scripts" / "render_rooms.py")
    render_rooms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render_rooms)
    out = tmp_path / "renders"
    with pytest.raises(SystemExit) as e:
        render_rooms.main([str(tmp_path / arg), "--out", str(out)])
    assert e.value.code == 2
    assert "no scan.json or roomplan.json at" in capsys.readouterr().err
    assert not out.exists()
