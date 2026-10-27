"""TWN-1a: twin dimensions vs tape measurements. Real rooms are checked once a scans/<room>/measurements.json
exists (CAP-3); until then the harness is exercised on the hero twin and synthetic layouts."""

import json
import shutil
import sys

import pytest

from arrival.twin.accuracy import MeasurementError, compare, counter_runs, extract_dimensions, load_measurements
from arrival.twin.compile import compile_twin

from conftest import REPO, SCANS

sys.path.insert(0, str(REPO / "scripts"))
import scan_accuracy  # noqa: E402

HERO_FIXTURE = REPO / "tests" / "fixtures" / "measurements" / "hero_kitchen.json"
MEASURED = sorted(SCANS.glob("*/measurements.json"))


def _twin(fixtures, walls=()):
    return compile_twin({"schema_version": "1", "room_id": "t", "units": "meters", "frame": "z_up",
                         "geometry_source": "roomplan", "blocks_claims": [], "walls": list(walls),
                         "fixtures": fixtures, "objects": []})


def _counter(i, x, y, sx, yaw=0.0, sy=0.65):
    return {"id": i, "category": "counter", "size": [sx, sy, 0.92], "position": [x, y, 0.46], "yaw": yaw}


@pytest.mark.skipif(not MEASURED, reason="no scans/<room>/measurements.json yet (CAP-3)")
@pytest.mark.parametrize("path", MEASURED, ids=lambda p: p.parent.name)
def test_real_room_within_tolerance(path):
    from arrival.scan import bundle

    res = compare(compile_twin(bundle.load_room(path.parent.name, SCANS)), load_measurements(path))
    errs = {k: round(r["error_m"] * 100, 1) if r["error_m"] is not None else None
            for k, r in res["dimensions"].items()}
    assert res["ok"], f"{path.parent.name} errors in cm (tolerance {res['tolerance_m'] * 100:g}): {errs}"


def test_hero_dimensions_match_scan_json(hero_twin):
    scan = json.loads((SCANS / "hero_kitchen" / "scan.json").read_text())
    fx = {f["id"]: f for f in scan["fixtures"]}
    # counter_a, the stove and counter_b share one worktop line: read its ends straight off scan.json
    left = min(fx[i]["position"][0] - fx[i]["size"][0] / 2 for i in ("counter_a", "stove_1", "counter_b"))
    right = max(fx[i]["position"][0] + fx[i]["size"][0] / 2 for i in ("counter_a", "stove_1", "counter_b"))
    dims = extract_dimensions(hero_twin)
    assert dims["counter_length_m"] == pytest.approx(right - left, abs=1e-3)
    assert dims["counter_height_m"] == pytest.approx(fx["counter_a"]["position"][2] + fx["counter_a"]["size"][2] / 2,
                                                     abs=1e-3)
    assert dims["sink_width_m"] == pytest.approx(fx["sink_1"]["size"][0], abs=1e-3)
    assert dims["fridge_width_m"] == pytest.approx(fx["fridge_1"]["size"][0], abs=1e-3)
    assert set(counter_runs(hero_twin)[0]["members"]) == {"counter_a", "stove_1", "counter_b"}


def test_hero_authored_fixture_passes_and_is_labelled(hero_twin):
    m = load_measurements(HERO_FIXTURE)
    assert m["authored"] is True and "not tape-measured" in m["measured_by"]
    res = compare(hero_twin, m)
    assert res["ok"] and res["tape_measured"] is False
    assert all(abs(r["error_m"]) < 1e-3 for r in res["dimensions"].values())
    assert not (SCANS / "hero_kitchen" / "measurements.json").exists()


def test_six_cm_fails_four_cm_passes(hero_twin):
    base = load_measurements(HERO_FIXTURE)
    off = compare(hero_twin, {**base, "counter_length_m": base["counter_length_m"] + 0.06})
    assert not off["ok"]
    row = off["dimensions"]["counter_length_m"]
    assert row["error_m"] == pytest.approx(-0.06) and not row["ok"]  # twin is shorter than the tape
    near = compare(hero_twin, {**base, "counter_length_m": base["counter_length_m"] - 0.04})
    assert near["ok"] and near["dimensions"]["counter_length_m"]["error_m"] == pytest.approx(0.04)


def test_tolerance_is_per_file(hero_twin):
    base = load_measurements(HERO_FIXTURE)
    tape = {**base, "sink_width_m": 0.66}
    assert not compare(hero_twin, tape)["ok"]
    assert compare(hero_twin, {**tape, "tolerance_m": 0.07})["ok"]
    no_tol = {k: v for k, v in tape.items() if k != "tolerance_m"}
    assert compare(hero_twin, no_tol)["tolerance_m"] == 0.05


def test_gap_splits_runs_and_touching_pieces_join():
    twin = _twin([_counter("a", 1.0, 0.4, 2.0), _counter("b", 3.3, 0.4, 1.0)])  # 0.3 m gap
    runs = counter_runs(twin)
    assert [round(r["length"], 3) for r in runs] == [2.0, 1.0]
    twin = _twin([_counter("a", 1.0, 0.4, 2.0), _counter("b", 2.5, 0.4, 1.0)])  # flush
    assert counter_runs(twin)[0]["length"] == pytest.approx(3.0)


def test_l_shape_takes_longest_leg():
    # leg 1 along X (1.8 m), leg 2 along Y (2.4 m) rotated 90 degrees
    twin = _twin([_counter("a", 0.9, 0.325, 1.8), _counter("b", 0.325, 2.0, 2.4, yaw=1.5707963267948966)])
    runs = counter_runs(twin)
    assert len(runs) == 2
    assert extract_dimensions(twin)["counter_length_m"] == pytest.approx(2.4)


def test_island_and_synthesized_counters_are_not_measured():
    walls = [{"id": "w", "start": [0, 0], "end": [6, 0], "height": 2.4, "thickness": 0.1}]
    island = {"id": "isl", "category": "table", "size": [2.5, 0.9, 0.92], "position": [3.0, 2.5, 0.46], "yaw": 0}
    twin = _twin([_counter("a", 1.0, 0.4, 1.5), island], walls)
    dims = extract_dimensions(twin)
    assert dims["counter_length_m"] == pytest.approx(1.5)
    assert dims["island_length_m"] == pytest.approx(2.5)
    sink_only = _twin([{"id": "s", "category": "sink", "size": [0.6, 0.5, 0.25], "position": [1, 1, 0.85], "yaw": 0}])
    with pytest.raises(MeasurementError, match="no scanned counter"):
        extract_dimensions(sink_only)


def test_missing_twin_dimension_fails(hero_twin):
    res = compare(hero_twin, {**load_measurements(HERO_FIXTURE), "island_length_m": 1.0})
    row = res["dimensions"]["island_length_m"]
    assert row["twin_m"] is None and not row["ok"] and not res["ok"]


def test_measurements_validation(tmp_path):
    def write(d):
        p = tmp_path / "m.json"
        p.write_text(json.dumps(d))
        return p

    good = {"counter_length_m": 3, "counter_height_m": 0.9, "sink_width_m": 0.6}
    assert load_measurements(write(good)) == good
    with pytest.raises(MeasurementError, match="missing"):
        load_measurements(write({"counter_length_m": 3}))
    with pytest.raises(MeasurementError, match="unknown key"):
        load_measurements(write({**good, "counter_lenght_m": 3}))
    with pytest.raises(MeasurementError, match="positive"):
        load_measurements(write({**good, "sink_width_m": "60cm"}))
    with pytest.raises(MeasurementError, match="positive"):
        load_measurements(write({**good, "tolerance_m": -1}))


def test_script_rejects_missing_room(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        scan_accuracy.main([str(tmp_path / "no_such_room")])
    assert e.value.code == 2
    assert "no_such_room" in capsys.readouterr().err


def _room(tmp_path, name, measurements=None):
    d = tmp_path / name
    d.mkdir()
    shutil.copy(SCANS / "hero_kitchen" / "scan.json", d / "scan.json")
    if measurements is not None:
        (d / "measurements.json").write_text(json.dumps(measurements))
    return d


def test_script_exit_codes_and_skip(tmp_path, capsys):
    good = json.loads(HERO_FIXTURE.read_text())
    ok_room = _room(tmp_path, "ok_room", good)
    bad_room = _room(tmp_path, "bad_room", {**good, "counter_length_m": good["counter_length_m"] + 0.06})
    none_room = _room(tmp_path, "none_room")
    assert scan_accuracy.main([str(ok_room), str(none_room)]) == 0
    out = capsys.readouterr().out
    assert "none_room: no measurements.json, skipped" in out and "1 room(s) measured, 0 failing" in out
    assert "AUTHORED, not tape-measured" in out  # the fixture is flagged in the report, not just in the data
    assert scan_accuracy.main([str(ok_room), str(bad_room)]) == 1
    out = capsys.readouterr().out
    assert "-6.0 cm  FAIL" in out and "1 failing" in out
    assert scan_accuracy.main([str(none_room)]) == 0  # nothing measured is not a failure


def test_script_reads_an_intake_scan_with_only_roomplan_json(tmp_path, capsys):
    d = tmp_path / "intake_room"  # what scripts/intake_scan.py writes: roomplan.json, no scan.json
    d.mkdir()
    shutil.copy(SCANS / "hero_kitchen" / "roomplan.json", d / "roomplan.json")
    (d / "measurements.json").write_text(HERO_FIXTURE.read_text())
    scan_accuracy.main([str(d)])
    out = capsys.readouterr().out
    assert "1 room(s) measured" in out and "intake_room" in out
