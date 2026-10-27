"""CAP-3a: scan intake lands a correct, EXIF-free scans/<room_id>/ and refuses bad input. Every test writes to tmp_path."""

import importlib.util
import json
import re
from pathlib import Path

import pytest
from PIL import Image

from arrival.scan import intake as it
from arrival.twin.accuracy import load_measurements

from conftest import REPO

OFFICE = REPO / "tests" / "fixtures" / "roomplan" / "office.json"
KITCHEN = REPO / "tests" / "fixtures" / "roomplan" / "kitchen_real_01.json"
MEASURE = "counter_length=3.20,counter_height=0.92,sink_width=0.55"
META = dict(location_type="kitchen", phone="iPhone 15 Pro", ios="18.1", lighting="daylight", clutter="low")


@pytest.fixture
def clean_plan(tmp_path):
    """office.json with its objects dropped: the walls and floor only, so the geometry checks have nothing to flag."""
    plan = json.loads(OFFICE.read_text())
    plan["objects"] = []
    path = tmp_path / "clean.json"
    path.write_text(json.dumps(plan))
    return path


def _jpeg(path: Path, size=(2400, 1800)) -> Path:
    exif = Image.Exif()
    exif[0x010F] = "Apple"
    exif[0x0132] = "2026:10:01 10:00:00"
    exif[0xA431] = "SERIAL123456"
    gps = exif.get_ifd(it.GPS_IFD)
    gps[1], gps[2], gps[3], gps[4] = "N", (37.0, 46.0, 29.0), "W", (122.0, 25.0, 9.0)
    Image.new("RGB", size, (120, 90, 60)).save(path, "JPEG", exif=exif)
    return path


@pytest.fixture
def photos(tmp_path):
    return [_jpeg(tmp_path / "a.jpg"), _jpeg(tmp_path / "b.jpg", (800, 600))]


def _run(tmp_path, photos, room="kitchen_01", plan=OFFICE, **kw):
    args = dict(scans=tmp_path / "scans", out_dir=tmp_path / "renders", measures=MEASURE, **META)
    args.update(kw)
    return it.intake(room, plan, photos, **args)


def test_fixture_photos_really_carry_gps(photos):
    assert it._photo_problems(photos[0]) == ["a.jpg carries an EXIF GPS tag"]


def test_office_is_no_go_for_its_footprint_overlaps_but_the_folder_validates(tmp_path, photos):
    r = _run(tmp_path, photos)
    assert it.validate_scan_dir(r["dir"]) == []
    assert not r["go"] and len(r["problems"]) == 1 and "overlapping fixtures" in r["problems"][0]


def test_real_kitchen_scan_is_go(tmp_path, photos):
    """Flush cabinets and a dishwasher in a real scan touch to within float noise; that is not an overlap."""
    r = _run(tmp_path, photos, plan=KITCHEN)
    assert r["go"], r["problems"]


def test_intake_lands_a_valid_folder_without_exif(tmp_path, photos, clean_plan):
    r = _run(tmp_path, photos, plan=clean_plan)
    d = r["dir"]
    assert d == tmp_path / "scans" / "kitchen_01"
    assert it.validate_scan_dir(d) == []
    assert r["go"], r["problems"]
    assert r["render"].is_file() and r["fixtures"] == []
    assert (d / "roomplan.json").read_bytes() == clean_plan.read_bytes()
    saved = sorted(d.glob("reference_*.jpg"))
    assert len(saved) == 2
    for p in saved:
        with Image.open(p) as im:
            assert len(im.getexif()) == 0 and not im.getexif().get_ifd(it.GPS_IFD)
            assert "exif" not in im.info and max(im.size) <= it.MAX_PHOTO_PX
    with Image.open(d / "reference_1.jpg") as im:
        assert im.size == (1600, 1200)
    with Image.open(d / "reference_2.jpg") as im:
        assert im.size == (800, 600)  # never upscaled
    assert load_measurements(d / "measurements.json") == {
        "counter_length_m": 3.2, "counter_height_m": 0.92, "sink_width_m": 0.55}
    assert not list((tmp_path / "scans").glob(".*"))


def test_no_photos_flag_and_missing_notes_are_no_go(tmp_path):
    r = _run(tmp_path, [], no_photos=True, measures=None)
    problems = " ".join(r["problems"])
    assert not r["go"]
    assert "'Counter length (m)' is missing" in problems and "0 photos" in problems
    assert "TODO" in (r["dir"] / "notes.md").read_text()


@pytest.mark.parametrize("room", ["Kitchen", "kitchen-1", "my kitchen", "../x", ""])
def test_bad_room_id_refused(tmp_path, photos, room):
    with pytest.raises(it.IntakeError, match="lowercase letters, digits and underscores"):
        _run(tmp_path, photos, room=room)
    assert not (tmp_path / "scans").exists()


def test_existing_folder_refused_and_untouched(tmp_path, photos):
    d = tmp_path / "scans" / "kitchen_01"
    d.mkdir(parents=True)
    (d / "keep.txt").write_text("mine")
    with pytest.raises(it.IntakeError, match="already exists"):
        _run(tmp_path, photos)
    assert [p.name for p in d.iterdir()] == ["keep.txt"]


def test_fewer_than_two_photos_refused(tmp_path, photos):
    with pytest.raises(it.IntakeError, match="at least 2 are required"):
        _run(tmp_path, photos[:1])
    assert not (tmp_path / "scans" / "kitchen_01").exists()


@pytest.mark.parametrize("name", ["IMG_0001.HEIC", "IMG_0001.jpg"])
def test_heic_photo_refused_with_how_to_get_jpeg(tmp_path, photos, name):
    box = b"ftypheic" + bytes(4) + b"mif1heic"
    heic = tmp_path / name
    heic.write_bytes((len(box) + 4).to_bytes(4, "big") + box + bytes(64))
    with pytest.raises(it.IntakeError, match="HEIC/HEIF.*Most Compatible"):
        _run(tmp_path, [photos[0], heic])
    assert not (tmp_path / "scans" / "kitchen_01").exists()


@pytest.mark.parametrize("spec", ["counter_length=abc", "counter_length=3.2,sink_width=", "counter_height=92cm",
                                  "counter_length=-1", "counter_length=320", "counter_length=nan", "desk=1.0"])
def test_bad_measurement_refused(tmp_path, photos, spec):
    with pytest.raises(it.IntakeError, match="measurement|unknown"):
        _run(tmp_path, photos, measures=spec)
    assert not (tmp_path / "scans" / "kitchen_01").exists()


def test_roomplan_with_no_walls_refused(tmp_path, photos):
    plan = json.loads(OFFICE.read_text())
    plan["walls"] = []
    bad = tmp_path / "nowalls.json"
    bad.write_text(json.dumps(plan))
    with pytest.raises(it.IntakeError, match="0 walls"):
        _run(tmp_path, photos, plan=bad)
    junk = tmp_path / "junk.json"
    junk.write_text("[1, 2")
    with pytest.raises(it.IntakeError, match="does not load"):
        _run(tmp_path, photos, plan=junk)
    assert not (tmp_path / "scans" / "kitchen_01").exists()


def test_validate_names_each_missing_or_bad_notes_field(tmp_path, photos):
    d = _run(tmp_path, photos)["dir"]
    notes = d / "notes.md"
    good = notes.read_text()
    notes.write_text(good.replace("- Lighting: daylight\n", "").replace("iOS version: 18.1", "iOS version:   "))
    problems = it.validate_scan_dir(d)
    assert any("'Lighting' is missing" in p for p in problems)
    assert any("'iOS version' is missing" in p for p in problems)
    notes.write_text(good.replace("Sink width (m): 0.55", "Sink width (m): about half a metre"))
    assert [p for p in it.validate_scan_dir(d) if "Sink width" in p and "half a metre" in p]
    notes.write_text(good.replace("Counter height (m): 0.92", "Counter height (m): 92"))
    assert any("Counter height" in p and "'92'" in p for p in it.validate_scan_dir(d))
    notes.unlink()
    assert "notes.md is missing" in it.validate_scan_dir(d)


def test_intake_requires_accuracy_file_and_matching_tape_notes(tmp_path, photos, clean_plan):
    d = _run(tmp_path, photos, plan=clean_plan)["dir"]
    measures = d / "measurements.json"
    good = measures.read_text()
    measures.unlink()
    assert any("measurements.json is missing" in p for p in it.validate_scan_dir(d))
    measures.write_text(good.replace('"sink_width_m": 0.55', '"sink_width_m": 0.65'))
    assert any("sink_width does not match" in p for p in it.validate_scan_dir(d))
    measures.write_text('{"counter_length": 3.2}')
    assert any("measurements.json is invalid" in p for p in it.validate_scan_dir(d))


def test_validate_flags_gps_photo_and_bad_roomplan(tmp_path, photos):
    d = _run(tmp_path, photos)["dir"]
    _jpeg(d / "reference_1.jpg", (100, 100))
    assert any("reference_1.jpg carries an EXIF GPS tag" in p for p in it.validate_scan_dir(d))
    (d / "roomplan.json").write_text("{}")
    assert any("0 walls" in p for p in it.validate_scan_dir(d))
    assert it.validate_scan_dir(tmp_path / "nope") == [f"{tmp_path / 'nope'} is not a directory"]


def test_cli_go_and_refusal(tmp_path, photos, clean_plan, capsys):
    spec = importlib.util.spec_from_file_location("intake_scan", REPO / "scripts" / "intake_scan.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    base = ["kitchen_01", "--roomplan", str(clean_plan), "--scans", str(tmp_path / "scans"), "--out", str(tmp_path / "r"),
            "--location-type", "kitchen", "--phone", "iPhone 15 Pro", "--ios", "18.1", "--lighting", "daylight",
            "--clutter", "low", "--measure", MEASURE]
    assert cli.main(base + ["--photos", *map(str, photos)]) == 0
    assert "GO" in capsys.readouterr().out
    assert cli.main(base + ["--photos", *map(str, photos)]) == 2
    assert "already exists" in capsys.readouterr().err


def test_check_reruns_every_check_on_an_existing_folder(tmp_path, photos, clean_plan):
    amal = it.check(_run(tmp_path, photos, room="amal")["dir"].name, scans=tmp_path / "scans")
    assert not amal["go"] and "overlapping fixtures" in " ".join(amal["problems"])  # geometry, not only notes
    d = _run(tmp_path, photos, plan=clean_plan, measures="counter_length=3.2,counter_height=0.92")["dir"]
    before = it.check("kitchen_01", scans=tmp_path / "scans")
    assert not before["go"] and "notes.md: 'Sink width (m)' is missing or blank" in before["problems"]
    notes = d / "notes.md"
    notes.write_text(re.sub(r"(Sink width \(m\)[^\n]*?)TODO", r"\g<1>0.55", notes.read_text(encoding="utf-8")),
                     encoding="utf-8")
    (d / "measurements.json").write_text(json.dumps(
        {"counter_length_m": 3.2, "counter_height_m": 0.92, "sink_width_m": 0.55}))
    after = it.check("kitchen_01", scans=tmp_path / "scans")
    assert after["go"], after["problems"]
    with pytest.raises(it.IntakeError, match="does not exist"):
        it.check("kitchen_02", scans=tmp_path / "scans")


def test_cli_no_go_points_at_check(tmp_path, photos, clean_plan, capsys):
    spec = importlib.util.spec_from_file_location("intake_scan", REPO / "scripts" / "intake_scan.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    scans = ["--scans", str(tmp_path / "scans")]
    assert cli.main(["kitchen_01", "--roomplan", str(clean_plan), "--out", str(tmp_path / "r"), *scans,
                     "--photos", *map(str, photos)]) == 1
    out = capsys.readouterr().out
    assert f"intake_scan.py kitchen_01 --check --scans {tmp_path / 'scans'}" in out
    assert cli.main(["kitchen_01", "--check", *scans]) == 1
    assert "NO-GO" in capsys.readouterr().out
    assert cli.main(["kitchen_02", "--check", *scans]) == 2
    with pytest.raises(SystemExit) as e:
        cli.main(["kitchen_01", *scans])
    assert e.value.code == 2 and "--roomplan is required" in capsys.readouterr().err
