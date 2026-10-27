"""CAP-3a: land one real scan (RoomPlan JSON, reference photos, tape measurements) as scans/<room_id>/.

    intake(room_id, roomplan_path, photos, measures=...)   # creates the folder, runs the checks
    validate_scan_dir(path)                                 # list of problems, empty when the folder is complete
    check(room_id)                                          # validate_scan_dir plus the geometry checks

The folder holds roomplan.json as exported, reference_N.jpg re-encoded without EXIF (GPS, device serial,
timestamps) and downscaled to MAX_PHOTO_PX, notes.md from NOTES_TEMPLATE, and measurements.json, the
machine-readable copy of the three tape values that TWN-1 reads. Nothing is written when an input is refused.
"""

from __future__ import annotations

import importlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps

from arrival.scan import roomplan

ROOT = Path(__file__).resolve().parents[2]
ROOM_ID = re.compile(r"[a-z0-9_]+")
MAX_PHOTO_PX = 1600
MIN_PHOTOS = 2
GPS_IFD = 0x8825
HEIF_BRANDS = {b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"hevm", b"hevs", b"mif1", b"msf1"}
MEASURES = ("counter_length", "counter_height", "sink_width")
MAX_MEASURE_M = 20.0  # a tape value above this is almost certainly centimetres
# notes.md key -> label shown in the template. Keys are what validate_scan_dir looks for.
FIELDS = {
    "location_type": "Location type",
    "phone_model": "Phone model",
    "ios_version": "iOS version",
    "lighting": "Lighting",
    "clutter": "Clutter",
    "counter_length_m": "Counter length (m)",
    "counter_height_m": "Counter height (m)",
    "sink_width_m": "Sink width (m)",
}
NOTES_TEMPLATE = """# {room_id}

Fill in every field below; `validate_scan_dir` refuses a folder with a blank one. Tape values are metres, a
plain number (0.92), measured with a tape measure and not read off the scan.

- Location type: {location_type}
- Phone model: {phone_model}
- iOS version: {ios_version}
- Lighting: {lighting}
- Clutter: {clutter}
- Counter length (m): {counter_length_m}
- Counter height (m): {counter_height_m}
- Sink width (m): {sink_width_m}

Optional, see docs/SCANNING.md: capture date, reflective or occluded surfaces, scan issues or objects moved,
what each tape measurement spanned, and that the owner permitted public geometry and photos.

measurements.json is the machine-readable copy of the three tape values; TWN-1 reads it. Keep it equal to
the notes.
"""


class IntakeError(ValueError):
    """An input was refused; nothing was written."""


def _key(label: str) -> str:
    label = re.sub(r"\(.*?\)", "", label.lower())
    return re.sub(r"[^a-z0-9]+", "_", label).strip("_")


def _parse_notes(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*(?:[-*]\s*)?([^:]{1,40}):\s*(.*?)\s*$", line)
        if m:
            out.setdefault(_key(m.group(1)) + ("_m" if "(m)" in m.group(1) else ""), m.group(2))
    return out


def _number(text: str) -> float | None:
    try:
        v = float(text.strip())
    except ValueError:
        return None
    return v if math.isfinite(v) and 0 < v <= MAX_MEASURE_M else None


def parse_measures(spec: str | dict | None) -> dict[str, float]:
    """'counter_length=3.2,counter_height=0.92,sink_width=0.5' (or a dict) -> floats in metres. Empty -> {}."""
    if not spec:
        return {}
    pairs = spec.items() if isinstance(spec, dict) else (p.partition("=")[::2] for p in spec.split(",") if p.strip())
    out = {}
    for k, v in pairs:
        k = str(k).strip()
        if k not in MEASURES:
            raise IntakeError(f"unknown measurement {k!r}; expected {', '.join(MEASURES)}")
        if k in out:
            raise IntakeError(f"duplicate measurement {k!r}")
        n = _number(str(v))
        if n is None:
            raise IntakeError(f"measurement {k}={str(v).strip()!r} must be a number of metres, above 0 and at most "
                              f"{MAX_MEASURE_M:g} (a value in centimetres?)")
        out[k] = n
    return out


def _is_heif(path: Path) -> bool:
    """An iPhone's default photo format; Pillow has no HEIF decoder unless pillow-heif is installed."""
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return False
    return head[4:8] == b"ftyp" and head[8:12] in HEIF_BRANDS


def _photo_problems(path: Path) -> list[str]:
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if GPS_IFD in exif or exif.get_ifd(GPS_IFD):
                return [f"{path.name} carries an EXIF GPS tag"]
    except Exception as e:  # noqa: BLE001 - any unreadable file is a problem, not a crash
        return [f"{path.name} is not a readable image ({e})"]
    return []


def _load_roomplan(path: Path, room_id: str) -> dict:
    try:
        bundle = roomplan.load(path, room_id)
    except Exception as e:  # noqa: BLE001
        raise IntakeError(f"{path.name} does not load as a RoomPlan export ({type(e).__name__}: {e})") from e
    if len(bundle["walls"]) < 3:
        raise IntakeError(f"{path.name} has {len(bundle['walls'])} walls; a room needs at least 3")
    return bundle


def validate_scan_dir(path: str | Path) -> list[str]:
    """Every reason scans/<room_id>/ is not a complete intake; [] when it is."""
    d = Path(path)
    if not d.is_dir():
        return [f"{d} is not a directory"]
    problems = []
    fields = {}
    notes = d / "notes.md"
    if not notes.is_file():
        problems.append("notes.md is missing")
    else:
        fields = _parse_notes(notes.read_text(encoding="utf-8"))
        for key, label in FIELDS.items():
            value = fields.get(key, "")
            if not value or value.startswith("<") or value.upper() == "TODO":
                problems.append(f"notes.md: '{label}' is missing or blank")
            elif key.endswith("_m") and _number(value) is None:
                problems.append(f"notes.md: '{label}' must be a number of metres above 0 and at most "
                                f"{MAX_MEASURE_M:g}, got {value!r}")
    measurements = d / "measurements.json"
    if not measurements.is_file():
        problems.append("measurements.json is missing; three tape measurements are required")
    else:
        from arrival.twin.accuracy import MeasurementError, load_measurements

        try:
            measured = load_measurements(measurements)
        except (MeasurementError, ValueError, OSError) as e:
            problems.append(f"measurements.json is invalid: {e}")
        else:
            for key in MEASURES:
                note = _number(fields.get(f"{key}_m", ""))
                if note is not None and not math.isclose(note, measured[f"{key}_m"], abs_tol=1e-9):
                    problems.append(f"notes.md {key} does not match measurements.json")
    photos = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len(photos) < MIN_PHOTOS:
        problems.append(f"{len(photos)} photos; at least {MIN_PHOTOS} are required")
    for p in photos:
        problems += _photo_problems(p)
    plan = d / "roomplan.json"
    if not plan.is_file():
        problems.append("roomplan.json is missing")
    else:
        try:
            _load_roomplan(plan, d.name)
        except IntakeError as e:
            problems.append(str(e))
    return problems


def _reencode(src: Path, dst: Path) -> None:
    """Orientation applied, then a fresh image from the pixels only: no EXIF, ICC profile or comments survive."""
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((MAX_PHOTO_PX, MAX_PHOTO_PX))
    clean = Image.new("RGB", im.size)
    clean.paste(im)
    clean.save(dst, "JPEG", quality=88)


def _checks(bundle: dict) -> dict:
    """The tests/test_scans.py checks (its helpers, unchanged) on the compiled twin, as a report."""
    from arrival.twin.compile import compile_twin, place_default_objects

    tests_dir = str(ROOT / "tests")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    ts = importlib.import_module("test_scans")
    twin = compile_twin(bundle)
    place_default_objects(twin)
    problems = []
    fx = twin["fixtures"]
    overlaps = sorted({tuple(sorted((a["id"], b["id"]))) for i, a in enumerate(fx) for b in fx[i + 1:]
                       if ts._overlap(a, b) and not ts._allowed(a, b)})
    if overlaps:
        problems.append(f"overlapping fixtures: {overlaps}")
    for item in fx + twin["objects"]:
        if item["position"][2] - item["size"][2] / 2 < -ts.FLOOR_TOL:
            problems.append(f"{item['id']} is below the floor")
    walls = twin["walls"]
    for i, w in enumerate(walls):
        others = [p for j, o in enumerate(walls) if j != i for p in (o["start"], o["end"])]
        for end in (w["start"], w["end"]):
            if min(math.dist(end, p) for p in others) > ts.WALL_TOL:
                problems.append(f"wall {w['id']} is not closed at {end}")
    return {"twin": twin, "problems": problems}


def _render(room_id: str, twin: dict, out_dir: Path) -> Path:
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    render_rooms = importlib.import_module("render_rooms")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{room_id}_roomplan_top.png"
    render_rooms.render(f"{room_id}_roomplan", twin, 100.0).save(path)
    return path


def check(room_id: str, *, scans: str | Path | None = None) -> dict:
    """Re-check an existing scans/<room_id>/ (a NO-GO intake whose notes were filled in since): validate_scan_dir
    plus the same geometry checks intake ran. Returns {"dir", "go", "problems", "fixtures"}."""
    if not ROOM_ID.fullmatch(room_id or ""):
        raise IntakeError(f"room id {room_id!r} must be lowercase letters, digits and underscores only")
    dest = (Path(scans) if scans else ROOT / "scans") / room_id
    if not dest.is_dir():
        raise IntakeError(f"{dest} does not exist; land the scan with intake first")
    problems = validate_scan_dir(dest)
    fixtures = []
    try:
        bundle = _load_roomplan(dest / "roomplan.json", room_id)
    except IntakeError:
        bundle = None  # validate_scan_dir already reported it
    if bundle is not None:
        report = _checks(bundle)
        problems += report["problems"]
        fixtures = [f"{f['id']} ({f['type']})" for f in report["twin"]["fixtures"]]
    return {"dir": dest, "go": not problems, "problems": problems, "fixtures": fixtures}


def intake(room_id: str, roomplan_path: str | Path, photos: list | None = None, *, scans: str | Path | None = None,
           no_photos: bool = False, location_type: str = "", phone: str = "", ios: str = "", lighting: str = "",
           clutter: str = "", measures: str | dict | None = None, out_dir: str | Path | None = None) -> dict:
    """Create scans/<room_id>/ and check it. Raises IntakeError (nothing written) on a bad id, an existing
    folder, too few or unreadable photos, a RoomPlan that does not load or has fewer than 3 walls, or a bad
    measurement. Returns {"dir", "go", "problems", "fixtures", "render"}; go is False while the notes are
    incomplete or any geometry check fails."""
    if not ROOM_ID.fullmatch(room_id or ""):
        raise IntakeError(f"room id {room_id!r} must be lowercase letters, digits and underscores only")
    scans = Path(scans) if scans else ROOT / "scans"
    dest = scans / room_id
    if dest.exists():
        raise IntakeError(f"{dest} already exists; intake never overwrites a scan")
    photos = [Path(p) for p in photos or []]
    if len(photos) < MIN_PHOTOS and not no_photos:
        raise IntakeError(f"{len(photos)} photos given; at least {MIN_PHOTOS} are required (or pass --no-photos)")
    plan = Path(roomplan_path)
    bundle = _load_roomplan(plan, room_id)
    tape = parse_measures(measures)
    for p in photos:
        if _is_heif(p):
            raise IntakeError(f"photo {p} is HEIC/HEIF, which intake cannot read; convert it to JPEG, or "
                              "take the photos with the iPhone's Settings > Camera > Formats set to Most Compatible")
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as e:  # noqa: BLE001
            raise IntakeError(f"photo {p} is not a readable image ({e})") from e

    scans.mkdir(parents=True, exist_ok=True)
    tmp = scans / f".{room_id}.intake"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        shutil.copyfile(plan, tmp / "roomplan.json")
        for i, p in enumerate(photos, 1):
            _reencode(p, tmp / f"reference_{i}.jpg")
        values = {"location_type": location_type, "phone_model": phone, "ios_version": ios, "lighting": lighting,
                  "clutter": clutter, **{f"{k}_m": tape.get(k, "") for k in MEASURES}}
        values = {k: v if v != "" else "TODO" for k, v in values.items()}
        (tmp / "notes.md").write_text(NOTES_TEMPLATE.format(room_id=room_id, **values), encoding="utf-8")
        if tape:
            (tmp / "measurements.json").write_text(json.dumps(
                {f"{k}_m": tape[k] for k in MEASURES if k in tape}, indent=2) + "\n")
        tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    report = _checks(bundle)
    problems = validate_scan_dir(dest) + report["problems"]
    twin = report["twin"]
    render = _render(room_id, twin, Path(out_dir) if out_dir else ROOT / "out" / "renders")
    fixtures = [f"{f['id']} ({f['type']})" for f in twin["fixtures"]]
    return {"dir": dest, "go": not problems, "problems": problems, "fixtures": fixtures, "render": render}
