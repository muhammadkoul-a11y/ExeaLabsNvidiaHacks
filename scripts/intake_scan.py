"""Land one real scan under scans/<room_id>/ and check it (CAP-3a).

    python scripts/intake_scan.py kitchen_01 --roomplan ~/Downloads/roomplan.json --photos a.jpg b.jpg \\
        --location-type kitchen --phone "iPhone 15 Pro" --ios 18.1 --lighting daylight --clutter low \\
        --measure counter_length=3.20,counter_height=0.92,sink_width=0.55
    python scripts/intake_scan.py kitchen_01 --check    # after filling in a NO-GO folder's notes

Copies roomplan.json, re-encodes the photos without EXIF (GPS, serial, timestamps) at 1600 px or less,
writes notes.md and measurements.json, then normalizes, compiles, places the default objects, runs the
tests/test_scans.py geometry checks, writes out/renders/<room>_roomplan_top.png and prints GO or NO-GO
with the fixture list. Refuses a bad room id, an existing folder, fewer than two photos (unless
--no-photos) and a RoomPlan that does not load or has fewer than three walls. A NO-GO folder is kept;
--check re-runs every check on it without writing anything. Exit 0 on GO, 1 on NO-GO, 2 on a refusal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrival.scan.intake import IntakeError, check, intake  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("room_id", help="lowercase letters, digits and underscores only")
    ap.add_argument("--roomplan", help="the exported CapturedRoom JSON")
    ap.add_argument("--check", action="store_true", help="re-check the existing scans/<room_id>/, write nothing")
    ap.add_argument("--photos", nargs="*", default=[], help="reference photos (at least 2)")
    ap.add_argument("--no-photos", action="store_true", help="allow fewer than 2 photos")
    ap.add_argument("--location-type", default="", help="kitchen, office, ...")
    ap.add_argument("--phone", default="", help="e.g. 'iPhone 15 Pro'")
    ap.add_argument("--ios", default="")
    ap.add_argument("--lighting", default="")
    ap.add_argument("--clutter", default="")
    ap.add_argument("--measure", default="", help="counter_length=..,counter_height=..,sink_width=.. in metres")
    ap.add_argument("--scans", default=str(ROOT / "scans"), help="scans directory (default: the repo's)")
    ap.add_argument("--out", default=str(ROOT / "out" / "renders"), help="where the top-down PNG goes")
    a = ap.parse_args(argv)
    if a.check:
        try:
            r = check(a.room_id, scans=a.scans)
        except IntakeError as e:
            print(f"refused: {e}", file=sys.stderr)
            return 2
        return _report(r)
    if not a.roomplan:
        ap.error("--roomplan is required (or --check to re-check an existing folder)")
    try:
        r = intake(a.room_id, a.roomplan, a.photos, scans=a.scans, no_photos=a.no_photos,
                   location_type=a.location_type, phone=a.phone, ios=a.ios, lighting=a.lighting,
                   clutter=a.clutter, measures=a.measure, out_dir=a.out)
    except IntakeError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(f"wrote {r['dir']}")
    print(f"render {r['render']}")
    return _report(r)


def _report(r: dict) -> int:
    print("fixtures: " + (", ".join(r["fixtures"]) or "none"))
    for p in r["problems"]:
        print(f"  - {p}")
    d = Path(r["dir"])
    scans = "" if d.parent.resolve() == (ROOT / "scans").resolve() else f" --scans {d.parent}"
    print("GO" if r["go"] else f"NO-GO: fix the items above in {d}, then run "
          f"python scripts/intake_scan.py {d.name} --check{scans}")
    return 0 if r["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
