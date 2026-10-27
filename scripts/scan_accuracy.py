"""Twin dimensions vs tape measurements for every scans/<room> that has a measurements.json (TWN-1a).

    python scripts/scan_accuracy.py                  # all rooms under scans/
    python scripts/scan_accuracy.py scans/my_kitchen

Prints one row per measured dimension with the signed error (twin minus tape) and exits 1 when any
dimension is off by more than the room's tolerance_m (default 5 cm). Rooms without measurements.json
are skipped. The rule for reading each dimension off the twin is in arrival/twin/accuracy.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrival.scan import bundle, roomplan  # noqa: E402
from arrival.twin.accuracy import MeasurementError, compare, load_measurements  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402


def load_scan(d: Path) -> dict:
    """scans/<room>/scan.json, or the roomplan.json that scripts/intake_scan.py lands a new scan with."""
    if (d / "scan.json").is_file() or not (d / "roomplan.json").is_file():
        return bundle.load_room(d.name, d.parent)
    return roomplan.load(d / "roomplan.json", d.name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("rooms", nargs="*", type=Path, help="room directories (default: every one under scans/)")
    args = ap.parse_args(argv)
    dirs = args.rooms or sorted(p for p in (ROOT / "scans").glob("*") if p.is_dir())
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing:
        ap.error(f"no such room folder: {', '.join(missing)}")
    failed = measured = 0
    for d in dirs:
        mfile = d / "measurements.json"
        if not mfile.is_file():
            print(f"{d.name}: no measurements.json, skipped")
            continue
        measured += 1
        try:
            res = compare(compile_twin(load_scan(d)), load_measurements(mfile))
        except (MeasurementError, ValueError, KeyError) as e:
            print(f"{d.name}: ERROR {e}")
            failed += 1
            continue
        by = f"{res['measured_by']}, {res['date']}" if res["measured_by"] else "unknown"
        if not res["tape_measured"]:
            by += "; AUTHORED, not tape-measured, not evidence of scan accuracy"
        print(f"{d.name}  (tolerance {res['tolerance_m'] * 100:.1f} cm, measured by {by})")
        for key, r in res["dimensions"].items():
            if r["error_m"] is None:
                print(f"  {key:<20} tape {r['measured_m']:.3f} m  twin n/a  FAIL")
            else:
                print(f"  {key:<20} tape {r['measured_m']:.3f} m  twin {r['twin_m']:.3f} m  "
                      f"error {r['error_m'] * 100:+.1f} cm  {'ok' if r['ok'] else 'FAIL'}")
        failed += not res["ok"]
    print(f"{measured} room(s) measured, {failed} failing")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
