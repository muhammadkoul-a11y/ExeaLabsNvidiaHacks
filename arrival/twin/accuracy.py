"""Twin dimensions vs tape measurements (TWN-1a): scans/<room>/measurements.json -> per-room signed error.

measurements.json:
    {"counter_length_m": 3.25, "counter_height_m": 0.92, "sink_width_m": 0.6,
     "island_length_m": 1.2,            # optional extras, see EXTRAS
     "tolerance_m": 0.05,               # optional, default 0.05
     "measured_by": "name", "date": "2026-10-19"}

How each dimension is read off the compiled twin (all metres):
  counter_length_m  the longest contiguous counter run. The pieces are the scanned `counter` fixtures that are
                    not islands, not dining tables and not synthesized (a synthesized counter was never seen by
                    the scanner). A built-in stove or dishwasher no taller than a counter (+5 cm) counts as part
                    of the run, because the worktop is measured across it. Two pieces join when their long axes
                    are parallel (0.05 rad), their centres are within LATERAL of each other across the run and
                    the gap along it is at most ADJACENT_GAP (a 5 cm overlap is tolerated). The length is the
                    extent of the run along its axis. Pieces at another angle never join, so each leg of an
                    L or U is its own run and the longest leg is the one compared; a second counter elsewhere
                    in the room is another run and loses unless it is longer. Ties go to the lower fixture id.
  counter_height_m  the top surface (position z + size z / 2) of the longest counter piece in that run.
  sink_width_m      size[0] of a sink (its extent along the counter front). With several sinks, the lowest id
                    among those over the longest run's counters, else the lowest id.
Extras: island_length_m (longest island, long side), stove_width_m, fridge_width_m, dishwasher_width_m
(size[0] of the lowest-id fixture of that type; the fridge is the one the scan calls refrigerator).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from arrival.twin.compile import ADJACENT_GAP, COUNTER_HEIGHT, to_local

DEFAULT_TOLERANCE_M = 0.05
REQUIRED = ("counter_length_m", "counter_height_m", "sink_width_m")
EXTRAS = {"island_length_m": None, "stove_width_m": "stove", "fridge_width_m": "fridge_bottom_freezer",
          "dishwasher_width_m": "dishwasher"}
META = ("tolerance_m", "measured_by", "date", "authored", "note")
BUILT_IN = ("stove", "dishwasher")
BUILT_IN_HEIGHT_SLACK = 0.05
LATERAL = 0.25
YAW_TOL = 0.05
EPS = 1e-9  # float slack on the tolerance comparison


class MeasurementError(ValueError):
    """measurements.json is malformed, or the twin has no fixture to measure."""


def load_measurements(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise MeasurementError(f"{path}: expected a JSON object")
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise MeasurementError(f"{path}: missing {', '.join(missing)}")
    unknown = [k for k in data if k not in REQUIRED and k not in EXTRAS and k not in META]
    if unknown:
        raise MeasurementError(f"{path}: unknown key {', '.join(unknown)} (extras: {', '.join(EXTRAS)})")
    for k in (*REQUIRED, *(k for k in EXTRAS if k in data), "tolerance_m"):
        if k in data and (isinstance(data[k], bool) or not isinstance(data[k], (int, float))
                          or not math.isfinite(data[k]) or data[k] <= 0):
            raise MeasurementError(f"{path}: {k} must be a positive number of metres")
    return data


def _pieces(twin: dict) -> list[dict]:
    out = []
    for f in twin["fixtures"]:
        if f["type"] == "counter":
            if f.get("island") or f.get("default_config_name") == "dining" or f["source_category"] == "synthesized":
                continue
            out.append(f)
        elif f["type"] in BUILT_IN and f["size"][2] <= COUNTER_HEIGHT + BUILT_IN_HEIGHT_SLACK:
            out.append(f)
    return sorted(out, key=lambda f: f["id"])


def _joined(a: dict, b: dict) -> bool:
    d = (b["yaw"] - a["yaw"]) % math.pi
    if min(d, math.pi - d) > YAW_TOL:
        return False
    lx, ly = to_local(b["position"][:2], a)
    gap = abs(lx) - (a["size"][0] + b["size"][0]) / 2
    return abs(ly) <= LATERAL and -ADJACENT_GAP <= gap <= ADJACENT_GAP


def counter_runs(twin: dict) -> list[dict]:
    """Contiguous counter runs, longest first: {"length", "members", "counters"} (see the module docstring)."""
    pieces = _pieces(twin)
    parent = list(range(len(pieces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(pieces):
        for j in range(i + 1, len(pieces)):
            if _joined(a, pieces[j]):
                parent[find(j)] = find(i)
    groups: dict[int, list[dict]] = {}
    for i, p in enumerate(pieces):
        groups.setdefault(find(i), []).append(p)
    runs = []
    for members in groups.values():
        ref = members[0]
        spans = []
        for m in members:
            c = to_local(m["position"][:2], ref)[0]
            spans.append((c - m["size"][0] / 2, c + m["size"][0] / 2))
        counters = [m for m in members if m["type"] == "counter"]
        if counters:  # a run of a lone stove is not a counter
            runs.append({"length": max(s[1] for s in spans) - min(s[0] for s in spans),
                         "members": [m["id"] for m in members], "counters": counters})
    runs.sort(key=lambda r: (-r["length"], min(r["members"])))
    return runs


def _over(sink: dict, counter: dict) -> bool:
    """A sink centre lies over the counter's footprint."""
    lx, ly = to_local(sink["position"][:2], counter)
    return abs(lx) <= counter["size"][0] / 2 and abs(ly) <= counter["size"][1] / 2


def extract_dimensions(twin: dict) -> dict[str, float]:
    runs = counter_runs(twin)
    if not runs:
        raise MeasurementError(f"{twin['room_id']}: no scanned counter to measure")
    best = runs[0]
    top = sorted(best["counters"], key=lambda c: (-c["size"][0], c["id"]))[0]
    dims = {"counter_length_m": best["length"], "counter_height_m": top["position"][2] + top["size"][2] / 2}
    sinks = sorted((f for f in twin["fixtures"] if f["type"] == "sink"), key=lambda f: f["id"])
    if sinks:
        on_run = [s for s in sinks if any(_over(s, c) for c in best["counters"])]
        dims["sink_width_m"] = (on_run or sinks)[0]["size"][0]
    islands = [f for f in twin["fixtures"] if f.get("island")]
    if islands:
        dims["island_length_m"] = max(max(f["size"][:2]) for f in islands)
    for key, ftype in EXTRAS.items():
        found = sorted((f for f in twin["fixtures"] if f["type"] == ftype), key=lambda f: f["id"]) if ftype else []
        if found:
            dims[key] = found[0]["size"][0]
    return dims


def compare(twin: dict, measurements: dict) -> dict:
    """Per-dimension {"twin_m", "measured_m", "error_m" (twin minus tape, signed), "ok"} plus the overall verdict.
    A dimension the twin cannot supply (no sink, no island...) is reported with twin_m None and fails."""
    tol = float(measurements.get("tolerance_m", DEFAULT_TOLERANCE_M))
    dims = extract_dimensions(twin)
    rows = {}
    for key in (*REQUIRED, *(k for k in EXTRAS if k in measurements)):
        tape, got = float(measurements[key]), dims.get(key)
        err = None if got is None else got - tape
        rows[key] = {"twin_m": got, "measured_m": tape, "error_m": err,
                     "ok": err is not None and abs(err) <= tol + EPS}
    return {"room_id": twin["room_id"], "tolerance_m": tol, "dimensions": rows,
            "ok": all(r["ok"] for r in rows.values()),
            "tape_measured": not measurements.get("authored", False),
            "measured_by": measurements.get("measured_by"), "date": measurements.get("date")}
