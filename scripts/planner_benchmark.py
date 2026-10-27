"""Score the Nemotron task planner (Token Factory) against the keyword parser on instructions it has not seen.

    python scripts/planner_benchmark.py                 # writes docs/results/planner-benchmark-<UTC date>.json
    python scripts/planner_benchmark.py --per-family 4

Two sets, both on the hero kitchen's inventory (the ids `sim` passes the planner):
  generated  wordings Nemotron Lightning writes through arrival.tasks.instructions (train and held-out together)
  indirect   requests written by us that name the goal, not the action (labelled as ours in the report)
A plan is right when its family is the one the instruction asks for, and, where the family moves an object, when the
object is the right one. Every call uses a throwaway cache, so each plan is a live call. Needs TOKEN_FACTORY_API_KEY
with NEMOTRON_SUPER_MODEL and NEMOTRON_LIGHTNING_MODEL; exits 2 without them.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OBJECTS, FIXTURES = ["mug_1", "bowl_1", "sponge_1"], ["counter_a", "sink_1", "counter_b", "fridge_1", "stove_1"]
# family -> (object id the plan must name or None, object label, fixture label) for the generated wordings
FAMILIES = {
    "counter_to_sink": ("mug_1", "mug", None),
    "counter_to_counter": ("bowl_1", "bowl", None),
    "put_in_drawer": ("mug_1", "mug", None),
    "open_drawer": (None, None, "drawer"),
    "close_drawer": (None, None, "drawer"),
    "open_door": (None, None, "cabinet door"),
    "close_door": (None, None, "cabinet door"),
    "wipe_counter": (None, None, "counter"),
    "open_fridge": (None, None, "fridge"),
}
INDIRECT = [  # written by us; the family and object each one asks for
    ("The mug is dirty, get it ready to be washed.", "counter_to_sink", "mug_1"),
    ("I left my mug out, can you deal with it so I can rinse it later?", "counter_to_sink", "mug_1"),
    ("I need this counter clear for cooking; the bowl can go on the other side.", "counter_to_counter", "bowl_1"),
    ("Make some room by shifting the bowl down the counter.", "counter_to_counter", "bowl_1"),
    ("Put the mug away somewhere I won't see it, in the drawer.", "put_in_drawer", "mug_1"),
    ("Stash the mug where the cutlery lives.", "put_in_drawer", "mug_1"),
    ("I need a spoon, get the cutlery out where I can reach it.", "open_drawer", None),
    ("Someone left the cutlery drawer hanging out, fix that.", "close_drawer", None),
    ("I'll bump my hip on that drawer, push it back in.", "close_drawer", None),
    ("I want to grab a plate from the cupboard under the counter.", "open_door", None),
    ("The cupboard is gaping open, shut it.", "close_door", None),
    ("There are crumbs everywhere, clean up the counter.", "wipe_counter", None),
    ("The counter is sticky, give it a once-over with the sponge.", "wipe_counter", None),
    ("I want to see what we have for dinner in the fridge.", "open_fridge", None),
    ("Let me check if we're out of milk.", "open_fridge", None),
]
MOVES_OBJECT = {"counter_to_sink", "counter_to_counter", "put_in_drawer"}


def right(plan: dict, family: str, obj: str | None) -> bool:
    return plan.get("family") == family and (family not in MOVES_OBJECT or obj is None or plan.get("object") == obj)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-family", type=int, default=6, help="generated wordings asked for per family (default 6)")
    ap.add_argument("--out", type=Path, help="report path (default docs/results/planner-benchmark-<UTC date>.json)")
    a = ap.parse_args(argv)
    if a.per_family < 1:
        print("--per-family must be >= 1", file=sys.stderr)
        return 2

    from arrival import config as settings
    from arrival.models.token_factory import config
    from arrival.sim.policy import VLMPlanner, rule_plan
    from arrival.tasks.instructions import generate_instructions

    if not (settings.get("TOKEN_FACTORY_API_KEY") and config()["models"]["super"] and config()["models"]["lightning"]):
        print("set TOKEN_FACTORY_API_KEY, NEMOTRON_SUPER_MODEL and NEMOTRON_LIGHTNING_MODEL (see .env.example)",
              file=sys.stderr)
        return 2

    now = datetime.now(timezone.utc)
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cases = []
        for fam, (obj, obj_label, fx_label) in FAMILIES.items():
            gen = generate_instructions(fam, obj_label, fx_label, n=a.per_family, cache_path=d / "instr.json")
            if gen.get("source") != "token_factory":
                print(f"FAIL wordings for {fam}: source={gen.get('source')}", file=sys.stderr)
                return 1
            cases += [("generated", w, fam, obj) for w in gen["train"] + gen["eval"]]
        cases += [("indirect", w, fam, obj) for w, fam, obj in INDIRECT]

        for i, (kind, text, fam, obj) in enumerate(cases):
            planner = VLMPlanner(log=lambda *_: None, cache_path=d / f"plan{i}.json")  # one file each: no cache hits
            t = time.perf_counter()
            plan = planner.plan(text, OBJECTS, FIXTURES)
            secs = time.perf_counter() - t
            rule = rule_plan(text, OBJECTS, FIXTURES)
            rows.append({"set": kind, "instruction": text, "family": fam, "object": obj, "nemotron": plan,
                         "nemotron_source": planner.source, "nemotron_ok": right(plan, fam, obj),
                         "nemotron_seconds": round(secs, 3), "rule": rule, "rule_ok": right(rule, fam, obj)})

    live = [r for r in rows if r["nemotron_source"] == "token_factory"]
    summary = {}
    for kind in ("generated", "indirect"):
        rs = [r for r in rows if r["set"] == kind]
        summary[kind] = {"n": len(rs), "nemotron_right": sum(r["nemotron_ok"] for r in rs),
                         "rule_right": sum(r["rule_ok"] for r in rs),
                         "nemotron_live_calls": sum(r["nemotron_source"] == "token_factory" for r in rs)}
    secs = sorted(r["nemotron_seconds"] for r in live)
    report = {"run_at": now.isoformat(timespec="seconds"), "planner_model": config()["models"]["super"],
              "wording_model": config()["models"]["lightning"], "objects": OBJECTS, "fixtures": FIXTURES,
              "summary": summary, "median_seconds": secs[len(secs) // 2] if secs else None,
              "note": "The indirect set is written by us, not sampled from users. A plan the planner could not parse "
                      "falls back to the rule parser and is counted under nemotron with source rule_based.",
              "rows": rows}
    for kind, s in summary.items():
        print(f"{kind:9} n={s['n']:3}  nemotron {s['nemotron_right']}/{s['n']}  rule parser {s['rule_right']}/{s['n']}"
              f"  (live calls {s['nemotron_live_calls']})")
    out = a.out or REPO / "docs" / "results" / f"planner-benchmark-{now:%Y-%m-%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0 if len(live) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
