"""Score the Nemotron task planner (Token Factory) against the keyword parser on instructions it has not seen.

    python scripts/planner_benchmark.py                 # writes docs/results/planner-benchmark-<UTC date>.json
    python scripts/planner_benchmark.py --per-family 4
    python scripts/planner_benchmark.py --dry-run       # no key, no network: the inventories and the call count
    python scripts/planner_benchmark.py --sets generated,indirect --blind path/to/blind.json

Five sets (report version 2; the v1 keys keep their v1 meaning):
  generated          wordings Nemotron Lightning writes through arrival.tasks.instructions (train and held-out
                     together), on the hero kitchen
  indirect           requests written by us that name the goal, not the action (labelled as ours in the report),
                     on the hero kitchen
  blind              requests from --blind, written by people who had not seen the planner prompt; scored over all
                     rows and again over the rows asking for one of the 9 families the prompt lists
  out_of_scope       requests written by us that match no family: there is no right answer, the report gives the
                     distribution of what each planner picks
  unlisted_families  requests written by us for the 6 sim families the planner prompt does not list; expected to fail
                     by construction (the keyword parser cannot return them at all)
Inventories: the hero kitchen's ids (hard-coded, as in v1) and tests/fixtures/roomplan/kitchen_real_01.json and
office.json built the way `arrival demo --from-roomplan` builds a twin (place_default_objects with seed 0 on a first
compile, then compile again), with the ids read the way arrival.sim.collect reads them.

A plan is right when its family is the one the instruction asks for, and, where the family moves an object, when the
object is the right one. The v1 keys (nemotron_ok, nemotron_right) count a reply _parse_plan rejected under nemotron
with the rule parser's answer; every v2 rate counts only plans from an accepted reply, and a rejected reply as wrong.
token_factory.complete is wrapped inside this script (production code is unchanged) to keep each raw reply. Every
call uses a throwaway cache, so each plan is a live call at temperature 0. Needs TOKEN_FACTORY_API_KEY with
NEMOTRON_SUPER_MODEL and NEMOTRON_LIGHTNING_MODEL; exits 2 without them, or while the blind file has placeholder rows.
Exits 1 when a case got no reply from Token Factory. An HTTP 429 or 5xx is retried up to 3 times, after 2, 4 and
8 s, each retry logged with its case and counted in rows[].retries; timeouts, other errors and rejected replies are
not retried. The report keeps no key or header: of each call it records the
model id, the temperature and the base URL only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OBJECTS, FIXTURES = ["mug_1", "bowl_1", "sponge_1"], ["counter_a", "sink_1", "counter_b", "fridge_1", "stove_1"]
ROOMPLAN_ROOMS = ("kitchen_real_01", "office")  # tests/fixtures/roomplan/<room>.json
DEFAULT_BLIND = REPO / "docs" / "results" / "planner-blind-requests.json"
SETS = ("generated", "indirect", "blind", "out_of_scope", "unlisted_families")
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
OUT_OF_SCOPE = [  # written by us; no sim family does any of these, so no plan is right: (request, inventory). None
    # names a family's verb, fixture or object (tests/test_planner_benchmark.py checks a keyword list)
    ("Turn off the kitchen light.", "hero_kitchen"),
    ("Set a timer for ten minutes.", "hero_kitchen"),
    ("Order more paper towels.", "hero_kitchen"),
    ("Play some music.", "hero_kitchen"),
    ("Tell me what time it is.", "hero_kitchen"),
    ("What's the weather going to be like tomorrow?", "hero_kitchen"),
    ("Add eggs to the shopping list.", "hero_kitchen"),
    ("Feed the cat.", "kitchen_real_01"),
    ("Remind me to call my sister at six.", "kitchen_real_01"),
    ("Read me today's news headlines.", "kitchen_real_01"),
    ("How many grams are in a pound?", "kitchen_real_01"),
    ("Arm the alarm system.", "kitchen_real_01"),
    ("Tell me a joke.", "kitchen_real_01"),
    ("Dim the lights a little.", "kitchen_real_01"),
    ("What's on my calendar this afternoon?", "office"),
    ("Translate good morning into Spanish.", "office"),
    ("Call me a taxi for eight o'clock.", "office"),
    ("Make it a bit warmer in here.", "office"),
    ("Make the music quieter.", "office"),
    ("Send a message saying I'm running late.", "office"),
]
# the room each unlisted family runs in: the one whose twin has the fixture the sim's task needs (stove, dishwasher,
# a table, the hero kitchen's sink in a counter), so a miss is about the family the prompt leaves out, not the room
UNLISTED_ROOM = {"open_oven": "kitchen_real_01", "open_dishwasher": "kitchen_real_01",
                 "load_dishwasher": "kitchen_real_01", "unload_dishwasher": "kitchen_real_01",
                 "turn_on_faucet": "hero_kitchen", "move_on_table": "office"}
UNLISTED = [  # written by us, for the sim families the planner prompt leaves out: (request, family, object). Objects
    # are ones arrival.sim.tasks.make_tasks uses for that family in that room (checked in the test)
    ("Open the oven so I can check on the bread.", "open_oven", None),
    ("Pull the oven door down.", "open_oven", None),
    ("Get the oven open, I want to slide the tray in.", "open_oven", None),
    ("Open the dishwasher, I want to see if it's done.", "open_dishwasher", None),
    ("Fold the dishwasher door down.", "open_dishwasher", None),
    ("Is the dishwasher finished? Open it up and let me look.", "open_dishwasher", None),
    ("Put the cup in the dishwasher.", "load_dishwasher", "cup_1"),
    ("The cup is dirty, load it into the dishwasher.", "load_dishwasher", "cup_1"),
    ("Put the apple in the dishwasher rack.", "load_dishwasher", "apple_1"),
    ("Rack the apple in the dishwasher with the other things.", "load_dishwasher", "apple_1"),
    ("Get the clean cup out of the dishwasher.", "unload_dishwasher", "cup_1"),
    ("Take the cup out of the dishwasher and put it on the counter.", "unload_dishwasher", "cup_1"),
    ("Unload the apple from the dishwasher.", "unload_dishwasher", "apple_1"),
    ("The apple is in the dishwasher, take it out.", "unload_dishwasher", "apple_1"),
    ("Turn on the faucet.", "turn_on_faucet", None),
    ("I need water running in the sink to rinse my hands.", "turn_on_faucet", None),
    ("Turn the tap on for me.", "turn_on_faucet", None),
    ("Move the book further along the table.", "move_on_table", "book_1"),
    ("Slide the book down the table a bit, it's in my way.", "move_on_table", "book_1"),
    ("Shift the book to another spot on the table.", "move_on_table", "book_1"),
]
LISTED = tuple(FAMILIES)  # the 9 families the planner's system prompt names; SIM_FAMILIES has 15
MOVES_OBJECT = {"counter_to_sink", "counter_to_counter", "put_in_drawer", "move_on_table", "load_dishwasher",
                "unload_dishwasher"}
RETRY_WAITS = (2, 4, 8)  # seconds before each retry of a 429 or 5xx: the first try and up to 3 retries
SIM_SAMPLES = 200  # tasks make_tasks draws per family to list the objects it uses; it picks from the reachable pool
# at random, so a draw this size leaves out none of the few objects a pool holds
REJECT_KINDS = (("not in the scene", "object_or_target_not_in_inventory"), ("no known family", "unknown_family"),
                ("substring not found", "no_json_object"), ("Expecting", "bad_json"), ("JSONDecodeError", "bad_json"))


def right(plan: dict, family: str, obj: str | None) -> bool:
    return plan.get("family") == family and (family not in MOVES_OBJECT or obj is None or plan.get("object") == obj)


def inventories() -> dict:
    """{name: {"objects", "fixtures", "source"}}: the hero kitchen's ids, then each RoomPlan fixture built into a twin
    the way the CLI's --from-roomplan path builds it, with the ids collect() hands the planner."""
    inv = {"hero_kitchen": {"objects": OBJECTS, "fixtures": FIXTURES, "source": "hard-coded in this script (as v1)"}}
    for room in ROOMPLAN_ROOMS:
        twin = roomplan_twin(room)
        inv[room] = {"objects": [o["id"] for o in twin.get("objects", [])],
                     "fixtures": [f["id"] for f in twin.get("fixtures", [])],
                     "fixture_types": {f["id"]: f["type"] for f in twin.get("fixtures", [])},  # for reading only
                     "source": f"tests/fixtures/roomplan/{room}.json, built as `arrival demo --from-roomplan` "
                               "builds it (seed 0)"}
    return inv


def roomplan_twin(room: str, seed: int = 0) -> dict:
    """The twin arrival.cli.run_demo builds with --from-roomplan (its seed defaults to 0): default objects placed on a
    first compile, then the bundle compiled again with them."""
    from arrival.scan import roomplan
    from arrival.twin.compile import compile_twin, place_default_objects

    bundle = roomplan.load(REPO / "tests" / "fixtures" / "roomplan" / f"{room}.json", room)
    bundle["objects"] = place_default_objects(compile_twin(bundle), seed=seed)
    return compile_twin(bundle)


def room_twins() -> dict:
    """The twins behind the inventories: the hero kitchen from scans/, the RoomPlan rooms from roomplan_twin."""
    from arrival.scan.bundle import load_room
    from arrival.twin.compile import compile_twin

    return {"hero_kitchen": compile_twin(load_room("hero_kitchen", REPO / "scans")),
            **{room: roomplan_twin(room) for room in ROOMPLAN_ROOMS}}


def sim_executable(twin: dict, family: str, obj: str | None, cache: dict | None = None) -> bool | None:
    """True when arrival.sim.tasks.make_tasks makes a task of this family in this twin, with this object when one is
    named: whether the request is one the sim could run, not whether the planner understood it. None without MuJoCo
    (the reach check needs it). `cache` maps (id(twin), family) to the objects used; keep the twins alive with it."""
    from arrival.sim.tasks import make_tasks

    key = (id(twin), family)
    if cache is None or key not in cache:
        try:
            made = {t.get("object_id") for t in make_tasks(twin, [family], SIM_SAMPLES, seed=0, log=lambda *_: None)}
        except ImportError:
            return None
        if cache is not None:
            cache[key] = made
    made = cache[key] if cache is not None else made
    return bool(made) and (obj is None or obj in made)


def load_blind(path: Path, inv: dict) -> list[dict]:
    """The blind requests, checked against the schema in docs/results/planner-blind-requests.json. Raises ValueError."""
    from arrival.sim.tasks import SIM_FAMILIES

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    reqs = data.get("requests") if isinstance(data, dict) else None
    if not isinstance(reqs, list):
        raise ValueError(f"{path}: expected an object with a 'requests' list")
    for i, r in enumerate(reqs):
        where = f"{path}: requests[{i}]"
        if not isinstance(r, dict) or not isinstance(r.get("text"), str) or not r["text"].strip():
            raise ValueError(f"{where}: needs a non-empty 'text'")
        if r.get("inventory") not in inv:
            raise ValueError(f"{where}: 'inventory' must be one of {sorted(inv)}")
        if r.get("family") not in SIM_FAMILIES:
            raise ValueError(f"{where}: 'family' must be one of arrival.sim.tasks.SIM_FAMILIES")
        if r.get("object") is not None and r["object"] not in inv[r["inventory"]]["objects"]:
            raise ValueError(f"{where}: 'object' {r['object']!r} is not in {r['inventory']}'s objects")
    return reqs


def reject_reason(raw: str, objects: list[str], fixtures: list[str]) -> str | None:
    """Why VLMPlanner._parse_plan refuses this reply, or None when it accepts it."""
    from arrival.sim.policy import VLMPlanner

    try:
        VLMPlanner._parse_plan(raw, objects, fixtures)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


def reject_kind(reason: str) -> str:
    return next((kind for needle, kind in REJECT_KINDS if needle in reason), "other")


def call_error_kind(e: Exception) -> str:
    """Why a call got no reply, kept apart from reject_kind (a reply _parse_plan refused): "timeout" (the 60 s in
    token_factory.complete, while connecting or reading), "http_<status>", else "other". A reply whose message
    content is null is "no_content", set by the wrapper in main, which checks the reply rather than an exception."""
    import urllib.error

    if isinstance(e, TimeoutError) or isinstance(getattr(e, "reason", None), TimeoutError):
        return "timeout"
    if isinstance(e, urllib.error.HTTPError):
        return f"http_{e.code}"
    return "other"


def retryable(e: Exception) -> bool:
    """HTTP 429 or 5xx: Token Factory was busy, not the request wrong. Timeouts, other errors and replies
    _parse_plan rejects are not retried."""
    import urllib.error

    return isinstance(e, urllib.error.HTTPError) and (e.code == 429 or 500 <= e.code < 600)


def rate(k: int, n: int) -> dict | None:
    from arrival.eval.harness import wilson

    return {"k": k, "n": n, "rate": round(k / n, 4), "wilson_95": wilson(k, n)} if n else None


def score(rows: list[dict]) -> dict:
    """v2 fields for one set: rates with Wilson 95% intervals, confusion counts, rejections, latency."""
    live = [r for r in rows if r["live_call"]]
    scored = [r for r in rows if r["family"] is not None]
    with_obj = [r for r in scored if r["family"] in MOVES_OBJECT and r["object"] is not None]
    confusion = {"nemotron": {}, "rule": {}}
    for r in rows:
        want = r["family"] or "out_of_scope"
        got = r["nemotron"]["family"] if r["reply_accepted"] else ("rejected" if r["live_call"] else "no_reply")
        for who, pick in (("nemotron", got), ("rule", r["rule"]["family"])):
            row = confusion[who].setdefault(want, {})
            row[pick] = row.get(pick, 0) + 1
    secs = [r["nemotron_seconds"] for r in live]
    out = {
        "live_calls": len(live),
        "rejected": rate(sum(not r["reply_accepted"] for r in live), len(live)),
        "reject_reasons": dict(Counter(reject_kind(r["reject_reason"]) for r in live if r["reject_reason"])),
        "call_errors": dict(Counter(r["call_error_kind"] for r in rows if r["call_error_kind"])),  # no reply at all
        "family": None, "object": None, "right": None,
        "confusion": confusion,
        "median_seconds": round(statistics.median(secs), 3) if secs else None,
        "sim_not_executable": sum(r["sim_executable"] is False for r in rows),
    }
    if scored:
        for key, rs, field in (("family", scored, "family_ok"), ("object", with_obj, "object_ok"),
                               ("right", scored, "plan_ok")):
            out[key] = {"nemotron": rate(sum(bool(r[f"nemotron_{field}"]) for r in rs), len(rs)),
                        "rule": rate(sum(bool(r[f"rule_{field}"]) for r in rs), len(rs))}
    return out


def static_cases(sets: list[str], blind: list[dict]) -> list[tuple]:
    """(set, text, family, object, inventory) for every set but generated."""
    cases = []
    if "indirect" in sets:
        cases += [("indirect", w, fam, obj, "hero_kitchen") for w, fam, obj in INDIRECT]
    if "blind" in sets:
        cases += [("blind", r["text"], r["family"], r.get("object"), r["inventory"]) for r in blind]
    if "out_of_scope" in sets:
        cases += [("out_of_scope", w, None, None, room) for w, room in OUT_OF_SCOPE]
    if "unlisted_families" in sets:
        cases += [("unlisted_families", w, fam, obj, UNLISTED_ROOM[fam]) for w, fam, obj in UNLISTED]
    return cases


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-family", type=int, default=6, help="generated wordings asked for per family (default 6)")
    ap.add_argument("--out", type=Path, help="report path (default docs/results/planner-benchmark-<UTC date>.json)")
    ap.add_argument("--blind", type=Path, default=DEFAULT_BLIND,
                    help="blind requests (default docs/results/planner-blind-requests.json)")
    ap.add_argument("--sets", default=",".join(SETS),
                    help=f"comma-separated sets to run (default all: {','.join(SETS)})")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the inventories and how many live calls the run would make; no key, no network")
    a = ap.parse_args(argv)
    if a.per_family < 1:
        print("--per-family must be >= 1", file=sys.stderr)
        return 2
    sets = [s.strip() for s in a.sets.split(",") if s.strip()]
    if not sets or any(s not in SETS for s in sets):
        print(f"--sets takes a comma-separated list of {', '.join(SETS)}", file=sys.stderr)
        return 2

    import arrival.models.token_factory as tf
    from arrival import config as settings
    from arrival.models.token_factory import config
    from arrival.sim.policy import VLMPlanner, rule_plan
    from arrival.tasks.instructions import generate_instructions

    inv = inventories()
    blind = []
    if "blind" in sets:
        try:
            blind = load_blind(a.blind, inv)
        except (OSError, ValueError) as e:
            print(f"blind set: {e}", file=sys.stderr)
            return 2
    placeholders = sum(bool(r.get("placeholder")) for r in blind)
    cases = static_cases(sets, blind)
    twins, made = room_twins(), {}
    blind_not_sim = sum(sim_executable(twins[r["inventory"]], r["family"], r.get("object"), made) is False
                        for r in blind)

    if a.dry_run:
        for name, i in inv.items():
            fixtures = [f"{f} ({i['fixture_types'][f]})" if "fixture_types" in i else f for f in i["fixtures"]]
            print(f"{name:16} objects {i['objects']}\n{'':16} fixtures {fixtures}")
        per_set = Counter(c[0] for c in cases)
        n_gen = len(FAMILIES) * a.per_family if "generated" in sets else 0
        for s in sets:
            extra = sum(r["family"] not in LISTED for r in blind) if s == "blind" else 0
            print(f"{s:18} {'at most ' + str(n_gen) if s == 'generated' else per_set[s]} planner calls"
                  + (f" ({extra} ask for a family the prompt does not list)" if extra else "")
                  + (f" ({blind_not_sim} the sim could not run: sim_executable false)"
                     if s == "blind" and blind_not_sim else ""))
        lightning = len(FAMILIES) if "generated" in sets else 0
        print(f"live calls: at most {n_gen + len(cases)} to the planner (Super) "
              f"and {lightning} for wordings (Lightning)")
        if placeholders:
            print(f"{a.blind}: {placeholders} placeholder rows; a live run refuses to start until they are replaced",
                  file=sys.stderr)
        return 0

    if not (settings.get("TOKEN_FACTORY_API_KEY") and config()["models"]["super"] and config()["models"]["lightning"]):
        print("set TOKEN_FACTORY_API_KEY, NEMOTRON_SUPER_MODEL and NEMOTRON_LIGHTNING_MODEL (see .env.example)",
              file=sys.stderr)
        return 2
    if placeholders:
        print(f"{a.blind} still has {placeholders} placeholder rows: replace them, or leave blind out with --sets",
              file=sys.stderr)
        return 2

    calls: list[dict] = []
    real_complete = tf.complete
    case = ""  # what the next call is for, for the retry log

    def recording(*args, **kw):  # the planner and generate_instructions import complete per call, so they get this
        rec = {"role": kw.get("role", args[1] if len(args) > 1 else "super"),
               "messages": kw.get("messages", args[0] if args else None),
               "model": kw.get("model") or config()["models"].get(kw.get("role", "super")),
               "temperature": kw.get("temperature")}
        calls.append(rec)
        rec["retries"] = 0
        while True:
            try:
                reply = real_complete(*args, **kw)
                break
            except Exception as e:
                if retryable(e) and rec["retries"] < len(RETRY_WAITS):
                    wait = RETRY_WAITS[rec["retries"]]
                    rec["retries"] += 1
                    print(f"retry {rec['retries']}/{len(RETRY_WAITS)} in {wait} s after {call_error_kind(e)}: {case}",
                          file=sys.stderr)
                    time.sleep(wait)
                    continue
                rec["error"] = settings.redact(f"{type(e).__name__}: {e}")
                rec["error_kind"] = call_error_kind(e)
                raise
        if reply is None:  # complete returns message.content as sent: null when no answer text came back
            rec["error"], rec["error_kind"] = "the reply's message content was null", "no_content"
        else:
            rec["reply"] = reply
        return reply

    now = datetime.now(timezone.utc)
    rows: list[dict] = []
    wording_calls = 0
    tf.complete = recording
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            gen_cases = []
            if "generated" in sets:
                for fam, (obj, obj_label, fx_label) in FAMILIES.items():
                    case = f"wordings for {fam}"
                    gen = generate_instructions(fam, obj_label, fx_label, n=a.per_family, cache_path=d / "instr.json")
                    if gen.get("source") != "token_factory":
                        print(f"FAIL wordings for {fam}: source={gen.get('source')}", file=sys.stderr)
                        return 1
                    gen_cases += [("generated", w, fam, obj, "hero_kitchen") for w in gen["train"] + gen["eval"]]
                wording_calls = sum(c["role"] == "lightning" for c in calls)

            for i, (kind, text, fam, obj, room) in enumerate(gen_cases + cases):
                objects, fixtures = inv[room]["objects"], inv[room]["fixtures"]
                # one cache file each: no cache hits
                planner = VLMPlanner(log=lambda *_: None, cache_path=d / f"plan{i}.json")
                start = len(calls)
                case = f"{kind} {text!r}"
                t = time.perf_counter()
                plan = planner.plan(text, objects, fixtures)
                secs = time.perf_counter() - t
                mine = [c for c in calls[start:] if c["role"] == "super"]
                raw = mine[0].get("reply") if mine else None
                reason = reject_reason(raw, objects, fixtures) if raw is not None else None
                accepted = raw is not None and reason is None
                if accepted != (planner.source == "token_factory"):
                    print(f"accounting mismatch on {text!r}: source={planner.source}, reason={reason}",
                          file=sys.stderr)
                rule = rule_plan(text, objects, fixtures)
                scored = fam is not None
                moves = scored and fam in MOVES_OBJECT and obj is not None
                rows.append({
                    "set": kind, "instruction": text, "family": fam, "object": obj, "nemotron": plan,
                    "nemotron_source": planner.source, "nemotron_ok": right(plan, fam, obj) if scored else None,
                    "nemotron_seconds": round(secs, 3), "rule": rule,
                    "rule_ok": right(rule, fam, obj) if scored else None,
                    # v2: the inventory, the raw reply and what _parse_plan made of it; nemotron_* here count only
                    # plans from an accepted reply
                    "inventory": room, "live_call": raw is not None, "calls": len(mine),
                    "raw_reply": None if raw is None else settings.redact(raw), "reply_accepted": accepted,
                    "reject_reason": None if reason is None else settings.redact(reason),
                    "call_error": mine[0].get("error") if mine else "no call reached Token Factory",
                    "call_error_kind": mine[0].get("error_kind") if mine else "no_call",
                    "retries": sum(c["retries"] for c in mine),  # 429/5xx retries; nemotron_seconds includes the waits
                    "nemotron_family_ok": (accepted and plan["family"] == fam) if scored else None,
                    "nemotron_object_ok": (accepted and plan.get("object") == obj) if moves else None,
                    "nemotron_plan_ok": (accepted and right(plan, fam, obj)) if scored else None,
                    "rule_family_ok": rule["family"] == fam if scored else None,
                    "rule_object_ok": rule.get("object") == obj if moves else None,
                    "rule_plan_ok": right(rule, fam, obj) if scored else None,
                    # the sim can make this task in this room (with this object): the blind set is a language test
                    # and keeps rows where it cannot
                    "sim_executable": sim_executable(twins[room], fam, obj, made) if scored else None,
                })
    finally:
        tf.complete = real_complete

    live = [r for r in rows if r["nemotron_source"] == "token_factory"]
    planner_calls = [c for c in calls if c["role"] == "super"]
    summary = {}
    for kind in sets:
        rs = [r for r in rows if r["set"] == kind]
        summary[kind] = {"n": len(rs),
                         "nemotron_right": sum(bool(r["nemotron_ok"]) for r in rs) if kind != "out_of_scope" else None,
                         "rule_right": sum(bool(r["rule_ok"]) for r in rs) if kind != "out_of_scope" else None,
                         "nemotron_live_calls": sum(r["nemotron_source"] == "token_factory" for r in rs),
                         **score(rs)}
        if kind == "blind":
            listed = [r for r in rs if r["family"] in LISTED]
            summary[kind]["listed_families"] = {"n": len(listed), **{k: v for k, v in score(listed).items()
                                                                     if k in ("family", "object", "right")}}
    secs = sorted(r["nemotron_seconds"] for r in live)
    report = {"version": 2, "run_at": now.isoformat(timespec="seconds"), "planner_model": config()["models"]["super"],
              "wording_model": config()["models"]["lightning"], "objects": OBJECTS, "fixtures": FIXTURES,
              "inventories": inv,
              "planner_system_prompt": planner_calls[0]["messages"][0]["content"] if planner_calls else None,
              # model id, temperature and base URL, nothing else: no key, no header
              "planner_params": {"model": planner_calls[0]["model"], "temperature": planner_calls[0]["temperature"],
                                 "base_url": settings.redact(config()["base_url"])} if planner_calls else None,
              "live_calls": {"planner": sum("reply" in c for c in planner_calls), "wordings": wording_calls},
              "sets": {
                  "generated": "wordings Nemotron Lightning wrote for the 9 families the planner prompt lists; "
                               "hero kitchen",
                  "indirect": "written by us, naming the goal rather than the action; hero kitchen",
                  "blind": f"from {a.blind.name}, written by people who had not seen the planner prompt but were "
                           "given all 15 sim families in plain words; scored over all rows, and under "
                           "listed_families over the rows asking for one of the 9 families the prompt lists. "
                           "A language test: rows the sim could not run are kept and counted in "
                           "sim_not_executable (rows[].sim_executable)",
                  "out_of_scope": "written by us; no family fits, so there is no right answer: the confusion counts "
                                  "under out_of_scope are what each planner picked",
                  "unlisted_families": "written by us for open_oven, open_dishwasher, load_dishwasher, "
                                       "unload_dishwasher, turn_on_faucet and move_on_table, which the planner prompt "
                                       "does not list: expected to fail by construction, and the keyword parser "
                                       "cannot return these families at all. Each runs in the room whose twin has "
                                       "the fixture its sim task needs (rows[].inventory): oven and dishwasher in "
                                       "kitchen_real_01, faucet in hero_kitchen, table in office",
              },
              "summary": summary, "median_seconds": secs[len(secs) // 2] if secs else None,
              "note": "The indirect, out_of_scope and unlisted_families sets are written by us, not sampled from "
                      "users. A reply _parse_plan rejected falls back to the rule parser: the v1 fields (nemotron_ok, "
                      "nemotron_right) count it under nemotron with source rule_based, every v2 rate (family, "
                      "object, right) counts it as wrong. Rates carry Wilson 95% intervals (arrival.eval.harness).",
              "rows": rows}
    for kind, s in summary.items():
        line = f"{kind:18} n={s['n']:3}  live {s['live_calls']}  rejected {s['rejected']['k'] if s['rejected'] else 0}"
        if s["call_errors"]:
            line += f"  no reply {s['call_errors']}"
        if s["family"]:
            nf, rf = s["family"]["nemotron"], s["family"]["rule"]
            line += (f"  family: nemotron {nf['k']}/{nf['n']} {nf['wilson_95']}"
                     f"  rule {rf['k']}/{rf['n']} {rf['wilson_95']}")
        else:
            line += f"  nemotron picked {s['confusion']['nemotron'].get('out_of_scope', {})}"
        print(line)
        if "listed_families" in s and s["listed_families"]["family"]:
            nf, rf = s["listed_families"]["family"]["nemotron"], s["listed_families"]["family"]["rule"]
            print(f"{'':18} listed families only: n={nf['n']}  nemotron {nf['k']}/{nf['n']} {nf['wilson_95']}"
                  f"  rule {rf['k']}/{rf['n']} {rf['wilson_95']}")
    out = a.out or REPO / "docs" / "results" / f"planner-benchmark-{now:%Y-%m-%d}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0 if all(r["live_call"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
