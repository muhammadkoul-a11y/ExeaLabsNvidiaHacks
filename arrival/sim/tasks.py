"""Sim task families: counter_to_sink, counter_to_counter (when reachable), open_drawer, open_door, the close and
put-away families, wipe_counter, open_fridge, move_on_table (objects on a scanned table) and the appliance
families: open_oven, open_dishwasher, turn_on_faucet, load_dishwasher and unload_dishwasher."""

from __future__ import annotations

import math
import random

from arrival.catalog.objects import GRASPABLE, PICKABLE, TABLE_MOVABLE, WIPE_TOOLS
from arrival.sim.scene import has_door, has_drawers, is_table
from arrival.tasks.propose import LIGHTING, POSE_OFFSETS, WORDINGS, top_z
from arrival.sim.preflight import DISHWASHER_SIDE, check_task
from arrival.sim.reach import find_shift
from arrival.twin.compile import is_counter_host, to_local, to_world

TABLE_MOVE = 0.35  # how far along the table edge move_on_table carries the object (m)
TABLE_REACH = 0.44  # objects further than this from the robot's table edge are out of reach for a fixed base
# (a 0.745 m horizontal reach from a base 0.28 m out, with the object 0.175 m along the edge from the base); a
# candidate filter only, the IK pre-flight (arrival.sim.reach) decides
TARGET_GAP = 0.04  # clear space between the object at its target and any other object on the table (m)
BASE_CLEAR = 0.25  # the base column (r 0.12 + plate) must stay this far from furniture and walls
REACH = 0.78  # max horizontal eef reach from the arm base used when placing the base
SIM_FAMILIES = ("counter_to_sink", "counter_to_counter", "open_drawer", "open_door", "close_drawer", "close_door",
                "put_in_drawer", "wipe_counter", "open_fridge", "move_on_table", "open_oven", "open_dishwasher",
                "turn_on_faucet", "load_dishwasher", "unload_dishwasher")
HOLDOUT_WORDING = 3  # docs/07-eval.md: template 4 and the (0, +0.08) offset are for evaluation only
HOLDOUT_OFFSET = [0.0, 0.08]
EDGE_GAP = 0.02  # an offset start keeps the object's footprint this far inside the counter's edge
SIM_WORDINGS = {
    **WORDINGS,
    "open_drawer": ["Open the top drawer.", "Pull the top drawer of the {name} open.",
                    "Slide open the drawer under the counter.", "Pull out the top drawer on the {name}."],
    "open_door": ["Open the cabinet door.", "Swing the {name} cabinet door open.", "Pull the cabinet door open.",
                  "Swing open the lower cabinet door on the {name}."],
    "close_drawer": ["Close the drawer.", "Push the open drawer of the {name} shut.", "Slide the drawer closed.",
                     "Push the drawer on the {name} back in."],
    "close_door": ["Close the cabinet door.", "Push the {name} cabinet door shut.", "Swing the open door closed.",
                   "Shut the cabinet door on the {name}."],
    "put_in_drawer": ["Put the {name} in the open drawer.", "Place the {name} into the drawer.",
                      "Move the {name} from the counter into the drawer.", "Store the {name} in the drawer."],
    "wipe_counter": ["Wipe the counter with the {name}.", "Clean the counter top with the {name}.",
                     "Scrub the counter using the {name}.", "Use the {name} to wipe down the counter."],
    "open_fridge": ["Open the fridge.", "Pull the {name} door open.", "Swing the fridge door open.",
                    "Get the fridge open."],
    "move_on_table": ["Move the {name} further along the table.", "Pick up the {name} and set it down along the table.",
                      "Shift the {name} to another spot on the table.", "Reposition the {name} on the table."],
    "open_oven": ["Open the oven.", "Pull the oven door down.", "Open the oven door of the {name}.",
                  "Lower the oven door until it is flat."],
    "open_dishwasher": ["Open the dishwasher.", "Pull the dishwasher door open.", "Open the {name} door.",
                        "Fold the dishwasher door down."],
    "turn_on_faucet": ["Turn on the faucet.", "Lift the faucet lever to start the water.", "Turn the tap on.",
                       "Push the faucet handle up to run the water."],
    "load_dishwasher": ["Put the {name} in the dishwasher.", "Load the {name} into the dishwasher rack.",
                        "Move the {name} from the counter into the dishwasher.",
                        "Set the {name} on the pulled-out dishwasher rack."],
    "unload_dishwasher": ["Take the {name} out of the dishwasher.", "Unload the {name} onto the counter.",
                          "Move the {name} from the dishwasher rack to the counter.",
                          "Lift the {name} off the dishwasher rack and put it on the counter."],
}


def finish_task(t: dict) -> dict:
    """Add the handle/joint names the scripted policy and the success check read; shared by collect and eval."""
    if t["family"] in ("open_drawer", "close_drawer", "put_in_drawer"):
        t["handle"], t["joint"] = f'{t["fixture"]}_drawer0_handle', f'{t["fixture"]}_drawer0'
    elif t["family"] in ("open_door", "close_door", "open_fridge"):
        t["joint"] = f'{t["fixture"]}_door'
    elif t["family"] == "open_oven":
        t["joint"], t["handle"] = f'{t["fixture"]}_oven_door', f'{t["fixture"]}_oven_door_handle'
    elif t["family"] in ("open_dishwasher", "load_dishwasher", "unload_dishwasher"):
        t["joint"], t["handle"] = f'{t["fixture"]}_door', f'{t["fixture"]}_door_handle'
    elif t["family"] == "turn_on_faucet":
        t["joint"], t["handle"] = f'{t["fixture"]}_faucet', f'{t["fixture"]}_faucet_handle'
    return t


def table_move(table: dict, obj: dict, fixtures: list[dict], walls: list[dict], objects: list[dict] = ()) -> dict | None:
    """move_on_table variation for an object on a scanned table: which long edge the robot stands at (the one
    nearer the object), the target spot TABLE_MOVE along that edge (toward the middle of the table, else the other
    way), and the base position. None when the object is out of reach of that edge, or in neither direction is
    the target on the table, free of other objects (it would be set down on one) and reachable from a base that
    clears furniture and walls."""
    from arrival.twin.compile import _rect_distance

    sx, sy = table["size"][0], table["size"][1]
    lx, ly = to_local(obj["position"][:2], table)
    if abs(lx) > sx / 2 or abs(ly) > sy / 2:
        return None
    side = 1.0 if ly >= 0 else -1.0
    if sy / 2 - abs(ly) > TABLE_REACH:
        return None
    toward_middle = -TABLE_MOVE if lx > 0 else TABLE_MOVE
    half = max(obj["size"][:2]) / 2
    for step in (toward_middle, -toward_middle):  # the other way along the edge when the first spot is taken
        tx = lx + step
        if abs(tx) > sx / 2 - 0.12:
            continue
        target = to_world((tx, ly), table)
        if any(math.dist(target, o["position"][:2]) < half + max(o["size"][:2]) / 2 + TARGET_GAP
               for o in objects if o is not obj):  # an object set down on another one lands on it and misses
            continue
        base = to_world(((lx + tx) / 2, side * (sy / 2 + 0.28)), table)
        xs = [p[0] for w in walls for p in (w["start"], w["end"])]
        ys = [p[1] for w in walls for p in (w["start"], w["end"])]
        if xs and not (min(xs) + BASE_CLEAR <= base[0] <= max(xs) - BASE_CLEAR
                       and min(ys) + BASE_CLEAR <= base[1] <= max(ys) - BASE_CLEAR):
            continue
        if any(_rect_distance(base, f) < BASE_CLEAR for f in fixtures if f is not table and f["type"] != "sink"):
            continue
        return {"side": side, "target_xy": list(target), "base_xy": list(base)}
    return None


def _fx(twin: dict, fid: str) -> dict:
    return next(f for f in twin["fixtures"] if f["id"] == fid)


def _holder(o: dict, counters: list[dict]) -> dict | None:
    """The counter whose top the object stands on, judged in the counter's own frame (any wall, any yaw)."""
    for c in counters:
        lx, ly = to_local(o["position"], c)
        if abs(lx) < c["size"][0] / 2 and abs(ly) < c["size"][1] / 2:
            return c
    return None


def _footprint_half(twin: dict, object_id: str) -> float:
    o = next(o for o in twin["objects"] if o["id"] == object_id)
    return max(o["size"][0], o["size"][1]) / 2


def _offset_start(src: dict, start, off, half: float | None = None) -> list[float]:
    """`start` moved by a POSE_OFFSETS entry, which runs along the source counter (its local x, and y at half).
    With `half` (the object's footprint half-width) the move stops EDGE_GAP short of the counter's edge, so the
    object never stands half off the top (unload_dishwasher sets it down there); a start the twin already put
    closer to the edge is not moved in."""
    dx, dy = off[0], off[1] * 0.5
    if half is not None:
        local = to_local(start, src)

        def clamp(l0, d, size):
            lim = max(size / 2 - half - EDGE_GAP, abs(l0))
            return d if abs(l0 + d) <= lim else math.copysign(lim, l0 + d) - l0

        dx, dy = clamp(local[0], dx, src["size"][0]), clamp(local[1], dy, src["size"][1])
    co, si = math.cos(src.get("yaw", 0.0)), math.sin(src.get("yaw", 0.0))
    return [start[0] + co * dx - si * dy, start[1] + si * dx + co * dy]


def _start_variants(twin: dict, v: dict) -> list:
    """The object's xy for every offset make_tasks may apply (train and held-out); [None] for tasks without one."""
    if "start" not in v:
        return [None]
    src, half = _fx(twin, v["source"]), _footprint_half(twin, v["object_id"])
    return [_offset_start(src, v["start"], off, half) for off in POSE_OFFSETS]


def make_tasks(twin: dict, families: list[str], n: int, seed: int = 0, people: int = 0, holdout: bool = False,
               split: str = "train", use_generated_instructions: bool = False,
               mobile: bool = False, log=print, preflight: dict | None = None, arm: str = "procedural") -> list[dict]:
    """n task variations cycling through the requested families. Every candidate variation first passes the
    pre-flight (`arrival.sim.preflight.check_task`: stand spot inside the room and clear of walls and fixtures, the
    expert's waypoints within `arm`'s reach by an IK solve). When the stand rule's spot fails, the reach planner
    (`arrival.sim.reach.find_shift`) looks for a nearby spot that works and the variation keeps it as `base_shift`;
    one with none is logged with its reason and never becomes a task, and a family left with none is skipped. mobile=True means the base drives (MAN-3): a
    counter_to_counter target may then be any counter of the same run, not only one within 1 m of the object, and the
    pre-flight checks the drive's stand spot and path. When `preflight` is a dict it is filled with
    {"rejected": count, "shifted": count, "families": [skipped families], "reasons": [{"family", "reason"}, ...]}.
    holdout=False draws the training set: wording HOLDOUT_WORDING (families with 4 templates) and HOLDOUT_OFFSET
    are excluded. holdout=True draws only those; families with 3 templates keep all 3 either way.
    use_generated_instructions draws from the Nemotron cache (or the template fallback) for `split` instead."""
    rng = random.Random(seed)
    counters = [f for f in twin["fixtures"] if is_counter_host(f)]
    cabinets = [c for c in counters if not is_table(c)]  # tables have no drawers or doors
    sinks = [f for f in twin["fixtures"] if f["type"] == "sink"]
    pools: dict[str, list[dict]] = {}
    pre = preflight if preflight is not None else {}
    pre.update({"rejected": 0, "shifted": 0, "families": [], "reasons": []})
    for fam in families:
        pool = []
        if fam in ("counter_to_sink", "counter_to_counter"):
            for o in twin["objects"]:
                if o["catalog_id"] not in GRASPABLE:  # the laptop stays out of the pick pool
                    continue
                src = _holder(o, counters)
                if src is None:
                    continue
                ox = to_local(o["position"], src)[0]  # local x: distance along the source run, whatever wall it is on
                if fam == "counter_to_sink":  # a sink set into the same run, within reach along it
                    tgts = [s for s in sinks if abs(to_local(s["position"], src)[0] - ox) < 2 * REACH - 0.3
                            and abs(to_local(s["position"], src)[1]) < src["size"][1] / 2]
                    tgt_xy = [(s, s["position"][0], s["position"][1]) for s in tgts]
                else:
                    tgt_xy = []
                    for c in counters:
                        if c is src or math.cos(c.get("yaw", 0.0) - src.get("yaw", 0.0)) < 0.9:
                            continue  # the target must continue the source run
                        side = 1 if to_local(c["position"], src)[0] > ox else -1
                        tx, ty = to_world((-side * (c["size"][0] / 2 - 0.12), 0.0), c)  # the target's near end
                        if mobile or abs(to_local((tx, ty), src)[0] - ox) < 1.0:  # a fixed base sits between source and target
                            tgt_xy.append((c, tx, ty))
                for tgt, tx, ty in tgt_xy:
                    pool.append({"object_id": o["id"], "object_label": o["label"], "source": src["id"],
                                 "target": tgt["id"], "target_x": tx, "target_y": ty, "start": list(o["position"])})
        elif fam == "put_in_drawer":
            from arrival.sim.scene import drawer_x
            for o in twin["objects"]:
                if o["catalog_id"] != "mug":
                    continue
                src = _holder(o, cabinets)
                if src is None or not has_drawers(src, twin["fixtures"]) or (
                        abs(to_local(o["position"], src)[0] - drawer_x(src, twin["fixtures"])) > 0.6):
                    continue  # the base parks beside the open drawer, so the mug must be within reach of that spot
                pool.append({"object_id": o["id"], "object_label": o["label"], "source": src["id"], "fixture": src["id"],
                             "start": list(o["position"])})
        elif fam == "wipe_counter":
            for o in twin["objects"]:
                if o["catalog_id"] not in WIPE_TOOLS:
                    continue
                src = _holder(o, counters)
                if src is not None:
                    pool.append({"object_id": o["id"], "object_label": o["label"], "source": src["id"],
                                 "fixture": src["id"], "start": list(o["position"])})
        elif fam in ("open_drawer", "open_door", "close_drawer", "close_door"):
            has = has_door if fam.endswith("_door") else has_drawers  # a narrow cabinet is one door, no drawers
            pool = [{"fixture": c["id"], "name": c["id"].replace("_", " ")} for c in cabinets
                    if has(c, twin["fixtures"])]
        elif fam == "move_on_table":
            for c in (c for c in counters if is_table(c)):
                for o in twin["objects"]:
                    if o["catalog_id"] not in TABLE_MOVABLE:
                        continue
                    m = table_move(c, o, twin["fixtures"], twin["walls"], twin["objects"])
                    if m:
                        pool.append({"object_id": o["id"], "object_label": o["label"], "source": c["id"],
                                     "target": c["id"], "start": list(o["position"]), **m})
        elif fam == "open_oven":
            pool = [{"fixture": f["id"], "name": f["id"].replace("_", " ")} for f in twin["fixtures"] if f["type"] == "stove"]
        elif fam == "open_dishwasher":
            pool = [{"fixture": f["id"], "name": f["id"].replace("_", " ")} for f in twin["fixtures"]
                    if f["type"] == "dishwasher"]
        elif fam == "turn_on_faucet":  # the sink sits in a counter run; the lever is pushed from in front of that counter
            pool = [{"fixture": s["id"], "name": "faucet"} for s in sinks if _holder(s, counters) is not None]
        elif fam in ("load_dishwasher", "unload_dishwasher"):
            # the base parks beside the dishwasher in front of the counter, so the object (or its spot on the
            # counter for unloading) has to be within reach of that parking spot
            for d in (f for f in twin["fixtures"] if f["type"] == "dishwasher"):
                for o in twin["objects"]:
                    if o["catalog_id"] not in PICKABLE:
                        continue
                    src = _holder(o, counters)
                    if src is None:
                        continue
                    ox = to_local(o["position"], d)[0]  # along the dishwasher's run, from its centre
                    park = math.copysign(DISHWASHER_SIDE, ox)
                    if abs(ox - park) > 0.45 or abs(ox) < d["size"][0] / 2 + 0.1:
                        continue
                    pool.append({"object_id": o["id"], "object_label": o["label"], "source": src["id"],
                                 "fixture": d["id"], "start": list(o["position"])})
        elif fam == "open_fridge":
            pool = [{"fixture": f["id"], "name": "fridge"} for f in twin["fixtures"]
                    if f["type"].startswith("fridge")]
        kept = []
        for v in pool:
            why = None
            starts = _start_variants(twin, v)
            for xy in starts:  # every start pose the offsets can give must be reachable
                why = why or check_task(twin, {"family": fam, **v}, xy, mobile, arm)
            shift = find_shift(twin, {"family": fam, **v}, starts, mobile, arm) if why else None
            if shift is not None:
                pre["shifted"] += 1
                log(f"[sim] family {fam}: variation {v.get('object_id') or v['fixture']} stands {shift[0]:+.2f} m "
                    f"along and {shift[1]:+.2f} m out from the usual spot: {why}")
                v, why = {**v, "base_shift": list(shift)}, None
            if why is None:
                kept.append(v)
            else:
                pre["rejected"] += 1
                pre["reasons"].append({"family": fam, "reason": why})
                log(f"[sim] family {fam}: variation {v.get('object_id') or v['fixture']} rejected by pre-flight: {why}")
        if not kept:
            log(f"[sim] family {fam}: no reachable variation in this room for a {'mobile' if mobile else 'fixed'}-base arm; skipped")
            pre["families"].append(fam)
            continue
        pools[fam] = kept
    tasks = []
    fams = list(pools)
    for i in range(n if fams else 0):
        fam = fams[i % len(fams)]
        v = dict(rng.choice(pools[fam]))
        light = rng.choice(LIGHTING)
        name = v.get("object_label") or v.get("name")
        made_by = {}
        if use_generated_instructions:
            from arrival.tasks.instructions import generate_instructions
            gen = generate_instructions(fam, object_name=v.get("object_label"), fixture_name=v.get("name"))
            options = gen.get(split, gen.get("train", SIM_WORDINGS[fam]))
            instr = options[rng.randrange(len(options))]
            w = -1  # the generated lists have their own indexing: no template index, and no holdout wording
            made_by = {"instruction_source": gen.get("source")}
        else:
            words = SIM_WORDINGS[fam]
            w = rng.choice([k for k in range(len(words)) if len(words) <= HOLDOUT_WORDING or (k == HOLDOUT_WORDING) == holdout])
            instr = words[w].format(name=name)
        t = {"family": fam, "lighting": light, "wording_index": w, "people": people, "seed": seed * 1000 + i,
             "holdout": holdout, "instruction": instr, "split": split, **made_by, **v}
        if "start" in v:
            off = rng.choice([o for o in POSE_OFFSETS if (list(o) == HOLDOUT_OFFSET) == holdout])
            src = _fx(twin, v["source"])
            t["start"] = [*_offset_start(src, v["start"], off, _footprint_half(twin, v["object_id"])),
                          top_z(src) + 0.06]  # offsets run along the counter
        t["task_id"] = f"sim_{fam}_{i:04d}"
        tasks.append(t)
    return tasks
