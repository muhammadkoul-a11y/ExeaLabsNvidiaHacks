"""Where the base stands for a task, and a pre-flight on that spot, before any scene model is built.

`base_pose` is the one place the stand rule lives (SimEnv uses it too), written in the fixture's own frame so it holds
on any wall. `check_task` then rejects a task whose stand spot is outside the room, too close to a wall or a fixture,
or from which the arm cannot reach the expert's waypoints: an IK solve on an arm-only model (`arrival.sim.reach`),
joint limits included, collisions not. A task that passes can still miss in physics (that is counted as a dropped
attempt, not a skip), but a layout the base cannot stand in or reach from never costs sim time or a traceback.
"""

from __future__ import annotations

import math

import numpy as np

from arrival.sim.scene import DRAWER_W, door_hinge_x, door_w, drawer_x, front_stand
from arrival.twin.compile import is_counter_host, to_local, to_world

BASE_CLEAR = 0.15  # the base column is 0.12 m in radius
PART_CLEAR = 0.13  # base centre to an open door or drawer: the column's radius plus 1 cm (the put_in_drawer stand
# rule leaves 0.148 m to the pulled-out handle)
DRAWER_TRAVEL = {"put_in_drawer": 0.14, "open_drawer": 0.325, "close_drawer": 0.30}  # how far each has it pulled
DRAWER_HANDLE = 0.062  # the bar's outer face past the closed carcass front (front 2 cm in, 7 cm standoff, 12 mm bar)
WALL_CLEAR = 0.15  # base centre to the wall's face
OPEN_DOOR_WALL_OUT = 0.12  # open_door: the base stands this much further out when the run's hinge end is drawn at least
# OPEN_DOOR_INSIDE into a wall and the bar ends the pull within OPEN_DOOR_ROOM of that wall. The pull ends with the outer
# face turned to the wall and the hand between them, and from the usual spot link7 grazes it (hero counter_b, 0.01 m
# in: 22 steps, 0.9 mm; 15 steps at 0.06 out). In fuzzed rooms no door that opened from a run flush with a wall
# grazed it, and there the further spot cost one door its last few degrees (1.067 to 1.024 rad)
OPEN_DOOR_INSIDE, OPEN_DOOR_ROOM = 0.005, 0.06
DRAWER_OPEN_PUT = 0.14  # start state for put_in_drawer: wide enough for a mug on the exposed tray, and the base
# can then park straight in front without the pulled-out front reaching its column
FRIDGE_BASE_ALONG = 0.1  # base centre past the door's free edge, along the front
FRIDGE_BASE_OUT = 0.75  # base centre out from the fridge front: nearer, the handle's arc folds the elbow onto its stop
FRIDGE_BASE_YAW = 0.0  # degrees the base turns toward the hinge from facing the front
FLAP_MARGIN = 0.06  # open_oven / open_dishwasher: base column edge to where the open door's top edge lands
OVEN_ALONG = 0.25  # open_oven: base centre this far along the run from the bar, +x side first. Straight in front,
# the hand pointing into the panel holds joint 5 on its stop for 71 steps; from 0.20 to 0.30 either side no joint nears one
FAUCET_STANDOFF = 0.34  # base column to counter front for turn_on_faucet (0.28 elsewhere): the lever is at the back
# of the counter above shoulder height, and a horizontal hand reaches it with less joint strain from further out
ARM_STANDOFF_TRIM = {"procedural": 0.0, "menagerie": 0.02}  # the real hand's closed fingertips reach 8 mm past the eef,
# the procedural fingers 23 mm: the faucet push stands 2 cm closer with it (0.34 misses the lever, 0.32 turns it to 1.10 rad)
DISHWASHER_SIDE = 0.55  # load/unload_dishwasher: the base parks this far beside the dishwasher centre, clear of its door
RACK_FRONT = 0.15  # the rack spot an object is set into (or taken from) lies this far out from the dishwasher front
FLAP_DOOR_BELOW = {"open_oven": 0.24, "open_dishwasher": 0.18}  # fixture height minus the door's height, per family
PICK_PLACE = ("counter_to_sink", "counter_to_counter")
MOBILE_STEP = 0.3  # a mobile base parks this far along the source run toward the target, as counter_to_sink parks
# between the object and the sink; it then drives to `drive_spot` once the object is held
CLOSE_DOOR_WALL_ALONG = 0.06  # close_door: the base stands this much nearer the hinge when the run ends flush against a
# wall (hinge within CLOSE_DOOR_FLUSH of it) and the open door's handle starts within CLOSE_DOOR_ROOM of that wall. The
# side push then starts with the hand squeezed between the panel and the wall, and from the usual spot link6 grazes it
# (hero counter_b, hinge 0.01 m inside the wall face, handle 0.117 m from it: 27 steps, 2.0 mm). With counter_b slid
# along the wall the usual spot grazes with the hinge at 0 and neither spot does at 0.01 m. Off the wall (0.16 m doors
# hinged 0.11 m from it) or with the handle further out (0.86 m doors flush with it) the nearer spot can miss the bar
# where the usual one closes the door
CLOSE_DOOR_FLUSH, CLOSE_DOOR_ROOM = 0.005, 0.13


def front(f: dict) -> np.ndarray:
    """Unit normal out of a fixture's front face (local +y), in the world."""
    y = f.get("yaw", 0.0)
    return np.array([-np.sin(y), np.cos(y), 0.0])


def corners(f: dict) -> list[tuple[float, float]]:
    hx, hy = f["size"][0] / 2, f["size"][1] / 2
    return [to_world((dx, dy), f) for dx in (-hx, hx) for dy in (-hy, hy)]


def top(f: dict) -> float:
    return f["position"][2] + f["size"][2] / 2


def park(c: dict, along: float, out: float) -> tuple[float, float, float]:
    """Base centre `along` the fixture's run (its local x) and `out` beyond its front face."""
    yaw, d = c.get("yaw", 0.0), c["size"][1] / 2 + out
    return (c["position"][0] + np.cos(yaw) * along - np.sin(yaw) * d,
            c["position"][1] + np.sin(yaw) * along + np.cos(yaw) * d, 0.75)


def place_point(twin: dict, task: dict) -> np.ndarray:
    """Where a counter_to_sink / counter_to_counter / move_on_table task sets the object down (eef target, world)."""
    fx = {f["id"]: f for f in twin["fixtures"]}
    tgt = fx[task["target"]]
    hh = next(o for o in twin["objects"] if o["id"] == task["object_id"])["size"][2] / 2
    if "target_xy" in task:  # move_on_table: a spot on the same table
        return np.array([task["target_xy"][0], task["target_xy"][1], top(tgt) + hh + 0.012])
    tops = max(top(f) for f in twin["fixtures"] if is_counter_host(f)) if tgt["type"] == "sink" else top(tgt)
    if tgt["type"] == "sink":
        return np.array([tgt["position"][0], tgt["position"][1], tops + hh + 0.05])
    return np.array([task["target_x"], task.get("target_y", tgt["position"][1]), tops + hh + 0.012])


def mobile_drive(task: dict, mobile: bool) -> bool:
    """True when the base drives during this task: only counter_to_counter does, and only on a mobile base."""
    return mobile and task["family"] == "counter_to_counter"


def drive_spot(twin: dict, task: dict) -> tuple[float, float, float]:
    """Where a mobile base stands to set the object down: in front of the place point, the same 0.28 m out from the
    target counter's front that the parked pose keeps from the source's. World xy and the stand height."""
    tgt = next(f for f in twin["fixtures"] if f["id"] == task["target"])
    return park(tgt, to_local(place_point(twin, task)[:2], tgt)[0], 0.28)


def faucet_counter(twin: dict, sink: dict) -> dict:
    """The counter run the sink is set into (its frame holds the sink's centre); the lever is pushed from in front of it."""
    for c in twin["fixtures"]:
        if is_counter_host(c):
            lx, ly = to_local(sink["position"][:2], c)
            if abs(lx) < c["size"][0] / 2 and abs(ly) < c["size"][1] / 2:
                return c
    raise StopIteration(f"no counter run holds {sink['id']}")


def rack_spot(d: dict) -> tuple[float, float]:
    """World xy in the middle of the part of a dishwasher's upper rack that is out past its front."""
    n = front(d)
    out = d["size"][1] / 2 + RACK_FRONT
    return float(d["position"][0] + n[0] * out), float(d["position"][1] + n[1] * out)


def base_pose(twin: dict, task: dict, obj_xy=None, mobile: bool = False, arm: str = "procedural"
              ) -> tuple[tuple[float, float, float], float, dict]:
    """(base position, base yaw in degrees, anchor fixture): in front of the fixture the task works at, facing it,
    on any wall. `obj_xy` is the object's xy at the start (the task's `start` when omitted). `mobile` moves the
    counter_to_counter stand spot MOBILE_STEP toward the target instead of midway: the base drives the rest.
    `arm` trims the stand spots that push with the closed fingertips (ARM_STANDOFF_TRIM). open_door stands
    OPEN_DOOR_WALL_OUT further out when the run's hinge end is drawn into a wall and the bar ends the pull beside it,
    and only when that further spot still passes the stand check. A task carrying
    `base_shift` (metres along the anchor fixture's local x, and out from the face of it the base stands at; set by
    the reach planner when the rule's spot cannot reach) stands that far from the rule's spot."""
    base, yaw, src = _stand_rule(twin, task, obj_xy, mobile, arm)
    if task["family"] == "open_door":
        hinge_gap, handle_gap = _door_wall_gaps(twin, src, math.radians(80))  # where the expert's pull ends
        if hinge_gap < -OPEN_DOOR_INSIDE and handle_gap < OPEN_DOOR_ROOM:
            n = front(src)
            out = (base[0] + n[0] * OPEN_DOOR_WALL_OUT, base[1] + n[1] * OPEN_DOOR_WALL_OUT, base[2])
            if _stand_problem(twin, np.array(out[:2]), src, task=task) is None:
                base = out
    if task.get("base_shift"):
        along, out = task["base_shift"]
        a = src.get("yaw", 0.0)
        side = 1.0 if to_local(base[:2], src)[1] >= 0 else -1.0  # move_on_table may stand at the table's back edge
        n = side * front(src)
        base = (base[0] + math.cos(a) * along + n[0] * out, base[1] + math.sin(a) * along + n[1] * out, base[2])
    return base, yaw, src


def _stand_rule(twin: dict, task: dict, obj_xy, mobile: bool, arm: str) -> tuple[tuple[float, float, float], float, dict]:
    fx = {f["id"]: f for f in twin["fixtures"]}
    fam, fixtures = task["family"], twin["fixtures"]
    if obj_xy is None and "object_id" in task:
        obj_xy = task["start"][:2]
    yaw_off = 0.0
    if mobile_drive(task, mobile):
        src = fx[task["source"]]
        ox, tx = to_local(obj_xy[:2], src)[0], to_local(place_point(twin, task)[:2], src)[0]
        base = park(src, ox + MOBILE_STEP * float(np.sign(tx - ox)), 0.28)
    elif fam == "move_on_table":
        # at the table edge nearer the object (task["side"] is +1 or -1 along the table's local y), halfway between
        # where the object starts and where it goes, facing in
        src = fx[task["source"]]
        ox, tx = to_local(obj_xy[:2], src)[0], to_local(task["target_xy"], src)[0]
        base = to_world(((ox + tx) / 2, task["side"] * (src["size"][1] / 2 + 0.28)), src) + (0.75,)
        n = task["side"] * front(src)
        return base, float(np.degrees(np.arctan2(-n[1], -n[0]))), src
    elif fam in PICK_PLACE:
        src = fx[task["source"]]
        mid = (np.array(obj_xy[:2]) + place_point(twin, task)[:2]) / 2
        base = park(src, to_local(mid, src)[0], 0.28)
    elif fam == "wipe_counter":
        src = fx[task["source"]]
        base = park(src, to_local(obj_xy[:2], src)[0], 0.28)
    elif fam == "put_in_drawer":
        # straight in front, between the mug and the drawer, just beyond the pulled-out front and its handle;
        # parking beside the drawer instead puts the elbow into whatever appliance flanks the counter
        src = fx[task["fixture"]]
        along = (drawer_x(src, fixtures) + to_local(obj_xy[:2], src)[0]) / 2
        base = park(src, along, DRAWER_OPEN_PUT + 0.07 + 0.14)  # front + handle bar + base column + margin
    elif fam in FLAP_DOOR_BELOW:
        # in front of the bar (the oven OVEN_ALONG to either side, still in front of the door, whichever side has room,
        # else straight in front), just far enough out that the door swings all the way down to flat without its top
        # edge reaching the base column; parking beside the swing instead puts the wrist under the arm for a
        # horizontal grasp
        src = fx[task["fixture"]]
        out = src["size"][2] - FLAP_DOOR_BELOW[fam] + 0.12 + FLAP_MARGIN
        spots = [park(src, a, out) for a in ((OVEN_ALONG, -OVEN_ALONG, 0.0) if fam == "open_oven" else (0.0,))]
        base = next((b for b in spots if _stand_problem(twin, np.array(b[:2]), src, task=task) is None), spots[-1])
    elif fam == "turn_on_faucet":
        # in front of the counter run at the sink, closer than for counter_to_sink: the lever is at the back of the
        # counter, at the edge of what a horizontal hand reaches from the usual parking spot
        sink = fx[task["fixture"]]
        src = faucet_counter(twin, sink)
        base = park(src, to_local(sink["position"][:2], src)[0], FAUCET_STANDOFF - ARM_STANDOFF_TRIM[arm])
    elif fam in ("load_dishwasher", "unload_dishwasher"):
        # beside the dishwasher, in front of the counter, so the flat door and the rolled-out rack stay clear of the
        # base column and both the rack and the counter spot are within reach
        src = fx[task["fixture"]]
        base = park(src, math.copysign(DISHWASHER_SIDE, to_local(obj_xy[:2], src)[0]), 0.28)
    elif fam == "open_fridge":
        # beside the door's free edge and out of its swing (the hinge is on the fixture's left edge), facing the
        # front like the cabinet door tasks
        src = fx[task["fixture"]]
        from arrival.sim.scene import fridge_mirror

        m = fridge_mirror(src)  # a right-hinged door's free edge is the fixture's left one
        base = park(src, m * (src["size"][0] / 2 + FRIDGE_BASE_ALONG), FRIDGE_BASE_OUT)
        yaw_off = -m * FRIDGE_BASE_YAW
    else:
        src = fx[task["fixture"]]
        along, out = front_stand(src, fam, fixtures)
        base = park(src, along, out)
        hinge_gap, handle_gap = _close_door_gaps(twin, src) if fam == "close_door" else (math.inf, math.inf)
        if hinge_gap < CLOSE_DOOR_FLUSH and handle_gap < CLOSE_DOOR_ROOM:
            nearer = park(src, along + CLOSE_DOOR_WALL_ALONG, out)
            if _stand_problem(twin, np.array(nearer[:2]), src, task=task) is None:
                base = nearer
    n = front(src)
    return base, float(np.degrees(np.arctan2(-n[1], -n[0])) + yaw_off), src


def _segment_distance(p, a, b) -> float:
    a, b, p = np.asarray(a, float), np.asarray(b, float), np.asarray(p, float)
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - p))


def _close_door_gaps(twin: dict, c: dict) -> tuple[float, float]:
    """Distances from a cabinet door's hinge edge and from its handle, at the angle close_door starts from, to the
    nearest wall face square to the run (within 5 deg; a skewed wall leaves a different gap). The offsets are
    counter_xml's: the hinge on the door's right edge (door_hinge_x) and 1 cm behind the top's edge, the
    handle 6 cm in from the free edge and 5 cm out from the panel, the door turning clockwise from above."""
    from arrival.sim.collect import DOOR_OPEN

    r, a = door_w(c, twin["fixtures"]) - 0.06, DOOR_OPEN
    hx = door_hinge_x(c, twin["fixtures"]) - r * math.cos(a) + 0.05 * math.sin(a)
    hy = c["size"][1] / 2 - 0.01 + r * math.sin(a) + 0.05 * math.cos(a)
    n = front(c)

    def square(w):
        dx, dy = w["end"][0] - w["start"][0], w["end"][1] - w["start"][1]
        return abs(dx * n[1] - dy * n[0]) < math.sin(math.radians(5)) * math.hypot(dx, dy)

    walls = [w for w in twin.get("walls") or [] if square(w)]
    return tuple(min((_segment_distance(p, w["start"], w["end"]) - w.get("thickness", 0.0) / 2 for w in walls),
                     default=math.inf)
                 for p in (to_world((door_hinge_x(c, twin["fixtures"]) + 0.02, c["size"][1] / 2), c),
                           to_world((hx, hy), c)))


def _footprint_distance(xy, f: dict) -> float:
    lx, ly = to_local(xy, f)
    return math.hypot(max(abs(lx) - f["size"][0] / 2, 0.0), max(abs(ly) - f["size"][1] / 2, 0.0))


def _door_wall_gaps(twin: dict, c: dict, a: float) -> tuple[float, float]:
    """Distances from a cabinet run's hinge end and from its door handle with the door at angle `a`, to the nearest
    wall face square to the run (within 5 deg; a skewed wall leaves a different gap). The offsets are counter_xml's:
    the hinge on the door's right edge (door_hinge_x) and 1 cm behind the top's edge, the handle 6 cm in
    from the free edge and 5 cm out from the panel, the door turning clockwise from above."""
    r = door_w(c, twin["fixtures"]) - 0.06
    hx = door_hinge_x(c, twin["fixtures"]) - r * math.cos(a) + 0.05 * math.sin(a)
    hy = c["size"][1] / 2 - 0.01 + r * math.sin(a) + 0.05 * math.cos(a)
    n = front(c)

    def square(w):
        dx, dy = w["end"][0] - w["start"][0], w["end"][1] - w["start"][1]
        return abs(dx * n[1] - dy * n[0]) < math.sin(math.radians(5)) * math.hypot(dx, dy)

    walls = [w for w in twin.get("walls") or [] if square(w)]
    return tuple(min((_segment_distance(p, w["start"], w["end"]) - w.get("thickness", 0.0) / 2 for w in walls),
                     default=math.inf)
                 for p in (to_world((door_hinge_x(c, twin["fixtures"]) + 0.02, c["size"][1] / 2), c),
                           to_world((hx, hy), c)))


def open_parts(twin: dict, task: dict) -> list[tuple[str, dict, tuple[float, float, float, float]]]:
    """(name, fixture, (local x0, x1, y0, y1)) of the parts the task opens or starts open, out past the fixture front:
    a bottom-hinged door lying flat, a pulled-out drawer with its handle. The closed footprint misses them, and a
    base column standing on one starts the episode inside it. Hinged doors sweep an arc and are left out."""
    fam = task["family"]
    f = next((x for x in twin["fixtures"] if x["id"] == task.get("fixture")), None)
    if f is None:
        return []
    sx, sy, sz = f["size"]
    if fam in ("open_oven", "open_dishwasher", "load_dishwasher", "unload_dishwasher"):
        below = FLAP_DOOR_BELOW["open_oven" if f["type"] == "stove" else "open_dishwasher"]
        return [(f"{f['id']} door lying open", f, (-sx / 2, sx / 2, sy / 2, sy / 2 + 0.016 + sz - below))]
    if fam in DRAWER_TRAVEL:
        dx = drawer_x(f, twin["fixtures"])
        return [(f"{f['id']} drawer pulled out", f,
                 (dx - DRAWER_W / 2, dx + DRAWER_W / 2, sy / 2, sy / 2 + DRAWER_TRAVEL[fam] + DRAWER_HANDLE))]
    return []


def _part_distance(xy, f: dict, rect) -> float:
    lx, ly = to_local(xy, f)
    x0, x1, y0, y1 = rect
    return math.hypot(max(x0 - lx, lx - x1, 0.0), max(y0 - ly, ly - y1, 0.0))


def _stand_problem(twin: dict, b, anchor: dict, what: str = "base stand spot", task: dict | None = None) -> str | None:
    walls = twin.get("walls") or []
    if walls:
        xs = [p[0] for w in walls for p in (w["start"], w["end"])]
        ys = [p[1] for w in walls for p in (w["start"], w["end"])]
        if not (min(xs) < b[0] < max(xs) and min(ys) < b[1] < max(ys)):
            return f"{what} ({b[0]:.2f}, {b[1]:.2f}) at {anchor['id']} is outside the room"
        for w in walls:
            d = _segment_distance(b, w["start"], w["end"]) - w.get("thickness", 0.0) / 2
            if d < WALL_CLEAR:
                return f"{what} at {anchor['id']} is {max(d, 0.0):.2f} m from {w.get('id', 'a wall')}, needs {WALL_CLEAR:.2f} m"
    for f in twin["fixtures"]:
        d = _footprint_distance(b, f)
        if d < BASE_CLEAR:
            return f"{what} at {anchor['id']} is {d:.2f} m from {f['id']}, needs {BASE_CLEAR:.2f} m"
    for name, f, rect in open_parts(twin, task) if task else []:
        d = _part_distance(b, f, rect)
        if d < PART_CLEAR:
            return f"{what} at {anchor['id']} is {d:.2f} m from the {name}, needs {PART_CLEAR:.2f} m"
    return None


def check_task(twin: dict, task: dict, obj_xy=None, mobile: bool = False, arm: str = "procedural") -> str | None:
    """None when the base can stand where the task needs it and reach what it needs; otherwise one sentence saying
    why not. `task` is a pool entry (fixture, or object_id, source and start) or a finished task. On a mobile base
    a counter_to_counter task is checked at both stand spots and along the straight drive between them. Reach is an
    IK solve of the expert's waypoints from the stand spot (`arrival.sim.reach`); families whose hand targets come
    from the built model (doors, drawers, the fridge, flap doors, the faucet) get the stand checks only."""
    from arrival.sim import reach

    try:
        base, yaw, anchor = base_pose(twin, task, obj_xy, mobile, arm)
    except (KeyError, StopIteration, IndexError, TypeError, ValueError) as e:  # a task naming a fixture the twin lacks
        return f"no stand spot: the task does not match the twin ({type(e).__name__}: {e})"
    b = np.array(base[:2])
    why = _stand_problem(twin, b, anchor, task=task)
    if why:
        return why
    wps = reach.waypoints(twin, task, obj_xy, mobile, b)
    spots = [(b, "the base", [w for w in wps if not w[3]])]
    if mobile_drive(task, mobile):
        end = np.array(drive_spot(twin, task)[:2])
        for k in range(1, 9):  # the drive is one straight line: its whole length must be clear, not just its ends
            why = _stand_problem(twin, b + (end - b) * k / 8, anchor, "base drive path", task)
            if why:
                return why
        spots.append((end, "the drive spot", [w for w in wps if w[3]]))
    for xy, name, chain in spots:
        if not chain:
            continue
        m = reach.miss(reach.chain_errors(chain, xy, math.radians(yaw), arm), task["family"])
        if m:
            k, at, err, tol = m
            d = float(np.linalg.norm(chain[k][1][:2] - xy))
            return (f"{at} ({d:.2f} m out) is beyond the arm's reach from {name} at {anchor['id']}: "
                    f"the IK solve misses it by {err * 100:.1f} cm (allowed {tol * 100:.1f} cm)")
    return None
