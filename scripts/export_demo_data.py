"""Export everything a static demo site needs under site/data/ from the REAL pipeline.

    .venv/bin/python scripts/export_demo_data.py [--out site/data] [--room hero_kitchen] [--seed 5]

Loads scans/<room>, compiles the twin, runs the physics sim (arrival.sim.collect.SimEnv + ScriptedExpert) for one
episode per task family with two walking people, and writes per-body world poses at 15 fps, the visual geoms,
the room layout, stats (dataset facts, loader check, a tiny eval run, cost report), a simulated LiDAR scan and
preview frames. Schema: site/data/README.md (written by this script). One process, tiny renders, no mp4 in site/.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arrival.catalog.objects import OBJECT_SPECS  # noqa: E402
from arrival.scan import bundle as scan_bundle  # noqa: E402
from arrival.sim import robot  # noqa: E402
from arrival.sim.collect import FPS, SUBSTEPS, SimEnv, run_episode  # noqa: E402
from arrival.sim.export import BASE_SLICE  # noqa: E402
from arrival.sim.people import limb_segments  # noqa: E402
from arrival.sim.policy import ScriptedExpert  # noqa: E402
from arrival.sim.scene import build_mjcf, part_names  # noqa: E402
from arrival.sim.tasks import finish_task, make_tasks  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

FAMILIES = ["counter_to_sink", "open_drawer", "open_door", "open_fridge", "put_in_drawer", "wipe_counter"]
OUT_FPS = 15
KEEP = ("todo.json",)  # hand-edited files in site/data that this export does not generate; kept across a re-run
PREVIEW_PX = 256
PEOPLE = 2
VISIBLE_GROUPS = (0, 1, 2)  # group 3 holds the hidden collision geoms
SCAN_POINTS = 20_000
SCAN_STEP = 0.001  # quantisation of scan points, metres
GEOM_SIZE_N = {"box": 3, "sphere": 1, "capsule": 2, "cylinder": 2, "plane": 3, "ellipsoid": 3}
LIMBS = tuple(limb_segments({"hip_w": 0, "hip_z": 0, "thigh": 1, "shank": 1, "foot": 1, "shoulder_w": 0,
                             "shoulder_z": 0, "upper_arm": 1, "forearm": 1}, 0.0, 0.0))


# ---------------------------------------------------------------- helpers
def rnd(v, n=4):
    """Round floats (recursively) and turn -0.0 into 0.0, so the JSON stays short."""
    if isinstance(v, (float, np.floating)):
        r = round(float(v), n)
        return 0.0 if r == 0 else r
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return rnd(v.tolist(), n)
    if isinstance(v, dict):
        return {k: rnd(x, n) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [rnd(x, n) for x in v]
    if isinstance(v, Path):
        return str(v)
    return v


def dump(path: Path, obj, indent=None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=indent, separators=(",", ":") if indent is None else None, allow_nan=False)
    path.write_text(text, newline="\n")  # same bytes on every OS, so the sizes in stats.json match what git serves
    return len(text)


def hexcolour(rgba) -> str:
    return "#" + "".join(f"{int(round(min(max(c, 0), 1) * 255)):02x}" for c in rgba[:3])


def mat_to_quat(mat9) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat9, dtype=np.float64).ravel())
    return q


# ---------------------------------------------------------------- geoms
def geom_entry(m, g: int, identity: bool = False) -> dict:
    t = mujoco.mjtGeom(m.geom_type[g]).name.removeprefix("mjGEOM_").lower()
    n = GEOM_SIZE_N.get(t, 3)
    mat = int(m.geom_matid[g])
    return {
        "name": m.geom(g).name or f"g{g}",
        "type": t,
        "size": rnd(m.geom_size[g][:n], 4),
        "pos": [0.0, 0.0, 0.0] if identity else rnd(m.geom_pos[g], 5),
        "quat": [1.0, 0.0, 0.0, 0.0] if identity else rnd(m.geom_quat[g], 5),
        "rgba": rnd(m.geom_rgba[g], 3),
        "colour": hexcolour(m.geom_rgba[g]),
        "material": m.material(mat).name if mat >= 0 else None,
    }


def visible_geoms(m, body: int) -> list[int]:
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_group[g] in VISIBLE_GROUPS
            and m.geom_rgba[g][3] > 0]


def is_person(m, b: int) -> bool:
    name = m.body(b).name
    return name.startswith("person_") and int(m.body_mocapid[b]) >= 0


def export_geoms(m) -> tuple[dict, dict, dict]:
    """(bodies, person_bodies, cameras). Person limbs are their own one-geom 'bodies' named after the geom, because the
    sim moves them by rewriting geom_pos/geom_quat: their world pose is read per geom, not per body."""
    bodies, persons, cams = {}, {}, {}
    for b in range(m.nbody):
        name = "world" if b == 0 else m.body(b).name
        gs = visible_geoms(m, b)
        if not gs:
            continue
        parent = None if b == 0 else ("world" if m.body_parentid[b] == 0 else m.body(int(m.body_parentid[b])).name)
        if is_person(m, b):
            torso = [g for g in gs if not any(m.geom(g).name.endswith("_" + k) for k in LIMBS)]
            persons[name] = {"parent": "world", "role": "person", "geoms": [geom_entry(m, g) for g in torso]}
            for g in gs:
                if g not in torso:
                    persons[m.geom(g).name] = {"parent": name, "role": "person_limb",
                                               "geoms": [geom_entry(m, g, identity=True)]}
        else:
            bodies[name] = {"parent": parent, "geoms": [geom_entry(m, g) for g in gs]}
    for c in range(m.ncam):
        cams[m.camera(c).name] = {"fovy": rnd(m.cam_fovy[c], 2), "body": m.body(int(m.cam_bodyid[c])).name
                                  if m.cam_bodyid[c] else "world", "node": f"cam:{m.camera(c).name}"}
    return bodies, persons, cams


def export_lights(m) -> list[dict]:
    return [{"name": m.light(i).name, "directional": bool(m.light_type[i] == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL), "pos": rnd(m.light_pos[i], 3),
             "dir": rnd(m.light_dir[i], 3), "diffuse": rnd(m.light_diffuse[i], 3), "cutoff": rnd(m.light_cutoff[i], 1),
             "castshadow": bool(m.light_castshadow[i])} for i in range(m.nlight)]


# ---------------------------------------------------------------- pose recording
class Recorder:
    """Reads world poses of every visible body after each control step. Pose is read from a scratch MjData that is
    re-run through kinematics, so mocap people and rewritten limb geoms are current and the sim's own data is untouched."""

    def __init__(self, env: SimEnv, task: dict):
        self.env, self.task = env, task
        m = env.model
        self.m, self.d2 = m, mujoco.MjData(m)
        self.body_nodes = []  # (name, body id)
        self.limb_nodes = []  # (name, geom id)
        for b in range(1, m.nbody):
            if visible_geoms(m, b):
                self.body_nodes.append((m.body(b).name, b))
        for b in range(1, m.nbody):
            if is_person(m, b):
                self.limb_nodes += [(m.geom(g).name, g) for g in visible_geoms(m, b)
                                    if any(m.geom(g).name.endswith("_" + k) for k in LIMBS)]
        self.cam_nodes = [(f"cam:{m.camera(c).name}", c) for c in range(m.ncam)]
        self.names = [n for n, _ in self.body_nodes] + [n for n, _ in self.limb_nodes] + [n for n, _ in self.cam_nodes]
        self.pos, self.quat, self.states, self.people_xy, self.joint_q = [], [], [], [], []
        self.previews: list[tuple[int, np.ndarray, np.ndarray]] = []
        self.steps = 0

    def capture(self) -> None:
        env, m, d, d2 = self.env, self.m, self.env.data, self.d2
        d2.qpos[:] = d.qpos
        if m.nmocap:
            d2.mocap_pos[:] = d.mocap_pos
            d2.mocap_quat[:] = d.mocap_quat
        mujoco.mj_kinematics(m, d2)
        mujoco.mj_camlight(m, d2)
        p = [d2.xpos[b].copy() for _, b in self.body_nodes] + [d2.geom_xpos[g].copy() for _, g in self.limb_nodes] \
            + [d2.cam_xpos[c].copy() for _, c in self.cam_nodes]
        q = [d2.xquat[b].copy() for _, b in self.body_nodes] \
            + [mat_to_quat(d2.geom_xmat[g]) for _, g in self.limb_nodes] \
            + [mat_to_quat(d2.cam_xmat[c]) for _, c in self.cam_nodes]
        self.pos.append(p)
        self.quat.append(q)
        self.states.append(env.state().astype(np.float64))
        self.people_xy.append([[float(p[0]), float(p[1])] for p in env.walkers.pos] if env.walkers else [])
        if "joint" in self.task:
            self.joint_q.append(float(d.joint(self.task["joint"]).qpos[0]))

    def install(self) -> None:
        orig = self.env.step

        def step(action):
            if not self.pos:
                self.capture()
                self.render_preview()
            orig(action)
            self.steps += 1
            self.capture()
            if self.steps % 3 == 0:
                self.render_preview()

        self.env.step = step

    def render_preview(self) -> None:
        self.previews.append((len(self.pos) - 1, self.env.render("ego_view"), self.env.render("wrist")))


def resample(arr: np.ndarray, src_fps: float, dst_fps: float, quat: bool = False) -> np.ndarray:
    """arr (N, ...) sampled at src_fps -> samples at dst_fps by linear interpolation (nlerp for quaternions)."""
    n = arr.shape[0]
    if quat:
        s = np.where((arr[1:] * arr[:-1]).sum(-1, keepdims=True) < 0, -1.0, 1.0)
        arr = arr * np.concatenate([np.ones_like(s[:1]), np.cumprod(s, axis=0)], axis=0)  # keep the hemisphere
    t_end = (n - 1) / src_fps
    k = int(math.floor(t_end * dst_fps + 1e-9)) + 1
    u = np.arange(k) * src_fps / dst_fps
    i0 = np.minimum(np.floor(u + 1e-9).astype(int), n - 2)
    a = (u - i0).reshape((-1,) + (1,) * (arr.ndim - 1))
    out = arr[i0] * (1 - a) + arr[i0 + 1] * a
    if quat:
        out = out / np.linalg.norm(out, axis=-1, keepdims=True)
    return out


def pack_episode(rec: Recorder, fps: int) -> dict:
    P = resample(np.array(rec.pos), FPS, fps)  # (T, K, 3)
    Q = resample(np.array(rec.quat), FPS, fps, quat=True)  # (T, K, 4)
    P, Q = np.round(P, 3), np.round(Q, 4)
    moves = (np.ptp(P, axis=0).max(-1) > 6e-4) | (np.ptp(Q, axis=0).max(-1) > 6e-4)
    dyn = [i for i in range(len(rec.names)) if moves[i]]
    static = {rec.names[i]: rnd(list(P[0, i]) + list(Q[0, i]), 4) for i in range(len(rec.names)) if not moves[i]}
    frames = np.concatenate([P[:, dyn], Q[:, dyn]], axis=-1).reshape(P.shape[0], -1)  # per node: x y z qw qx qy qz
    # arm joints and gripper only: the site reads 8 values per frame; the sim state also carries the base pose
    states = np.round(resample(np.array(rec.states)[:, :BASE_SLICE["start"]], FPS, fps), 4)
    return {
        "n_frames": int(P.shape[0]),
        "nodes": {"dynamic": [rec.names[i] for i in dyn], "static": static},
        "frames": rnd(frames, 4),
        "joint_states": rnd(states, 4),
        "people_xy": rnd(np.round(resample(np.array(rec.people_xy, dtype=np.float64), FPS, fps), 3), 3)
        if PEOPLE else [],
        "task_joint_q": rnd(np.round(resample(np.array(rec.joint_q)[:, None], FPS, fps)[:, 0], 4), 4)
        if rec.joint_q else None,
    }


# ---------------------------------------------------------------- scan (simulated LiDAR)
def simulate_scan(twin: dict, seed: int = 7) -> tuple[bytes, dict]:
    """Ray-cast the twin from three standing positions (iPhone height) with a Fibonacci-sphere ray pattern, add range
    and lateral noise, keep SCAN_POINTS hits and quantise to 1 mm. Visible static + object geoms only; no robot, no people."""
    rng = np.random.default_rng(seed)
    xml = build_mjcf(twin, (2.0, 1.5, 0.75), 0.0, None, "neutral", 0)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    robot_bodies = {m.body(n).id for n in ("robot_base",) + tuple(f"link{i}" for i in range(1, 8))
                    + ("hand", "finger_l", "finger_r")}
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) in robot_bodies:
            m.geom_group[g] = 5  # excluded by the ray mask below
    mujoco.mj_forward(m, d)
    mask = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)
    xs = [p for w in twin["walls"] for p in (w["start"][0], w["end"][0])]
    ys = [p for w in twin["walls"] for p in (w["start"][1], w["end"][1])]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    stations = [(x0 + fx * (x1 - x0), y0 + fy * (y1 - y0), 1.3) for fx, fy in ((0.28, 0.6), (0.72, 0.6), (0.5, 0.88))]
    n_rays = 14_000
    i = np.arange(n_rays) + 0.5
    phi = np.arccos(1 - 2 * i / n_rays)
    theta = math.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    dirs = dirs[dirs[:, 2] < 0.6]  # the twin has no ceiling; a scanner would not aim far above the horizon
    pts, cols = [], []
    gid = np.zeros(1, dtype=np.int32)
    floor_rgb = np.array([0.68, 0.61, 0.53])  # mean of the floor tile texture
    for s in stations:
        o = np.array(s)
        for v in dirs:
            dist = mujoco.mj_ray(m, d, o, v, mask, 1, -1, gid)
            if dist < 0 or dist > 7.0:
                continue
            g = int(gid[0])
            rgb = floor_rgb if m.geom(g).name == "floor" else m.geom_rgba[g][:3]
            sigma = 0.003 + 0.0025 * dist  # range noise grows with distance, a few mm
            hit = o + v * (dist + rng.normal(0, sigma)) + rng.normal(0, 0.002, 3)
            pts.append(hit)
            cols.append(np.clip(rgb * (1 + rng.normal(0, 0.04)), 0, 1))
    pts, cols = np.array(pts), np.array(cols)
    keep = rng.permutation(len(pts))[:SCAN_POINTS]
    pts, cols = pts[keep], cols[keep]
    origin = pts.min(0) - 0.01
    q = np.round((pts - origin) / SCAN_STEP).astype(np.uint16)
    rgb8 = np.round(cols * 255).astype(np.uint8)
    rec = np.zeros(len(q), dtype=[("xyz", "<u2", 3), ("rgb", "u1", 3)])
    rec["xyz"], rec["rgb"] = q, rgb8
    meta = {"file": "scan_points.bin", "count": int(len(q)), "bytes_per_point": 9,
            "record": "uint16 x, uint16 y, uint16 z (little endian), uint8 r, uint8 g, uint8 b",
            "decode": "position_m = origin + uint16 * scale", "origin": rnd(origin, 4), "scale": SCAN_STEP,
            "bounds_min": rnd(pts.min(0), 3), "bounds_max": rnd(pts.max(0), 3),
            "stations": rnd([list(s) for s in stations], 3), "rays_per_station": int(len(dirs)),
            "noise": {"range_sigma_m": "0.003 + 0.0025 * range", "lateral_sigma_m": 0.002, "colour_jitter": 0.04},
            "max_range_m": 7.0, "seed": seed, "frame": "MuJoCo z-up, metres, same as rooms.json",
            "note": "Simulated from the compiled twin (not a real iPhone capture): ray hits on visible geoms, robot and "
                    "people excluded."}
    return rec.tobytes(), meta


# ---------------------------------------------------------------- rooms.json
def rooms_json(twin: dict, bodies: dict) -> dict:
    def geom_rgba(name):
        for b in bodies.values():
            for g in b["geoms"]:
                if g["name"] == name:
                    return g
        return None

    fixtures = []
    for f in twin["fixtures"]:
        main = next((geom_rgba(f"{f['id']}_{s}") for s in ("carcass", "box", "bottom") if geom_rgba(f"{f['id']}_{s}")), None)
        top = geom_rgba(f"{f['id']}_top") or geom_rgba(f"{f['id']}_top_l")
        e = {k: f[k] for k in ("id", "type", "source_category", "size", "position", "yaw") if k in f}
        e["island"] = bool(f.get("island", False))
        e["colour"] = main["colour"] if main else "#888888"
        e["colour_rgba"] = main["rgba"] if main else [0.53, 0.53, 0.53, 1.0]
        if top:
            e["top_colour"] = top["colour"]
        fixtures.append(e)
    parts = part_names(twin)
    xs = [p for w in twin["walls"] for p in (w["start"][0], w["end"][0])]
    ys = [p for w in twin["walls"] for p in (w["start"][1], w["end"][1])]
    catalogue = {k: {"shape": v[0], "nominal_size": v[1], "mass_kg": v[2], "friction": v[3], "rgba": v[4],
                     "colour": hexcolour([float(c) for c in v[4].split()])} for k, v in OBJECT_SPECS.items()}
    return {
        "room_id": twin["room_id"], "geometry_source": twin["geometry_source"], "frame": "z_up", "units": "meters",
        "bounds": {"min": [min(xs), min(ys), 0.0], "max": [max(xs), max(ys), max(w["height"] for w in twin["walls"])]},
        "wall_colour": hexcolour([0.93, 0.91, 0.86]), "floor_colour": "#ad9c87",
        "walls": twin["walls"], "fixtures": fixtures, "parts": parts,
        "objects": [{**o, "colour": catalogue.get(o["catalog_id"], {}).get("colour", "#8080e6")} for o in twin["objects"]],
        "object_catalogue": catalogue,
    }


# ---------------------------------------------------------------- one episode
def run_one(twin: dict, task: dict, out: Path, previews: Path) -> dict:
    env = SimEnv(twin, task, res=PREVIEW_PX, cameras=())
    rec = Recorder(env, task)
    rec.install()
    m = env.model
    geoms = export_geoms(m)
    lights = export_lights(m)
    ep = run_episode(env, ScriptedExpert(), task)  # closes the env
    packed = pack_episode(rec, OUT_FPS)
    # preview PNGs: ego_view and wrist at mid-episode and at the end
    from PIL import Image
    n = len(rec.pos)
    for label, frac in (("mid", 0.35), ("end", 1.0)):
        want = int(round(frac * (n - 1)))
        _, ego, wr = min(rec.previews, key=lambda p: abs(p[0] - want))
        Image.fromarray(ego).save(previews / f"{task['family']}_ego_view_{label}.png", optimize=True)
        Image.fromarray(wr).save(previews / f"{task['family']}_wrist_{label}.png", optimize=True)
    s = ep["safety"]
    doc = {
        "family": task["family"], "task_id": task["task_id"], "instruction": task["instruction"],
        "seed": task["seed"], "lighting": task["lighting"], "people": task.get("people", 0),
        "success": bool(ep["success"]), "human_contact": bool(s["human_contact"]),
        "physics_detail": ep["detail"], "min_human_distance_m": s["min_human_distance_m"], "safety": s,
        "ik_error": ep["ik_error"], "fridge_visit": ep["fridge_visit"],
        "fps": OUT_FPS, "source_control_hz": FPS, "physics_hz": round(FPS * SUBSTEPS),
        "duration_s": round((packed["n_frames"] - 1) / OUT_FPS, 3),
        "frame_layout": "frames[t] = [x,y,z,qw,qx,qy,qz] for each name in nodes.dynamic, concatenated per node "
                        "(positions are 3 numbers followed by 4 quaternion numbers: 7 numbers per node)",
        "frame": "MuJoCo z-up, metres, quaternion order w,x,y,z, world poses of body frames",
        **packed,
        "joint_names": robot.ARM_JOINTS + ["gripper_opening_m"],
        "task": {k: task[k] for k in ("fixture", "object_id", "source", "target", "start", "joint", "handle")
                 if k in task},
        "lights": lights,
        "person_geoms": geoms[1],
    }
    p = out / "episodes" / f"{task['family']}.json"
    size = dump(p, rnd(doc, 4))
    return {"doc": doc, "geoms": geoms, "bytes": size, "states_full": ep["states"], "ep": ep}


# ---------------------------------------------------------------- README
README = """# site/data: demo data schema

Generated by `scripts/export_demo_data.py` from the real Arrival pipeline (scan, twin, MuJoCo physics, scripted expert).
Nothing here is hand-drawn. Re-run `.venv/bin/python scripts/export_demo_data.py` to regenerate.

Conventions (everything): **MuJoCo world frame, z up, metres, radians, seconds.** Quaternions are **w, x, y, z**
(MuJoCo order; three.js wants x, y, z, w, so reorder). A pose is `[x, y, z, qw, qx, qy, qz]` of a body *frame*.
Geom `size` follows MuJoCo: box = half extents, sphere = [radius], capsule and cylinder = [radius, half length along the
geom's local z], plane = [half x, half y, grid]. The sim is z up; three.js is y up, so rotate the whole scene by -90 deg
about x (or swap axes) once at the root.

## Files

| file | what |
|---|---|
| `rooms.json` | the compiled twin: walls, fixtures (type, size, position, yaw, colour), moving parts, objects, object catalogue |
| `geoms.json` | the visual geoms of every body, local to the body, plus cameras. Hidden collision geoms (group 3) are left out |
| `episodes/<family>.json` | one successful physics episode: poses per frame, joint states, success, physics detail, safety |
| `stats.json` | episode counts, success per family, dataset facts, offline loader check, tiny eval run, cost report |
| `scan_points.bin` + `scan_points.json` | simulated LiDAR point cloud of the room (20k points, quantised) |
| `frames/<family>_<cam>_<mid|end>.png` | 256 px previews: `ego_view` (fixed camera on the robot base) and `wrist` (eye in hand), mid-episode and final |

## rooms.json

```
room_id, geometry_source ("authored_sample" | "roomplan"), frame "z_up", units "meters"
bounds {min:[x,y,z], max:[x,y,z]}
wall_colour, floor_colour                       "#rrggbb"
walls[]     {id, start:[x,y], end:[x,y], height, thickness}          (vertical slabs, thickness at least 5 cm in the sim)
fixtures[]  {id, type ("counter"|"sink"|"stove"|"fridge"|...), source_category, size:[sx,sy,sz], position:[x,y,z] (box centre),
             yaw (rad, rotation about z; local +x along the fixture, front faces local +y rotated by yaw),
             island, colour "#rrggbb", colour_rgba [r,g,b,a], top_colour?}
parts[]     {fixture, joint, site, kind ("slide"|"hinge"), open}     every moving part the scene builds; `joint` names a
                                                                      body in geoms.json (e.g. counter_b_drawer0)
objects[]   {id, catalog_id, label, size:[x,y,z], position:[x,y,z], yaw, colour}   start positions in the twin
object_catalogue{label: {shape, nominal_size, mass_kg, friction, rgba, colour}}
```

## geoms.json

```
frame, units, quaternion_order, size_convention (strings)
bodies{ <body name>: {parent, geoms:[{name, type ("box"|"sphere"|"capsule"|"cylinder"|"plane"), size, pos:[x,y,z], quat:[w,x,y,z],
                                       rgba:[r,g,b,a], colour "#rrggbb", material ("wood"|"stone"|"panel"|"paint"|"floor"|null)}]} }
cameras{ <name>: {fovy (deg), body, node "cam:<name>"} }     MuJoCo camera: looks down its local -z, +y up, +x right
materials{name: description}
```

`bodies.world` is the room shell (floor plane, walls) and never moves. A body's geom pose in the world is
`body_pose * (geom.pos, geom.quat)`. Where a body is not listed in an episode's `nodes.dynamic` it is in
`nodes.static` (constant for that episode). The robot base position differs per episode (it is placed for the task), so
always take body poses from the episode file, never from geoms.json.

People are **not** in geoms.json: their proportions and colours depend on the episode seed, so each episode carries
`person_geoms` (same schema as `bodies`). Limbs are separate one-geom nodes named after the geom
(`person_0_thigh_l`, with `role: "person_limb"`, geom pose identity), because the sim rotates limbs relative to the
person's body. The pose of a limb node is the geom's own world pose.

## episodes/<family>.json

```
family, task_id, instruction, seed, lighting ("dim"|"neutral"|"bright"), people (2)
success (bool, physics-checked), human_contact (bool), physics_detail {...}  (family-specific, e.g. {joint, qpos} or {object_pos, object_speed})
min_human_distance_m   smallest robot to person clearance over the whole episode (m)
safety {min_human_distance_m, min_distance_step, human_contact, people, door_contact, arm_contact_steps, ...}
ik_error, fridge_visit (a walker opened the fridge during the episode)
fps (15), source_control_hz (20), physics_hz (500), duration_s, n_frames
nodes {dynamic:[name...], static:{name:[x,y,z,qw,qx,qy,qz]}}
frames[t]            numbers: for each name in nodes.dynamic, in order, [x,y,z,qw,qx,qy,qz]  (length 7 * len(dynamic))
                     poses are world poses of body frames (limb nodes and cameras: of the geom / camera frame)
                     positions rounded to 1 mm, quaternions to 4 decimals, hemisphere-continuous (safe to nlerp)
joint_states[t]      [joint1..joint7 (rad), gripper_opening_m (mean finger slide, 0..0.05)]  == the first 8 values of the LeRobot
                     observation.state (the dataset's last 3, the parked base pose, are left out)
joint_names, people_xy[t][person] = [x,y] floor position, task_joint_q[t] (the drawer/door joint the task moves, or null)
task {fixture, object_id, source, target, start, joint, handle}  (only the keys the family has)
lights[]             {name, directional, pos, dir, diffuse, cutoff, castshadow} as built by the sim (key spot, fill, window)
person_geoms{}       see above
```

Nodes are named after MuJoCo bodies (arm: `robot_base`, `link1`..`link7`, `hand`, `finger_l`, `finger_r`; parts:
`<fixture>_drawer0`, `<fixture>_door`, ...; objects: their id, e.g. `mug_1`), person nodes (`person_0`,
`person_0_thigh_l`, ...) and cameras (`cam:wrist`, `cam:ego_view`, `cam:overview`). Every node in `nodes` has geoms in
`geoms.json.bodies` or `person_geoms`, except cameras. Frame t is at time t / fps; frames were resampled from the 20 Hz
control loop (linear, nlerp for rotations).

## stats.json

```
room_id, seed, people, out_fps
episodes {attempted, successful, exported}
families{<family>: {exported, success, human_contact, frames, duration_s, min_human_distance_m, physics_detail, episode_file, instruction}}
skipped{<family>: reason}
dataset {physically_valid (arrival_physically_valid), generator, frames, episodes, attempts, successes, fps, cameras, resolution,
         state_dim, action_dim, families, loader_check {ok, errors, warnings, info{...}}, matches_exported_episodes}
eval {...}    arrival.eval.harness.evaluate(), 1 episode per family, held-out seeds, 2 people
cost {...}    arrival.eval.cost.build() over the local reports; null if it did not run offline
sim {physics_hz, control_hz, arm, people_model, ...}
files {name: bytes}
```

## scan_points.bin / scan_points.json

Little-endian records of 9 bytes: `uint16 x, uint16 y, uint16 z, uint8 r, uint8 g, uint8 b`. Metres = `origin + value * scale`
(origin and scale in `scan_points.json`, along with the scanner stations and noise model). Same frame as rooms.json.
It is simulated: rays cast from three standing positions at 1.3 m against the twin, with range noise
(about 3 mm plus 0.25% of range) and 2 mm lateral jitter. It is not a real iPhone capture.
"""


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="site/data")
    ap.add_argument("--room", default="hero_kitchen")
    ap.add_argument("--scans", default="scans")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--scratch", default="out/demo_export", help="scratch dir for the dataset, eval and reports (git-ignored)")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--skip-dataset", action="store_true")
    a = ap.parse_args(argv)
    try:  # before the output folder is cleared: a mistyped --room must not wipe site/data
        bundle = scan_bundle.load_room(a.room, a.scans)
    except FileNotFoundError:
        ap.error(f"no scan for room {a.room!r} under {a.scans}")
    out, scratch = Path(a.out), Path(a.scratch)
    kept = {n: (out / n).read_bytes() for n in KEEP if (out / n).is_file()}
    if out.exists():
        shutil.rmtree(out)
    (out / "episodes").mkdir(parents=True)
    (out / "frames").mkdir()
    for n, data in kept.items():
        (out / n).write_bytes(data)
    if scratch.exists():
        shutil.rmtree(scratch)
    log = lambda *x: print(*x, flush=True)  # noqa: E731

    twin = compile_twin(bundle)
    log(f"twin {twin['room_id']}: {len(twin['fixtures'])} fixtures, {len(twin['objects'])} objects")
    # a second room: only scans/hero_kitchen exists; the RoomPlan test fixtures hold no kitchen fixtures,
    # so there is nothing to compile to a sim-able twin
    others = [p.name for p in Path(a.scans).iterdir() if p.is_dir() and p.name != a.room and (p / "scan.json").exists()]
    log(f"other rooms under {a.scans}: {others or 'none'}")

    wanted = [f for f in a.families.split(",") if f]
    skipped, reachable = {}, []
    for fam in wanted:
        if make_tasks(twin, [fam], 1, seed=a.seed, people=PEOPLE, log=lambda *x: None):
            reachable.append(fam)
        else:
            skipped[fam] = "no reachable variation in this room for the fixed-base arm"
    tasks = [finish_task(t) for t in make_tasks(twin, reachable, len(reachable), seed=a.seed, people=PEOPLE, log=log)]

    fam_stats, all_bodies, bytes_by_file, attempted, exported = {}, [], {}, 0, 0
    for task in tasks:
        fam = task["family"]
        attempted += 1
        try:
            r = run_one(twin, task, out, out / "frames")
        except Exception as e:  # skip a family that fails, and say so
            skipped[fam] = f"{type(e).__name__}: {e}"
            log(f"[export] {fam}: ERROR {skipped[fam]}")
            continue
        d = r["doc"]
        log(f"[export] {fam}: success={d['success']} contact={d['human_contact']} frames={d['n_frames']} "
            f"nodes={len(d['nodes']['dynamic'])} dyn/{len(d['nodes']['static'])} static {d['physics_detail']} "
            f"min_human_d={d['min_human_distance_m']} {r['bytes'] / 1e3:.0f} kB")
        if not d["success"] or d["human_contact"]:
            (out / "episodes" / f"{fam}.json").unlink()
            for pth in (out / "frames").glob(f"{fam}_*"):
                pth.unlink()
            skipped[fam] = "episode failed" if not d["success"] else "dropped: human contact"
            continue
        exported += 1
        all_bodies.append(r["geoms"])
        fam_stats[fam] = {"exported": True, "success": d["success"], "human_contact": d["human_contact"],
                          "frames": d["n_frames"], "duration_s": d["duration_s"],
                          "min_human_distance_m": d["min_human_distance_m"], "physics_detail": d["physics_detail"],
                          "episode_file": f"episodes/{fam}.json", "instruction": d["instruction"],
                          "task_id": d["task_id"], "seed": d["seed"], "lighting": d["lighting"]}
    if not all_bodies:
        log("no episode exported")
        return 1

    # geoms.json: the shared (non-person) bodies; the same in every episode of this twin
    bodies0 = all_bodies[0][0]
    for b in all_bodies[1:]:
        for name, v in b[0].items():
            assert bodies0.get(name) == v, f"body {name} differs between episodes"
    cams = all_bodies[0][2]
    geoms = {"frame": "MuJoCo z-up, metres", "quaternion_order": "w,x,y,z",
             "size_convention": "MuJoCo: box half extents; sphere [r]; capsule/cylinder [r, half length]; plane [hx, hy, grid]",
             "bodies": bodies0, "cameras": cams,
             "materials": {"wood": "plank-grain cabinet fronts", "stone": "speckled counter tops",
                           "panel": "backsplash boards", "paint": "wall paint", "floor": "grouted floor tile"}}
    dump(out / "geoms.json", rnd(geoms, 5))
    dump(out / "rooms.json", rnd(rooms_json(twin, bodies0), 4), indent=1)

    # simulated LiDAR scan
    blob, meta = simulate_scan(twin)
    (out / "scan_points.bin").write_bytes(blob)
    dump(out / "scan_points.json", meta, indent=1)
    log(f"scan: {meta['count']} points, {len(blob) / 1e3:.0f} kB")

    # dataset facts: the same tasks through arrival.sim.collect, written to scratch (mp4s stay out of site/)
    stats = {"room_id": twin["room_id"], "seed": a.seed, "people": PEOPLE, "out_fps": OUT_FPS,
             "episodes": {"attempted": attempted, "successful": exported, "exported": exported},
             "families": fam_stats, "skipped": skipped, "other_rooms_found": others,
             "sim": {"engine": f"mujoco {mujoco.__version__}", "physics_hz": round(FPS * SUBSTEPS), "control_hz": FPS,
                     "arm": "procedural 7-DoF Panda-style arm, friction grasp", "success_check": "physics",
                     "people_model": "kinematic capsule walkers, seeded waypoint loops, safety metric per step"}}
    if not a.skip_dataset:
        from arrival.eval import cost as cost_mod
        from arrival.eval.harness import evaluate
        from arrival.export.loader_check import check_loader
        from arrival.sim.collect import collect

        res = collect(twin, reachable, len(reachable), scratch, people=PEOPLE, res=128, seed=a.seed, log=log)
        info = res["info"]
        lc = check_loader(res["dataset"])
        rd = res["report_data"]
        stats["dataset"] = {
            "physically_valid": info["arrival_physically_valid"], "generator": info["arrival_generator"],
            "frames": info["total_frames"], "episodes": info["total_episodes"], "attempts": info["arrival_attempts"],
            "successes": info["arrival_successes"], "fps": info["fps"], "families": info["arrival_families"],
            "cameras": [k.removeprefix("observation.images.") for k in info["features"] if k.startswith("observation.images.")],
            "resolution": [128, 128], "state_dim": info["features"]["observation.state"]["shape"][0],
            "action_dim": info["features"]["action"]["shape"][0], "format": f"LeRobot {info['codebase_version']} (GR00T layout)",
            "success_check": info["arrival_success_check"], "grasp": info["arrival_grasp"],
            "loader_check": {"ok": lc["ok"], "errors": lc["errors"], "warnings": lc["warnings"], "info": lc["info"]},
            "report": {k: rd[k] for k in ("attempts", "successes", "dropped", "seconds", "demos_per_hour",
                                          "human_contacts", "families")},
            "matches_exported_episodes": all(rd["families"].get(f, {}).get("successes") == 1 for f in fam_stats),
        }
        log(f"loader_check ok={lc['ok']} frames={info['total_frames']}")
        ev = evaluate(twin, reachable, 1, seed=a.seed, people=PEOPLE, res=128, out_dir=scratch, log=log)
        for k in [k for k in ev if k == "path" or k.endswith("_path")]:  # local scratch paths, not site data
            ev.pop(k)
        stats["eval"] = ev
        try:
            stats["cost"] = cost_mod.build(scratch, baseline_path=cost_mod.default_baseline(scratch))
        except Exception as e:  # report, do not hide
            stats["cost"] = None
            stats["cost_error"] = f"{type(e).__name__}: {e}"
            log(f"cost report did not run: {stats['cost_error']}")
    (out / "README.md").write_text(README, newline="\n")
    sizes = write_stats(out, stats)
    log(f"total {sum(sizes.values()) / 1e6:.2f} MB in {len(sizes)} files")
    return verify(out)


def write_stats(out: Path, stats: dict) -> dict:
    """Write stats.json with `files` listing every file under `out`, stats.json included. Its own size depends on the
    listing, so it is rewritten until the listing stops changing (two or three passes)."""
    stats["files"] = {}
    while True:
        dump(out / "stats.json", rnd(stats, 4), indent=1)
        sizes = {p.relative_to(out).as_posix(): p.stat().st_size for p in sorted(out.rglob("*")) if p.is_file()}
        if sizes == stats["files"]:
            return sizes
        stats["files"] = sizes


def verify(out: Path) -> int:
    """Load every JSON back and cross-check names and lengths."""
    rooms = json.loads((out / "rooms.json").read_text())
    geoms = json.loads((out / "geoms.json").read_text())
    stats = json.loads((out / "stats.json").read_text())
    meta = json.loads((out / "scan_points.json").read_text())
    assert (out / "scan_points.bin").stat().st_size == meta["count"] * meta["bytes_per_point"]
    assert rooms["fixtures"] and rooms["walls"] and rooms["object_catalogue"]
    for fam, fs in stats["families"].items():
        ep = json.loads((out / fs["episode_file"]).read_text())
        dyn, n = ep["nodes"]["dynamic"], ep["n_frames"]
        assert len(ep["frames"]) == n and all(len(r) == 7 * len(dyn) for r in ep["frames"]), fam
        assert len(ep["joint_states"]) == n and all(len(r) == 8 for r in ep["joint_states"]), fam
        known = set(geoms["bodies"]) | set(ep["person_geoms"]) | {f"cam:{c}" for c in geoms["cameras"]}
        for name in dyn + list(ep["nodes"]["static"]):
            assert name in known, f"{fam}: node {name} has no geoms"
        for name, p in ep["nodes"]["static"].items():
            assert len(p) == 7
        q = np.array(ep["frames"]).reshape(n, len(dyn), 7)[..., 3:]
        assert np.allclose(np.linalg.norm(q, axis=-1), 1, atol=2e-3), fam
        assert ep["success"] is True
        for cam in ("ego_view", "wrist"):
            for lab in ("mid", "end"):
                assert (out / "frames" / f"{fam}_{cam}_{lab}.png").is_file(), (fam, cam, lab)
    for p in out.rglob("*.json"):  # site/data is public: no machine paths
        text = p.read_text()
        assert str(Path.home()) not in text and str(out.resolve().parent) not in text, f"{p.name} holds a local path"
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    assert total < 12e6, f"site/data is {total / 1e6:.1f} MB"
    assert not list(out.rglob("*.mp4"))
    print(f"verify ok: {len(stats['families'])} episodes, {total / 1e6:.2f} MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
