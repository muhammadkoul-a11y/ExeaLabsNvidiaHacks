"""Run sim episodes at 20 Hz, record real joint states / commanded targets / camera frames,
check success with physics, and write a LeRobot v2 dataset."""

from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np

from arrival.sim import menagerie, robot
from arrival.sim.people import Walkers, frame
from arrival.sim.safety import SafetyMonitor
from arrival.catalog.objects import HANDLE_HELD, RIM_HELD, THIN_HELD
from arrival.sim.export import STATE_DIM
from arrival.sim.preflight import DRAWER_OPEN_PUT, base_pose, corners, drive_spot, front, mobile_drive, place_point, rack_spot
from arrival.sim.scene import BOWL_WALL, DOOR_FLAT, PLATE_FLOOR, PLATE_RIM, RACK_OUT, build_mjcf
from arrival.twin.compile import is_counter_host, to_local

FPS = 20
SUBSTEPS = 25  # 0.002 s physics -> 20 Hz control
SETTLE_STEPS = 12
DRAWER_OPEN = 0.30  # start state (m) for close_drawer; open_drawer counts > 0.15 as open
HANDLE_ROOT = 0.017  # grasp point out from a pan's ring: pad half-width 0.01 plus 7 mm
PAD_BELOW_EEF = 0.023  # the finger pads reach this far below the eef site (robot.py hand geometry)
PAD_CLEAR = 0.003  # pad tips over the counter when pinching a flat object: lower and they drag on the top
DOOR_OPEN = np.radians(70)  # start state for close_door; open_door counts > 60 deg as open
FRIDGE_OPEN = np.radians(60)  # open_fridge counts the door open past this, the same bar as open_door
SKIP_REASONS_KEPT = 50  # reasons kept in the report and the dataset info
WIPE_MIN = 0.4  # metres the sponge must travel pressed to the counter for wipe_counter to count
FLAP_OPEN = 1.0  # rad; open_oven / open_dishwasher count a bottom-hinged door past this as open
FAUCET_ON = 0.6  # rad; the lever lifted past this is water on (range is 0 to 1.2)


_front, _corners = front, corners  # the stand-rule helpers live in preflight; the old names stay importable


BASE_MODES = ("fixed", "mobile")


DOOR_KEEP = 0.3  # m past the panel's width that walkers keep from the hinge (the hand pulls the free edge)


class SimEnv:
    def __init__(self, twin: dict, task: dict, res: int = 128, cameras=("ego_view", "wrist"), overview: bool = False,
                 base: str = "fixed", arm: str = "procedural"):
        if base not in BASE_MODES:
            raise ValueError(f"base must be one of {BASE_MODES}, got {base!r}")
        self.twin, self.task, self.mobile = twin, task, base == "mobile"
        self.arm = arm
        self.finger_open = menagerie.FINGER_OPEN if arm == "menagerie" else robot.FINGER_OPEN  # per finger, m
        self.hand = menagerie.HAND if arm == "menagerie" else robot.HAND
        self.fx = {f["id"]: f for f in twin["fixtures"]}
        self.objs = {o["id"]: o for o in twin["objects"]}
        obj_pos = {}
        if "object_id" in task:
            o = self.objs[task["object_id"]]
            src = self.fx[task["source"]]
            obj_pos[o["id"]] = [task["start"][0], task["start"][1], self._top(src) + o["size"][2] / 2 + 0.003]
        if task["family"] == "unload_dishwasher":  # the object starts in the pulled-out rack
            xy, z = self._rack_spot(task)
            obj_pos[task["object_id"]] = [xy[0], xy[1], z + self.objs[task["object_id"]]["size"][2] / 2 + 0.003]
        base, yaw = self._base_pose(task, obj_pos)
        self.base0 = np.array([base[0], base[1], np.radians(yaw)])  # parked pose, room frame; base joints are zero here
        xml = build_mjcf(twin, base, yaw, obj_pos, task.get("lighting", "neutral"), task.get("people", 0),
                         seed=task.get("seed"), people_seed=task.get("seed", 0), base="mobile" if self.mobile else "fixed", arm=arm)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.qadr = [int(self.model.joint(j).qposadr[0]) for j in robot.ARM_JOINTS]
        self.dofadr = [int(self.model.joint(j).dofadr[0]) for j in robot.ARM_JOINTS]
        self.fadr = [int(self.model.joint(f"finger_{s}").qposadr[0]) for s in ("l", "r")]
        self.badr = [int(self.model.joint(j).qposadr[0]) for j in robot.BASE_JOINTS] if self.mobile else []
        self.bact = [int(self.model.actuator(a).id) for a in ("b_x", "b_y", "b_yaw")] if self.mobile else []
        self.data.qpos[self.qadr] = robot.HOME
        self.data.qpos[self.fadr] = self.finger_open
        if task["family"] in ("close_drawer", "close_door", "put_in_drawer"):  # the part starts open
            jn = f'{task["fixture"]}_door' if task["family"] == "close_door" else f'{task["fixture"]}_drawer0'
            start = {"close_drawer": DRAWER_OPEN, "close_door": DOOR_OPEN, "put_in_drawer": DRAWER_OPEN_PUT}
            self.data.qpos[int(self.model.joint(jn).qposadr[0])] = start[task["family"]]
        if task["family"] in ("load_dishwasher", "unload_dishwasher"):  # door flat, upper rack rolled out over it
            self.data.qpos[int(self.model.joint(f'{task["fixture"]}_door').qposadr[0])] = DOOR_FLAT
            self.data.qpos[int(self.model.joint(f'{task["fixture"]}_rack').qposadr[0])] = RACK_OUT
        self.data.ctrl[:9] = list(robot.HOME) + [self.finger_open] * 2  # base actuators, if any, stay at zero
        self.wiped = 0.0  # xy distance the sponge travelled while pressed to the counter (wipe_counter)
        self.people_note = None
        self.walkers = None
        self.drive_to = None  # room-frame base pose the expert drives to after the grasp (mobile counter_to_counter)
        if mobile_drive(task, self.mobile):
            self.drive_to = np.array([*drive_spot(twin, task)[:2], self.base0[2]])
        if task.get("people"):
            free, bxy, front, self.people_note = self._free_floor(self.anchor, base)
            to = None if self.drive_to is None else frame(self.anchor.get("yaw", 0.0)) @ self.drive_to[:2]
            self.walkers = Walkers(task["people"], free, seed=task.get("seed", 0), base_xy=bxy, front_y=front,
                                   fixtures=twin["fixtures"], yaw=self.anchor.get("yaw", 0.0), base_to=to,
                                   keep=self._door_keepout(task))
        self.safety = SafetyMonitor(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.cameras, self.res = cameras, res
        self.renderer = mujoco.Renderer(self.model, res, res)
        self.big = mujoco.Renderer(self.model, 360, 480) if overview else None
        try:
            for _ in range(SETTLE_STEPS):
                self.step(np.array(list(robot.HOME) + [self.finger_open]))
        except BaseException:
            self.close()
            raise

    # ---- geometry helpers ----
    @staticmethod
    def _top(f: dict) -> float:
        return f["position"][2] + f["size"][2] / 2

    def _door_keepout(self, task: dict) -> list:
        """The disc the robot's door swings through, for the walkers to stay out of: centred on the hinge, with the
        panel's width plus 0.3 m as radius. Nothing for a task without a hinged door."""
        name = f'{task.get("fixture")}_door'
        try:
            b = self.model.body(name).id
        except KeyError:
            return []
        mujoco.mj_forward(self.model, self.data)
        w = max((2 * float(max(self.model.geom_size[g][:2])) for g in range(self.model.ngeom)
                 if self.model.geom_bodyid[g] == b and self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX), default=0.0)
        return [(self.data.xpos[b][:2].copy(), w + DOOR_KEEP)] if w else []

    def _free_floor(self, anchor: dict, base):
        """Floor for the walkers, in the frame where `anchor` (the fixture the task works at) faces +y: the strip
        in front of its run, from 0.9 m past the run's deepest fixture to 0.3 m short of the facing wall or of a
        run on it, narrowed by runs on the side walls. Returns (floor, base_xy, front_y, note) in that frame.
        A room with no usable strip gets the whole floor and front_y None (no counter stop), and `note` says so."""
        M = frame(anchor.get("yaw", 0.0))
        n = _front(anchor)[:2]
        pts = np.array([M @ np.array(p, dtype=float) for w in self.twin["walls"] for p in (w["start"], w["end"])])
        bx, by = M @ np.array(base[:2])
        x0, x1, y1 = float(pts[:, 0].min()) + 0.4, float(pts[:, 0].max()) - 0.4, float(pts[:, 1].max()) - 0.3
        deep, sides = -np.inf, []
        for f in self.twin["fixtures"]:
            c = np.array([M @ np.array(p) for p in _corners(f)])
            d = float(np.dot(_front(f)[:2], n))
            if d > 0.9:
                deep = max(deep, float(c[:, 1].max()))
            elif d < -0.9:
                y1 = min(y1, float(c[:, 1].min()) - 0.3)
            else:
                sides.append(c)
        y0 = deep + 0.9
        for c in sides:
            if c[:, 1].max() > y0:  # a side-wall run reaching into the strip
                if c[:, 0].max() < bx:
                    x0 = max(x0, float(c[:, 0].max()) + 0.4)
                else:
                    x1 = min(x1, float(c[:, 0].min()) - 0.4)
        if max(x1 - bx, bx - x0) >= 0.8 and x1 - x0 >= 1.0 and y1 - y0 >= 0.6:
            return (x0, y0, x1, y1), (bx, by), deep, None
        whole = (float(pts[:, 0].min()) + 0.4, float(pts[:, 1].min()) + 0.4,
                 float(pts[:, 0].max()) - 0.4, float(pts[:, 1].max()) - 0.4)
        return whole, (bx, by), None, f"no free strip in front of {anchor['id']}: people walk the whole floor, no counter stop"

    def _base_pose(self, task: dict, obj_pos: dict):
        """Base pose in front of the fixture the task works at (kept as `self.anchor`), facing it, on any wall."""
        xy = obj_pos[task["object_id"]][:2] if "object_id" in task else None
        if task["family"] == "unload_dishwasher":
            xy = task["start"][:2]  # the object starts in the rack, but the base stands on the side of the counter spot
        base, yaw, self.anchor = base_pose(self.twin, task, xy, self.mobile, self.arm)
        return base, yaw

    def base_pose(self) -> np.ndarray:
        """Base x, y, yaw in the room frame (the parked pose when the base is fixed)."""
        return self.base0 + self.data.qpos[self.badr] if self.mobile else self.base0.copy()

    def body_pos(self, name: str) -> np.ndarray:
        return self.data.body(name).xpos.copy()

    def site_pos(self, name: str) -> np.ndarray:
        return self.data.site(name).xpos.copy()

    def obj_half_h(self, oid: str) -> float:
        return self.objs[oid]["size"][2] / 2

    def grasp(self, oid: str) -> dict | None:
        """How the expert holds an object it cannot grip by the body: xy offset from the body centre to the grasp
        point, the yaw for ik.down (fingers close across the wall or the handle) and the eef height. A bowl and a
        plate are pinched by the rim nearest the robot, the bowl 2 cm under its lip, the plate with the pads just
        above its bottom disc; a pan by its handle, 1.7 cm out from the ring; a spatula across its width with the pad
        tips 3 mm over the top it lies on. None for objects held by the body."""
        o = self.objs[oid]
        cid, p = o["catalog_id"], self.body_pos(oid)
        radius, sz = min(o["size"][0], o["size"][1]) / 2, o["size"][2]
        if cid in RIM_HELD:
            d = self.data.body("link1").xpos[:2] - p[:2]
            d = d / (np.linalg.norm(d) + 1e-9)
            wall = BOWL_WALL if cid == "bowl" else PLATE_RIM
            z = p[2] + sz / 2 - 0.02 if cid == "bowl" else p[2] - sz / 2 + PLATE_FLOOR + PAD_BELOW_EEF + 0.003
            return {"off": d * (radius - wall / 2), "yaw": float(np.arctan2(d[0], -d[1])), "z": z}
        if cid in HANDLE_HELD:  # the handle runs along +x from the ring (object_xml ignores the twin yaw); the pads
            # sit at its root, the thickest part; the pan tips and hangs from the handle during the carry
            return {"off": np.array([radius + HANDLE_ROOT, 0.0]), "yaw": 0.0, "z": p[2] + sz / 2 - 0.008}
        if cid in THIN_HELD:  # lying flat, too thin to hold from above the body: the pads close across the short
            # side with their tips PAD_CLEAR over the top it lies on (object_xml ignores the twin yaw, so that is x)
            across = 0 if o["size"][0] < o["size"][1] else 1
            return {"off": np.zeros(2), "yaw": np.pi / 2 if across == 0 else 0.0,
                    "z": p[2] - sz / 2 + self.pad_reach() + PAD_CLEAR}
        return None

    def pad_reach(self) -> float:
        """How far the fingers' collision geoms reach past the eef site along the approach axis: 23 mm on the
        procedural fingers, about 10 mm on the Menagerie hand. Read from the model so both arms use one rule."""
        m, d = self.model, self.data
        R, e = d.site("eef").xmat.reshape(3, 3), d.site("eef").xpos
        reach = 0.0
        for i in range(m.ngeom):
            if m.body(m.geom_bodyid[i]).name not in ("finger_l", "finger_r") or not m.geom_contype[i]:
                continue
            c, h = m.geom_aabb[i][:3], m.geom_aabb[i][3:]
            G = d.geom_xmat[i].reshape(3, 3)
            for s in np.array(np.meshgrid(*[(-1, 1)] * 3)).T.reshape(-1, 3):
                reach = max(reach, float((R.T @ (d.geom_xpos[i] + G @ (c + s * h) - e))[2]))
        return reach

    def place_point(self, task: dict) -> np.ndarray:
        return place_point(self.twin, task)

    def pull_axis(self, task: dict) -> np.ndarray:
        if task["family"] == "turn_on_faucet":  # the lever is pushed from in front of the counter run the sink is set in
            return _front(self.anchor)
        return _front(self.fx[task["fixture"]])

    def run_yaw(self) -> float:
        """Yaw of the run the task works at: 0 on the hero kitchen's north wall, so grasp yaws written for it
        (fingers across the counter, hand along it) turn with the fixture on any other wall."""
        return float(self.anchor.get("yaw", 0.0))

    def place_point_top(self, task: dict) -> float:
        return self._top(self.fx[task["source"]])

    def drawer_place_point(self, task: dict) -> np.ndarray:
        """Where to set an object down on the pulled-out part of the open drawer's tray."""
        c = self.fx[task["fixture"]]
        n = _front(c)
        tray = self.data.geom(f'{task["fixture"]}_drawer0_tray')
        half = self.model.geom(f'{task["fixture"]}_drawer0_tray').size
        front = np.array(c["position"][:2]) + n[:2] * (c["size"][1] / 2 - 0.02)  # a point on the carcass face
        along = float(np.dot(tray.xpos[:2] - front, n[:2]))  # tray centre, measured out from that face
        far = along + float(half[1])  # how far the tray reaches past the face
        mid = 0.02 + (far - 0.02) / 2  # middle of the exposed part of the tray
        xy = tray.xpos[:2] + n[:2] * (mid - along)  # keep the drawer's own x; the counter centre is not the drawer
        return np.array([xy[0], xy[1], tray.xpos[2] + half[2] + self.obj_half_h(task["object_id"]) + 0.01])

    def _rack_spot(self, task: dict):
        """(xy, z of an object's base) in the middle of the part of the rack that is out past the dishwasher front,
        computed from the twin (before the model exists)."""
        from arrival.sim.scene import RACK_Z

        return np.array(rack_spot(self.fx[task["fixture"]])), RACK_Z + 0.01

    def rack_place_point(self, task: dict) -> np.ndarray:
        xy, z = self._rack_spot(task)
        return np.array([xy[0], xy[1], z + self.obj_half_h(task["object_id"]) + 0.01])

    def counter_place_point(self, task: dict) -> np.ndarray:
        """Set-down spot on the counter for unload_dishwasher: where the object started in the twin."""
        p = task["start"]
        return np.array([p[0], p[1], self._top(self.fx[task["source"]]) + self.obj_half_h(task["object_id"]) + 0.012])

    def flap_geometry(self, task: dict):
        """Hinge axis point, handle site and the door body rotation for a bottom-hinged door."""
        b = self.data.body(task["joint"])
        return b.xpos.copy(), self.site_pos(task["handle"]), b.xmat.reshape(3, 3).copy()

    def door_geometry(self, task: dict):
        b = self.data.body(f'{task["fixture"]}_door')
        return b.xpos.copy(), self.site_pos(f'{task["fixture"]}_door_handle'), b.xmat.reshape(3, 3).copy()

    # ---- stepping ----
    def state(self) -> np.ndarray:
        q = self.data.qpos
        return np.array(list(q[self.qadr]) + [float(np.mean(q[self.fadr]))] + list(self.base_pose()), dtype=np.float32)

    def step(self, action: np.ndarray) -> None:
        """action: 7 arm targets, gripper opening, and optionally base x, y, yaw in the room frame (held when
        absent or when the base is fixed)."""
        self.data.ctrl[:7] = action[:7]
        self.data.ctrl[7:9] = action[7]
        if self.mobile and len(action) >= STATE_DIM:
            self.data.ctrl[self.bact] = np.asarray(action[8:11], dtype=float) - self.base0
        before = self.body_pos(self.task["object_id"])[:2] if self.task["family"] == "wipe_counter" else None
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self.people_xy = self.walkers.step(self.model, self.data, 1.0 / FPS, self.safety.yield_clearance(self.data)) if self.walkers else []
        if before is not None:
            p = self.body_pos(self.task["object_id"])
            if p[2] < self._top(self.fx[self.task["source"]]) + self.obj_half_h(self.task["object_id"]) + 0.01:
                self.wiped += float(np.linalg.norm(p[:2] - before))
        self.safety.update(self.model, self.data, getattr(self.walkers, "driven_geoms", ()))

    def render(self, cam: str, big: bool = False) -> np.ndarray:
        r = self.big if big else self.renderer
        r.update_scene(self.data, cam)
        return r.render().copy()

    def __enter__(self) -> "SimEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """Free the GL contexts. On Windows, letting an old renderer be collected after a new one exists leaves
        no context current, and every frame from then on is black."""
        for r in (self.renderer, self.big):
            if r is not None:
                r.close()
        self.renderer = self.big = None

    def obs(self) -> dict:
        o = {"state": self.state()}
        for c in self.cameras:
            o[c] = self.render(c)
        return o

    # ---- physics success ----
    def success(self, task: dict) -> tuple[bool, dict]:
        fam = task["family"]
        if fam == "move_on_table":  # resting on the table top within 6 cm of the target spot, nearly still
            oid = task["object_id"]
            p = self.body_pos(oid)
            v = float(np.linalg.norm(self.data.joint(oid).qvel[:3]))
            err = float(np.linalg.norm(p[:2] - np.asarray(task["target_xy"])))
            resting = abs(p[2] - (self._top(self.fx[task["target"]]) + self.obj_half_h(oid))) < 0.025
            return bool(err < 0.06 and resting and v < 0.05), {
                "object_pos": p.round(4).tolist(), "object_speed": round(v, 4), "target_error_m": round(err, 4)}
        if fam in ("counter_to_sink", "counter_to_counter"):
            oid = task["object_id"]
            p = self.body_pos(oid)
            v = float(np.linalg.norm(self.data.joint(oid).qvel[:3]))
            tgt = self.fx[task["target"]]
            (lx, ly), (sx, sy, _) = to_local(p[:2], tgt), tgt["size"]
            inside = abs(lx) < sx / 2 - 0.01 and abs(ly) < sy / 2 - 0.01
            if tgt["type"] == "sink":
                top = max(self._top(f) for f in self.twin["fixtures"] if is_counter_host(f))
                ok = inside and p[2] < top and v < 0.05
            else:
                ok = inside and abs(p[2] - (self._top(tgt) + self.obj_half_h(oid))) < 0.025 and v < 0.05
            return bool(ok), {"object_pos": p.round(4).tolist(), "object_speed": round(v, 4)}
        if fam == "put_in_drawer":
            oid = task["object_id"]
            p = self.body_pos(oid)
            v = float(np.linalg.norm(self.data.joint(oid).qvel[:3]))
            tray = self.data.geom(f'{task["fixture"]}_drawer0_tray')
            half = self.model.geom(f'{task["fixture"]}_drawer0_tray').size
            local = tray.xmat.reshape(3, 3).T @ (p - tray.xpos)
            inside = abs(local[0]) < half[0] - 0.02 and abs(local[1]) < half[1] - 0.02
            resting = abs(local[2] - (half[2] + self.obj_half_h(oid))) < 0.03
            return bool(inside and resting and v < 0.05), {
                "object_pos": p.round(4).tolist(), "object_speed": round(v, 4),
                "drawer_qpos": round(float(self.data.joint(task["joint"]).qpos[0]), 4)}
        if fam == "wipe_counter":
            oid = task["object_id"]
            p = self.body_pos(oid)
            v = float(np.linalg.norm(self.data.joint(oid).qvel[:3]))
            src = self.fx[task["source"]]
            (lx, ly), (sx, sy, _) = to_local(p[:2], src), src["size"]
            on_top = (abs(lx) < sx / 2 and abs(ly) < sy / 2
                      and abs(p[2] - (self._top(src) + self.obj_half_h(oid))) < 0.025)
            return bool(on_top and self.wiped >= WIPE_MIN and v < 0.05), {
                "object_pos": p.round(4).tolist(), "wiped_m": round(self.wiped, 3)}
        if fam == "load_dishwasher":
            oid = task["object_id"]
            p = self.body_pos(oid)
            v = float(np.linalg.norm(self.data.joint(oid).qvel[:3]))
            tray = self.data.geom(f'{task["fixture"]}_rack_tray')
            half = self.model.geom(f'{task["fixture"]}_rack_tray').size
            local = tray.xmat.reshape(3, 3).T @ (p - tray.xpos)
            inside = abs(local[0]) < half[0] - 0.02 and abs(local[1]) < half[1] - 0.02
            resting = abs(local[2] - (half[2] + self.obj_half_h(oid))) < 0.03
            return bool(inside and resting and v < 0.05), {"object_pos": p.round(4).tolist(), "object_speed": round(v, 4)}
        if fam == "unload_dishwasher":
            oid = task["object_id"]
            p = self.body_pos(oid)
            v = float(np.linalg.norm(self.data.joint(oid).qvel[:3]))
            c = self.fx[task["source"]]
            (lx, ly), (sx, sy, _) = to_local(p[:2], c), c["size"]
            inside = abs(lx) < sx / 2 - 0.01 and abs(ly) < sy / 2 - 0.01
            ok = inside and abs(p[2] - (self._top(c) + self.obj_half_h(oid))) < 0.025 and v < 0.05
            return bool(ok), {"object_pos": p.round(4).tolist(), "object_speed": round(v, 4)}
        jn = task["joint"]
        q = float(self.data.joint(jn).qpos[0])
        ok = {"open_drawer": q > 0.15, "open_door": q > np.radians(60), "open_fridge": q > FRIDGE_OPEN,
              "close_drawer": q < 0.03, "close_door": q < np.radians(5),
              "open_oven": q > FLAP_OPEN, "open_dishwasher": q > FLAP_OPEN, "turn_on_faucet": q > FAUCET_ON}[fam]
        return bool(ok), {"joint": jn, "qpos": round(q, 4)}


def robot_drawer_z() -> float:
    from arrival.sim.scene import CARCASS_H

    return CARCASS_H - 0.01


EXEC_NOISE_TAU = 10.0  # steps (0.5 s): how long a joint-noise excursion lasts
EXEC_NOISE_SPREAD = (0.4, 1.6)  # each episode draws its noise level from this range times --dart, for a spread of off-plan states
EXEC_NOISE_NEAR_GRIP = 0.25  # noise scale within 12 steps of a gripper command: a grasp needs a steady arm
GRIP_WINDOW = 12


def _grip_events(plan) -> np.ndarray:
    """Steps at which a scripted plan opens or closes the gripper (empty for a policy without a plan)."""
    if not plan:
        return np.zeros(0, dtype=int)
    return np.where(np.abs(np.diff([float(p[7]) for p in plan])) > 1e-6)[0]


def run_episode(env: SimEnv, policy, task: dict, max_steps: int = 400, preview: list | None = None,
                exec_noise: float = 0.0) -> dict:
    """exec_noise > 0 (rad, the typical std of smooth noise on the 7 arm joint targets; each episode uses 0.4 to 1.6
    times it) is DART: the arm is driven by
    target + noise, but the dataset keeps the clean target as the action, so a trained policy sees states that are
    slightly off the plan, each labelled with the move back onto it. The noise is seeded from the task, shrinks
    near gripper commands, and the physics check still has to pass, so a perturbation that spoils a grasp drops
    the episode like any other failure."""
    rng = np.random.default_rng(int(task.get("seed", 0)) + 1) if exec_noise > 0 else None
    level = exec_noise * rng.uniform(EXEC_NOISE_SPREAD[0], EXEC_NOISE_SPREAD[1]) if rng is not None else 0.0  # per episode
    rho, drift, events = float(np.exp(-1.0 / EXEC_NOISE_TAU)), np.zeros(7), None
    try:
        policy.reset(env, task)
        events = _grip_events(getattr(policy, "plan", None)) if exec_noise > 0 else None
        env.safety.reset()
        states, actions, frames, people = [], [], {c: [] for c in env.cameras}, []
        tail = 0
        max_steps = max(max_steps, len(getattr(policy, "plan", ())) + 100)  # a base drive, plus waits for people
        for t in range(max_steps):
            obs = env.obs()
            a = np.asarray(policy.act(obs), dtype=float)
            if len(a) < STATE_DIM:  # arm-only policies: the base holds where it is
                a = np.concatenate([a, obs["state"][len(a):STATE_DIM]])
            states.append(obs["state"])
            actions.append(a.astype(np.float32))
            for c in env.cameras:
                frames[c].append(obs[c])
            if preview is not None:
                preview.append(env.render("overview", big=True))
            if rng is not None:
                near = len(events) > 0 and np.min(np.abs(events - t)) < GRIP_WINDOW
                drift = rho * drift + np.sqrt(1 - rho * rho) * level * (EXEC_NOISE_NEAR_GRIP if near else 1.0) * rng.standard_normal(7)
                driven = a.copy()
                driven[:7] += drift
                env.step(driven)
            else:
                env.step(a)
            people.append(env.people_xy)
            if getattr(policy, "done", lambda: False)():
                tail += 1
                if tail > 10:
                    break
        ok, detail = env.success(task)
    finally:
        env.close()  # idempotent; frees both GL renderers when the policy, the step or the success check raises
    return {"states": np.array(states), "actions": np.array(actions), "videos": {c: np.array(v) for c, v in frames.items()},
            "success": ok, "detail": detail, "people_xy": people, "ik_error": getattr(policy, "ik_error", None),
            "safety": env.safety.summary(),
            "fridge_visit": bool(getattr(env.walkers, "visit", None) and getattr(env.walkers, "door_max", 0.0) > 0.0)}


def collect(twin: dict, families: list[str], episodes: int, out_dir: str | Path = "out", people: int = 2,
            policy_name: str = "scripted", res: int = 128, seed: int = 0, save_preview: bool = False,
            planner_name: str = "rule", cache_path: str | Path | None = None,
            split: str = "train", use_generated_instructions: bool = False, base: str = "fixed",
            shard: tuple[int, int] | None = None, resume: bool = False, arm: str = "procedural",
            exec_noise: float = 0.0, exocentric: bool = False, log=print) -> dict:
    """Collect `episodes` attempts; failed attempts are dropped and counted. A task whose stand spot or reach the
    pre-flight rejects never runs (`skipped_unreachable`, counted per rejected variation), and one whose env build
    or rollout raises is logged with its reason, counted as `skipped_error` and never written, so a bad geometry
    costs a task, not the run. base="mobile" gives the arm a driven base (MAN-3), which also admits the
    counter_to_counter variations a fixed base cannot reach. arm="menagerie" swaps in the real Franka asset
    (`python scripts/fetch_menagerie.py`); the dataset and report say which arm made them.
    shard=(i, n) runs only block i of n of the `episodes` tasks (arrival/jobs/shards.py) and writes under
    <out_dir>/shards/<i>of<n>/, so n processes cover the run with no overlap; `arrival merge-shards` joins them.
    exocentric=True adds the existing fixed overview camera to the exported observation at every control tick.
    With save_preview=True it also writes a labeled ego/exo pair from those same dataset frames.
    Every run journals each finished task; resume=True continues from that journal (a different run's journal is
    refused with a ShardError) and the result is the dataset an uninterrupted run would have written."""
    from arrival.jobs.shards import Journal, shard_dir, shard_range
    from arrival.sim.export import SimDatasetWriter
    from arrival.sim.policy import ScriptedExpert, VLAClient, VLANotConfigured, VLMPlanner
    from arrival.sim.tasks import finish_task, make_tasks

    cameras = ("ego_view", "wrist", "overview") if exocentric else ("ego_view", "wrist")
    out_dir = Path(out_dir)
    room = twin["room_id"]
    pre: dict = {}
    tasks = make_tasks(twin, families, episodes, seed=seed, people=people,
                       split=split, use_generated_instructions=use_generated_instructions, mobile=base == "mobile", log=log,
                       preflight=pre, arm=arm)
    lo, hi = shard_range(len(tasks), *shard) if shard else (0, len(tasks))
    if shard:
        out_dir = shard_dir(out_dir, *shard)
    skips = [{"stage": "preflight", **r} for r in pre["reasons"]]  # what never ran and why
    errors = 0
    for t in tasks[lo:hi]:
        finish_task(t)
    policy = ScriptedExpert() if policy_name == "scripted" else VLAClient()
    planner = VLMPlanner(log=log, cache_path=cache_path) if planner_name == "vlm" else None
    from arrival.export.provenance import run_record

    prov = run_record(twin, seed=seed, generator="mujoco_scripted_ik", policy=policy_name, planner=planner_name,
                      episodes=episodes, families=list(families), people=people, res=res, split=split,
                      exec_noise=exec_noise, cameras=list(cameras))
    fingerprint = {"room": room, "families": list(families), "episodes": episodes, "seed": seed, "people": people,
                   "policy": policy_name, "planner": planner_name, "split": split,
                   "instructions": use_generated_instructions, "base": base, "arm": arm, "res": res,
                   "shard": list(shard) if shard else None, "exec_noise": exec_noise, "cameras": list(cameras)}
    journal = Journal(out_dir / "progress" / f"{room}_sim.jsonl")
    done = journal.start(fingerprint, resume, tasks[lo:hi])
    writer = SimDatasetWriter(out_dir / "datasets" / f"{room}_sim", room, FPS, res, cameras, provenance=prov,
                              base=base, resume=bool(done), arm=arm)
    per: dict = {}
    prior_seconds = 0.0
    if done:
        writer.restore([r["episode"] for r in done if r["outcome"] == "ok"], [r["stats"] for r in done if r["outcome"] == "ok"])
        for r in done:
            prior_seconds += r["seconds"]
            if r["outcome"] == "skip":
                errors += 1
                skips.append({"stage": r["stage"], "task_id": r["task_id"], "family": r["family"], "reason": r["reason"]})
                continue
            fam = per.setdefault(r["family"], {"attempts": 0, "successes": 0, "human_contacts": 0,
                                               "fridge_visits": 0, "door_contacts": 0, "seconds": 0.0})
            fam["attempts"] += 1
            fam["seconds"] += r["seconds"]
            fam["successes"] += r["outcome"] == "ok"
            fam["human_contacts"] += r["human_contact"]
            fam["fridge_visits"] += r["fridge_visit"]
            fam["door_contacts"] += r["door_contact"]
        log(f"[sim] resume: {len(done)} of {hi - lo} tasks already done, {len(writer.episodes)} episodes kept")
    finished = {r["task_id"] for r in done}
    t0 = time.perf_counter()
    preview = [] if save_preview else None
    for i, task in enumerate(tasks[lo:hi]):
        if task["task_id"] in finished:
            continue
        t_task = time.perf_counter()
        if planner is not None:
            obj_ids = [o["id"] for o in twin.get("objects", [])]
            fx_ids = [f["id"] for f in twin.get("fixtures", [])]
            research_ctx = None
            if "tavily_research" in twin:
                target_fx = task.get("target") or task.get("fixture") or ""
                target_type = next((f.get("type", "") for f in twin.get("fixtures", []) if f.get("id") == target_fx), "")
                if target_type and target_type in twin["tavily_research"]:
                    r_info = twin["tavily_research"][target_type]
                    research_ctx = f"{r_info.get('summary', '')} {' '.join(r_info.get('safety_guidelines', []))}"
            plan = planner.plan(task["instruction"], obj_ids, fx_ids, research_context=research_ctx)
            task["vlm_plan"] = plan
            task["planned_by"] = planner.source
            if research_ctx:
                task["tavily_enriched"] = True
            # the planner only refines a task's own fields: its family, and the pick/place fields other families lack,
            # stay as make_tasks set them (a rewritten family would miss its source/target/fixture or flip open/close)
            if plan.get("family") and plan["family"] != task["family"]:
                log(f"[sim] task {task['task_id']}: planner family {plan['family']} != {task['family']}; keeping the task's")
            if plan.get("object") and plan["object"] in obj_ids and "object_id" in task:
                task["object_id"] = plan["object"]
            if plan.get("target") and plan["target"] in fx_ids and "target" in task:
                task["target"] = plan["target"]
            log(f"[sim] task {task['task_id']}: planned by {planner.source} -> {plan}")
        stage = "build"
        try:
            env = SimEnv(twin, task, res=res, cameras=cameras, overview=save_preview and i == 0, base=base, arm=arm)
            stage = "rollout"
            try:
                if env.people_note:
                    log(f"[sim] {task['task_id']}: {env.people_note}")
                ep = run_episode(env, policy, task, preview=preview if i == 0 else None, exec_noise=exec_noise)
            finally:
                env.close()  # idempotent; frees both GL renderers when the policy or the episode raises
        except VLANotConfigured:
            raise
        except Exception as e:  # noqa: BLE001 - one bad geometry must cost this task, not the run; nothing is written for it
            if isinstance(e, OSError) and policy_name == "vla":  # the policy server did not answer: not this task's fault
                raise
            errors += 1
            skips.append({"stage": stage, "task_id": task["task_id"], "family": task["family"],
                          "reason": one_line(e)})
            log(f"[sim] {task['task_id']}: SKIP at {stage}: {skips[-1]['reason']}")
            journal.append({"task_id": task["task_id"], "outcome": "skip", "family": task["family"], "stage": stage,
                            "reason": skips[-1]["reason"], "seconds": round(time.perf_counter() - t_task, 3)})
            continue
        fam = per.setdefault(task["family"], {"attempts": 0, "successes": 0, "human_contacts": 0,
                                              "fridge_visits": 0, "door_contacts": 0, "seconds": 0.0})
        fam["attempts"] += 1
        contact = ep["safety"]["human_contact"]
        fam["human_contacts"] += contact
        fam["fridge_visits"] += ep["fridge_visit"]
        fam["door_contacts"] += ep["safety"]["door_contact"]
        tag = "DROP human_contact" if contact else ("OK " if ep["success"] else "MISS")
        log(f"[sim] {task['task_id']}: {tag} {ep['detail']} ik_err={ep['ik_error'] if ep['ik_error'] is None else round(ep['ik_error'], 4)} "
            f"min_human_d={ep['safety']['min_human_distance_m']}" + (" fridge_visit" if ep["fridge_visit"] else ""))
        kept = ep["success"] and not contact
        if kept:
            fam["successes"] += 1
            writer.add_episode(ep, task)
        task_seconds = round(time.perf_counter() - t_task, 3)
        fam["seconds"] += task_seconds
        journal.append({"task_id": task["task_id"], "outcome": "ok" if kept else ("contact" if contact else "miss"),
                        "family": task["family"], "human_contact": contact, "fridge_visit": int(ep["fridge_visit"]),
                        "door_contact": ep["safety"]["door_contact"], "seconds": task_seconds,
                        **({"episode": writer.episodes[-1], "stats": writer.stats[-1]} if kept else {})})  # after the files
        if save_preview and i == 0:
            _write_preview(out_dir / "previews", room, preview, ep)
            preview = None  # up to ~200 MB of overview frames; not needed past the first episode
    seconds = prior_seconds + time.perf_counter() - t0
    attempts = sum(f["attempts"] for f in per.values())  # tasks that ran to the physics check; a skipped one is not an attempt
    skipped = {"skipped_unreachable": pre["rejected"], "skipped_error": errors,
               "skipped": pre["rejected"] + errors, "skip_reasons": skips[:SKIP_REASONS_KEPT]}
    shard_info = ({"index": shard[0], "count": shard[1], "task_range": [lo, hi], "total_tasks": len(tasks),
                   "episodes": episodes, "seed": seed, "families": list(families), "people": people,
                   "policy": policy_name, "planner": planner_name, "split": split,
                   "instructions": use_generated_instructions, "base": base, "arm": arm, "res": res}) if shard else None
    extra = {"shard": shard_info} if shard else {}
    if exec_noise > 0:  # DART: say so in info.json, next to arrival_generator, not only in the provenance block
        extra["exec_noise_rad"] = exec_noise
    info = writer.finish(attempts, skipped, extra or None)
    succ = sum(f["successes"] for f in per.values())
    report = {"room_id": room, "generator": "mujoco_scripted_ik", "policy": policy_name,
              "planner": planner_name, "attempts": attempts, **skipped,
              "successes": succ, "dropped": attempts - succ, "seconds": round(seconds, 2),
              "demos_per_hour": round(succ / seconds * 3600, 1) if seconds else 0.0,
              "human_contacts": sum(f["human_contacts"] for f in per.values()),
              "fridge_visits": sum(f["fridge_visits"] for f in per.values()),
              "door_contacts": sum(f["door_contacts"] for f in per.values()),
              "families": {k: {**v, "seconds": round(v["seconds"], 2), "success_rate": round(v["successes"] / v["attempts"], 3)}
                           for k, v in per.items()},
              "people": people, "resolution": res, "physics_checked_success": True, "grasp": "friction",
              "base": base, "arm": arm, "cameras": list(cameras), **({"shard": shard_info} if shard else {}),
              **({"resumed_tasks": len(done)} if done else {})}
    rp = out_dir / "reports" / f"{room}_sim.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))
    return {"dataset": writer.root, "report": rp, "report_data": report, "info": info}


def one_line(e: BaseException) -> str:
    """An exception as one line: its type, message and the innermost frame it came from."""
    import traceback

    frames = traceback.extract_tb(e.__traceback__)
    at = f" (at {Path(frames[-1].filename).name}:{frames[-1].lineno})" if frames else ""
    return f"{type(e).__name__}: {e}{at}"[:300]


def _write_preview(d: Path, room: str, frames: list, ep: dict) -> None:
    from PIL import Image

    from arrival.export.lerobot import write_video

    d.mkdir(parents=True, exist_ok=True)
    write_video(d / f"{room}_sim_overview.mp4", frames, FPS)
    picks = [frames[int(k)] for k in np.linspace(0, len(frames) - 1, 6)]
    ego = ep["videos"]["ego_view"]
    egos = [np.array(Image.fromarray(ego[int(k)]).resize((480, 360))) for k in np.linspace(0, len(ego) - 1, 6)]
    sheet = np.concatenate([np.concatenate(picks[:3], 1), np.concatenate(picks[3:], 1),
                            np.concatenate(egos[:3], 1), np.concatenate(egos[3:], 1)], 0)
    Image.fromarray(sheet).save(d / f"{room}_sim_contact.png")
    if "overview" in ep["videos"]:
        from arrival.render.multiview import write_paired_preview

        write_paired_preview(d / f"{room}_sim_ego_exo", ep["videos"]["ego_view"], ep["videos"]["overview"], FPS)
