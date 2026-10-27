"""Object-centric SE(3) retargeting of executed counter_to_sink demonstrations.

This native Arrival implementation does not call the upstream MimicGen framework. S20's translation-only
seed replay stays separate. Only MuJoCo rollouts that pass the sink oracle without human contact are exported.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from arrival.export.provenance import canonical_sha256, run_record
from arrival.sim import ik
from arrival.sim.collect import FPS, SimEnv, run_episode
from arrival.sim.export import SimDatasetWriter
from arrival.sim.policy import Policy, ScriptedExpert
from arrival.sim.tasks import finish_task, make_tasks

GENERATOR = "object_centric_se3_retarget"
POSITION_TOL = 0.02  # m, reject an IK plan before rolling out if any sample exceeds this
ROTATION_TOL = 0.15  # rad; the existing IK reports position error only
BRIDGE_STEPS = 20  # one second at 20 Hz, zero velocity/acceleration at each endpoint


def pose(position, rotation) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rotation, position
    return t


def rigid(value) -> np.ndarray:
    """Validate a finite homogeneous SE(3) matrix, including proper (not reflected) rotation."""
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all() or not np.allclose(t[3], [0, 0, 0, 1])
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1.0, atol=1e-6)):
        raise ValueError("expected a finite rigid SE(3) transform")
    return t


def inverse(t) -> np.ndarray:
    t = rigid(t)
    r = t[:3, :3].T
    return pose(-r @ t[:3, 3], r)


def sink_frame(env, task) -> np.ndarray:
    sink = env.fx[task["target"]]
    if task["family"] != "counter_to_sink" or sink["type"] != "sink":
        raise ValueError("SE(3) retargeting supports counter_to_sink with a sink target only")
    c, s = np.cos(sink.get("yaw", 0.0)), np.sin(sink.get("yaw", 0.0))
    return pose(sink["position"], [[c, -s, 0], [s, c, 0], [0, 0, 1]])


def object_frame(env, task) -> np.ndarray:
    b = env.data.body(task["object_id"])
    return pose(b.xpos.copy(), b.xmat.reshape(3, 3).copy())


def eef_frame(env) -> np.ndarray:
    site = env.data.site("eef")
    return pose(site.xpos.copy(), site.xmat.reshape(3, 3).copy())


def interpolate(a, b, steps=BRIDGE_STEPS) -> list[np.ndarray]:
    """Minimum-jerk translation + shortest-arc quaternion SLERP; excludes a, includes b."""
    a, b = rigid(a), rigid(b)
    if steps < 1:
        raise ValueError("bridge steps must be positive")
    qa, qb = np.empty(4), np.empty(4)
    mujoco.mju_mat2Quat(qa, a[:3, :3].flatten())
    mujoco.mju_mat2Quat(qb, b[:3, :3].flatten())
    if qa @ qb < 0:
        qb = -qb
    dot = float(np.clip(qa @ qb, -1.0, 1.0))
    angle = np.arccos(dot)
    frames = []
    for k in range(1, steps + 1):
        u = k / steps
        h = u ** 3 * (10 - 15 * u + 6 * u ** 2)
        q = ((1 - h) * qa + h * qb if dot > 0.9995 else
             (np.sin((1 - h) * angle) * qa + np.sin(h * angle) * qb) / np.sin(angle))
        r = np.empty(9)
        mujoco.mju_quat2Mat(r, q / np.linalg.norm(q))
        frames.append(pose(a[:3, 3] + h * (b[:3, 3] - a[:3, 3]), r.reshape(3, 3)))
    return frames


class _Recorder(ScriptedExpert):
    def reset(self, env, task):
        super().reset(env, task)
        self.samples = []
        self.task = task

    def act(self, obs):
        action = super().act(obs)
        self.samples.append((eef_frame(self.env), float(action[7]),
                             float(self.env.body_pos(self.task["object_id"])[2])))
        return action


def record_source(env, task: dict) -> dict:
    """Record actual pre-action eef poses and commanded gripper openings during a verified physics demo.

    The object anchor covers approach, grasp and lift. The sink anchor covers transport, place and retreat.
    The switch is the first closed-gripper sample with the object lifted at least 15 cm off its start.
    """
    target, obj = sink_frame(env, task), object_frame(env, task)
    recorder = _Recorder()
    episode = run_episode(env, recorder, task)
    if not episode["success"] or episode["safety"]["human_contact"]:
        raise ValueError("source demonstration failed physics or had human contact")
    split = next((i for i, (_, g, z) in enumerate(recorder.samples)
                  if g < env.finger_open / 2 and z > obj[2, 3] + 0.15), None)
    if split is None or split == 0:
        raise ValueError("source demonstration has no closed-gripper lift")
    samples = []
    for i, (eef, g, _) in enumerate(recorder.samples):
        segment = "object" if i < split else "sink"
        anchor = obj if segment == "object" else target
        samples.append({"segment": segment, "eef_relative": (inverse(anchor) @ eef).tolist(), "gripper": g})
    source = {"schema_version": 1, "family": "counter_to_sink", "task": dict(task),
              "source_generator": "mujoco_scripted_ik", "source_seed": task.get("seed"),
              "provenance": run_record(env.twin, seed=task.get("seed"), generator="mujoco_scripted_ik"),
              "object_spec": {k: env.objs[task["object_id"]][k] for k in ("catalog_id", "size")},
              "object_frame": obj.tolist(), "sink_frame": target.tolist(), "split": split, "fps": FPS,
              "samples": samples, "states": episode["states"].tolist(), "actions": episode["actions"].tolist(),
              "success_detail": episode["detail"], "safety": episode["safety"]}
    source["source_id"] = canonical_sha256(source)
    return source


def validate_source(source):
    if (not isinstance(source, dict) or source.get("schema_version") != 1
            or source.get("family") != "counter_to_sink" or source.get("fps") != FPS):
        raise ValueError("source must be a counter_to_sink SE(3) recording at 20 Hz")
    rigid(source["object_frame"])
    rigid(source["sink_frame"])
    if source.get("source_id") != canonical_sha256({k: v for k, v in source.items() if k != "source_id"}):
        raise ValueError("source recording hash mismatch")
    samples = source["samples"]
    split = source["split"]
    if not 0 < split < len(samples):
        raise ValueError("source needs object and sink segments")
    for i, sample in enumerate(samples):
        rigid(sample["eef_relative"])
        if (sample["segment"] != ("object" if i < split else "sink")
                or not np.isfinite(sample["gripper"]) or not 0 <= sample["gripper"] <= 0.05):
            raise ValueError("invalid source segment or gripper opening")


def retarget_trajectory(samples, anchors, initial):
    """Re-anchor each recorded pose and bridge the object-to-sink handoff while holding the grasp."""
    targets, previous, segment = [], rigid(initial), None
    anchors = {name: rigid(value) for name, value in anchors.items()}
    for sample in samples:
        t = anchors[sample["segment"]] @ rigid(sample["eef_relative"])
        if sample["segment"] != segment:
            g = sample["gripper"] if segment is None else targets[-1][1]
            targets.extend((p, g) for p in interpolate(previous, t))
        targets.append((t, sample["gripper"]))
        previous, segment = t, sample["segment"]
    return targets


class RetargetPolicy(Policy):
    """Transform the recorded trajectory, bridge frames, and solve IK without re-running the expert."""
    def __init__(self, source):
        validate_source(source)
        self.source = source

    def reset(self, env, task):
        spec = {k: env.objs[task["object_id"]][k] for k in ("catalog_id", "size")}
        if task["object_id"] != self.source["task"]["object_id"] or spec != self.source["object_spec"]:
            raise ValueError("retargeting requires the source object's identity and geometry")
        anchors = {"object": object_frame(env, task), "sink": sink_frame(env, task)}
        self.anchors = {k: v.tolist() for k, v in anchors.items()}
        targets = retarget_trajectory(self.source["samples"], anchors, eef_frame(env))
        q = env.data.qpos[env.qadr].copy()
        scratch = mujoco.MjData(env.model)
        scratch.qpos[:] = env.data.qpos
        self.plan, self.ik_error, self.rotation_error = [], 0.0, 0.0
        for t, g in targets:
            q, _ = ik.solve(env.model, env.data, t[:3, 3], t[:3, :3], q, env.qadr, env.dofadr, rest=q)
            scratch.qpos[env.qadr] = q
            mujoco.mj_forward(env.model, scratch)
            site = scratch.site("eef")
            err = float(np.linalg.norm(site.xpos - t[:3, 3]))
            angle = float(np.arccos(np.clip((np.trace(site.xmat.reshape(3, 3).T @ t[:3, :3]) - 1) / 2, -1, 1)))
            self.ik_error = max(self.ik_error, err)
            self.rotation_error = max(self.rotation_error, angle)
            self.plan.append(np.r_[q, g, env.base_pose()])
        if self.ik_error > POSITION_TOL or self.rotation_error > ROTATION_TOL:
            raise Unreachable(f"IK residual {self.ik_error:.4f} m, {self.rotation_error:.4f} rad")
        self.t = 0

    def act(self, obs):
        a = self.plan[min(self.t, len(self.plan) - 1)]
        self.t += 1
        return a

    def done(self):
        return self.t >= len(self.plan)


class Unreachable(ValueError):
    """A transformed pose cannot be reached within the position or orientation tolerance."""


def generate(twin, out_dir="out", *, source=None, attempts=2, seed=0, res=64, offsets=None, log=print):
    """Record one source if absent; export successful replays at sampled (dx, dy, yaw) object offsets.

    Sink anchors always come from the actual target twin. A source may be reused in another compatible twin
    with the same object identity. Nothing changes object qpos after the initial episode setup.
    """
    if attempts < 1 or seed < 0 or res <= 0 or res % 16:
        raise ValueError("attempts must be positive, seed nonnegative, res a positive multiple of 16")
    if offsets is None:
        rng = np.random.default_rng(seed)
        offsets = rng.uniform([-.06, -.025, -.15], [.06, .025, .15], size=(attempts, 3)).tolist()
    offsets = np.asarray(offsets, dtype=float)
    if offsets.ndim != 2 or offsets.shape[1] != 3 or not len(offsets) or not np.isfinite(offsets).all():
        raise ValueError("offsets must be a nonempty finite list of (dx, dy, yaw_radians)")
    if source is None:
        tasks = make_tasks(twin, ["counter_to_sink"], 1, seed=seed, people=0, log=log)
        if not tasks:
            raise ValueError("no reachable counter_to_sink source task in the twin")
        finish_task(tasks[0])
        with SimEnv(twin, tasks[0], res=res, cameras=()) as env:
            source = record_source(env, tasks[0])
    validate_source(source)
    out_dir = Path(out_dir)
    writer = SimDatasetWriter(out_dir / "datasets" / f"{twin['room_id']}_se3_retarget", twin["room_id"], FPS, res,
                              ("ego_view", "wrist"), provenance=run_record(twin, seed=seed, generator=GENERATOR))
    (writer.root / "meta" / "se3_source.json").write_text(json.dumps(source, indent=2, allow_nan=False))
    rows, start_time = [], time.perf_counter()
    for i, (dx, dy, yaw) in enumerate(offsets):
        task = dict(source["task"], task_id=f"se3_{i:06d}", seed=seed * 1000 + i,
                    start=[source["task"]["start"][0] + dx, source["task"]["start"][1] + dy,
                           source["task"]["start"][2]])
        row = {"attempt": i, "offset": [float(dx), float(dy), float(yaw)]}
        policy = RetargetPolicy(source)
        try:
            with SimEnv(twin, task, res=res) as env:
                # Set only the initial object orientation; all subsequent motion is physics, never teleporting.
                adr = int(env.model.joint(task["object_id"]).qposadr[0])
                source_rotation = rigid(source["object_frame"])[:3, :3]
                c, s = np.cos(yaw), np.sin(yaw)
                r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ source_rotation
                mujoco.mju_mat2Quat(env.data.qpos[adr + 3:adr + 7], r.flatten())
                mujoco.mj_forward(env.model, env.data)
                episode = run_episode(env, policy, task)
        except Unreachable as error:
            row.update(outcome="drop_unreachable", reason=str(error))
        else:
            row.update(ik_error_m=policy.ik_error, rotation_error_rad=policy.rotation_error,
                       object_frame=policy.anchors["object"], sink_frame=policy.anchors["sink"],
                       detail=episode["detail"], human_contact=episode["safety"]["human_contact"])
            if episode["success"] and not episode["safety"]["human_contact"]:
                k = writer.add_episode(episode, task)
                writer.episodes[k].update(arrival_generator=GENERATOR, arrival_source_id=source["source_id"],
                                          arrival_source_seed=source["source_seed"], arrival_retarget_frames=policy.anchors)
                row["outcome"] = "success"
            else:
                row["outcome"] = "drop_physics_fail"
            del episode
        rows.append(row)
        log(f"[se3_retarget] attempt {i}: {row['outcome']}")
    elapsed = time.perf_counter() - start_time
    successes = len(writer.episodes)
    info = writer.finish(len(rows), extra={"generator": GENERATOR, "source_id": source["source_id"],
                         "source_seed": source["source_seed"], "source_ref": "meta/se3_source.json",
                         "dropped": len(rows) - successes,
                         "retarget": "executed eef SE(3) segments with minimum-jerk bridges; native Arrival, not upstream MimicGen"})
    report = {"generator": GENERATOR, "source_id": source["source_id"], "attempts": len(rows),
              "source_seed": source["source_seed"], "source_generator": source["source_generator"],
              "successes": successes, "dropped": len(rows) - successes, "seconds": elapsed,
              "success_rate": successes / len(rows), "demos_per_hour": successes * 3600 / elapsed,
              "timing_scope": "attempt loop: IK, physics, rendering and episode export; excludes source and final metadata",
              "drop_reasons": {name: sum(r["outcome"] == f"drop_{name}" for r in rows)
                               for name in ("unreachable", "physics_fail")},
              "sampled_offset_range": np.ptp(offsets, axis=0).tolist(),
              "rows": rows}
    rp = out_dir / "reports" / f"{twin['room_id']}_se3_retarget.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, allow_nan=False))
    return {"dataset": writer.root, "info": info, "report": rp, "report_data": report}


def main(argv=None):
    from arrival.scan.bundle import load_room
    from arrival.twin.compile import compile_twin

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--room", default="hero_kitchen")
    ap.add_argument("--scans", default="scans")
    ap.add_argument("--out", default="out")
    ap.add_argument("--source", type=Path, help="reuse a previously recorded meta/se3_source.json")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=64)
    args = ap.parse_args(argv)
    if args.attempts < 1 or args.seed < 0 or args.res <= 0 or args.res % 16:
        ap.error("attempts must be positive, seed nonnegative, res a positive multiple of 16")
    try:
        source = json.loads(args.source.read_text()) if args.source else None
        if source is not None:
            validate_source(source)
        twin = compile_twin(load_room(args.room, args.scans))
        result = generate(twin, args.out, source=source, attempts=args.attempts, seed=args.seed, res=args.res)
    except (OSError, ValueError, KeyError, TypeError) as error:
        ap.error(str(error))
    print(json.dumps({k: v for k, v in result["report_data"].items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
