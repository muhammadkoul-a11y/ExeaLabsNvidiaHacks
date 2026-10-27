"""Seed-demo replay retargeting in the twin: one scripted-expert seed episode is recorded as eef waypoints per
segment (approach, grasp, lift: relative to the object; carry, place, release, retreat: relative to the place
point), then replayed for new object start poses and target offsets by shifting each segment with the pose delta.
Every replay is judged by the same physics oracle as the expert and failures are dropped, so the only claim made
is arrival_physically_valid. This is object-centric segment replay in the spirit of MimicGen, not MimicGen: no
RoboCasa, no learned model and no seed-set pipeline are involved, and the generator is labelled
seed_replay_retarget. Pick-and-place families only (counter_to_sink, counter_to_counter); translation-only
transforms (the object's yaw is not tracked)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from arrival.sim.collect import FPS, SimEnv, run_episode
from arrival.sim.export import SimDatasetWriter
from arrival.sim.policy import ScriptedExpert
from arrival.sim.tasks import finish_task, make_tasks

GENERATOR = "seed_replay_retarget"
FAMILIES = ("counter_to_sink", "counter_to_counter")
TARGET_SEGMENTS = ("carry", "place", "release", "retreat")
IK_TOL = 0.02  # worst eef position error (m) a retargeted plan may have before the attempt counts as out of reach
START_DX, START_DY, TARGET_SHIFT = 0.10, 0.05, 0.03  # default sampling box (m) around the seed's poses


def _segment_names(n: int) -> list[str]:
    """Segment label per waypoint of the pick-and-place plan: 2 approach, 2 grasp, the lift (1 or 4 waypoints),
    then carry, place, release, retreat at the target."""
    return ["approach"] * 2 + ["grasp"] * 2 + ["lift"] * (n - 8) + list(TARGET_SEGMENTS)


class _Recorder(ScriptedExpert):
    """The scripted expert, keeping the waypoint list it planned."""

    def _waypoints(self, env, task):
        self.recorded = super()._waypoints(env, task)
        return self.recorded


def record_seed(env, task: dict, seed_id: str) -> dict:
    """Run the scripted expert on `task` and return the seed: each eef waypoint relative to its anchor (the
    object's body position for object segments, the place point for target ones)."""
    pol = _Recorder()
    obj0, tgt0 = env.body_pos(task["object_id"]), env.place_point(task)
    ep = run_episode(env, pol, task)
    wps = []
    for name, (pos, R, g, steps, *via) in zip(_segment_names(len(pol.recorded)), pol.recorded):
        anchor = tgt0 if name in TARGET_SEGMENTS else obj0
        wps.append({"segment": name, "rel": (np.asarray(pos) - anchor).tolist(), "R": np.asarray(R).tolist(),
                    "gripper": float(g), "steps": int(steps), "via": bool(via and via[0])})
    return {"seed_episode_id": seed_id, "task_id": task["task_id"], "family": task["family"],
            "object_id": task["object_id"], "start": [float(x) for x in obj0], "place_point": [float(x) for x in tgt0],
            "instruction": task["instruction"], "success": bool(ep["success"]), "frames": int(len(ep["states"])),
            "waypoints": wps}


class ReplayPolicy(ScriptedExpert):
    """Replays a recorded seed: each waypoint is re-anchored to the object's current pose (object segments) or to
    the current place point plus `target_shift` (target segments). Joint targets come from the same IK and
    min-jerk playback as the expert, so only the waypoint source differs."""

    def __init__(self, seed: dict, target_shift=(0.0, 0.0), **kw):
        super().__init__(**kw)
        self.seed, self.target_shift = seed, np.array([target_shift[0], target_shift[1], 0.0])

    def _waypoints(self, env, task):
        obj = env.body_pos(task["object_id"])
        tgt = env.place_point(task) + self.target_shift
        return [((tgt if w["segment"] in TARGET_SEGMENTS else obj) + np.array(w["rel"]), np.array(w["R"]),
                 w["gripper"], w["steps"], w["via"]) for w in self.seed["waypoints"]]


def sample_offsets(n: int, rng, dx=START_DX, dy=START_DY, shift=TARGET_SHIFT) -> list[tuple]:
    """(start dx, start dy, target shift x, target shift y) per attempt."""
    return [tuple(float(v) for v in rng.uniform([-dx, -dy, -shift, -shift], [dx, dy, shift, shift])) for _ in range(n)]


def retarget(twin: dict, out_dir: str | Path = "out", attempts: int = 6, seed: int = 0, res: int = 64,
             family: str = "counter_to_sink", offsets: list[tuple] | None = None, baseline: bool = False,
             log=print) -> dict:
    """One seed episode, then `attempts` retargeted replays; writes the successes as a LeRobot v2 dataset."""
    if family not in FAMILIES:
        raise ValueError(f"retarget supports {FAMILIES}, not {family}")
    out_dir = Path(out_dir)
    room = twin["room_id"]
    rng = np.random.default_rng(seed)
    pool = make_tasks(twin, [family], 8, seed=seed, people=0, log=log)
    if not pool:
        raise ValueError(f"no {family} variation in {room}")
    seed_info, seed_task, seed_tries = None, None, 0
    for t in pool:  # the first seed the expert completes
        finish_task(t)
        seed_tries += 1
        with SimEnv(twin, t, res=res) as env:
            info = record_seed(env, t, f"seed_{room}_{t['task_id']}")
        log(f"[retarget] seed {info['seed_episode_id']}: {'OK' if info['success'] else 'MISS'}")
        if info["success"]:
            seed_info, seed_task = info, t
            break
    if seed_info is None:
        raise RuntimeError("the scripted expert completed none of the candidate seeds; nothing to retarget")
    seed_path = out_dir / "seeds" / f"{seed_info['seed_episode_id']}.json"
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(json.dumps(seed_info, indent=2))

    offsets = offsets if offsets is not None else sample_offsets(attempts, rng)
    writer = SimDatasetWriter(out_dir / "datasets" / f"{room}_seed_replay", room, FPS, res, ("ego_view", "wrist"))
    t0 = time.perf_counter()
    rows, drops = [], {"unreachable": 0, "physics_fail": 0}
    for i, (dx, dy, tx, ty) in enumerate(offsets):
        task = dict(seed_task, task_id=f"{seed_task['task_id']}_rt{i:03d}", seed=seed * 1000 + 100 + i,
                    start=[seed_task["start"][0] + dx, seed_task["start"][1] + dy, seed_task["start"][2]])
        pol = ReplayPolicy(seed_info, (tx, ty))
        row = {"attempt": i, "start_xy": task["start"][:2], "target_shift": [tx, ty]}
        with SimEnv(twin, task, res=res) as env:
            pol.reset(env, task)  # IK for the shifted waypoints; out of reach is dropped before any rollout
            if pol.ik_error > IK_TOL:
                drops["unreachable"] += 1
                row.update(outcome="drop_unreachable", ik_error=round(float(pol.ik_error), 4))
                log(f"[retarget] {task['task_id']}: DROP unreachable ik_err={row['ik_error']}")
                rows.append(row)
                continue
            ep = run_episode(env, pol, task)
        row["ik_error"] = round(float(pol.ik_error), 4)
        if ep["success"] and not ep["safety"]["human_contact"]:
            k = writer.add_episode(ep, task)
            writer.episodes[k].update({"arrival_seed_episode_id": seed_info["seed_episode_id"],
                                       "arrival_start_xy": task["start"][:2], "arrival_target_shift": [tx, ty]})
            row["outcome"] = "success"
        else:
            drops["physics_fail"] += 1
            row["outcome"] = "drop_physics_fail"
        log(f"[retarget] {task['task_id']}: {row['outcome']} {ep['detail']}")
        rows.append(row)
        del ep  # one episode of frames in memory at a time
    seconds = time.perf_counter() - t0
    info = writer.finish(len(offsets))
    succ = len(writer.episodes)
    info.update({"arrival_generator": GENERATOR, "arrival_physically_valid": True,
                 "arrival_success_check": "physics", "arrival_seed_episode_id": seed_info["seed_episode_id"],
                 "arrival_seed_ref": str(seed_path.relative_to(out_dir)), "arrival_attempts": len(offsets),
                 "arrival_successes": succ, "arrival_dropped": len(offsets) - succ,
                 "arrival_retarget": "object-centric segment replay of one scripted-expert seed; translation only"})
    (writer.root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    ok_xy = np.array([r["start_xy"] for r in rows if r["outcome"] == "success"]).reshape(-1, 2)
    report = {"room_id": room, "generator": GENERATOR, "family": family, "seed_episode_id": seed_info["seed_episode_id"],
              "seed_candidates_tried": seed_tries, "attempts": len(offsets), "successes": succ,
              "dropped": len(offsets) - succ, "drop_reasons": drops,
              "success_rate": round(succ / len(offsets), 3) if offsets else 0.0,
              "start_xy_std_m": ok_xy.std(axis=0).round(4).tolist() if succ else None,
              "start_xy_range_m": np.ptp(ok_xy, axis=0).round(4).tolist() if succ else None,
              "seconds": round(seconds, 2), "demos_per_hour": round(succ / seconds * 3600, 1) if seconds else 0.0,
              "resolution": res, "people": 0, "physics_checked_success": True, "rows": rows}
    if baseline:  # the scripted expert on the same number of tasks, same room, family and resolution
        from arrival.sim.collect import collect

        b = collect(twin, [family], len(offsets), out_dir / "baseline", people=0, res=res, seed=seed, log=lambda *_: None)
        bd = b["report_data"]
        report["scripted_baseline"] = {k: bd[k] for k in ("attempts", "successes", "seconds", "demos_per_hour")}
    rp = out_dir / "reports" / f"{room}_seed_replay.json"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))
    return {"dataset": writer.root, "report": rp, "report_data": report, "info": info}


def main(argv=None) -> int:
    from arrival.scan.bundle import load_room, rooms_hint
    from arrival.twin.compile import compile_twin

    ap = argparse.ArgumentParser(prog="python -m arrival.generate.retarget", description=__doc__.split("\n\n")[0])
    ap.add_argument("--room", default="hero_kitchen")
    ap.add_argument("--scans", default="scans")
    ap.add_argument("--out", default="out")
    ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--family", default="counter_to_sink", choices=FAMILIES)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", action="store_true", help="also time the scripted expert on the same count")
    a = ap.parse_args(argv)
    if a.attempts < 1:
        ap.error("--attempts must be >= 1")
    if a.seed < 0:
        ap.error("--seed must be >= 0")
    if a.res <= 0 or a.res % 16:
        ap.error("--res must be a positive multiple of 16")
    try:
        bundle = load_room(a.room, a.scans)
    except FileNotFoundError:
        ap.error(f"no scan for room {a.room!r} under {a.scans} ({rooms_hint(a.room, a.scans)})")
    r = retarget(compile_twin(bundle), a.out, a.attempts, a.seed, a.res, a.family, baseline=a.baseline)
    print(json.dumps({k: v for k, v in r["report_data"].items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
