"""Closed-loop eval in the twin: N episodes per family for one policy, scored without writing a dataset.

Success is the physics check from collect.py; an episode with robot-human contact counts as a failure.
Time is sim time (frames / 20 Hz) and wall time per episode. `holdout` draws only the held-out wording and
start offset of docs/07-eval.md (make_tasks(holdout=True)); `sim` datasets never contain them.
"""

from __future__ import annotations

import json
import math
import platform
import time
import uuid
from importlib.metadata import version
from pathlib import Path

from arrival.sim.collect import FPS, SimEnv, run_episode
from arrival.eval.manifest import EVAL_SEED_BASE, make_manifest, validate_manifest


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score interval for k successes in n trials; [0, 1] when n is 0."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    zz = z * z / n
    centre = (p + zz / 2) / (1 + zz)
    half = z * math.sqrt(p * (1 - p) / n + zz / (4 * n)) / (1 + zz)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _family_summary(rows: list[dict]) -> dict:
    n, k = len(rows), sum(r["success"] for r in rows)
    observed = [r for r in rows if r["error"] is None]
    dists = [r["safety"]["min_human_distance_m"] for r in rows if r["safety"]["min_human_distance_m"] is not None]
    iks = [r["ik_error"] for r in rows if r["ik_error"] is not None]
    return {"episodes": n, "successes": k, "success_rate": round(k / n, 4), "wilson_95": wilson(k, n),
            "observed_episodes": len(observed),
            "mean_seconds": round(sum(r["seconds"] for r in observed) / len(observed), 3) if observed else None,
            "mean_wall_seconds": round(sum(r["wall_seconds"] for r in rows) / n, 3),
            "human_contacts": sum(r["safety"]["human_contact"] for r in observed) if observed else None,
            "errors": [r["error"] for r in rows if r["error"]],
            "min_human_distance_m": round(min(dists), 4) if dists else None,
            "ik_error_max": round(max(iks), 4) if iks else None,
            "arm_contact_episodes": sum(r["safety"]["arm_contact_steps"] > 0 for r in observed) if observed else None,
            "arm_contact_max_depth_m": round(max(r["safety"]["arm_contact_max_depth_m"] for r in observed), 4) if observed else None}


def evaluate(twin: dict | None = None, families: list[str] | None = None, episodes: int = 4,
             policy_name: str = "scripted", seed: int = 0, people: int = 2, holdout: bool = False,
             res: int = 128, out_dir: str | Path = "out", log=print, *, manifest: dict | None = None,
             checkpoint_id: str | None = None, base: str = "fixed", arm: str = "procedural") -> dict:
    from arrival.sim.policy import ScriptedExpert, VLAClient

    if policy_name not in ("scripted", "vla"):
        raise ValueError(f"unknown evaluation policy {policy_name!r}")
    if policy_name == "vla" and (not isinstance(checkpoint_id, str) or not checkpoint_id.strip()):
        raise ValueError("VLA evaluation requires --checkpoint-id (the caller's checkpoint label)")
    frozen = (validate_manifest(manifest) if manifest is not None else
              make_manifest(twin, families, episodes, seed, people, holdout, res, log, base=base, arm=arm))
    twin, cfg = frozen["twin"], frozen["settings"]
    policy = ScriptedExpert() if policy_name == "scripted" else VLAClient()
    policy_server = policy.server_info() if hasattr(policy, "server_info") else None
    run_id = uuid.uuid4().hex
    run_dir = Path(out_dir) / "eval" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path, episodes_path = run_dir / "manifest.json", run_dir / "episodes.jsonl"
    manifest_path.write_text(json.dumps(frozen, indent=2, allow_nan=False), encoding="utf-8")
    checkpoint_id = checkpoint_id.strip() if checkpoint_id else "scripted-expert"
    rows = []
    t_all = time.perf_counter()
    with episodes_path.open("x", encoding="utf-8") as stream:
        for index, task in enumerate(frozen["tasks"]):
            row = {"run_id": run_id, "checkpoint_id": checkpoint_id, "manifest_id": frozen["manifest_id"],
                   "trial_index": index, "task_id": task["task_id"], "family": task["family"],
                   "physics_success": None, "success": False, "seconds": None, "wall_seconds": None,
                   "safety": {"human_contact": None, "min_human_distance_m": None,
                              "arm_contact_steps": None, "arm_contact_max_depth_m": None},
                   "ik_error": None, "detail": None, "error": None}
            t0, abort = time.perf_counter(), None
            try:
                with SimEnv(twin, task, res=cfg["resolution"], base=cfg["base"], arm=cfg.get("arm", "procedural")) as env:
                    ep = run_episode(env, policy, task)
                safety = ep["safety"]
                outcome = dict(physics_success=bool(ep["success"]), success=bool(ep["success"] and not safety["human_contact"]),
                               seconds=len(ep["states"]) / FPS, safety=safety, ik_error=ep["ik_error"], detail=ep["detail"])
                json.dumps(outcome, allow_nan=False)
                tag = "DROP human_contact" if safety["human_contact"] else ("OK " if outcome["success"] else "MISS")
                arm_note = (f" arm_contact={safety['arm_contact_steps']} steps {safety['arm_contact_worst']}"
                       if safety["arm_contact_steps"] else "")
                log(f"[eval] {task['task_id']}: {tag} {ep['detail']} sim={outcome['seconds']:.2f}s "
                    f"min_human_d={safety['min_human_distance_m']}{arm_note}")
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                if isinstance(e, OSError) and not rows:
                    abort = e
                log(f"[eval] {task['task_id']}: ERROR {row['error']}")
            else:
                row.update(outcome)
            row["wall_seconds"] = time.perf_counter() - t0
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
            if abort is not None:
                raise abort
    per = {}
    for family in cfg["families"]:
        family_rows = [r for r in rows if r["family"] == family]
        if family_rows:
            per[family] = _family_summary(family_rows)
    n, k = len(rows), sum(r["success"] for r in rows)
    observed = [r for r in rows if r["error"] is None]
    report = {"room_id": twin["room_id"], "policy": policy_name, "run_id": run_id,
              "checkpoint_id": checkpoint_id, "checkpoint_id_source": "caller" if policy_name == "vla" else "scripted",
              "policy_server": policy_server,
              "manifest_id": frozen["manifest_id"], "twin_sha256": frozen["twin_sha256"],
              "manifest_path": str(manifest_path), "episodes_path": str(episodes_path),
              "runtime": {"python": platform.python_version(), "mujoco": version("mujoco"), "numpy": version("numpy")},
              **{k: cfg[k] for k in ("episodes_per_family", "seed", "people", "holdout", "resolution", "base")},
              "arm": cfg.get("arm", "procedural"),
              "omitted_families": frozen["omitted_families"], "families": per,
              "totals": {"episodes": n, "successes": k, "success_rate": round(k / n, 4) if n else None,
                         "wilson_95": wilson(k, n), "observed_episodes": len(observed),
                         "human_contacts": sum(r["safety"]["human_contact"] for r in observed) if observed else None,
                         "errors": sum(r["error"] is not None for r in rows),
                         "arm_contact_episodes": sum(r["safety"]["arm_contact_steps"] > 0 for r in observed) if observed else None,
                         "wall_seconds": round(time.perf_counter() - t_all, 2)}}
    path = run_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    report["path"] = path
    return report


def table(report: dict) -> str:
    head = (f"{'family':<20}{'ep':>4}{'succ':>6}{'rate':>7}  {'wilson95':<16}{'sim_s':>7}{'wall_s':>8}{'contact':>9}"
            f"{'min_d_m':>9}{'ik_err':>8}{'arm_hit':>9}{'arm_mm':>8}{'obs':>5}")
    lines = [head]
    for fam, f in report["families"].items():
        lo, hi = f["wilson_95"]
        md = "-" if f["min_human_distance_m"] is None else f"{f['min_human_distance_m']:.3f}"
        ik = "-" if f["ik_error_max"] is None else f"{f['ik_error_max']:.3f}"
        seconds = "-" if f["mean_seconds"] is None else f"{f['mean_seconds']:.2f}"
        depth = "-" if f["arm_contact_max_depth_m"] is None else f"{f['arm_contact_max_depth_m'] * 1000:.1f}"
        human = "-" if f["human_contacts"] is None else str(f["human_contacts"])
        arm = "-" if f["arm_contact_episodes"] is None else str(f["arm_contact_episodes"])
        lines.append(f"{fam:<20}{f['episodes']:>4}{f['successes']:>6}{f['success_rate']:>7.3f}  [{lo:.3f}, {hi:.3f}]  "
                     f"{seconds:>7}{f['mean_wall_seconds']:>8.1f}{human:>9}{md:>9}{ik:>8}"
                     f"{arm:>9}{depth:>8}{f['observed_episodes']:>5}")
    t = report["totals"]
    lo, hi = t["wilson_95"]
    rate = "-" if t["success_rate"] is None else f"{t['success_rate']:.3f}"
    human = "-" if t["human_contacts"] is None else str(t["human_contacts"])
    arm = "-" if t["arm_contact_episodes"] is None else str(t["arm_contact_episodes"])
    lines.append(f"{'total':<20}{t['episodes']:>4}{t['successes']:>6}{rate:>7}  [{lo:.3f}, {hi:.3f}]"
                 f"{'':>24}{human:>9}{'':>17}{arm:>9}")
    if t["errors"]:
        lines.append("Safety and sim time cover only observed episodes (obs); errors count as failures.")
    return "\n".join(lines)
