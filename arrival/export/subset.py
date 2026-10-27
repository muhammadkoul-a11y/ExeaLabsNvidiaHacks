"""Keep a rule-selected part of a sim dataset: `python -m arrival subset SRC --out OUT [--families ...] [--min-drawer-opening M]`.

Why: DART runs (sim --dart) drive the arm off the plan, and a drawer pull is fragile under that: on the hero kitchen the
clean expert ends every drawer pull at 0.300 m, but with even 0.01 rad of noise about 30% of the kept demos end under
0.28 m (the success bar is 0.15 m), and a policy trained on them under-pulled (83% -> 38% open_drawer). This drops such
marginal demos, or whole families, and writes the rest as a new dataset through the same writer merge-shards uses, so
episodes are re-indexed and the result validates. Nothing is regenerated and nothing is changed in the episodes kept.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from arrival.export.lerobot import chunk_of
from arrival.jobs.shards import _find_dataset, _jsonl


class SubsetError(ValueError):
    """The source is not a sim dataset or the rule keeps nothing."""


def subset_dataset(src: str | Path, out_dir: str | Path = "out", families: list[str] | None = None,
                   min_drawer_opening: float | None = None, log=print) -> dict:
    from arrival.sim.export import SimDatasetWriter

    if min_drawer_opening is not None and (type(min_drawer_opening) not in (int, float) or
                                           not math.isfinite(min_drawer_opening) or min_drawer_opening < 0):
        raise SubsetError("min_drawer_opening must be a finite, nonnegative distance in metres")
    root = _find_dataset(Path(src))
    meta = root / "meta"
    info = json.loads((meta / "info.json").read_text())
    if "arrival_room_id" not in info:
        raise SubsetError(f"{root} is not a dataset written by `arrival sim` (no arrival_room_id in info.json)")
    rows = _jsonl(meta / "episodes.jsonl")
    kept, dropped = [], {"family": 0, "drawer_opening": 0}
    for r in rows:
        if families and r.get("arrival_family") not in families:
            dropped["family"] += 1
            continue
        if min_drawer_opening is not None and r.get("arrival_family") == "open_drawer":
            q = (r.get("arrival_success_detail") or {}).get("qpos")
            if q is None or q < min_drawer_opening:  # no recorded opening counts as unknown, not as good
                dropped["drawer_opening"] += 1
                continue
        kept.append(r)
    if not kept:
        raise SubsetError(f"the rule keeps none of the {len(rows)} episodes")
    room, fps = info["arrival_room_id"], info["fps"]
    cams = [k.removeprefix("observation.images.") for k, v in info["features"].items() if v.get("dtype") == "video"]
    if not cams:
        raise SubsetError(f"{root} has no video features; subset requires at least one camera")
    res = info["features"][f"observation.images.{cams[0]}"]["shape"][0]
    embodiment = json.loads((meta / "embodiment.json").read_text())
    out_root = Path(out_dir) / "datasets" / f"{room}_sim"
    if out_root.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(out_root.resolve()):
        raise SubsetError("the output would overwrite or be inside the source; pick another --out")
    w = SimDatasetWriter(out_root, room, fps, res, tuple(cams), base="mobile" if embodiment.get("base_actuated") else "fixed",
                         arm=embodiment.get("arm", "procedural"), provenance=info.get("arrival_provenance"))
    size = info.get("chunks_size", 1000)
    for r in kept:
        j = r["episode_index"]
        t = pq.read_table(root / info["data_path"].format(episode_chunk=chunk_of(j, size), episode_index=j))
        videos = {c: root / info["video_path"].format(episode_chunk=chunk_of(j, size), episode_index=j,
                                                      video_key=f"observation.images.{c}") for c in cams}
        w.add_copied(np.array(t.column("observation.state").to_pylist(), dtype=np.float32),
                     np.array(t.column("action").to_pylist(), dtype=np.float32), r, videos)
    rule = {"families": families, "min_drawer_opening_m": min_drawer_opening}
    extra = {"subset_of": str(root), "subset_rule": rule, "subset_source_episodes": len(rows), "subset_dropped": dropped}
    noise = info.get("arrival_exec_noise_rad")
    if noise:
        extra["exec_noise_rad"] = noise  # the kept episodes are exactly as perturbed as in the source
    out_info = w.finish(len(kept), None, extra)  # `attempts` is the kept count: the generation stats stay with the source
    log(f"[subset] {len(rows)} -> {len(kept)} episodes ({dropped['family']} by family, {dropped['drawer_opening']} drawer pulls "
        f"under {min_drawer_opening} m): {out_root}")
    return {"dataset": out_root, "info": out_info, "kept": len(kept), "dropped": dropped}
