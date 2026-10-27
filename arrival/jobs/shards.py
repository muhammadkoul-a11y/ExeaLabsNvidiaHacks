"""Shardable, resumable sim collection (L5).

`arrival sim --episodes N --shard i/n` runs a contiguous block of the N tasks `make_tasks` draws (the whole list is
built in every shard, from the same seed, and the shard keeps tasks [i*N//n, (i+1)*N//n)), so shards need no
coordination: the union of the n shards is exactly the task list one process would run, each task keeps the seed and
id (`sim_<family>_<global index>`) it has there, and the families stay balanced inside a shard. Each shard writes under
`<out>/shards/<i>of<n>/`. `arrival merge-shards` joins the finished shard datasets, in shard order, into one LeRobot
v2 dataset whose episode indices, task indices, frame indices and stats are rebuilt, not copied.

A run also keeps a journal (`<out>/progress/<room>_sim.jsonl`): one line per finished task, written after that task's
episode files, so `--resume` can skip what is done, drop the files of the task a kill interrupted and carry on. A
journal from a run with different settings is refused rather than mixed in."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from arrival.export.lerobot import chunk_of

SHARD_RE = re.compile(r"^(\d+)/(\d+)$")
SHARD_DIR_RE = re.compile(r"^\d+of\d+$")
FINGERPRINT_KEYS = ("room", "families", "episodes", "seed", "people", "policy", "planner", "split", "instructions",
                    "base", "arm", "res", "shard", "exec_noise", "cameras")
SKIP_REASONS_KEPT = 50
# the shard fields that must agree across shards (index and task_range differ by design)
SHARED_SHARD_KEYS = ("count", "total_tasks", "episodes", "seed", "families", "people", "policy", "planner", "split",
                     "instructions", "base", "arm", "res")


class ShardError(ValueError):
    """A shard spec, a journal or a set of shard datasets that cannot be used; the message says why."""


def parse_shard(spec: str) -> tuple[int, int]:
    """'i/n' -> (i, n), 0 <= i < n. Shards are numbered from 0."""
    m = SHARD_RE.match(spec.strip())
    if not m:
        raise ShardError(f"--shard must look like i/n, e.g. 0/4, got {spec!r}")
    i, n = int(m.group(1)), int(m.group(2))
    if n < 1 or not 0 <= i < n:
        raise ShardError(f"--shard i/n needs n >= 1 and 0 <= i < n (shards count from 0), got {spec!r}")
    return i, n


def shard_range(total: int, i: int, n: int) -> tuple[int, int]:
    """Task indices [start, end) of shard i of n over `total` tasks: contiguous blocks that differ in size by at most one."""
    return i * total // n, (i + 1) * total // n


def shard_dir(out_dir: str | Path, i: int, n: int) -> Path:
    return Path(out_dir) / "shards" / f"{i}of{n}"


# ---- journal ----

class Journal:
    """Append-only record of the tasks a collect run finished. Line 1 is the run's fingerprint, then one JSON object
    per task, in task order: {"task_id", "outcome": ok|miss|contact|skip, "family", "seconds", and for ok the
    episode's "episode" and "stats" rows, for a skip its "stage" and "reason", else the safety counters}."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def start(self, fingerprint: dict, resume: bool, expected_tasks: list[dict] | None = None) -> list[dict]:
        """Begin the run. Fresh: truncate and write the fingerprint, return []. Resume: return the journaled records
        (a torn last line, from a kill mid-write, is cut off), or start fresh when there is no journal. Raises
        ShardError when the journal belongs to a run with other settings."""
        fp = {k: fingerprint.get(k) for k in FINGERPRINT_KEYS}
        fp["exec_noise"] = fp["exec_noise"] or 0.0
        fp["cameras"] = fingerprint.get("cameras", ["ego_view", "wrist"])
        if resume and self.path.is_file():
            body = self.path.read_text()
            lines = body.splitlines()
            rows = []
            for k, line in enumerate(lines):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    if k != len(lines) - 1 or body.endswith("\n"):
                        raise ShardError(f"{self.path}: line {k + 1} is not JSON; delete the progress file to start over")
                    break  # a kill mid-write: that task counts as not done
            if rows:
                if not isinstance(rows[0], dict):
                    raise ShardError(f"{self.path}: invalid fingerprint record")
                saved = rows[0].get("fingerprint")
                if isinstance(saved, dict):
                    saved = {**saved, "exec_noise": saved.get("exec_noise") or 0.0}  # journals from before DART have none
                if isinstance(saved, dict):
                    saved = {"cameras": ["ego_view", "wrist"], **saved}
                if saved != fp:
                    diff = sorted(k for k in set(fp) | set(saved or {}) if (saved or {}).get(k) != fp.get(k))
                    raise ShardError(f"--resume: the saved progress in {self.path} is for a run with different "
                                     f"{', '.join(diff)}; rerun without --resume to start over")
                for index, record in enumerate(rows[1:], 1):
                    if (not isinstance(record, dict) or not isinstance(record.get("task_id"), str)
                            or record.get("outcome") not in ("ok", "miss", "contact", "skip")):
                        raise ShardError(f"{self.path}: record {index} is not a completed task")
                if expected_tasks is not None:
                    done = rows[1:]
                    if len(done) > len(expected_tasks):
                        raise ShardError(f"{self.path}: more completed tasks than this run contains")
                    for index, (record, task) in enumerate(zip(done, expected_tasks)):
                        if (not isinstance(record, dict) or record.get("task_id") != task["task_id"]
                                or record.get("family") != task["family"]
                                or record.get("outcome") not in ("ok", "miss", "contact", "skip")):
                            raise ShardError(f"{self.path}: record {index + 1} does not match the task sequence")
                self.path.write_text("".join(json.dumps(r) + "\n" for r in rows))  # drops a torn last line
                return rows[1:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"fingerprint": fp}) + "\n")
        return []

    def append(self, record: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()


# ---- merge ----

def _jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ShardError(f"{path}: line {number} is not JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ShardError(f"{path}: line {number} must be a JSON object")
            rows.append(row)
    return rows


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ShardError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ShardError(f"{path}: expected a JSON object")
    return value


def _find_dataset(path: Path) -> Path:
    """`path` is a dataset folder, or a shard's out folder holding datasets/<room>_sim."""
    if (path / "meta" / "info.json").is_file():
        return path
    found = sorted(p.parent.parent for p in path.glob("datasets/*/meta/info.json"))
    if len(found) == 1:
        return found[0]
    if not found:
        raise ShardError(f"{path}: no dataset here (no meta/info.json, none under datasets/); is the shard finished?")
    raise ShardError(f"{path}: several datasets under datasets/ ({', '.join(p.name for p in found)}); pass one")


def expand_sources(paths: list[str | Path]) -> list[Path]:
    """One folder holding `<i>of<n>` shard folders (the layout `arrival sim --shard` writes) stands for them all."""
    paths = [Path(p) for p in paths]
    if len(paths) == 1 and not (paths[0] / "meta" / "info.json").is_file():
        home = paths[0] / "shards" if (paths[0] / "shards").is_dir() else paths[0]  # `out` or `out/shards`
        kids = sorted(p for p in home.glob("*of*") if p.is_dir() and SHARD_DIR_RE.match(p.name))
        if kids:
            return kids
    return paths


def _load(path: Path) -> dict:
    root = _find_dataset(path)
    info = _json(root / "meta" / "info.json")
    shard = info.get("arrival_shard")
    if not isinstance(shard, dict):
        raise ShardError(f"{root} is not a shard dataset (no arrival_shard in info.json); "
                         f"produce it with arrival sim --shard i/n")
    report = root.parent.parent / "reports" / f"{info.get('arrival_room_id')}_sim.json"
    return {"root": root, "info": info, "shard": shard, "episodes": _jsonl(root / "meta" / "episodes.jsonl"),
            "modality": _json(root / "meta" / "modality.json"),
            "embodiment": _json(root / "meta" / "embodiment.json"),
            "report": _json(report) if report.is_file() else None}


def _exec_noise(info: dict) -> float:
    """The DART level of a shard: the info key, or (shards written before it existed) the provenance block's."""
    return float(info.get("arrival_exec_noise_rad") or ((info.get("arrival_provenance") or {}).get("run") or {}).get("exec_noise") or 0.0)


def merge_shards(sources: list[str | Path], out_dir: str | Path = "out", log=print) -> dict:
    """Join finished shard datasets into out_dir/datasets/<room>_sim. Refuses an incomplete or mixed set (a missing
    shard, shards of different runs, an unfinished shard with no info.json) instead of writing a short dataset.
    Returns {"dataset", "report", "info"}; the caller validates the dataset."""
    from arrival.sim.export import SimDatasetWriter

    shards = [_load(p) for p in expand_sources(list(sources))]
    if not shards:
        raise ShardError("no shard datasets given")
    n = shards[0]["shard"]["count"]
    got = sorted(s["shard"]["index"] for s in shards)
    if got != list(range(n)):
        missing = sorted(set(range(n)) - set(got))
        raise ShardError(f"the run was split in {n} shards (0..{n - 1}); got {got}"
                         + (f", missing {missing}" if missing else ", some given twice"))
    shards.sort(key=lambda s: s["shard"]["index"])
    first = shards[0]
    for s in shards[1:]:
        for k in SHARED_SHARD_KEYS:
            if s["shard"].get(k) != first["shard"].get(k):
                raise ShardError(f"shard {s['shard']['index']} differs from shard 0 in {k}: "
                                 f"{s['shard'].get(k)!r} vs {first['shard'].get(k)!r}; they are not from one run")
        for k in ("arrival_room_id", "fps", "features"):
            if s["info"].get(k) != first["info"].get(k):
                raise ShardError(f"shard {s['shard']['index']} differs from shard 0 in info.{k}")
        if _exec_noise(s["info"]) != _exec_noise(first["info"]):
            raise ShardError(f"shard {s['shard']['index']} differs from shard 0 in arrival_exec_noise_rad: "
                             f"{_exec_noise(s['info'])!r} vs {_exec_noise(first['info'])!r}; they are not from one run")
        if s["modality"] != first["modality"] or s["embodiment"] != first["embodiment"]:
            raise ShardError(f"shard {s['shard']['index']} has a different modality or embodiment than shard 0")
    cursor = 0
    for s in shards:  # the blocks must tile [0, total_tasks)
        lo, hi = s["shard"]["task_range"]
        if lo != cursor:
            raise ShardError(f"shard {s['shard']['index']} starts at task {lo}, expected {cursor}")
        cursor = hi
    if cursor != first["shard"]["total_tasks"]:
        raise ShardError(f"the shards cover {cursor} tasks, the run had {first['shard']['total_tasks']}")
    seen_tasks: set[str] = set()
    for s in shards:
        if len(s["episodes"]) != s["info"]["total_episodes"]:
            raise ShardError(f"{s['root']}: episodes.jsonl has {len(s['episodes'])} rows, "
                             f"info.total_episodes={s['info']['total_episodes']}; the shard is damaged")
        lo, hi = s["shard"]["task_range"]
        for row in s["episodes"]:
            task_id = row.get("arrival_task_id")
            match = re.fullmatch(r"sim_(.+)_(\d+)", task_id) if isinstance(task_id, str) else None
            if not match or match.group(1) not in first["shard"]["families"]:
                raise ShardError(f"shard {s['shard']['index']} has an invalid task id {task_id!r}")
            task_index = int(match.group(2))
            if task_id in seen_tasks or not lo <= task_index < hi:
                raise ShardError(f"shard {s['shard']['index']} has duplicate or out-of-range task {task_id!r}")
            seen_tasks.add(task_id)
    if sum(len(s["episodes"]) for s in shards) == 0:
        raise ShardError("no shard holds an episode; nothing to merge")

    info0 = first["info"]
    room, fps = info0["arrival_room_id"], info0["fps"]
    cams = [k.removeprefix("observation.images.") for k, v in info0["features"].items() if v.get("dtype") == "video"]
    res = info0["features"][f"observation.images.{cams[0]}"]["shape"][0]

    def episode_files(s: dict, row: dict) -> tuple[Path, dict[str, Path]]:
        index = row["episode_index"]
        size = s["info"].get("chunks_size", 1000)
        data = s["root"] / s["info"]["data_path"].format(episode_chunk=chunk_of(index, size), episode_index=index)
        videos = {cam: s["root"] / s["info"]["video_path"].format(
            episode_chunk=chunk_of(index, size), episode_index=index, video_key=f"observation.images.{cam}")
            for cam in cams}
        return data, videos

    root = Path(out_dir) / "datasets" / f"{room}_sim"
    for s in shards:
        if root.resolve().is_relative_to(s["root"].resolve()) or s["root"].resolve().is_relative_to(root.resolve()):
            raise ShardError(f"the output {root} would overwrite or be inside shard {s['root']}; pick another --out")
        for row in s["episodes"]:
            data, videos = episode_files(s, row)
            for source in (data, *videos.values()):
                if not source.is_file():
                    raise ShardError(f"missing source file {source}; merge did not touch {root}")
    w = SimDatasetWriter(root, room, fps, res, tuple(cams), base="mobile" if first["embodiment"].get("base_actuated") else "fixed",
                        arm=first["embodiment"].get("arm", "procedural"), provenance=info0.get("arrival_provenance"))
    for s in shards:
        for row in s["episodes"]:
            data, videos = episode_files(s, row)
            t = pq.read_table(data)
            w.add_copied(np.array(t.column("observation.state").to_pylist(), dtype=np.float32),
                         np.array(t.column("action").to_pylist(), dtype=np.float32), row, videos)

    infos = [s["info"] for s in shards]
    reasons = [r for r in infos[0].get("arrival_skip_reasons", []) if r.get("stage") == "preflight"]
    for i in infos:  # the pre-flight runs on the whole task list in every shard: take it once
        reasons += [r for r in i.get("arrival_skip_reasons", []) if r.get("stage") != "preflight"]
    unreachable = infos[0].get("arrival_skipped_unreachable", 0)
    errored = sum(i.get("arrival_skipped_error", 0) for i in infos)
    skipped = {"skipped_unreachable": unreachable, "skipped_error": errored, "skipped": unreachable + errored,
               "skip_reasons": reasons[:SKIP_REASONS_KEPT]}
    attempts = sum(i.get("arrival_attempts", 0) for i in infos)
    merged = {"merged_shards": n, "merged_total_tasks": first["shard"]["total_tasks"]}
    if _exec_noise(info0):  # DART shards (they agree, checked above): the merged set is just as perturbed
        merged["exec_noise_rad"] = _exec_noise(info0)
    out_info = w.finish(attempts, skipped, merged)
    log(f"[merge] {n} shards -> {root}: {out_info['total_episodes']} episodes, {out_info['total_frames']} frames")
    rp = _merged_report(shards, out_info, skipped, Path(out_dir) / "reports" / f"{room}_sim.json", n)
    return {"dataset": root, "report": rp, "info": out_info}


def _merged_report(shards: list[dict], info: dict, skipped: dict, path: Path, n: int) -> Path | None:
    """Sum the shard reports into one (None when a shard has no report). `seconds` is the sum over shards, i.e.
    compute time; `seconds_longest_shard` is what the wall clock was if they ran in parallel."""
    reps = [s["report"] for s in shards]
    if any(r is None for r in reps):
        return None
    r0 = reps[0]
    fams: dict[str, dict] = {}
    for r in reps:
        for fam, v in r["families"].items():
            f = fams.setdefault(fam, {"attempts": 0, "successes": 0, "human_contacts": 0, "fridge_visits": 0,
                                      "door_contacts": 0, "seconds": 0.0})
            for k in f:
                if k == "seconds":  # a shard report written before per-family seconds makes the sum unknown
                    f[k] = f[k] + v[k] if f[k] is not None and k in v else None
                else:
                    f[k] += v.get(k, 0)
    secs = sum(r["seconds"] for r in reps)
    succ = info["total_episodes"]
    attempts = info["arrival_attempts"]
    report = {**{k: v for k, v in r0.items() if k not in ("shard", "resumed_tasks")},
              "attempts": attempts, **skipped, "successes": succ, "dropped": attempts - succ,
              "seconds": round(secs, 2), "seconds_longest_shard": round(max(r["seconds"] for r in reps), 2),
              "demos_per_hour": round(succ / secs * 3600, 1) if secs else 0.0,
              "human_contacts": sum(r["human_contacts"] for r in reps),
              "fridge_visits": sum(r.get("fridge_visits", 0) for r in reps),
              "door_contacts": sum(r.get("door_contacts", 0) for r in reps),
              "families": {k: {**v, "seconds": None if v["seconds"] is None else round(v["seconds"], 2),
                               "success_rate": round(v["successes"] / v["attempts"], 3)} for k, v in fams.items()
                           if v["attempts"]},
              "merged_shards": n}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path
