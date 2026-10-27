import itertools
import json
import math

import pyarrow.parquet as pq
import pytest

from arrival.cli import main, run_demo
from arrival.export.validate import validate
from arrival.generate.mimicgen import probe
from arrival.generate.scripted import count_attempts, generate, target_point
from arrival.render import sim_texture
from arrival.render.splat import SplatCompositeNotImplemented, render as splat_render
from arrival.tasks.propose import propose

from conftest import SCANS


@pytest.fixture
def hero_tasks(hero_twin):
    return propose(hero_twin)["tasks"]


def test_generator_and_oracle(hero_twin, hero_tasks):
    attempts = list(itertools.islice(generate(hero_tasks, hero_twin, episodes=100), 20))
    assert len(attempts) == 20
    assert [a["attempt_index"] for a in attempts if not a["success"]] == [9, 19]
    ok = next(a for a in attempts if a["success"])
    assert len(ok["frames"]) == 80
    assert ok["frames"][40]["gripper"] > 0.5
    tgt = target_point(ok["task"], hero_twin)
    last = ok["frames"][-1]["object_position"]
    assert math.hypot(last[0] - tgt[0], last[1] - tgt[1]) <= 0.05
    miss = attempts[9]
    assert all(f["object_position"] == miss["task"]["start_position"] for f in miss["frames"])
    assert max(f["gripper"] for f in miss["frames"]) == 0.0


def test_generator_is_deterministic(hero_twin, hero_tasks):
    a = [x["task"]["task_id"] for x in generate(hero_tasks, hero_twin, episodes=30)]
    b = [x["task"]["task_id"] for x in generate(hero_tasks, hero_twin, episodes=30)]
    assert a == b


def test_retarget_translates_seed(hero_twin, hero_tasks):
    from arrival.generate.scripted import build_seeds, make_attempt
    seeds = build_seeds(hero_tasks, hero_twin)
    t = next(t for t in hero_tasks if t["object_id"] == "mug_1" and t["pose_index"] == 1
             and t["family"] == "counter_to_sink")
    att = make_attempt(0, t, hero_twin, seeds)
    seed_frames = seeds[("counter_to_sink", "mug_1")][1]
    assert att["frames"][0]["eef"][0] == pytest.approx(seed_frames[0]["eef"][0] + 0.08)
    assert att["frames"][-1]["object_position"] == pytest.approx(seed_frames[-1]["object_position"])


@pytest.mark.parametrize("n,attempts,dropped", [(4, 4, 0), (20, 22, 2), (1000, 1111, 111)])
def test_attempt_counts(n, attempts, dropped):
    assert count_attempts(n) == {"attempts": attempts, "successes": n, "dropped": dropped}


def test_sim_texture(hero_twin, hero_tasks):
    from arrival.generate.scripted import build_seeds, make_attempt
    t = next(t for t in hero_tasks if t["object_id"] == "mug_1" and t["lighting"] == "dim")
    att = make_attempt(0, t, hero_twin, build_seeds(hero_tasks, hero_twin))
    frames = sim_texture.render_episode(att["frames"], "dim")
    assert frames.shape == (80, 64, 64, 3)
    assert tuple(frames[0][0, 0]) == (28, 32, 38)
    assert (frames[0] != frames[40]).any()
    with pytest.raises(SplatCompositeNotImplemented):
        splat_render()


def test_mimicgen_probe_skips():
    assert probe()["skipped"] is True


def test_demo_4_episodes(tmp_path):
    out = tmp_path / "out"
    res = run_demo("hero_kitchen", 4, "all", out_dir=out, scans_dir=SCANS, log=lambda *a: None)
    ds = res["dataset"]
    assert validate(ds) == []
    info = json.loads((ds / "meta" / "info.json").read_text())
    assert info["arrival_physically_valid"] is False
    assert info["arrival_generator"] == "scripted_retarget"
    assert info["total_episodes"] == 4 and info["total_frames"] == 320
    tasks = json.loads((out / "twins" / "hero_kitchen" / "tasks.json").read_text())
    assert {t["family"] for t in tasks["tasks"]} == {"counter_to_sink", "counter_to_counter"}
    assert set(info["arrival_families"]) == {"counter_to_sink", "counter_to_counter"}
    modality = json.loads((ds / "meta" / "modality.json").read_text())
    assert [modality["state"]["gripper"]["start"], modality["state"]["gripper"]["end"]] == [7, 8]
    emb = json.loads((ds / "meta" / "embodiment.json").read_text())
    assert emb["joints_are_placeholder"] is True
    assert emb["cartesian_sidecar"] == "arrival_cartesian.jsonl"
    assert (ds / "meta" / emb["cartesian_sidecar"]).is_file()
    assert info["features"]["observation.images.ego_view"]["info"]["video.height"] == 64
    assert len((ds / "meta" / "episodes_stats.jsonl").read_text().splitlines()) == 4
    assert json.loads((ds / "meta" / "stats.json").read_text())["observation.state"]["count"] == [320]
    assert len(list(ds.glob("videos/chunk-000/observation.images.ego_view/*.mp4"))) == 4
    t = pq.read_table(ds / "data" / "chunk-000" / "episode_000003.parquet")
    assert t.column("index")[0].as_py() == 240
    assert t.column("next.done").to_pylist()[-1] is True
    assert t.column("next.reward").to_pylist()[-1] == 1.0
    assert t.column("observation.state").to_pylist() == t.column("action").to_pylist()
    rep = json.loads(res["report"].read_text())
    assert (rep["successes"], rep["attempts"], rep["dropped"]) == (4, 4, 0)
    assert rep["dollars"]["local_cpu"] == {"usd": 0, "measured": True}
    assert all(v["measured"] is False for k, v in rep["dollars"].items() if k != "local_cpu")
    assert rep["teleop_baseline"] is None and rep["usefulness"] is None and rep["realism"] is None


def test_demo_20_report(tmp_path):
    res = run_demo("hero_kitchen", 20, "none", out_dir=tmp_path, scans_dir=SCANS, log=lambda *a: None)
    rep = res["report_data"]
    assert (rep["successes"], rep["attempts"], rep["dropped"]) == (20, 22, 2)
    assert validate(res["dataset"]) == []


def test_validator_rejects_broken(tmp_path):
    res = run_demo("hero_kitchen", 2, "all", out_dir=tmp_path, scans_dir=SCANS, log=lambda *a: None)
    ds = res["dataset"]
    m = json.loads((ds / "meta" / "modality.json").read_text())
    m["state"]["gripper"]["start"] = 6
    (ds / "meta" / "modality.json").write_text(json.dumps(m))
    assert validate(ds)
    (ds / "meta" / "embodiment.json").unlink()
    assert main(["validate", str(ds)]) == 1


def test_cli_usage_errors(tmp_path):
    assert main([]) == 2
    assert main(["bogus"]) == 2
    assert main(["demo", "--episodes", "0", "--out", str(tmp_path), "--scans", str(SCANS)]) == 2
    assert main(["demo", "--videos", "bogus"]) == 2


def test_from_roomplan(tmp_path):
    res = run_demo("hero_kitchen", 2, "none", out_dir=tmp_path, scans_dir=SCANS, from_roomplan=True,
                   log=lambda *a: None)
    twin = json.loads((res["twin_dir"] / "twin.json").read_text())
    sink = next(f for f in twin["fixtures"] if f["id"] == "sink_1")
    assert sink["position"] == pytest.approx([1.6, 0.4, 0.85])
    assert any(f["source_category"] == "synthesized" for f in twin["fixtures"])


def test_release_is_smooth_and_offset_constant(hero_twin, hero_tasks):
    from arrival.generate.scripted import GRASP, SEG, build_seeds, make_attempt
    seeds = build_seeds(hero_tasks, hero_twin)
    for t in hero_tasks[:6]:
        fr = make_attempt(0, t, hero_twin, seeds)["frames"]
        for a, b in zip(fr, fr[1:]):
            assert math.dist(a["object_position"], b["object_position"]) < 0.12
        for f in fr[2 * SEG + 1:4 * SEG]:
            assert f["eef"][2] - f["object_position"][2] == pytest.approx(GRASP, abs=1e-5)


def test_misses_hit_both_families(hero_twin, hero_tasks):
    dropped = [a["task"]["family"] for a in generate(hero_tasks, hero_twin, episodes=1000) if not a["success"]]
    assert len(dropped) == 111
    assert {"counter_to_sink", "counter_to_counter"} <= set(dropped)
    assert min(dropped.count(f) for f in set(dropped)) > 30


def test_sparse_policy_declares_no_video(tmp_path):
    res = run_demo("hero_kitchen", 3, "sparse", out_dir=tmp_path, scans_dir=SCANS, log=lambda *a: None)
    ds = res["dataset"]
    info = json.loads((ds / "meta" / "info.json").read_text())
    assert not any(v["dtype"] == "video" for v in info["features"].values())
    assert "video" not in json.loads((ds / "meta" / "modality.json").read_text())
    assert not list(ds.glob("videos/**/*.mp4"))
    assert (tmp_path / "datasets" / "hero_kitchen_previews" / "ego_view" / "episode_000000.mp4").is_file()
    assert validate(ds) == []


def test_chunking_1001(tmp_path):
    res = run_demo("hero_kitchen", 1001, "none", out_dir=tmp_path, scans_dir=SCANS, log=lambda *a: None)
    ds = res["dataset"]
    assert res["info"]["total_chunks"] == 2
    assert (ds / "data" / "chunk-001" / "episode_001000.parquet").is_file()
    assert not (ds / "data" / "chunk-000" / "episode_001000.parquet").exists()
    assert validate(ds) == []
    (ds / "data" / "chunk-001" / "episode_001000.parquet").unlink()
    assert any("episode_001000" in e for e in validate(ds))


def test_validator_catches_corruption(tmp_path):
    import pyarrow as pa
    res = run_demo("hero_kitchen", 3, "all", out_dir=tmp_path, scans_dir=SCANS, log=lambda *a: None)
    ds = res["dataset"]
    p = ds / "data" / "chunk-000" / "episode_000001.parquet"
    t = pq.read_table(p)
    done = t.column("next.done").to_pylist()
    done[0] = True
    pq.write_table(t.set_column(t.schema.get_field_index("next.done"), "next.done", pa.array(done)), p)
    assert any("next.done" in e for e in validate(ds))
    pq.write_table(t.slice(0, 40), p)
    assert any("rows" in e for e in validate(ds))
    (ds / "videos" / "chunk-000" / "observation.images.ego_view" / "episode_000002.mp4").write_bytes(b"junk")
    assert any("episode_000002.mp4" in e for e in validate(ds))


def test_validator_catches_nan_and_bad_timestamps(tmp_path):
    import json

    import pyarrow as pa
    res = run_demo("hero_kitchen", 2, "none", out_dir=tmp_path, scans_dir=SCANS, log=lambda *a: None)
    ds = res["dataset"]
    assert validate(ds) == []
    p = ds / "data" / "chunk-000" / "episode_000001.parquet"
    t = pq.read_table(p)

    def put(col, values, typ):
        pq.write_table(t.set_column(t.schema.get_field_index(col), col, pa.array(values, type=typ)), p)

    act = t.column("action").to_pylist()
    act[5][0] = float("nan")
    put("action", act, pa.list_(pa.float32()))
    assert any("action has NaN or inf" in e for e in validate(ds))
    ts = t.column("timestamp").to_pylist()
    put("timestamp", [x * 2 for x in ts], pa.float32())  # written at the wrong fps
    assert any("timestamp row 1 is" in e for e in validate(ds))
    put("timestamp", ts, pa.float32())
    assert validate(ds) == []
    info_p = ds / "meta" / "info.json"
    info = json.loads(info_p.read_text())
    info["fps"] = 0
    info_p.write_text(json.dumps(info))
    assert any("info.fps must be a positive number" in e for e in validate(ds))
    info["fps"] = True
    info_p.write_text(json.dumps(info))
    assert any("info.fps must be a positive number" in e for e in validate(ds))


def test_generic_writer(tmp_path):
    import numpy as np
    from arrival.export.lerobot import LeRobotWriter
    modality = {"state": {"arm": {"start": 0, "end": 5}, "base": {"start": 5, "end": 7}},
                "action": {"arm": {"start": 0, "end": 5, "absolute": True}},
                "annotation": {"human.task_description": {}}}
    w = LeRobotWriter(tmp_path / "ds", "r", "all", state_dim=7, action_dim=5, modality=modality,
                      video_keys=["front", "wrist"], embodiment={"embodiment": "test"},
                      extra_info={"arrival_generator": "mujoco_test", "arrival_physically_valid": True})
    T = 12
    for ep in range(3):
        vids = {k: np.full((T, 32, 48, 3), 40 * ep, dtype=np.uint8) for k in ("front", "wrist")}
        w.add_episode(np.zeros((T, 7)), np.ones((T, 5)), np.arange(T) / 20, f"task {ep % 2}", True,
                      videos=vids, extra={"arrival_family": "f"})
    with pytest.raises(ValueError):
        w.add_episode(np.zeros((T, 6)), np.ones((T, 5)), np.arange(T) / 20, "x", True)
    info = w.finish(3)
    assert info["arrival_generator"] == "mujoco_test" and info["total_videos"] == 6
    assert info["features"]["observation.images.wrist"]["shape"] == [32, 48, 3]
    assert validate(tmp_path / "ds") == []
    m = json.loads((tmp_path / "ds" / "meta" / "modality.json").read_text())
    m["state"]["base"]["end"] = 8
    (tmp_path / "ds" / "meta" / "modality.json").write_text(json.dumps(m))
    assert any("slices cover 8" in e for e in validate(tmp_path / "ds"))


def _writer(tmp_path, **kw):
    from arrival.export.lerobot import LeRobotWriter

    return LeRobotWriter(tmp_path / "ds", "room", kw.pop("policy", "none"), state_dim=2, action_dim=2, **kw)


@pytest.mark.parametrize("fps", [0, -1, True, 20.5, "20"])
def test_writer_rejects_invalid_fps_before_creating_dataset(tmp_path, fps):
    with pytest.raises(ValueError, match="fps"):
        _writer(tmp_path, fps=fps)

@pytest.mark.parametrize("chunks_size", [0, -1, True, 2.5, "1000"])
def test_writer_rejects_invalid_chunk_size_before_creating_dataset(tmp_path, chunks_size):
    with pytest.raises(ValueError, match="chunks_size"):
        _writer(tmp_path, chunks_size=chunks_size)
    assert not (tmp_path / "ds").exists()



@pytest.mark.parametrize("names", [["../outside"], ["ego_view", "ego_view"], ["ego/view"], []])
def test_writer_rejects_unsafe_or_duplicate_camera_keys(tmp_path, names):
    with pytest.raises(ValueError, match="camera name|video_keys"):
        _writer(tmp_path, policy="all", video_keys=names)
    assert not (tmp_path / "ds").exists()


def test_writer_rejects_unsafe_supplied_video_name_before_writing(tmp_path):
    import numpy as np

    w = _writer(tmp_path, policy="all")
    z = np.zeros((2, 2))
    clip = np.zeros((2, 32, 32, 3), np.uint8)
    with pytest.raises(ValueError, match="camera name"):
        w.add_episode(z, z, [0, .05], "task", True, videos={"ego_view": clip, "../outside": clip})
    assert not list((tmp_path / "ds").glob("data/**/*.parquet"))
    assert not (tmp_path / "outside").exists()
    w.close()


def test_writer_rejects_nan_and_leaves_no_orphan(tmp_path):
    import numpy as np

    with _writer(tmp_path) as w:
        good = np.zeros((3, 2))
        bad = good.copy()
        bad[1, 1] = np.nan
        for s, a, t in ((bad, good, [0, .05, .1]), (good, bad, [0, .05, .1]), (good, good, [0, np.inf, .1])):
            with pytest.raises(ValueError, match="NaN or inf"):
                w.add_episode(s, a, t, "x", True)
        assert not list((tmp_path / "ds").glob("data/**/*.parquet"))
        w.add_episode(good, good, [0, .05, .1], "x", True)
        w.finish(1)
    json.loads((tmp_path / "ds" / "meta" / "stats.json").read_text(), parse_constant=lambda c: pytest.fail(c))


@pytest.mark.parametrize("instruction", [None, "", "  ", 5])
def test_writer_rejects_missing_task_instruction_before_writing(tmp_path, instruction):
    import numpy as np

    w = _writer(tmp_path)
    z = np.zeros((2, 2))
    with pytest.raises(ValueError, match="instruction must be a nonempty string"):
        w.add_episode(z, z, [0, .05], instruction, True)
    assert not list((tmp_path / "ds").glob("data/**/*.parquet"))
    w.close()


def test_writer_checks_videos_before_writing_and_closes_sidecar(tmp_path):
    import numpy as np

    w = _writer(tmp_path, policy="all")
    z = np.zeros((2, 2))
    with pytest.raises(ValueError, match="needs frames"):
        w.add_episode(z, z, [0, .05], "x", True)
    with pytest.raises(ValueError, match="multiples of 16"):
        w.add_episode(z, z, [0, .05], "x", True, videos={"ego_view": np.zeros((2, 40, 40, 3), np.uint8)})
    assert not list((tmp_path / "ds").glob("data/**/*.parquet"))
    w.close()
    assert w._sidecar.closed


def test_writer_rejects_video_pixel_coercion_before_writing(tmp_path):
    import numpy as np

    w = _writer(tmp_path, policy="all")
    z = np.zeros((2, 2))
    bad = np.full((2, 32, 32, 3), 300, dtype=np.int16)
    with pytest.raises(ValueError, match="uint8 RGB pixels"):
        w.add_episode(z, z, [0, .05], "task", True, videos={"ego_view": bad})
    assert not list((tmp_path / "ds").glob("data/**/*.parquet"))
    w.close()


@pytest.mark.parametrize("size", [0, -1, True, 2.5, "1000", None])
def test_validator_reports_invalid_chunk_size_without_crashing(tmp_path, size):
    import numpy as np

    modality = {"state": {"arm": {"start": 0, "end": 2}},
                "action": {"arm": {"start": 0, "end": 2, "absolute": True}},
                "annotation": {"human.task_description": {}}}
    writer = _writer(tmp_path, modality=modality)
    writer.add_episode(np.zeros((2, 2)), np.zeros((2, 2)), np.arange(2) / 20, "task", True)
    writer.finish(1)
    root = tmp_path / "ds"
    assert validate(root) == []
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["chunks_size"] = size
    info_path.write_text(json.dumps(info))
    errors = validate(root)
    assert any("chunks_size" in error and "positive integer" in error for error in errors)
