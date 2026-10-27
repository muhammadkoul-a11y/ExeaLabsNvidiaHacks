"""scripts/export_demo_data.py packs a recorded episode the way site/js/data.js reads it: 8 joint values per frame
(arm and gripper), even though the sim state also carries the 3-float base pose."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")

from conftest import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
import export_demo_data  # noqa: E402
from arrival.sim.collect import FPS  # noqa: E402
from arrival.sim.export import STATE_DIM  # noqa: E402


def test_joint_states_are_arm_and_gripper_only():
    n = 2 * FPS  # two seconds of control steps
    states = np.arange(n * STATE_DIM, dtype=np.float64).reshape(n, STATE_DIM) / 100
    rec = SimpleNamespace(names=["link1"], pos=[[[0.0, 0.0, float(i) / n]] for i in range(n)],
                          quat=[[[1.0, 0.0, 0.0, 0.0]]] * n, states=list(states), people_xy=[[]] * n, joint_q=[])
    packed = export_demo_data.pack_episode(rec, export_demo_data.OUT_FPS)
    rows = packed["joint_states"]
    assert len(rows) == packed["n_frames"]
    assert all(len(r) == 8 for r in rows)
    assert rows[0] == pytest.approx(states[0, :8].tolist())  # the first 8 state values, not the base pose


def test_stats_lists_every_file_with_its_real_size(tmp_path):
    # keys are site paths ("episodes/a.json", not "episodes\\a.json" on Windows), files are written with LF on every OS,
    # and stats.json's own entry is its final size, not the size it had before `files` was filled in
    out = tmp_path / "data"
    export_demo_data.dump(out / "episodes" / "a.json", {"k": [1, 2]}, indent=1)
    (out / "scan_points.bin").write_bytes(bytes(9))
    sizes = export_demo_data.write_stats(out, {"room_id": "r", "families": {"a": {"frames": 3}}})
    on_disk = {p.relative_to(out).as_posix(): p.stat().st_size for p in out.rglob("*") if p.is_file()}
    assert sizes == on_disk
    assert set(sizes) == {"episodes/a.json", "scan_points.bin", "stats.json"}
    assert b"\r" not in (out / "episodes" / "a.json").read_bytes() + (out / "stats.json").read_bytes()
    assert json.loads((out / "stats.json").read_text())["files"] == sizes


def test_committed_stats_match_the_committed_files():
    # the site serves site/data as git stores it (LF), so sizes are compared on LF bytes whatever the checkout does;
    # stats.json (its own size) and the hand-edited KEEP files change without a re-export, so they are left out
    data = REPO / "site" / "data"
    files = json.loads((data / "stats.json").read_text())["files"]
    assert not [k for k in files if "\\" in k]
    skip = {"stats.json", *export_demo_data.KEEP}
    def lf_size(path):  # git converts line endings in text files only; every PNG has \r\n in its header
        raw = path.read_bytes()
        return len(raw.replace(b"\r\n", b"\n") if path.suffix in (".json", ".md") else raw)

    actual = {k: lf_size(data / k) for k in files if k not in skip}
    assert {k: files[k] for k in actual} == actual


def test_rerun_keeps_the_hand_edited_todo_list(tmp_path, monkeypatch):
    # the export clears its output folder; site/data/todo.json is the team's to-do list and must survive that
    out = tmp_path / "data"
    out.mkdir()
    (out / "todo.json").write_text('{"items": []}')
    (out / "stale.json").write_text("{}")
    def stop(*_args, **_kwargs):  # stop right after the output folder is reset, before any sim runs
        raise RuntimeError("stop")

    monkeypatch.setattr(export_demo_data, "compile_twin", stop)
    with pytest.raises(RuntimeError, match="stop"):
        export_demo_data.main(["--out", str(out), "--scratch", str(tmp_path / "scratch")])
    assert (out / "todo.json").read_text() == '{"items": []}'
    assert not (out / "stale.json").exists()


def test_unknown_room_leaves_the_output_folder_alone(tmp_path, capsys):
    out = tmp_path / "data"
    out.mkdir()
    (out / "rooms.json").write_text("{}")
    with pytest.raises(SystemExit) as e:
        export_demo_data.main(["--room", "nope_room", "--out", str(out), "--scratch", str(tmp_path / "scratch")])
    assert e.value.code == 2
    assert "no scan for room 'nope_room'" in capsys.readouterr().err
    assert (out / "rooms.json").read_text() == "{}"
