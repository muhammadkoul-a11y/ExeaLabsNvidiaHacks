"""L20: keyboard teleop drives the same env, oracle and writer as `sim`, and writes a baseline the cost report reads."""

import itertools
import json

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from arrival.eval.cost import load_baseline  # noqa: E402
from arrival.export.validate import validate  # noqa: E402
from arrival.scan.bundle import load_room  # noqa: E402
from arrival.sim import teleop  # noqa: E402
from arrival.sim.collect import SimEnv  # noqa: E402
from arrival.sim.tasks import finish_task, make_tasks  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

from conftest import SCANS  # noqa: E402

QUIET = lambda *_: None  # noqa: E731


@pytest.fixture(scope="module")
def twin():
    """The hero kitchen with only the mug to pick, so the scripted keyboard demo grips by the body (a bowl needs
    the rim grasp, which the key script below does not do)."""
    t = compile_twin(load_room("hero_kitchen", SCANS))
    t["objects"] = [o for o in t["objects"] if o["catalog_id"] in ("mug", "sponge")]
    return t


def _clock(step=0.05):
    c = itertools.count()
    return lambda: next(c) * step


def _moves(ctl, goal) -> list[list[str]]:
    """One key per frame that walks the controller's target to `goal` in 1 cm steps, in the robot frame."""
    d = np.asarray(goal) - ctl.pos
    out = []
    for comp, (pos_key, neg_key) in ((d @ ctl.fwd, ("w", "s")), (d @ ctl.left, ("a", "d")), (d[2], ("r", "f"))):
        n = int(round(comp / teleop.STEP))
        out += [[pos_key if n > 0 else neg_key]] * abs(n)
    ctl.pos = ctl.pos + sum(np.array({"w": ctl.fwd, "s": -ctl.fwd, "a": ctl.left, "d": -ctl.left,
                                       "r": [0, 0, 1], "f": [0, 0, -1]}[k[0]]) * teleop.STEP for k in out) if out else ctl.pos
    return out


def _demo_frames(twin) -> list[list[str]]:
    """The key presses an operator would make for the first counter_to_sink task: over the mug, down, close,
    up, over the sink, down, open, judge."""
    task = finish_task(make_tasks(twin, ["counter_to_sink"], 60, seed=0, people=0, log=QUIET)[0])
    with SimEnv(twin, task, res=64) as env:
        ctl = teleop.TeleopController(env)
        mug = env.body_pos(task["object_id"])
        gz = mug[2] + min(0.01, env.obj_half_h(task["object_id"]) * 0.2)
        tgt = env.place_point(task)
    wait = lambda n: [[]] * n  # noqa: E731
    f = []
    f += _moves(ctl, [ctl.pos[0], ctl.pos[1], gz + 0.18]) + wait(10)
    f += _moves(ctl, [mug[0], mug[1], gz + 0.18]) + wait(30)
    f += _moves(ctl, [mug[0], mug[1], gz]) + wait(25)
    f += [[" "]] + wait(15)
    f += _moves(ctl, [mug[0], mug[1], gz + 0.22]) + wait(20)
    f += _moves(ctl, [tgt[0], tgt[1], gz + 0.22]) + wait(30)
    f += _moves(ctl, [tgt[0], tgt[1], tgt[2]]) + wait(20)
    f += [[" "]] + wait(15)
    return f + [["\r"]]


def test_keys_move_the_target_in_the_robot_frame(twin):
    task = finish_task(make_tasks(twin, ["counter_to_sink"], 1, seed=0, people=0, log=QUIET)[0])
    with SimEnv(twin, task, res=64) as env:
        ctl = teleop.TeleopController(env)
        p0, yaw0 = ctl.pos.copy(), ctl.yaw
        ctl.apply("w")
        np.testing.assert_allclose(ctl.pos - p0, teleop.STEP * ctl.fwd, atol=1e-12)
        ctl.apply("D")
        np.testing.assert_allclose(ctl.pos - p0, teleop.STEP * ctl.fwd - teleop.STEP_FAST * ctl.left, atol=1e-12)
        ctl.apply("r")
        assert ctl.pos[2] - p0[2] == pytest.approx(teleop.STEP)
        ctl.apply("q")
        assert ctl.yaw == pytest.approx(yaw0 + teleop.YAW_STEP)
        g = ctl.gripper
        ctl.apply(" ")
        assert ctl.gripper != g
        assert ctl.apply("\r") == "judge" and ctl.apply("x") == "give_up" and ctl.apply("\x1b") == "quit"
        assert ctl.apply("z") is None
        assert abs(float(np.dot(ctl.fwd, ctl.left))) < 1e-9 and ctl.fwd[2] == 0.0


def test_scripted_keyboard_demo_succeeds_and_writes_a_valid_baseline(twin, tmp_path):
    s = teleop.run(twin, _demo_frames(twin), tmp_path, operator_usd_per_hour=36.0, setup_usd=5.0,
                   setup_assumptions=["test operator"], target_successes=1, res=64, record=True, realtime=False,
                   clock=_clock(), log=QUIET)
    assert s["n_success"] == 1 and s["attempts"] == 1
    b = load_baseline(tmp_path / "teleop_baseline.json")  # the cost report accepts it
    assert b["task"] == "counter_to_sink" and b["room"] == "hero_kitchen" and b["n_success"] == 1
    assert b["cost_basis_usd"] == pytest.approx(b["operator_seconds_total"] / 3600 * 36.0 + 5.0, abs=1e-4)
    assert "test operator" in b["setup_assumptions"]
    rows = [json.loads(ln) for ln in (tmp_path / "reports" / "hero_kitchen_teleop_attempts.jsonl").read_text().splitlines()]
    assert rows[0]["success"] and rows[0]["ended_by"] == "judge"
    ds = tmp_path / "datasets" / "hero_kitchen_teleop"
    assert validate(ds) == []
    info = json.loads((ds / "meta" / "info.json").read_text())
    assert info["arrival_generator"] == "teleop_keyboard" and info["total_episodes"] == 1
    assert info["arrival_physically_valid"] is True


def test_failed_and_abandoned_attempts_count_time_but_write_no_baseline(twin, tmp_path):
    frames = [["w"], ["\r"]] + [["a"]] * 3 + [["x"]]  # judged at once (mug still on the counter), then given up
    s = teleop.run(twin, frames, tmp_path, operator_usd_per_hour=36.0, target_successes=2, res=64,
                   realtime=False, clock=_clock(), log=QUIET)
    assert s["n_success"] == 0 and s["attempts"] == 2 and s["operator_seconds_total"] > 0
    assert not (tmp_path / "teleop_baseline.json").exists()
    rows = [json.loads(ln) for ln in (tmp_path / "reports" / "hero_kitchen_teleop_attempts.jsonl").read_text().splitlines()]
    assert [r["ended_by"] for r in rows] == ["judge", "give_up"] and not any(r["success"] for r in rows)


@pytest.mark.parametrize("extra, why", [
    (["--operator-usd-per-hour", "nan"], "finite"),
    (["--operator-usd-per-hour", "40", "--setup-usd", "inf"], "finite"),
    (["--operator-usd-per-hour", "40", "--people", "-1"], "--people"),
    (["--operator-usd-per-hour", "40", "--seed", "-1"], "--seed"),
    (["--operator-usd-per-hour", "40", "--res", "100"], "--res"),
])
def test_bad_arguments_stop_before_the_session(extra, why, tmp_path, capsys, monkeypatch):
    def session_started(*_a, **_k):
        raise AssertionError("the teleop session started")
    monkeypatch.setattr(teleop, "TeleopWindow", session_started)
    with pytest.raises(SystemExit) as e:
        teleop.main([*extra, "--scans", str(SCANS), "--out", str(tmp_path)])
    assert e.value.code == 2
    assert why in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_window_turns_typed_keys_into_frames(twin):
    """The teleop window, not the MuJoCo viewer, receives the keys (the viewer would eat w/s/r as render toggles)."""
    try:
        win = teleop.TeleopWindow(visible=False)
    except Exception as e:  # no display (CI)
        pytest.skip(f"no window: {e}")
    try:
        task = finish_task(make_tasks(twin, ["counter_to_sink"], 1, seed=0, people=0, log=QUIET)[0])
        with SimEnv(twin, task, res=64) as env:
            win.attach(env)
            frames = iter(win)
            assert next(frames) == []
            for ch in "wA ":
                win._char(ch)
            win._on_key(win.win, win.glfw.KEY_ENTER, 0, win.glfw.PRESS, 0)
            win._on_key(win.win, win.glfw.KEY_ESCAPE, 0, win.glfw.RELEASE, 0)  # releases are not keys
            assert next(frames) == ["w", "A", " ", "\r"]
            az, dist = win.cam.azimuth, win.cam.distance
            win._on_scroll(win.win, 0.0, 1.0)  # zoom in
            win._drag = "rotate"
            win._on_cursor(win.win, win._last[0] + 40, win._last[1])
            assert win.cam.distance < dist and win.cam.azimuth != az
            win.sync()
    finally:
        win.shutdown()


def test_unknown_room_is_a_usage_error_not_a_traceback(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        teleop.main(["--room", "nope_room", "--scans", str(tmp_path), "--operator-usd-per-hour", "10"])
    assert e.value.code == 2
    assert "no scan for room 'nope_room'" in capsys.readouterr().err
