"""Keyboard teleoperation in the twin, timed, for the generated-vs-teleop baseline (L20, OPS-3 #67).

The operator types into a window that shows the scene (`TeleopWindow`; `--no-viewer` reads the terminal instead).
Keys move an end-effector target in the robot's own frame (forward = toward the counter the base faces), turn the
gripper about the vertical and open or close it; every control step (20 Hz) the target goes through the same IK as
the scripted expert and the joint targets drive the same `SimEnv`. Enter ends an attempt: the arm holds still for
one second and then the same physics oracle as `sim` decides success. A human contact fails the attempt, as it does
for generated data.

Each attempt's operator time runs from the moment its scene is ready to the moment it is judged, so thinking,
mistakes and failed attempts all count. The run writes `teleop_baseline.json` in the shape
`arrival.eval.cost.load_baseline` validates, with the cost computed from an hourly rate the operator passes in
(nothing is assumed), every attempt in a JSONL log, and with --record the successful demos as a LeRobot v2
dataset labelled `teleop_keyboard` (usable as DAT-4 seeds).
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np

from arrival.sim import ik
from arrival.sim.robot import FINGER_OPEN

STEP = 0.01  # metres per key press (lowercase); uppercase moves STEP_FAST
STEP_FAST = 0.04
YAW_STEP = np.radians(10)
HOLD_FRAMES = 20  # one second of holding still before the oracle judges an attempt
GENERATOR = "teleop_keyboard"

# key -> (forward, left, up) in the robot frame
MOVES = {"w": (1, 0, 0), "s": (-1, 0, 0), "a": (0, 1, 0), "d": (0, -1, 0), "r": (0, 0, 1), "f": (0, 0, -1)}
WINDOW_KEYS = ("w / s\na / d\nr / f\nq / e\nspace\nenter\nx\nesc",
               "forward / back (capital: 4 cm)\nleft / right\nup / down\nturn gripper\nopen / close gripper\n"
               "judge the attempt\ngive up the attempt\nquit")
HELP = """keys (type in the teleop window; with --no-viewer, in this terminal):
  w/s forward/back   a/d left/right   r/f up/down   (capital letter: 4 cm, else 1 cm)
  q/e turn gripper   space open/close   enter judge the attempt   x give up the attempt   esc or ctrl-c quit"""


def _full_action(a, state) -> np.ndarray:
    """The dataset action carries base x, y, yaw after the gripper; an arm-only command holds the base where it is."""
    a = np.asarray(a, dtype=np.float32)
    return a if len(a) >= len(state) else np.concatenate([a, state[len(a):]]).astype(np.float32)


class TeleopController:
    """End-effector target driven by keys; `action()` solves IK for it (warm-started from the arm's posture)."""

    def __init__(self, env):
        self.env = env
        self.pos = env.site_pos("eef").copy()
        x = env.data.site("eef").xmat.reshape(3, 3)[:, 0]
        self.yaw = float(np.arctan2(x[1], x[0]))
        b = env.data.body("robot_base").xmat.reshape(3, 3)
        self.fwd, self.left = b[:, 0].copy(), b[:, 1].copy()
        self.fwd[2] = self.left[2] = 0.0
        self.gripper = FINGER_OPEN
        self.q = env.data.qpos[env.qadr].copy()
        self.ik_error = 0.0
        self._dirty = True

    def apply(self, key: str) -> str | None:
        """Update the target for one key. Returns 'judge', 'give_up' or 'quit' for those keys, else None."""
        k = key.lower()
        if k in MOVES:
            f, l, u = MOVES[k]
            step = STEP_FAST if key.isupper() else STEP
            self.pos = self.pos + step * (f * self.fwd + l * self.left + np.array([0.0, 0.0, u]))
        elif k in ("q", "e"):
            self.yaw += YAW_STEP if k == "q" else -YAW_STEP
        elif key == " ":
            self.gripper = 0.0 if self.gripper > 0 else FINGER_OPEN
        elif key in ("\r", "\n"):
            return "judge"
        elif k == "x":
            return "give_up"
        elif key in ("\x1b", "\x03"):
            return "quit"
        else:
            return None
        self._dirty = True
        return None

    def action(self) -> np.ndarray:
        if self._dirty:
            env = self.env
            self.q, self.ik_error = ik.solve(env.model, env.data, self.pos, ik.down(self.yaw), self.q,
                                             env.qadr, env.dofadr)
            self._dirty = False
        return np.concatenate([self.q, [self.gripper]])


class TerminalKeys:
    """Non-blocking key reader for a real terminal (raw mode on POSIX, msvcrt on Windows). Iterating yields the
    keys typed since the last frame, as a list, forever."""

    def __enter__(self):
        if sys.platform != "win32":
            import termios
            import tty

            self.fd = sys.stdin.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if sys.platform != "win32":
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def __iter__(self) -> Iterator[list[str]]:
        while True:
            yield self._read()

    def _read(self) -> list[str]:
        keys = []
        if sys.platform == "win32":
            import msvcrt

            while msvcrt.kbhit():
                keys.append(msvcrt.getwch())
            return keys
        import os
        import select

        while select.select([self.fd], [], [], 0)[0]:
            ch = os.read(self.fd, 1).decode(errors="ignore")
            if not ch:
                break
            keys.append(ch)
        return keys


def run(twin: dict, frames: Iterable[list[str]], out_dir: str | Path = "out", operator_usd_per_hour: float = 0.0,
        setup_usd: float = 0.0, setup_assumptions: list[str] | None = None, target_successes: int = 20,
        family: str = "counter_to_sink", people: int = 0, seed: int = 0, res: int = 128, record: bool = False,
        realtime: bool = True, viewer_factory: Callable | None = None, clock: Callable[[], float] = time.monotonic,
        log=print) -> dict:
    """Teleoperate attempts until `target_successes` succeed or the operator quits (or `frames` runs out).
    `frames` yields the keys typed during each 1/20 s control step. Returns a summary with the paths written."""
    from arrival.sim.collect import FPS, SimEnv
    from arrival.sim.export import SimDatasetWriter
    from arrival.sim.tasks import finish_task, make_tasks

    out_dir = Path(out_dir)
    room = twin["room_id"]
    tasks = [finish_task(t) for t in make_tasks(twin, [family], max(target_successes * 3, 1), seed=seed,
                                                    people=people, log=log)]
    if not tasks:
        raise ValueError(f"no {family} task in {room}")
    log_path = out_dir / "reports" / f"{room}_teleop_attempts.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    writer = SimDatasetWriter(out_dir / "datasets" / f"{room}_teleop", room, FPS, res, ("ego_view", "wrist")) \
        if record else None
    frames = iter(frames)
    attempts, successes, operator_seconds, quit_ = [], 0, 0.0, False
    while successes < target_successes and not quit_:
        task = tasks[len(attempts) % len(tasks)]
        env = SimEnv(twin, task, res=res)
        viewer = viewer_factory(env) if viewer_factory else None
        ctl = TeleopController(env)
        rec = {"states": [], "actions": [], "videos": {"ego_view": [], "wrist": []}, "people_xy": []}
        env.safety.reset()
        log(f"[teleop] attempt {len(attempts) + 1}: {task['instruction']}  ({successes}/{target_successes} done)")
        t0, outcome, steps = clock(), None, 0
        try:
            while outcome is None:
                tick = time.perf_counter()
                keys = next(frames, None)
                if keys is None:
                    outcome = "quit"
                    break
                for k in keys:
                    outcome = ctl.apply(k) or outcome
                a = ctl.action()
                if record:
                    rec["states"].append(env.state())
                    rec["actions"].append(_full_action(a, rec["states"][-1]))
                    for cam in rec["videos"]:
                        rec["videos"][cam].append(env.render(cam))
                env.step(a)
                rec["people_xy"].append(env.people_xy)
                steps += 1
                if viewer is not None:
                    viewer.sync()
                if realtime:
                    time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - tick)))
            if outcome == "judge":
                for _ in range(HOLD_FRAMES):  # the object must have come to rest for the oracle
                    a = ctl.action()
                    if record:
                        rec["states"].append(env.state())
                        rec["actions"].append(_full_action(a, rec["states"][-1]))
                        for cam in rec["videos"]:
                            rec["videos"][cam].append(env.render(cam))
                    env.step(a)
                    rec["people_xy"].append(env.people_xy)
                    if viewer is not None:
                        viewer.sync()
                ok, detail = env.success(task)
                safety = env.safety.summary()
                ok = bool(ok and not safety["human_contact"])
            else:
                ok, detail, safety = False, {"ended_by": outcome}, env.safety.summary()
        finally:
            if viewer is not None:
                viewer.close()
            env.close()
        seconds = clock() - t0
        if outcome == "quit" and steps == 0:
            break  # quit before touching anything: not an attempt
        operator_seconds += seconds
        row = {"attempt": len(attempts), "task_id": task["task_id"], "instruction": task["instruction"],
               "success": ok, "operator_seconds": round(seconds, 3), "control_steps": steps,
               "ended_by": outcome, "detail": detail, "human_contact": safety["human_contact"],
               "max_ik_error_m": round(float(ctl.ik_error), 4)}
        attempts.append(row)
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        if ok:
            successes += 1
            if writer is not None:
                ep = {"states": np.array(rec["states"]), "actions": np.array(rec["actions"]),
                      "videos": {c: np.array(v) for c, v in rec["videos"].items()}, "detail": detail,
                      "safety": safety, "people_xy": rec["people_xy"]}
                writer.add_episode(ep, task)
        log(f"[teleop] {'SUCCESS' if ok else 'fail'} in {seconds:.1f}s ({outcome}) {detail}")
        quit_ = outcome == "quit"
    summary = {"room": room, "task": family, "attempts": len(attempts), "n_success": successes,
               "operator_seconds_total": round(operator_seconds, 3), "attempts_log": str(log_path)}
    if writer is not None:
        info = writer.finish(len(attempts))
        info.update({"arrival_generator": GENERATOR, "arrival_physically_valid": True,
                     "arrival_teleop_input": "keyboard eef target + IK, binary gripper"})
        (writer.root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
        summary["dataset"] = str(writer.root)
    if successes == 0:
        log("[teleop] no successful attempt, so no teleop_baseline.json (the cost report needs n_success >= 1)")
        return summary
    baseline = {
        "task": family, "room": room, "n_success": successes, "n_attempts": len(attempts),
        "operator_seconds_total": round(operator_seconds, 3),
        "cost_basis_usd": round(operator_seconds / 3600 * operator_usd_per_hour + setup_usd, 4),
        "operator_usd_per_hour": operator_usd_per_hour, "setup_usd": setup_usd,
        "setup_assumptions": list(setup_assumptions or []) + [
            f"keyboard teleop in the MuJoCo twin of {room}, {people} walking people, success by the sim oracle",
            "operator time per attempt runs from scene ready to judgement and includes failed attempts"],
        "measured": True, "created_unix": int(time.time()), "attempts_log": str(log_path),
    }
    bp = out_dir / "teleop_baseline.json"
    bp.write_text(json.dumps(baseline, indent=2) + "\n")
    summary["baseline"] = str(bp)
    return summary


def frame_camera(env, cam) -> None:
    """Point a free camera over the robot's shoulder at its workspace. MuJoCo's default free camera frames the
    whole model from ~12 m away, which in a twin is outside the walls."""
    import mujoco

    b = env.model.body("robot_base").id
    base, fwd = env.data.xpos[b], env.data.xmat[b].reshape(3, 3)[:, 0]
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [base[0] + 0.45 * fwd[0], base[1] + 0.45 * fwd[1], 0.95]
    cam.azimuth = float(np.degrees(np.arctan2(fwd[1], fwd[0]))) + 30.0
    cam.elevation, cam.distance = -48.0, 1.9


class TeleopWindow:
    """Our own GLFW window: it draws the scene and is where the operator types. (The MuJoCo viewer binds most
    letters to its own render toggles, so keys typed into it switch on wireframe and shadows instead of moving the
    arm.) Iterating yields the keys typed since the last frame; closing the window quits. Left drag orbits the
    camera, right drag pans, scroll zooms. One window serves every attempt: `attach(env)` points it at a new scene.
    GLFW must run on the main thread, so on macOS start it with plain `python`, not `mjpython`."""

    def __init__(self, width: int = 1280, height: int = 800, visible: bool = True):
        import glfw
        import mujoco

        self.glfw, self.mj = glfw, mujoco
        if not glfw.init():
            raise RuntimeError("could not start GLFW (no display?)")
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
        self.win = glfw.create_window(width, height, "Arrival teleop", None, None)
        if not self.win:
            glfw.terminate()
            raise RuntimeError("could not open a window (no display?)")
        glfw.make_context_current(self.win)
        glfw.swap_interval(0)
        self.keys: list[str] = []
        self.env = self.ctx = self.scn = None
        self.cam, self.opt = mujoco.MjvCamera(), mujoco.MjvOption()
        self._framed, self._drag, self._last = False, None, (0.0, 0.0)
        glfw.set_char_callback(self.win, lambda w, cp: self._char(chr(cp)))
        glfw.set_key_callback(self.win, self._on_key)
        glfw.set_mouse_button_callback(self.win, self._on_button)
        glfw.set_cursor_pos_callback(self.win, self._on_cursor)
        glfw.set_scroll_callback(self.win, self._on_scroll)

    def _char(self, ch: str) -> None:
        """Printable keys, with the case the keyboard produced (capital letters move 4 cm)."""
        self.keys.append(ch)

    def _on_key(self, w, key, scancode, action, mods):
        g = self.glfw
        if action == g.RELEASE:
            return
        if key in (g.KEY_ENTER, g.KEY_KP_ENTER):
            self.keys.append("\r")
        elif key == g.KEY_ESCAPE:
            self.keys.append("\x1b")

    def _on_button(self, w, button, action, mods):
        g = self.glfw
        left = g.get_mouse_button(w, g.MOUSE_BUTTON_LEFT) == g.PRESS
        right = g.get_mouse_button(w, g.MOUSE_BUTTON_RIGHT) == g.PRESS
        self._drag = "rotate" if left else "move" if right else None
        self._last = g.get_cursor_pos(w)

    def _on_cursor(self, w, x, y):
        dx, dy = x - self._last[0], y - self._last[1]
        self._last = (x, y)
        if self._drag and self.env is not None:
            m = self.mj.mjtMouse
            act = m.mjMOUSE_ROTATE_V if self._drag == "rotate" else m.mjMOUSE_MOVE_V
            h = max(1, self.glfw.get_window_size(w)[1])
            self._move_camera(act, dx / h, dy / h)

    def _on_scroll(self, w, xoff, yoff):
        if self.env is not None:
            self._move_camera(self.mj.mjtMouse.mjMOUSE_ZOOM, 0.0, 0.05 * yoff)  # scroll up zooms in

    def _move_camera(self, action, dx: float, dy: float) -> None:
        """mjv_moveCamera dropped its scene argument in MuJoCo 3.16; pyproject allows >= 3.1, so try both."""
        try:
            self.mj.mjv_moveCamera(self.env.model, int(action), dx, dy, self.cam)
        except TypeError:
            self.mj.mjv_moveCamera(self.env.model, int(action), dx, dy, self.scn, self.cam)

    def attach(self, env) -> TeleopWindow:
        """Show `env` (the first time, from over the robot's shoulder; after that the operator's camera stays)."""
        mj = self.mj
        self.glfw.make_context_current(self.win)  # SimEnv's offscreen renderers switch the GL context
        if self.ctx is not None:
            self.ctx.free()
        self.env = env
        self.scn = mj.MjvScene(env.model, maxgeom=10000)
        self.ctx = mj.MjrContext(env.model, mj.mjtFontScale.mjFONTSCALE_150)
        if not self._framed:
            frame_camera(env, self.cam)
            self._framed = True
        self.sync()
        self.glfw.focus_window(self.win)
        return self

    def sync(self) -> None:
        """Draw the current state, with the key help in a corner."""
        mj, env = self.mj, self.env
        self.glfw.make_context_current(self.win)
        w, h = self.glfw.get_framebuffer_size(self.win)
        rect = mj.MjrRect(0, 0, w, h)
        mj.mjv_updateScene(env.model, env.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL, self.scn)
        mj.mjr_render(rect, self.scn, self.ctx)
        task = getattr(env, "task", {}) or {}
        mj.mjr_overlay(mj.mjtFont.mjFONT_NORMAL, mj.mjtGridPos.mjGRID_TOPLEFT, rect, task.get("instruction", ""),
                       "", self.ctx)
        mj.mjr_overlay(mj.mjtFont.mjFONT_NORMAL, mj.mjtGridPos.mjGRID_BOTTOMLEFT, rect, WINDOW_KEYS[0],
                       WINDOW_KEYS[1], self.ctx)
        self.glfw.swap_buffers(self.win)
        self.glfw.poll_events()

    def __iter__(self) -> Iterator[list[str]]:
        while True:
            self.glfw.poll_events()
            if self.glfw.window_should_close(self.win):
                yield ["\x1b"]
                continue
            keys, self.keys = self.keys, []
            yield keys

    def close(self) -> None:
        """Per attempt: keep the window for the next one (`shutdown` closes it)."""

    def shutdown(self) -> None:
        if self.ctx is not None:
            self.glfw.make_context_current(self.win)
            self.ctx.free()
            self.ctx = None
        self.glfw.destroy_window(self.win)
        self.glfw.terminate()


def main(argv=None) -> int:
    import argparse

    from arrival.scan.bundle import load_room
    from arrival.twin.compile import compile_twin

    ap = argparse.ArgumentParser(description="Timed keyboard teleop in the twin -> out/teleop_baseline.json",
                                 epilog=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room", default="hero_kitchen")
    ap.add_argument("--scans", default="scans")
    ap.add_argument("--out", default="out")
    ap.add_argument("--operator-usd-per-hour", type=float, required=True,
                    help="what an hour of operator time costs; the baseline's dollars come only from this")
    ap.add_argument("--setup-usd", type=float, default=0.0, help="one-off setup cost to add to the cost basis")
    ap.add_argument("--assumption", action="append", default=[], help="extra setup assumption (repeatable)")
    ap.add_argument("--successes", type=int, default=20, help="stop after this many successful demos")
    ap.add_argument("--people", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--record", action="store_true", help="also write the successful demos as a LeRobot v2 dataset")
    ap.add_argument("--no-viewer", action="store_true", help="no window (for a quick terminal-only check)")
    a = ap.parse_args(argv)
    if not all(math.isfinite(c) and c >= 0 for c in (a.operator_usd_per_hour, a.setup_usd)) or a.successes < 1:
        ap.error("costs must be finite numbers >= 0 and --successes >= 1")
    if a.people < 0 or a.seed < 0:
        ap.error("--people and --seed must be >= 0")
    if a.res <= 0 or a.res % 16:
        ap.error("--res must be a positive multiple of 16")
    if a.no_viewer and not sys.stdin.isatty():
        ap.error("--no-viewer reads keys from the terminal, so it needs an interactive one")
    try:
        bundle = load_room(a.room, a.scans)
    except FileNotFoundError:
        ap.error(f"no scan for room {a.room!r} under {a.scans}")
    twin = compile_twin(bundle)
    print(HELP, flush=True)
    kw = dict(people=a.people, seed=a.seed, res=a.res, record=a.record)
    args = (a.out, a.operator_usd_per_hour, a.setup_usd, a.assumption, a.successes)
    if a.no_viewer:
        with TerminalKeys() as keys:
            s = run(twin, keys, *args, **kw)
    else:
        win = TeleopWindow()
        try:
            s = run(twin, win, *args, viewer_factory=win.attach, **kw)
        finally:
            win.shutdown()
    print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
