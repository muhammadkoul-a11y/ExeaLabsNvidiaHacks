"""`python -m arrival realism <scan>`: how far the twin's render is from the real held-out frames of that scan.

A thin driver over arrival/eval/realism.py (the metrics) and arrival/scan/rgbd.py (the frame manifest). It adds only
what a measurement needs to be quotable:

  * held-out frames only. Frames the manifest marks `train` are never scored; the JSON lists the ids it used and
    how many it left out. A scan with no held-out frame is refused.
  * invalid depth is masked: a pixel counts for depth only if the reference depth is finite and above 0 and its
    ARKit confidence (when the bundle has a confidence map) is at least --min-confidence (2 = high). The twin render
    hides the robot and people, and the recorder films the empty room, so no robot mask is needed or applied.
  * a misaligned-pose control: the same frames are scored again with the camera moved 0.15 m sideways and turned
    5 degrees. If depth error does not rise, the poses or the twin's frame are wrong and the number must not be
    quoted; the command then exits 1 and says so.
  * a four-photo or RoomPlan-only scan has no frames: it is refused, never scored. A bundle whose blocks_claims holds
    `realism` reports realism null with the reason.

The twin is a schematic box-and-colour render (MuJoCo, no textures), so rgb_mae and SSIM mostly measure missing
texture and lighting; depth_l1_m measures geometry. The JSON says so (`scope`). Frames larger than --max-side are
downscaled (intrinsics scaled with them, depth by nearest neighbour so no depth is invented by interpolation).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from arrival.export.provenance import git_state, sha256_file
from arrival.scan import rgbd

SHIFT_M, YAW_DEG = 0.15, 5.0  # the misaligned-pose control (eval/realism.perturb_pose)
MAX_SIDE = 192
MAX_FRAMES = 8
PNG_FRAMES = 3
MIN_CONFIDENCE = 2  # ARKit ARConfidenceLevel: 0 low, 1 medium, 2 high
ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # ARKit camera axes (y up, z back) -> OpenCV (y down, z forward)
SCOPE = ("Schematic MuJoCo box render against reference frames: depth_l1_m measures geometry; rgb_mae, "
         "rgb_one_minus_ssim and rgb_psnr_db mostly measure missing texture and lighting, not layout.")


class RealismError(ValueError):
    """The measurement cannot be made (or cannot be trusted); the message says why and what to do."""


def resolve_scan(scan: str | Path, scans_dir: str | Path = "scans") -> Path:
    p = Path(scan)
    if p.is_dir():
        return p
    q = Path(scans_dir) / str(scan)
    if q.is_dir():
        return q
    raise RealismError(f"no scan folder {scan!r} (looked at {p} and {q})")


def manifest_dir(scan_dir: Path) -> Path:
    for d in (scan_dir, scan_dir / "frames"):
        if (d / rgbd.MANIFEST_NAME).is_file():
            return d
    raise RealismError(
        f"{scan_dir} has no {rgbd.MANIFEST_NAME}: realism needs posed RGB-D frames next to the scan. A RoomPlan-only "
        f"or four-photo scan cannot support a realism claim; record frames with the RGB-D recorder (docs/SCANNING.md)")


def load_twin(scan_dir: Path, twin_path: str | Path | None = None) -> tuple[dict, str]:
    """(twin, where it came from). An explicit twin.json wins; else the scan's roomplan.json (a real capture),
    else its scan.json."""
    from arrival.scan import bundle as scan_bundle
    from arrival.scan import roomplan
    from arrival.twin.compile import compile_twin

    room = scan_dir.name
    if twin_path is not None:
        p = Path(twin_path)
        if not p.is_file():
            raise RealismError(f"--twin {p} does not exist")
        try:
            twin = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise RealismError(f"--twin {p} is not readable JSON: {e}") from e
        missing = [k for k in ("walls", "fixtures", "objects")
                   if not isinstance(twin, dict) or not isinstance(twin.get(k), list)]
        if missing:
            raise RealismError(f"--twin {p} is not a compiled twin (no {', '.join(missing)} list); pass "
                               f"out/twins/<room_id>/twin.json")
        return twin, str(p)
    for name, load in (("roomplan.json", lambda p: roomplan.load(p, room)), ("scan.json", scan_bundle.load)):
        p = scan_dir / name
        if p.is_file():
            try:
                return compile_twin(load(p)), str(p)
            except ValueError as e:
                raise RealismError(f"{p} cannot be compiled into a twin: {e}") from e
    raise RealismError(f"{scan_dir} has neither roomplan.json nor scan.json (or pass --twin)")


def _nearest(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resample an HxW array to (width, height) = size by nearest neighbour: no interpolated depth."""
    h, w = a.shape[:2]
    tw, th = size
    ys = np.minimum((np.arange(th) * h / th + h / (2 * th)).astype(int), h - 1)
    xs = np.minimum((np.arange(tw) * w / tw + w / (2 * tw)).astype(int), w - 1)
    return a[np.ix_(ys, xs)]


def _open_image(path: Path, mode: str, frame_id: str):
    from PIL import Image

    try:
        with Image.open(path) as im:
            return im.convert(mode)
    except OSError as e:
        raise RealismError(f"frame {frame_id}: {path} is not a readable image: {e}") from e


def load_frame(m: rgbd.Manifest, f: rgbd.Frame, max_side: int = MAX_SIDE, min_confidence: int = MIN_CONFIDENCE) -> dict:
    """One manifest frame as the dict evaluate_frames takes: RGB, depth in metres, valid-depth mask and K at the
    working size, and cam_to_room in OpenCV axes (the convention of eval/realism.py)."""
    from PIL import Image

    rgb = _open_image(f.rgb, "RGB", f.id)
    if rgb.size != tuple(m.rgb_size):
        raise RealismError(f"frame {f.id}: rgb is {rgb.size[0]}x{rgb.size[1]}, the manifest says "
                           f"{m.rgb_size[0]}x{m.rgb_size[1]}")
    try:
        depth = rgbd.load_depth(m, f)
    except (OSError, EOFError, ValueError) as e:
        raise RealismError(f"frame {f.id}: depth {f.depth} cannot be used: {e}") from e
    valid = np.isfinite(depth) & (depth > 0)
    if f.confidence is not None:
        conf = np.asarray(_open_image(f.confidence, "L", f.id))
        if conf.shape != depth.shape:
            raise RealismError(f"frame {f.id}: confidence map is {conf.shape[::-1]}, depth is {depth.shape[::-1]}")
        valid &= conf >= min_confidence
    w, h = m.rgb_size
    scale = min(1.0, max_side / max(w, h))
    tw, th = max(1, round(w * scale)), max(1, round(h * scale))
    if (tw, th) != (w, h):
        rgb = rgb.resize((tw, th), Image.LANCZOS)
    K = f.K.copy()
    K[0] *= tw / w
    K[1] *= th / h
    depth_t = _nearest(np.where(valid, depth, 0.0), (tw, th)).astype(np.float32)
    return {"id": f.id, "rgb": np.asarray(rgb), "depth": depth_t, "valid_depth_mask": depth_t > 0, "K": K,
            "cam_to_room": f.camera_to_room @ ARKIT_TO_CV}


def _pick(frames: list, n: int) -> list:
    if len(frames) <= n:
        return frames
    return [frames[round(i * (len(frames) - 1) / (n - 1))] for i in range(n)] if n > 1 else frames[:1]


def _stack_png(path: Path, rows: list[Path]) -> None:
    from PIL import Image

    ims = [Image.open(p).convert("RGB") for p in rows]
    out = Image.new("RGB", (max(i.width for i in ims), sum(i.height for i in ims)))
    y = 0
    for im in ims:
        out.paste(im, (0, y))
        y += im.height
    out.save(path)


def run_realism(scan: str | Path, out_dir: str | Path = "out", scans_dir: str | Path = "scans",
                twin_path: str | Path | None = None, max_side: int = MAX_SIDE, max_frames: int = MAX_FRAMES,
                png_frames: int = PNG_FRAMES, min_confidence: int = MIN_CONFIDENCE, log=print,
                camera_transform=None) -> dict:
    """Score the held-out frames of a scan against the twin and write out/eval/<room>_realism.json and .png.
    Raises RealismError when no honest measurement can be made. Returns the report dict (with `path`, `png`)."""
    from arrival.eval.realism import evaluate_frames, perturb_pose, render_twin, side_by_side_png

    if type(max_side) is not int or max_side < 16 or type(max_frames) is not int or max_frames < 1:
        raise RealismError("--max-side must be at least 16 and --max-frames at least 1")
    if type(png_frames) is not int or png_frames < 0:
        raise RealismError("--png-frames must be a nonnegative integer")
    if type(min_confidence) is not int or min_confidence not in (0, 1, 2):
        raise RealismError("--min-confidence must be an ARKit confidence level: 0, 1, or 2")
    transform = None
    if camera_transform is not None:
        try:
            transform = np.asarray(camera_transform, dtype=float)
        except (TypeError, ValueError) as e:
            raise RealismError("camera_transform must be a finite 4x4 transform") from e
        if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(
                transform[3], [0, 0, 0, 1]):
            raise RealismError("camera_transform must be a finite 4x4 homogeneous transform")
    scan_dir = resolve_scan(scan, scans_dir)
    mdir = manifest_dir(scan_dir)
    try:
        manifest = rgbd.load_manifest(mdir)
    except rgbd.ManifestError as e:
        raise RealismError(f"{mdir / rgbd.MANIFEST_NAME} is not usable: {e}") from e
    twin, twin_source = load_twin(scan_dir, twin_path)
    heldout = [f for f in manifest.frames if f.split == "heldout"]
    train = [f for f in manifest.frames if f.split != "heldout"]
    if not heldout:
        raise RealismError(f"{mdir / rgbd.MANIFEST_NAME} has no held-out frame ({len(train)} train frames only); "
                           f"training frames cannot be used as evidence")
    used = _pick(heldout, max_frames)
    frames = [load_frame(manifest, f, max_side, min_confidence) for f in used]
    if transform is not None:
        for frame in frames:
            frame["cam_to_room"] = transform @ frame["cam_to_room"]
    ids = [f.id for f in used]
    size = frames[0]["rgb"].shape[:2]
    try:
        real = evaluate_frames(twin, frames, [f.id for f in heldout])
        control_frames = [{**f, "cam_to_room": perturb_pose(f["cam_to_room"], SHIFT_M, YAW_DEG)} for f in frames]
        control = evaluate_frames(twin, control_frames, [f.id for f in heldout])
    except ValueError as e:
        raise RealismError(str(e)) from e

    out_dir = Path(out_dir)
    stem = f"{twin.get('room_id') or scan_dir.name}_realism"
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "schema_version": 1,
        "room_id": twin.get("room_id") or scan_dir.name,
        "scan_dir": str(scan_dir),
        "twin_source": twin_source,
        "geometry_source": twin.get("geometry_source"),
        "frames": {"manifest": str(mdir / rgbd.MANIFEST_NAME), "manifest_sha256": sha256_file(mdir / rgbd.MANIFEST_NAME),
                   "total": len(manifest.frames), "heldout": len(heldout), "scored": ids,
                   "train_excluded": len(train), "working_size_hw": list(size)},
        "masks": {"invalid_depth": f"reference depth not finite, 0, or confidence < {min_confidence}",
                  "robot": "hidden in the twin render; the recorded frames contain no robot, so no robot mask"},
        "scope": SCOPE,
        "realism": real["realism"],
        "control": {"kind": "misaligned_pose", "shift_m": SHIFT_M, "yaw_deg": YAW_DEG, "realism": control["realism"]},
        "code": git_state(),
    }
    if real["realism"] is None:  # a four-photo bundle: the claim is blocked, no numbers
        report["reason"] = real["reason"]
        report["control"] = None
        report["alignment_check"] = None
        report["valid"] = False
    else:
        r, c = real["realism"], control["realism"]
        rises = {k: (r[k] is not None and c[k] is not None and c[k] > r[k]) for k in ("depth_l1_m", "rgb_mae")}
        report["alignment_check"] = {"error_rises_under_misalignment": rises,
                                     "passed": bool(rises["depth_l1_m"])}
        report["valid"] = bool(rises["depth_l1_m"])
        if not report["valid"]:
            report["reason"] = (f"depth error did not rise when the camera was moved {SHIFT_M} m and turned {YAW_DEG} "
                                f"degrees (real {r['depth_l1_m']}, control {c['depth_l1_m']}): the poses or the twin's "
                                f"frame are off, so these numbers must not be quoted")
    path = eval_dir / f"{stem}.json"
    report["path"] = str(path)
    png = None
    if real["realism"] is not None:
        rows = []
        pose_by_id = {f["id"]: f for f in frames}
        for index, fid in enumerate(ids[:png_frames]):
            f = pose_by_id[fid]
            rgb_sim, depth_sim = render_twin(twin, f["K"], f["cam_to_room"], f["rgb"].shape[:2])
            row = eval_dir / f"{stem}_{index:03d}.png"
            side_by_side_png(row, f["rgb"], rgb_sim, f["depth"] * f["valid_depth_mask"], depth_sim)
            rows.append(row)
        if rows:
            png = eval_dir / f"{stem}.png"
            _stack_png(png, rows)
    report["png"] = str(png) if png else None
    path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    log(f"[realism] {report['room_id']}: {len(ids)}/{len(heldout)} held-out frames, {len(train)} train excluded")
    return report


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def summary(report: dict) -> str:
    r = report["realism"]
    if r is None:
        return f"realism: null ({report['reason']})"

    def f(v, nd=4):
        return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"

    c = report["control"]["realism"]
    lines = [f"realism {report['room_id']}: {r['frames']} held-out frame(s), depth L1 {f(r['depth_l1_m'])} m, "
             f"RGB MAE {f(r['rgb_mae'])}, 1-SSIM {f(r['rgb_one_minus_ssim'])}",
             f"control (camera moved {SHIFT_M} m, turned {YAW_DEG} deg): depth L1 {f(c['depth_l1_m'])} m, "
             f"RGB MAE {f(c['rgb_mae'])}"]
    if not report["valid"]:
        lines.append(f"NOT VALID: {report['reason']}")
    return "\n".join(lines)
