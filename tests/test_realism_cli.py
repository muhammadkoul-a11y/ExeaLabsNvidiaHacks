"""L16: `arrival realism <scan>` scores only held-out frames, masks invalid depth, converts ARKit camera axes to the
renderer's, and refuses (or flags) a measurement it cannot trust. Frames are renders of the hero twin itself, so
a correct pipeline scores about zero and any axis, scale, mask or split mistake shows up as error."""

import json
import shutil

import numpy as np
import pytest
from PIL import Image

mujoco = pytest.importorskip("mujoco")

from arrival import cli  # noqa: E402
from arrival.eval import realism, realism_cli  # noqa: E402
from arrival.scan import rgbd  # noqa: E402
from arrival.scan.bundle import load_room  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

from conftest import SCANS  # noqa: E402

W, H = 64, 48
POSES = {"train_0": ((3.6, 0.5, 1.5), (2.0, 1.5, 0.8)),
         "held_0": ((3.6, 2.5, 1.5), (2.0, 1.5, 0.8)),
         "held_1": ((0.5, 2.5, 1.5), (2.0, 1.5, 0.8))}


@pytest.fixture(scope="module")
def twin():
    return compile_twin(load_room("hero_kitchen", SCANS))


def _write_scan(root, twin, size=(W, H), depth_div=1, depth_unit="m", bad_region=True, mutate=None):
    """scan.json + frames_manifest.json + frames rendered from `twin`. In every frame the leftmost 8 columns of depth
    are wrong by 5 m but marked confidence 0, so a mask that fails to apply shows up as error."""
    w, h = size
    root.mkdir(parents=True)
    shutil.copy(SCANS / "hero_kitchen" / "scan.json", root / "scan.json")
    for d in ("rgb", "depth", "conf"):
        (root / d).mkdir()
    fx = 0.9 * w
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1.0]])
    frames = []
    for i, (fid, (eye, target)) in enumerate(POSES.items()):
        T_cv = realism.look_at(eye, target)
        rgb, depth = realism.render_twin(twin, K, T_cv, (h, w))
        depth = depth.copy()
        conf = np.full(depth.shape, 2, np.uint8)
        if bad_region:
            depth[:, :8] += 5.0
            conf[:, :8] = 0
        if depth_div > 1:
            depth = depth[depth_div // 2::depth_div, depth_div // 2::depth_div]
            conf = conf[depth_div // 2::depth_div, depth_div // 2::depth_div]
        Image.fromarray(rgb).save(root / "rgb" / f"{fid}.png")
        if depth_unit == "m":
            np.save(root / "depth" / f"{fid}.npy", depth.astype(np.float32))
            dpath = f"depth/{fid}.npy"
        else:
            Image.fromarray(np.round(depth * 1000).astype(np.uint16)).save(root / "depth" / f"{fid}.png")
            dpath = f"depth/{fid}.png"
        Image.fromarray(conf).save(root / "conf" / f"{fid}.png")
        T_arkit = T_cv @ realism_cli.ARKIT_TO_CV  # the camera axes the manifest declares
        frames.append({"id": fid, "timestamp": 1.0 + i, "rgb": f"rgb/{fid}.png", "depth": dpath,
                       "confidence": f"conf/{fid}.png",
                       "intrinsics": {"fx": fx, "fy": fx, "cx": w / 2, "cy": h / 2},
                       "camera_to_room": T_arkit.ravel().tolist(),
                       "split": "train" if fid.startswith("train") else "heldout"})
    manifest = {"schema_version": rgbd.SCHEMA_VERSION, "room_id": "hero_kitchen",
                "axes": {"camera": rgbd.CAMERA_AXES, "room": rgbd.ROOM_AXES},
                "arkit_world_to_room": np.eye(4).ravel().tolist(), "rgb_size": [w, h],
                "depth_size": [w // depth_div, h // depth_div], "depth_unit": depth_unit, "frames": frames}
    if mutate:
        mutate(manifest)
    (root / rgbd.MANIFEST_NAME).write_text(json.dumps(manifest))
    return root


def _run(scan, tmp_path, *extra, capsys=None):
    code = cli.main(["realism", str(scan), "--out", str(tmp_path / "out"), *extra])
    report = None
    p = tmp_path / "out" / "eval" / "hero_kitchen_realism.json"
    if p.is_file():
        report = json.loads(p.read_text())
    return code, report


@pytest.fixture(scope="module")
def scan_dir(twin, tmp_path_factory):
    return _write_scan(tmp_path_factory.mktemp("scan") / "hero_kitchen", twin)


def test_twin_frames_score_near_zero_and_only_heldout_counts(scan_dir, tmp_path, capsys):
    code, rep = _run(scan_dir, tmp_path)
    assert code == 0, capsys.readouterr().err
    assert rep["frames"]["scored"] == ["held_0", "held_1"] and rep["frames"]["train_excluded"] == 1
    r = rep["realism"]
    assert r["heldout_only"] and r["frames"] == 2
    assert r["depth_l1_m"] < 1e-3 and r["rgb_mae"] < 1e-6  # exact: ARKit axes converted, wrong-depth region masked
    assert {p["id"] for p in r["per_frame"]} == {"held_0", "held_1"}
    assert rep["valid"] is True and rep["alignment_check"]["passed"] is True
    assert "texture" in rep["scope"] and rep["masks"]["robot"]


@pytest.mark.parametrize("transform", [np.eye(3), np.full((4, 4), np.nan),
                                       np.diag([1, 1, 1, 2]), [["bad"]] * 4])
def test_invalid_camera_transform_is_rejected_before_scoring(tmp_path, transform):
    with pytest.raises(realism_cli.RealismError, match="camera_transform"):
        realism_cli.run_realism(tmp_path / "missing_scan", tmp_path / "out", camera_transform=transform)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("option,value", [
    ("max_side", True), ("max_frames", 1.5), ("png_frames", -1),
    ("png_frames", True), ("min_confidence", -1), ("min_confidence", 3),
    ("min_confidence", True),
])
def test_invalid_realism_controls_rejected_before_loading_scan(tmp_path, option, value):
    with pytest.raises(realism_cli.RealismError):
        realism_cli.run_realism(tmp_path / "missing_scan", tmp_path / "out", **{option: value})
    assert not (tmp_path / "out").exists()


def test_masked_depth_is_not_scored(scan_dir, tmp_path):
    """The 5 m error in the confidence-0 columns must not reach the mean; with the mask off it does."""
    _, masked = _run(scan_dir, tmp_path)
    _, unmasked = _run(scan_dir, tmp_path / "u", "--min-confidence", "0")
    assert masked["realism"]["depth_l1_m"] < 1e-3
    assert unmasked["realism"]["depth_l1_m"] > 0.2  # 8 of 64 columns wrong by 5 m


def test_misaligned_pose_control_raises_the_error(scan_dir, tmp_path):
    _, rep = _run(scan_dir, tmp_path)
    r, c = rep["realism"], rep["control"]["realism"]
    assert rep["control"]["shift_m"] == 0.15 and rep["control"]["yaw_deg"] == 5.0
    assert c["depth_l1_m"] > r["depth_l1_m"] + 0.01 and c["rgb_mae"] > r["rgb_mae"]
    assert rep["alignment_check"]["error_rises_under_misalignment"] == {"depth_l1_m": True, "rgb_mae": True}


def test_wrong_camera_axes_would_be_caught(scan_dir, tmp_path, monkeypatch):
    """Control for the conversion: treating ARKit poses as OpenCV poses must wreck the score, so the near-zero
    result above really is evidence that the axes are converted."""
    monkeypatch.setattr(realism_cli, "ARKIT_TO_CV", np.eye(4))
    code, rep = _run(scan_dir, tmp_path)
    assert rep["realism"]["depth_l1_m"] > 0.3 or rep["realism"]["rgb_mae"] > 0.05


def test_a_control_that_does_not_raise_the_error_invalidates_the_run(scan_dir, tmp_path, monkeypatch, capsys):
    import arrival.eval.realism as r

    monkeypatch.setattr(r, "perturb_pose", lambda T, shift, yaw: T)
    code, rep = _run(scan_dir, tmp_path)
    assert code == 1 and rep["valid"] is False and "must not be quoted" in rep["reason"]
    assert "NOT VALID" in capsys.readouterr().out


def test_side_by_side_png_is_written(scan_dir, tmp_path):
    _, rep = _run(scan_dir, tmp_path)
    assert Image.open(rep["png"]).size == (W * 4, H * 2)  # reference | twin | depth | depth, one row per frame
    assert (tmp_path / "out" / "eval" / "hero_kitchen_realism_000.png").is_file()
    _, one = _run(scan_dir, tmp_path / "one", "--png-frames", "1")
    assert Image.open(one["png"]).size == (W * 4, H)


def test_max_frames_picks_evenly_and_never_a_train_frame(scan_dir, tmp_path):
    _, rep = _run(scan_dir, tmp_path, "--max-frames", "1")
    assert rep["frames"]["scored"] == ["held_0"] and rep["frames"]["heldout"] == 2


def test_downscaled_frames_with_half_resolution_depth(twin, tmp_path):
    """Frames 2x the working size, depth at half the frame size (as on a phone): K and depth follow the resize."""
    scan = _write_scan(tmp_path / "big" / "hero_kitchen", twin, size=(2 * W, 2 * H), depth_div=2, bad_region=False)
    code, rep = _run(scan, tmp_path, "--max-side", str(W))
    assert code == 0
    assert rep["frames"]["working_size_hw"] == [H, W]
    assert rep["realism"]["depth_l1_m"] < 0.25 and rep["control"]["realism"]["depth_l1_m"] > rep["realism"]["depth_l1_m"]


def test_millimetre_png_depth(twin, tmp_path):
    scan = _write_scan(tmp_path / "mm" / "hero_kitchen", twin, depth_unit="mm", bad_region=False)
    code, rep = _run(scan, tmp_path)
    assert code == 0 and rep["realism"]["depth_l1_m"] < 2e-3  # millimetre rounding only


def test_nearest_resample_never_interpolates():
    a = np.array([[0.0, 10.0], [20.0, 30.0]])
    out = realism_cli._nearest(a, (4, 4))
    assert set(np.unique(out)) <= {0.0, 10.0, 20.0, 30.0} and out.shape == (4, 4)
    assert out[0, 0] == 0.0 and out[3, 3] == 30.0 and out[0, 3] == 10.0 and out[3, 0] == 20.0


def test_no_heldout_frame_is_refused(twin, tmp_path, capsys):
    def all_train(m):
        for f in m["frames"]:
            f["split"] = "train"

    scan = _write_scan(tmp_path / "s" / "hero_kitchen", twin, mutate=all_train)
    code, rep = _run(scan, tmp_path)
    assert code == 1 and rep is None
    assert "no held-out frame" in capsys.readouterr().err


def test_scan_without_frames_is_refused_not_scored(tmp_path, capsys):
    d = tmp_path / "scans" / "hero_kitchen"
    d.mkdir(parents=True)
    shutil.copy(SCANS / "hero_kitchen" / "scan.json", d / "scan.json")
    code = cli.main(["realism", "hero_kitchen", "--scans", str(tmp_path / "scans"), "--out", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code == 1 and "frames_manifest.json" in err and "four-photo" in err
    assert not (tmp_path / "out" / "eval").exists()


def test_bad_manifest_is_a_readable_error(twin, tmp_path, capsys):
    scan = _write_scan(tmp_path / "s" / "hero_kitchen", twin)
    m = json.loads((scan / rgbd.MANIFEST_NAME).read_text())
    m["frames"][0]["camera_to_room"][0] = float("nan")
    (scan / rgbd.MANIFEST_NAME).write_text(json.dumps(m))
    code, _ = _run(scan, tmp_path)
    assert code == 1 and "nonfinite_transform" in capsys.readouterr().err


def test_four_photo_bundle_reports_null_with_a_reason(twin, tmp_path, capsys):
    scan = _write_scan(tmp_path / "s" / "hero_kitchen", twin)
    raw = json.loads((scan / "scan.json").read_text())
    raw["geometry_source"], raw["blocks_claims"] = "four_photo_unmeasured", ["geometry", "realism"]
    (scan / "scan.json").write_text(json.dumps(raw))
    code, rep = _run(scan, tmp_path)
    assert code == 1 and rep["realism"] is None and rep["valid"] is False and "realism" in rep["reason"]
    assert rep["png"] is None and rep["control"] is None
    assert "realism: null" in capsys.readouterr().out


def test_explicit_twin_json_is_used(twin, scan_dir, tmp_path):
    p = tmp_path / "twin.json"
    p.write_text(json.dumps(twin))
    _, rep = _run(scan_dir, tmp_path, "--twin", str(p))
    assert rep["twin_source"] == str(p) and rep["realism"]["depth_l1_m"] < 1e-3


@pytest.mark.parametrize("body", ['{"room_id": "hero_kitchen"', "[]", '"twin"', "{}", '{"walls": [], "fixtures": []}'])
def test_bad_twin_json_is_a_readable_error(scan_dir, tmp_path, capsys, body):
    p = tmp_path / "twin.json"
    p.write_text(body)
    code, rep = _run(scan_dir, tmp_path, "--twin", str(p))
    assert code == 1 and rep is None and "--twin" in capsys.readouterr().err


@pytest.mark.parametrize("name,body", [("rgb/held_0.png", b"not a png"), ("depth/held_0.npy", b"junk"),
                                       ("conf/held_0.png", b"junk"), ("depth/held_0.npy", None)])
def test_unreadable_frame_file_is_a_readable_error(twin, tmp_path, capsys, name, body):
    scan = _write_scan(tmp_path / "s" / "hero_kitchen", twin)
    if body is None:  # metric depth saved as float64
        np.save(scan / name, np.ones((H, W)))
    else:
        (scan / name).write_bytes(body)
    code, rep = _run(scan, tmp_path)
    assert code == 1 and rep is None and "frame held_0" in capsys.readouterr().err


def test_evaluate_frames_still_refuses_a_train_id(twin):
    """The library-level guard the command relies on."""
    with pytest.raises(ValueError, match="train_0"):
        realism.evaluate_frames(twin, [{"id": "train_0"}], ["held_0"])
