"""CAP-2: the iOS recorder's bundle writer (ios/ArrivalCapture/ArrivalCapture/FrameBundle.swift), compiled for the
Mac with swiftc and fed synthetic frames, writes a zip the server's frames upload path accepts, in the axes
arrival/scan/rgbd.py expects. ARKit itself is not exercised: the recorder that reads ARFrames needs a device."""

import http.client
import json
import os
import platform
import shutil
import subprocess
import threading
import zipfile

import numpy as np
import pytest
from PIL import Image

from arrival.scan import rgbd
from arrival.web import frames as frames_mod
from arrival.web.app import make_server

from conftest import REPO

pytestmark = pytest.mark.skipif(platform.system() != "Darwin" or shutil.which("swiftc") is None,
                                reason="needs swiftc on macOS")

APP = REPO / "ios" / "ArrivalCapture" / "ArrivalCapture"
SWIFT = [REPO / "ios" / "BundleCheck" / "main.swift", APP / "FrameBundle.swift", APP / "ArrivalAPI.swift"]
Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
W, H, DW, DH = 64, 48, 32, 24


@pytest.fixture(scope="module")
def bundle_check(tmp_path_factory):
    exe = tmp_path_factory.mktemp("swift") / "bundle_check"
    subprocess.run(["swiftc", "-O", *map(str, SWIFT), "-o", str(exe)], check=True, capture_output=True, timeout=600)
    return exe


def _depth(T_room, K):
    """Optical-axis depth of the floor (z = 0) and a wall (x = 3) seen from a room-frame camera pose."""
    v, u = np.mgrid[0:DH, 0:DW]
    ray = np.stack([(u - K[0, 2]) / K[0, 0], -(v - K[1, 2]) / K[1, 1], -np.ones(u.shape)], axis=-1)
    dirs, o = ray @ T_room[:3, :3].T, T_room[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(dirs[..., 2] < 0, -o[2] / dirs[..., 2], np.inf)
        t_wall = np.where(dirs[..., 0] > 0, (3.0 - o[0]) / dirs[..., 0], np.inf)
    d = np.minimum(t_floor, t_wall)
    return np.where(np.isfinite(d) & (d > 0), d, 0.0).astype(np.float32)


@pytest.fixture(scope="module")
def recorded(bundle_check, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bundle")
    K = np.array([[0.9 * W, 0, W / 2], [0, 0.9 * W, H / 2], [0, 0, 1.0]])
    Kd = K * np.array([[DW / W], [DH / H], [1]])
    poses = [rgbd.look_at((0.5 + 0.1 * i, 1.0 + 0.3 * i, 1.4), (2.6, 1.2 + 0.1 * i, 0.3)) for i in range(6)]
    spec, inputs = [], []
    for i, T_room in enumerate(poses):
        depth = _depth(T_room, Kd)
        conf = (np.arange(DW * DH) % 3).astype(np.uint8).reshape(DH, DW)
        Image.new("RGB", (W, H), (100 + 20 * i, 120, 140)).save(tmp / f"rgb{i}.jpg", quality=90)
        depth.astype("<f4").tofile(tmp / f"depth{i}.f32")
        conf.tofile(tmp / f"conf{i}.u8")
        to_world = np.linalg.inv(Y_UP_TO_Z_UP) @ T_room  # what ARKit reports: camera to its Y-up world
        spec.append({"timestamp": 10.0 + i, "rgb_jpeg": str(tmp / f"rgb{i}.jpg"), "rgb_size": [W, H],
                     "depth_f32": str(tmp / f"depth{i}.f32"), "conf_u8": str(tmp / f"conf{i}.u8"),
                     "depth_size": [DW, DH], "intrinsics": {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]},
                     "camera_to_world": to_world.ravel().tolist()})
        inputs.append({"T_room": T_room, "depth": depth, "conf": conf})
    repeat = dict(spec[2], timestamp=spec[2]["timestamp"])  # does not increase: skipped
    bad = dict(spec[3], timestamp=99.0, camera_to_world=["nan"] * 16)  # non-finite pose: skipped
    frames = spec[:3] + [repeat] + spec[3:] + [bad]
    (tmp / "spec.json").write_text(json.dumps({"room_id": "ios_kitchen", "frames": frames}))
    out = subprocess.run([str(bundle_check), str(tmp / "spec.json"), str(tmp / "bundle"), str(tmp / "frames.zip")],
                         capture_output=True, text=True, timeout=120)
    return tmp, out, inputs


def test_bad_frames_are_skipped_and_the_rest_kept(recorded):
    tmp, out, _ = recorded
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "6"


def test_zip_passes_the_server_upload_checks(recorded, tmp_path):
    tmp, _, _ = recorded
    zpath = tmp / "frames.zip"
    with zipfile.ZipFile(zpath) as zf:
        assert zf.testzip() is None  # every CRC matches
        assert "frames_manifest.json" in zf.namelist()  # at the archive root, as POST /jobs wants
    with open(zpath, "rb") as f:
        written = frames_mod.extract(f, 0, zpath.stat().st_size, tmp_path / "x")
    summary = frames_mod.validate(tmp_path / "x")
    assert written == 1 + 3 * 6
    assert summary["count"] == 6 and summary["heldout"] == 1 and summary["train"] == 5
    assert summary["rgb_size"] == [W, H] and summary["depth_size"] == [DW, DH] and summary["depth_unit"] == "m"


def test_depth_confidence_and_poses_round_trip(recorded):
    tmp, _, inputs = recorded
    m = rgbd.load_manifest(tmp / "bundle")
    assert m.room_id == "ios_kitchen"
    assert np.allclose(m.arkit_world_to_room, Y_UP_TO_Z_UP)
    for f, given in zip(m.frames, inputs):
        assert np.array_equal(rgbd.load_depth(m, f), given["depth"])
        assert np.array_equal(np.asarray(Image.open(f.confidence)), given["conf"])
        assert np.allclose(f.camera_to_room, given["T_room"], atol=1e-9)


def test_unprojected_depth_lands_on_the_floor_and_wall(recorded):
    # the axis conventions agree end to end: ARKit pose -> camera_to_room -> rgbd.unproject
    tmp, _, _ = recorded
    m = rgbd.load_manifest(tmp / "bundle")
    for f in m.frames:
        pts = rgbd.unproject(rgbd.load_depth(m, f), rgbd.depth_intrinsics(m, f), f.camera_to_room)
        assert len(pts) > 0
        off = np.minimum(np.abs(pts[:, 2]), np.abs(pts[:, 0] - 3.0))
        assert off.max() < 1e-4, f.id


def test_app_upload_body_with_frames_is_queued(bundle_check, recorded, tmp_path):
    # the body ArrivalAPI.writeMultipart builds, with a real RoomPlan export and the recorder's zip, is accepted
    tmp, _, _ = recorded
    boundary = "ArrivalBoundary-3F2504E0-4F89-11D3-9A0C-0305E82C3301"
    scan = REPO / "tests" / "fixtures" / "roomplan" / "office.json"
    subprocess.run([str(bundle_check), "multipart", boundary, str(scan), str(tmp / "frames.zip"), str(tmp_path / "body")],
                   check=True, capture_output=True, timeout=60)
    srv = make_server(port=0, out_dir=tmp_path / "out", scans_dir=REPO / "scans", rate_limit=0)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
        conn.request("POST", "/jobs", body=(tmp_path / "body").read_bytes(),
                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        conn.close()
    finally:
        srv.shutdown()
        srv.server_close()
    assert resp.status == 202, data
    job = tmp_path / "out" / "jobs" / data["id"]
    assert (job / "upload.bin").read_bytes() == scan.read_bytes()
    status = json.loads((job / "status.json").read_text())
    assert status["frames"]["count"] == 6 and status["frames"]["heldout"] == 1
    assert (job / "frames" / "frames_manifest.json").is_file()


@pytest.mark.parametrize("existing", [False, True])
def test_failed_ios_zip_preserves_destination(bundle_check, tmp_path, existing):
    folder = tmp_path / "frames"
    folder.mkdir()
    (folder / "a-readable.txt").write_text("first frame")
    unreadable = folder / "z-unreadable.txt"
    unreadable.write_text("cannot read this frame")
    unreadable.chmod(0)
    output = tmp_path / "frames.zip"
    previous = b"previous complete archive"
    if existing:
        output.write_bytes(previous)
    try:
        if os.access(unreadable, os.R_OK):
            pytest.skip("requires file read permission enforcement")
        result = subprocess.run([str(bundle_check), "zip", str(folder), str(output)],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 1
        if existing:
            assert output.read_bytes() == previous
        else:
            assert not output.exists()
        assert {p.name for p in tmp_path.iterdir()} == ({"frames", "frames.zip"} if existing else {"frames"})
    finally:
        unreadable.chmod(0o600)


def test_ios_zip_replaces_existing_archive(bundle_check, tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    (folder / "frame.txt").write_text("complete frame")
    output = tmp_path / "frames.zip"
    output.write_bytes(b"old archive")
    subprocess.run([str(bundle_check), "zip", str(folder), str(output)],
                   check=True, capture_output=True, timeout=10)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert archive.read("frame.txt") == b"complete frame"
    assert {p.name for p in tmp_path.iterdir()} == {"frames", "frames.zip"}
