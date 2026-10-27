"""CAP-2a: RGB-D manifest loader, unprojection and the synthetic two-camera bundle (offline, tiny)."""

import json

import numpy as np
import pytest
from PIL import Image

from arrival.scan import rgbd


def _edit(root, fn):
    p = root / rgbd.MANIFEST_NAME
    m = json.loads(p.read_text())
    fn(m)
    p.write_text(json.dumps(m))


def test_unproject_known_pixel_roundtrips():
    K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
    T = rgbd.look_at((1.0, 2.0, 1.5), (3.0, 2.0, 1.5))  # looks along +x, so camera -z is room +x
    depth = np.zeros((48, 64), np.float32)
    depth[14, 52] = 2.0  # 20 px right of centre, 10 px above
    pts = rgbd.unproject(depth, K, T)
    assert pts.shape == (1, 3)
    # right of the image is room -y when looking along +x with Z up; above centre is +z
    expected = [1.0 + 2.0, 2.0 - 20 / 100 * 2.0, 1.5 + 10 / 100 * 2.0]
    assert np.allclose(pts[0], expected, atol=1e-6)
    uv, d = rgbd.project(pts, K, T)
    assert np.allclose(uv[0], [52, 14], atol=1e-6) and abs(d[0] - 2.0) < 1e-6


@pytest.mark.parametrize("eye,target,up,reason", [
    ((1, 2, 3), (1, 2, 3), (0, 0, 1), "distinct"),
    ((0, 0, 0), (0, 0, 1), (0, 0, 1), "parallel"),
    ((float("nan"), 0, 0), (1, 0, 0), (0, 0, 1), "finite"),
])
def test_look_at_rejects_degenerate_camera_poses(eye, target, up, reason):
    with pytest.raises(ValueError, match=reason):
        rgbd.look_at(eye, target, up)


@pytest.mark.parametrize("unit", ["m", "mm"])
def test_synthetic_bundle_points_land_on_floor_plane(tmp_path, unit):
    m = rgbd.load_manifest(rgbd.write_synthetic_bundle(tmp_path, depth_unit=unit))
    assert [f.split for f in m.frames] == ["train", "heldout"]
    for f in m.frames:
        pts = rgbd.unproject(rgbd.load_depth(m, f), rgbd.depth_intrinsics(m, f), f.camera_to_room)
        on_wall = np.abs(pts[:, 0] - 3.0) < 1e-3
        floor = pts[~on_wall]
        assert on_wall.any() and len(floor) > 100  # each camera sees wall and floor
        assert np.abs(floor[:, 2]).max() < 1e-3  # 1 mm, also covers the mm PNG quantisation


@pytest.mark.parametrize("mutate,code", [
    (lambda m: m.update(depth_unit="mm"), "depth_unit"),  # files are .npy
    (lambda m: m["frames"][1].update(timestamp=0.5), "non_monotonic_timestamps"),
    (lambda m: m["frames"][0].update(camera_to_room=[2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
     "non_orthonormal_rotation"),
    (lambda m: m["frames"][0].update(rgb="rgb/nope.png"), "missing_file"),
    (lambda m: m.update(splits={"train": ["f000"], "heldout": ["f000", "f001"]}), "frame_in_both_splits"),
    (lambda m: m["axes"].update(camera="opencv"), "axis_convention"),
    (lambda m: m["frames"][0].update(camera_to_room=[float("nan")] * 16), "nonfinite_transform"),
    (lambda m: m.update(rgb_size=[True, 48]), "bad_size"),
    (lambda m: m.update(depth_size=[64, True]), "bad_size"),
    (lambda m: m["frames"][0]["intrinsics"].update(fx=True), "intrinsics"),
    (lambda m: m["frames"][0]["intrinsics"].update(cx="32"), "intrinsics"),
])
def test_malformed_manifest_rejected_with_distinct_code(tmp_path, mutate, code):
    root = rgbd.write_synthetic_bundle(tmp_path)
    _edit(root, mutate)
    with pytest.raises(rgbd.ManifestError) as e:
        rgbd.load_manifest(root)
    assert e.value.code == code and str(e.value).startswith(code + ":")


@pytest.mark.parametrize("body", ['[]', '{"schema_version": 1, "room_id": "a", "axes": [1]}'])
def test_non_object_manifest_raises_manifest_error(tmp_path, body):
    (tmp_path / rgbd.MANIFEST_NAME).write_text(body)
    with pytest.raises(rgbd.ManifestError):
        rgbd.load_manifest(tmp_path)


def test_non_object_frame_raises_manifest_error(tmp_path):
    root = rgbd.write_synthetic_bundle(tmp_path)
    _edit(root, lambda m: m["frames"].append(5))
    with pytest.raises(rgbd.ManifestError) as e:
        rgbd.load_manifest(root)
    assert e.value.code == "frames"


@pytest.mark.parametrize("unit", ["m", "mm"])
def test_depth_loader_rejects_wrong_pixel_type(tmp_path, unit):
    m = rgbd.load_manifest(rgbd.write_synthetic_bundle(tmp_path, depth_unit=unit))
    frame = m.frames[0]
    if unit == "m":
        np.save(frame.depth, np.ones((m.depth_size[1], m.depth_size[0]), np.float64))
    else:
        Image.fromarray(np.ones((m.depth_size[1], m.depth_size[0]), np.uint8)).save(frame.depth)
    with pytest.raises(rgbd.ManifestError) as e:
        rgbd.load_depth(m, frame)
    assert e.value.code == "depth_dtype"


def test_frame_symlink_cannot_escape_bundle(tmp_path):
    root = rgbd.write_synthetic_bundle(tmp_path / "bundle")
    outside = tmp_path / "outside.png"
    outside.write_bytes((root / "rgb" / "f000.png").read_bytes())
    (root / "rgb" / "f000.png").unlink()
    try:
        (root / "rgb" / "f000.png").symlink_to(outside)
    except OSError as e:  # Windows without Developer Mode or admin rights: WinError 1314, no symlink to test
        pytest.skip(f"this machine cannot create symlinks ({e.__class__.__name__}: {e.strerror or e})")
    with pytest.raises(rgbd.ManifestError) as e:
        rgbd.load_manifest(root)
    assert e.value.code == "bad_path"

@pytest.mark.parametrize("splits", [
    {"train": [], "heldout": ["f001"]},
    {"train": ["f000", "unknown"], "heldout": ["f001"]},
    {"train": ["f000", "f000"], "heldout": ["f001"]},
    {"train": "f000", "heldout": ["f001"]},
])
def test_explicit_split_lists_must_match_every_frame(tmp_path, splits):
    root = rgbd.write_synthetic_bundle(tmp_path)
    _edit(root, lambda m: m.update(splits=splits))
    with pytest.raises(rgbd.ManifestError) as e:
        rgbd.load_manifest(root)
    assert e.value.code == "split"


@pytest.mark.parametrize(("field", "code"), [
    ("timestamp", "timestamp"), ("fx", "intrinsics"), ("fy", "intrinsics"),
    ("cx", "intrinsics"), ("world", "bad_matrix"), ("pose", "bad_matrix"),
])
def test_oversized_numbers_raise_manifest_error(tmp_path, field, code):
    root = rgbd.write_synthetic_bundle(tmp_path)

    def mutate(manifest):
        frame = manifest["frames"][0]
        if field == "timestamp":
            frame[field] = 10 ** 400
        elif field == "world":
            manifest["arkit_world_to_room"][0] = 10 ** 400
        elif field == "pose":
            frame["camera_to_room"][0] = 10 ** 400
        else:
            frame["intrinsics"][field] = 10 ** 400

    _edit(root, mutate)
    with pytest.raises(rgbd.ManifestError) as error:
        rgbd.load_manifest(root)
    assert error.value.code == code
