import base64
import hashlib
import http.server
import io
import json
import threading
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from arrival.export.lerobot import LeRobotWriter
from arrival.export.validate import validate
from arrival.render import sim_texture, variation

N = 80
EGO = "observation.images.ego_view"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("var") / "src"
    with LeRobotWriter(root, "t", "all") as w:
        for ep in range(2):
            frames = [{"object_position": [2 + 0.01 * i, 0, 0.9], "eef": [2 - 0.01 * i, 0, 1.0 + 0.002 * i]}
                      for i in range(N)]
            vid = sim_texture.render_episode(frames, "neutral")
            z = np.zeros((N, 8), np.float32)
            w.add_episode(z, z, np.arange(N) / 20, f"task {ep}", True, videos={"ego_view": vid})
        w.finish(2)
    assert validate(root) == []
    return root


def digest(root):
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode() + p.read_bytes())
    return h.hexdigest()


def rows(dst):
    return [json.loads(x) for x in (dst / "meta" / variation.SIDECAR).read_text().splitlines()]


def video(path):
    return imageio.mimread(str(path), memtest=False)


def serve_video(handler_fn):
    """Mock the JSON video contract used by Cosmos Transfer NIM."""
    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert isinstance(payload.get("video"), str)
            assert isinstance(payload.get("edge"), dict)
            assert isinstance(payload["edge"].get("control"), str)
            assert payload.get("prompt") and payload.get("resolution") == "480"
            with tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "source.mp4"
                output = Path(tmp) / "output.mp4"
                source.write_bytes(base64.b64decode(payload["video"]))
                frames = imageio.mimread(str(source), memtest=False)
                result_frames = handler_fn(frames)
                if result_frames is None:
                    response = {}
                else:
                    imageio.mimwrite(str(output), result_frames, format="FFMPEG", fps=20,
                                     codec="libx264", macro_block_size=None)
                    response = {"b64_video": base64.b64encode(output.read_bytes()).decode("ascii")}
            data = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_selection_deterministic_and_ten_percent():
    a = [variation.select_frames(e, N, 7) for e in range(2)]
    assert a == [variation.select_frames(e, N, 7) for e in range(2)]
    assert a != [variation.select_frames(e, N, 8) for e in range(2)]
    assert a[0] != a[1]
    assert abs(sum(map(len, a)) / (2 * N) - 0.10) <= 0.02


def test_destination_inside_source_is_refused_before_copy(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    (src / "keep").write_text("original")
    with pytest.raises(ValueError, match="outside the source"):
        variation.vary_dataset(src, src / "nested")
    assert (src / "keep").read_text() == "original"
    assert not (src / "nested").exists()

def test_custom_client_cannot_mislabel_backend(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    dst = tmp_path / "copy"
    with pytest.raises(ValueError, match="does not match requested"):
        variation.vary_dataset(src, dst, backend="cosmos", client=variation.LocalStub())
    assert not dst.exists()


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan"), float("inf")])
def test_bad_fraction_rejected_before_copy(tmp_path, fraction):
    dst = tmp_path / "copy"
    with pytest.raises(ValueError, match="fraction"):
        variation.vary_dataset(tmp_path / "missing", dst, fraction=fraction)
    assert not dst.exists()


@pytest.mark.parametrize("option,value", [
    ("edge_min", float("nan")), ("edge_min", 1.1), ("edge_min", True),
    ("mask_max", float("nan")), ("mask_max", -0.1), ("mask_max", float("inf")),
])
def test_invalid_geometry_thresholds_rejected_before_copy(tmp_path, option, value):
    dst = tmp_path / "copy"
    with pytest.raises(ValueError, match=option):
        variation.vary_dataset(tmp_path / "missing", dst, **{option: value})
    assert not dst.exists()


def test_bad_frame_count_is_rejected():
    with pytest.raises(ValueError, match="n_frames"):
        variation.select_frames(0, -1, 0)


@pytest.mark.parametrize("fps", [0, -1, True, 20.5, "20"])
def test_bad_fps_rejected_before_copy(dataset, tmp_path, fps):
    dst = tmp_path / "copy"
    with pytest.raises(ValueError, match="fps"):
        variation.vary_dataset(dataset, dst, fps=fps)
    assert not dst.exists()


def test_local_stub_differs_keeps_geometry_and_source_untouched(dataset, tmp_path):
    before = digest(dataset)
    dst = tmp_path / "copy"
    s = variation.vary_dataset(dataset, dst, "local_stub", seed=3)
    assert digest(dataset) == before and validate(dataset) == []
    assert validate(dst) == []
    assert s["arrival_appearance_variation"] == "local_stub" and s["ok"] == s["requests"] == 16
    assert s["added_latency_s"] >= s["backend_latency_s"] > 0 and s["cost_usd"] is None
    assert json.loads((dst / "meta" / "info.json").read_text())["arrival_appearance_variation"] == "local_stub"
    rs = rows(dst)
    assert all(r["backend"] == "local_stub" and r["status"] == "ok" and r["edge_corr"] >= 0.6 for r in rs)
    r = rs[0]
    name = "episode_%06d.mp4" % r["episode_index"]
    src_f = video(dataset / "videos/chunk-000" / EGO / name)
    var_f = video(dst / variation.VIDEO_DIR / EGO / name)
    i = r["frame_index"]
    assert np.abs(src_f[i].astype(int) - var_f[i].astype(int)).mean() > 2
    picked = {x["frame_index"] for x in rs if x["episode_index"] == r["episode_index"]}
    j = next(j for j in range(N) if j not in picked)
    assert np.abs(src_f[j].astype(int) - var_f[j].astype(int)).mean() < 6  # codec noise only


def test_cosmos_mock_server_pairs_ids_and_label(dataset, tmp_path):
    srv = serve_video(lambda frames: [np.clip(img.astype(int) + 20, 0, 255).astype(np.uint8) for img in frames])
    try:
        client = variation.CosmosClient(f"http://127.0.0.1:{srv.server_port}/", timeout=5, retries=0)
        s = variation.vary_dataset(dataset, tmp_path / "c", "cosmos", seed=1, client=client)
    finally:
        srv.shutdown()
        srv.server_close()
    assert s["arrival_appearance_variation"] == "cosmos" and s["ok"] == 16 and s["failed"] == 0
    assert s["added_latency_s"] >= s["backend_latency_s"] > 0 and s["cost_usd"] is None
    rs = rows(tmp_path / "c")
    assert all(r["variant_id"] == r["original_id"] + "_v" and r["model"] == "cosmos-transfer2.5"
               for r in rs)
    assert any(r["latency_ms"] > 0 for r in rs)


def test_cosmos_invalid_response_is_not_labelled(dataset, tmp_path):
    srv = serve_video(lambda frames: None)
    try:
        client = variation.CosmosClient(f"http://127.0.0.1:{srv.server_port}/", timeout=5, retries=0)
        summary = variation.vary_dataset(dataset, tmp_path / "copy", "cosmos", client=client)
    finally:
        srv.shutdown()
        srv.server_close()
    assert summary["arrival_appearance_variation"] == "none"
    assert summary["failed"] == summary["requests"]
    assert all("missing non-empty b64_video" in row["reason"] for row in rows(tmp_path / "copy"))


def test_cosmos_client_rejects_legacy_single_frame_contract():
    client = variation.CosmosClient("http://127.0.0.1:1/v1/infer", retries=0)
    with pytest.raises(RuntimeError, match="video-based"):
        client(np.zeros((8, 8, 3), np.uint8), 0, 0)


def test_server_down_falls_back_cleanly(dataset, tmp_path):
    srv = serve_video(lambda frames: frames)
    port = srv.server_port
    srv.shutdown()
    srv.server_close()
    client = variation.CosmosClient(f"http://127.0.0.1:{port}/", timeout=1, retries=1, backoff=0)
    s = variation.vary_dataset(dataset, tmp_path / "d", "cosmos", client=client)
    assert s["arrival_appearance_variation"] == "none" and s["ok"] == 0 and s["failed"] == s["requests"] == 16
    assert s["skipped"] == 16 - variation.BREAKER
    assert {r["status"] for r in rows(tmp_path / "d")} == {"failed"}
    assert not (tmp_path / "d" / variation.VIDEO_DIR).exists()
    assert validate(tmp_path / "d") == []


def test_no_cosmos_url_is_skipped_not_labelled(dataset, tmp_path, monkeypatch):
    monkeypatch.delenv("COSMOS_URL", raising=False)
    s = variation.vary_dataset(dataset, tmp_path / "n", "cosmos")
    assert s["arrival_appearance_variation"] == "none" and s["skipped"] == 16


def test_geometry_shift_is_rejected(dataset, tmp_path):
    srv = serve_video(lambda frames: [np.roll(img, 12, axis=1) for img in frames])
    try:
        client = variation.CosmosClient(f"http://127.0.0.1:{srv.server_port}/", timeout=5, retries=0)
        s = variation.vary_dataset(dataset, tmp_path / "r", "cosmos", client=client)
    finally:
        srv.shutdown()
        srv.server_close()
    assert s["rejected"] == 16 and s["arrival_appearance_variation"] == "none"
    assert all(r["status"] == "rejected" and r["reason"] for r in rows(tmp_path / "r"))


def test_mask_check_flags_repainted_region():
    img = sim_texture.render_frame({"object_position": [2, 0, 0.9], "eef": [2.3, 0, 1.0]}, "neutral")
    mask = np.zeros(img.shape[:2], bool)
    mask[34:54, 10:54] = True  # straddles the counter edge
    same, _ = variation.LocalStub(0)(img, 0, 0)
    assert variation.check_frame(img, same, mask)["ok"]
    bad = img.copy()
    bad[mask] = bad[mask][::-1]
    chk = variation.check_frame(img, bad, mask)
    assert not chk["ok"] and chk["mask_change"] > variation.MASK_CHANGE_MAX


def test_non_image_backend_output_is_rejected():
    orig = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    bad = orig.astype(float)
    bad[0, 0, 0] = float("nan")
    chk = variation.check_frame(orig, bad)
    assert not chk["ok"] and "uint8" in chk["reason"]
    assert chk["edge_corr"] is None

@pytest.mark.parametrize("mask", [np.ones((2, 3), bool), np.ones((4, 4), np.uint8), [[True] * 4] * 4])
def test_malformed_mask_rejects_frame_without_crashing(mask):
    orig = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    chk = variation.check_frame(orig, orig.copy(), mask)
    assert not chk["ok"] and "mask" in chk["reason"]
    assert chk["edge_corr"] is None


def test_dataset_with_bad_mask_keeps_original_videos(dataset, tmp_path):
    dst = tmp_path / "masked"
    summary = variation.vary_dataset(dataset, dst, mask=np.ones((2, 3), bool))
    assert summary["rejected"] == summary["requests"] == 16
    assert summary["ok"] == 0 and not (dst / variation.VIDEO_DIR).exists()
    assert validate(dst) == []


@pytest.mark.parametrize("result", [(None, {}), (np.zeros((2, 2, 3), np.uint8), None),
                                     (np.zeros((2, 2, 3), np.uint8), {"bad": object()})])
def test_malformed_backend_result_fails_frames_without_aborting_copy(dataset, tmp_path, result):
    class Client:
        name, model, version = "local_stub", "test", "1"

        def __call__(self, frame, episode, index):
            return result

    dst = tmp_path / "copy"
    summary = variation.vary_dataset(dataset, dst, client=Client())
    assert summary["failed"] == summary["requests"] == 16
    assert summary["ok"] == 0
    assert all(row["status"] == "failed" and row["reason"] for row in rows(dst))
    assert validate(dst) == []
