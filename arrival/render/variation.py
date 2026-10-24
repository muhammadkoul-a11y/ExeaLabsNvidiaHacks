"""variation: post-render appearance variation over a finished sim dataset. Works on a COPY, never the source.

Backends: 'cosmos' (video-to-video client for NVIDIA Cosmos Transfer 2.5 NIM at COSMOS_URL) and 'local_stub'
(seeded colour jitter plus mild blur, geometry fixed). A restyled frame is kept only if edge structure (and an
optional robot/object mask) survive; otherwise the original frame stays. The dataset label is 'cosmos' only after
an accepted frame from a successful Cosmos request.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

SIDECAR = "arrival_variation.jsonl"
SUMMARY = "arrival_variation_summary.json"
VIDEO_DIR = "variation_videos"
FRACTION = 0.10
EDGE_CORR_MIN = 0.6
MASK_CHANGE_MAX = 0.35
BREAKER = 3  # consecutive failed cosmos requests before the rest are skipped


def _u64(*parts) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")


def select_frames(episode_index: int, n_frames: int, seed: int, fraction: float = FRACTION) -> list[int]:
    """The round(fraction * n) frames with the smallest hash of (episode_index, frame_index, seed), sorted."""
    if type(n_frames) is not int or n_frames < 0:
        raise ValueError("n_frames must be a non-negative integer")
    if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be finite and between 0 and 1")
    k = min(n_frames, int(round(fraction * n_frames)))
    ranked = sorted(range(n_frames), key=lambda i: (_u64(episode_index, i, seed), i))
    return sorted(ranked[:k])


def _gray(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def edge_map(img: np.ndarray) -> np.ndarray:
    g = np.pad(_gray(img), 1, mode="edge")
    gx = g[1:-1, 2:] - g[1:-1, :-2]
    gy = g[2:, 1:-1] - g[:-2, 1:-1]
    return np.hypot(gx, gy)


def edge_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ea, eb = edge_map(a).ravel(), edge_map(b).ravel()
    if ea.std() < 1e-9 or eb.std() < 1e-9:
        return 1.0 if ea.std() < 1e-9 and eb.std() < 1e-9 else 0.0
    return float(np.corrcoef(ea, eb)[0, 1])


def masked_change(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None) -> float | None:
    """Mean abs difference inside the mask of the grey values normalised to zero mean, unit std there, so a pure
    brightness or contrast change reads as ~0 and a moved or repainted object reads high. None without a mask."""
    if mask is None or not mask.any():
        return None
    ga, gb = _gray(a)[mask], _gray(b)[mask]
    na = (ga - ga.mean()) / (ga.std() + 1e-6)
    nb = (gb - gb.mean()) / (gb.std() + 1e-6)
    return float(np.abs(na - nb).mean() / 2)


def check_frame(orig: np.ndarray, var: np.ndarray, mask: np.ndarray | None = None,
                edge_min: float = EDGE_CORR_MIN, mask_max: float = MASK_CHANGE_MAX) -> dict:
    if var.shape != orig.shape:
        return {"ok": False, "reason": f"shape {list(var.shape)} != {list(orig.shape)}",
                "edge_corr": None, "mask_change": None}
    if var.dtype != np.uint8:
        return {"ok": False, "reason": f"expected uint8 RGB frame, got {var.dtype}",
                "edge_corr": None, "mask_change": None}
    if mask is not None and (not isinstance(mask, np.ndarray) or mask.dtype != np.bool_
                             or mask.shape != orig.shape[:2]):
        return {"ok": False, "reason": "mask must be a boolean array matching frame height and width",
                "edge_corr": None, "mask_change": None}
    if np.array_equal(orig, var):
        return {"ok": False, "reason": "backend returned the original frame unchanged",
                "edge_corr": 1.0, "mask_change": 0.0 if mask is not None and mask.any() else None}
    corr = edge_correlation(orig, var)
    mc = masked_change(orig, var, mask)
    reason = None
    if not math.isfinite(corr) or (mc is not None and not math.isfinite(mc)):
        reason = "non-finite geometry check"
    elif corr < edge_min:
        reason = f"edge correlation {corr:.3f} < {edge_min}"
    elif mc is not None and mc > mask_max:
        reason = f"masked region changed {mc:.3f} > {mask_max}"
    return {"ok": reason is None, "reason": reason, "edge_corr": round(corr, 4),
            "mask_change": None if mc is None else round(mc, 4)}


def _blur3(img: np.ndarray) -> np.ndarray:
    p = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    return sum(p[dy:dy + img.shape[0], dx:dx + img.shape[1]] for dy in range(3) for dx in range(3)) / 9.0


def _hue_matrix(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    k = 1 / 3
    r = math.sqrt(k)
    return np.array([
        [c + (1 - c) * k, k * (1 - c) - r * s, k * (1 - c) + r * s],
        [k * (1 - c) + r * s, c + k * (1 - c), k * (1 - c) - r * s],
        [k * (1 - c) - r * s, k * (1 - c) + r * s, c + k * (1 - c)]])


class LocalStub:
    """Seeded colour/brightness/contrast/hue jitter plus a mild blur. Not Cosmos: geometry is never moved."""

    name = "local_stub"
    model = "local_stub"
    version = "1"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def params(self, episode_index: int, frame_index: int) -> dict:
        rng = np.random.default_rng(_u64("stub", episode_index, frame_index, self.seed))
        return {"brightness": round(float(rng.uniform(0.8, 1.2)), 4), "contrast": round(float(rng.uniform(0.8, 1.25)), 4),
                "hue_deg": round(float(rng.uniform(-15, 15)), 3),
                "color_shift": [round(float(v), 2) for v in rng.uniform(-10, 10, 3)], "blur_mix": 0.4}

    def __call__(self, frame: np.ndarray, episode_index: int, frame_index: int) -> tuple[np.ndarray, dict]:
        p = self.params(episode_index, frame_index)
        x = frame.astype(np.float64) @ _hue_matrix(p["hue_deg"]).T
        x = (x - 128) * p["contrast"] + 128
        x = x * p["brightness"] + np.array(p["color_shift"])
        x = (1 - p["blur_mix"]) * x + p["blur_mix"] * _blur3(x)
        return np.clip(np.rint(x), 0, 255).astype(np.uint8), p


class CosmosClient:
    """Client for the Cosmos 3 Generator video API used by the Nebius vLLM-Omni endpoint.

    COSMOS_URL is the endpoint base URL, not an /v1/infer URL. The client uploads
    an episode to POST /v1/videos, requests derived edge transfer, polls the job,
    and downloads the MP4 from /content. A local mock validates the protocol;
    it does not prove that the hosted endpoint is reachable.
    """

    name = "cosmos"
    model = "nvidia/Cosmos3-Nano"
    version = "vllm-omni-videos-api"

    def __init__(self, url: str | None = None, timeout: float = 1800.0, retries: int = 1, backoff: float = 1.0,
                 prompt: str = "Restyle the scene with realistic lighting and surface materials. Preserve the exact scene layout, camera view, robot and object positions, and all task-relevant geometry.",
                 resolution: str = "480", control_weight: float = 1.5, poll_interval: float = 5.0):
        from arrival import config as settings

        self.url = (url if url is not None else settings.get("COSMOS_URL") or "").rstrip("/")
        self.api_key = settings.get("COSMOS_API_KEY")
        self.timeout, self.retries, self.backoff = timeout, retries, backoff
        self.prompt, self.resolution, self.control_weight = prompt, resolution, control_weight
        self.poll_interval = poll_interval

    @staticmethod
    def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
        import imageio.v2 as imageio

        with imageio.get_writer(str(path), format="FFMPEG", fps=fps, codec="libx264",
                                macro_block_size=None, pixelformat="yuv420p") as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame, dtype=np.uint8))

    @staticmethod
    def _size_for_shape(height: int, width: int) -> str:
        """Choose the 480-tier Cosmos3 output bucket matching the input aspect ratio."""
        ratio = width / height
        buckets = [
            (16 / 9, "832x480"),
            (1.0, "640x640"),
            (9 / 16, "480x832"),
            (4 / 3, "736x544"),
            (3 / 4, "544x736"),
        ]
        target, size = min(buckets, key=lambda item: abs(math.log(ratio / item[0])))
        if abs(math.log(ratio / target)) > 0.12:
            raise ValueError(f"input aspect ratio {width}:{height} is not close to a supported Cosmos3 480-tier bucket")
        return size

    @staticmethod
    def _multipart(fields: dict[str, str], video_bytes: bytes) -> tuple[bytes, str]:
        import uuid

        boundary = "----arrival-cosmos3-" + uuid.uuid4().hex
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").encode())
        chunks.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"input_reference\"; filename=\"episode.mp4\"\r\nContent-Type: video/mp4\r\n\r\n").encode())
        chunks.append(video_bytes)
        chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode())
        return b"".join(chunks), boundary

    def transform_video(self, frames: list[np.ndarray], fps: int | float, seed: int | None = None) -> tuple[list[np.ndarray], dict]:
        """Upload one episode to Cosmos3 video-to-video transfer and return decoded RGB frames."""
        import tempfile
        from PIL import Image
        import imageio.v2 as imageio

        if not self.url:
            raise RuntimeError("COSMOS_URL is not set")
        if not frames:
            return [], {"prompt": self.prompt, "request_scope": "episode_video"}
        original_frame_count = len(frames)
        if original_frame_count > 297:
            raise ValueError("Cosmos3 resolution 480 supports at most 297 frames per request; chunking is not implemented")
        if type(fps) not in (int, float) or isinstance(fps, bool) or not math.isfinite(fps) or fps < 1:
            raise ValueError("fps must be a finite positive number")
        first = np.asarray(frames[0])
        if first.dtype != np.uint8 or first.ndim != 3 or first.shape[2] != 3:
            raise ValueError("Cosmos3 expects uint8 RGB frames")
        if any(np.asarray(frame).shape != first.shape or np.asarray(frame).dtype != np.uint8 for frame in frames):
            raise ValueError("all episode frames must have the same uint8 RGB shape")
        if first.shape[0] % 8 or first.shape[1] % 8:
            raise ValueError("Cosmos3 input frame height and width must be multiples of 8")
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("seed must be a non-negative 32-bit integer")
        if self.resolution != "480":
            raise ValueError("this client currently supports only the Cosmos3 480 resolution tier")
        if not math.isfinite(self.control_weight) or not 0 <= self.control_weight <= 10:
            raise ValueError("control_weight must be finite and between 0 and 10")
        if self.timeout <= 0 or self.poll_interval <= 0:
            raise ValueError("timeout and poll_interval must be positive")

        # Cosmos3 video generation expects frame counts of 8n+1. Pad short or
        # misaligned episodes by repeating the final frame, then trim the output.
        inference_count = max(25, original_frame_count + ((1 - original_frame_count) % 8))
        if inference_count > 297:
            raise ValueError("padding to a valid Cosmos3 frame count would exceed the 297-frame limit")
        inference_frames = frames + [frames[-1]] * (inference_count - original_frame_count)
        size = self._size_for_shape(first.shape[0], first.shape[1])

        with tempfile.TemporaryDirectory(prefix="arrival-cosmos3-") as tmp:
            root = Path(tmp)
            source_path, output_path = root / "source.mp4", root / "output.mp4"
            self._write_video(source_path, inference_frames, float(fps))
            source_bytes = source_path.read_bytes()
            extra_params = {
                "edge": True,
                "resolution": self.resolution,
                "control_guidance": self.control_weight,
                "num_video_frames_per_chunk": inference_count,
                "max_frames": inference_count,
                "use_resolution_template": False,
                "use_duration_template": False,
                "guardrails": False,
            }
            fields = {
                "model": self.model,
                "prompt": self.prompt,
                "negative_prompt": "blurry, distorted, low quality, changed object positions, altered robot geometry",
                "size": size,
                "num_frames": str(inference_count),
                "fps": str(float(fps)),
                "num_inference_steps": "50",
                "guidance_scale": "3.0",
                "flow_shift": "10.0",
                "seed": str(seed if seed is not None else 0),
                "extra_params": json.dumps(extra_params, separators=(",", ":")),
            }
            body, boundary = self._multipart(fields, source_bytes)
            headers = {"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            last: Exception | None = None
            for attempt in range(self.retries + 1):
                if attempt:
                    time.sleep(self.backoff * attempt)
                try:
                    create_req = urllib.request.Request(
                        self.url + "/v1/videos", data=body, method="POST", headers=headers)
                    with urllib.request.urlopen(create_req, timeout=min(self.timeout, 120.0)) as response:
                        job = json.loads(response.read().decode("utf-8"))
                    if not isinstance(job, dict):
                        raise ValueError("Cosmos3 Generator job response must be a JSON object")
                    job_id = job.get("id")
                    if not isinstance(job_id, str) or not job_id:
                        raise ValueError("Cosmos3 Generator response is missing a job id")

                    deadline = time.monotonic() + self.timeout
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(f"Cosmos3 Generator job {job_id} timed out after {self.timeout:g}s")
                        status_req = urllib.request.Request(
                            self.url + "/v1/videos/" + urllib.parse.quote(job_id, safe=""),
                            headers={"Accept": "application/json", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
                        with urllib.request.urlopen(status_req, timeout=min(30.0, remaining)) as response:
                            status_body = json.loads(response.read().decode("utf-8"))
                        if not isinstance(status_body, dict):
                            raise ValueError("Cosmos3 Generator job status must be a JSON object")
                        status = status_body.get("status")
                        if status == "completed":
                            break
                        if status == "failed":
                            detail = status_body.get("error") or status_body.get("message") or "no error detail provided"
                            raise RuntimeError(f"Cosmos3 Generator job failed: {detail}")
                        if status not in ("queued", "in_progress"):
                            raise ValueError(f"Cosmos3 Generator returned unexpected job status {status!r}")
                        time.sleep(min(self.poll_interval, max(0.0, deadline - time.monotonic())))

                    content_req = urllib.request.Request(
                        self.url + "/v1/videos/" + urllib.parse.quote(job_id, safe="") + "/content",
                        headers={"Accept": "video/mp4", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
                    with urllib.request.urlopen(content_req, timeout=min(120.0, max(1.0, deadline - time.monotonic()))) as response:
                        output_path.write_bytes(response.read())
                    with imageio.get_reader(str(output_path), format="FFMPEG") as reader:
                        output_frames = [np.asarray(frame) for frame in reader]
                    if len(output_frames) != inference_count:
                        raise ValueError(f"Cosmos3 Generator returned {len(output_frames)} frames; expected {inference_count}")
                    height, width = first.shape[:2]
                    normalized = []
                    for output in output_frames:
                        if output.ndim != 3 or output.shape[2] < 3 or output.dtype != np.uint8:
                            raise ValueError(f"expected uint8 RGB output frames, got {output.dtype} {output.shape}")
                        rgb = output[..., :3]
                        if rgb.shape[:2] != (height, width):
                            rgb = np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.LANCZOS))
                        normalized.append(rgb)
                    return normalized[:original_frame_count], {
                        "prompt": self.prompt, "request_scope": "episode_video",
                        "model": self.model, "version": self.version, "resolution": self.resolution,
                        "control": "edge_derived", "control_weight": self.control_weight,
                        "input_frames": original_frame_count, "inference_frames": inference_count,
                        "output_frames": original_frame_count, "padding_frames": inference_count - original_frame_count,
                        "attempts": attempt + 1, "requested_seed": seed if seed is not None else 0,
                        "size": size, "job_id": job_id,
                    }
                except Exception as e:  # noqa: BLE001 - retry transport, job, and malformed-response failures
                    last = e
            raise RuntimeError(f"Cosmos3 Generator request failed after {self.retries + 1} attempts: {last}")

    def __call__(self, frame: np.ndarray, episode_index: int, frame_index: int) -> tuple[np.ndarray, dict]:
        raise RuntimeError("Cosmos3 Generator is video-based; call transform_video() with the full episode")

def _read_video(path: Path) -> list[np.ndarray]:
    import imageio.v2 as imageio

    with imageio.get_reader(str(path)) as r:
        return [np.asarray(f) for f in r]


def vary_dataset(src: str | Path, dst: str | Path, backend: str = "local_stub", seed: int = 0,
                 fraction: float = FRACTION, mask: np.ndarray | None = None, client=None, fps: int | None = None,
                 edge_min: float = EDGE_CORR_MIN, mask_max: float = MASK_CHANGE_MAX) -> dict:
    """Copy src to dst, restyle the selected frames of every video, write the sidecar, summary and variant mp4s."""
    from arrival.export.lerobot import write_video

    src, dst = Path(src), Path(dst)
    if backend not in ("cosmos", "local_stub"):
        raise ValueError(f"unknown backend {backend!r}")
    if client is not None and getattr(client, "name", None) != backend:
        raise ValueError(f"client backend {getattr(client, 'name', None)!r} does not match requested {backend!r}")
    select_frames(0, 0, seed, fraction)  # reject invalid fractions before copying the dataset
    if dst.resolve().is_relative_to(src.resolve()):
        raise ValueError("destination must be outside the source dataset")

    if type(edge_min) not in (int, float) or not math.isfinite(edge_min) or not -1 <= edge_min <= 1:
        raise ValueError("edge_min must be a finite correlation threshold between -1 and 1")
    if type(mask_max) not in (int, float) or not math.isfinite(mask_max) or mask_max < 0:
        raise ValueError("mask_max must be a finite, nonnegative change threshold")
    if dst.exists():
        raise FileExistsError(f"{dst} exists; the copy directory must be new")
    info = json.loads((src / "meta" / "info.json").read_text())
    fps = info.get("fps", 20) if fps is None else fps
    if type(fps) is not int or fps < 1:
        raise ValueError("fps must be a positive integer")

    stage_start = time.perf_counter()
    shutil.copytree(src, dst)
    cli = client if client is not None else (CosmosClient() if backend == "cosmos" else LocalStub(seed))
    counts = {"selected": 0, "ok": 0, "rejected": 0, "failed": 0, "skipped": 0}
    consecutive = 0
    cosmos_ok = 0
    cosmos_video_requests = 0
    cosmos_video_failures = 0
    rows: list[dict] = []
    for vid in sorted((src / "videos").glob("chunk-*/*/episode_*.mp4")):
        key, ep = vid.parent.name, int(vid.stem.split("_")[1])
        frames = _read_video(vid)
        changed = False
        selected_indices = select_frames(ep, len(frames), seed, fraction)
        cosmos_frames = None
        cosmos_error = None
        cosmos_params = {}
        cosmos_latency_ms = 0.0
        cosmos_attempted = False
        if cli.name == "cosmos" and selected_indices and cli.url and consecutive < BREAKER:
            cosmos_attempted = True
            cosmos_video_requests += 1
            cosmos_t0 = time.perf_counter()
            try:
                episode_seed = _u64("cosmos", seed, ep) % (2**32)
                cosmos_frames, cosmos_params = cli.transform_video(frames, fps, seed=episode_seed)
            except Exception as e:  # noqa: BLE001 - failed video request leaves selected source frames unchanged
                cosmos_error = str(e)
                cosmos_video_failures += 1
                consecutive += 1  # count failed video requests, not selected frames inside a video
            else:
                consecutive = 0
            cosmos_latency_ms = round((time.perf_counter() - cosmos_t0) * 1000, 2)
        for selected_position, fi in enumerate(selected_indices):
            counts["selected"] += 1
            orig = frames[fi]
            row = {"original_id": f"{key}/ep{ep:06d}/f{fi:06d}", "variant_id": f"{key}/ep{ep:06d}/f{fi:06d}_v",
                   "episode_index": ep, "frame_index": fi, "backend": cli.name, "model": cli.model,
                   "version": cli.version, "parameters": {"seed": seed, "fraction": fraction},
                   "latency_ms": None, "status": "failed", "skipped": False, "reason": None,
                   "edge_corr": None, "mask_change": None}
            if cli.name == "cosmos" and (not cli.url or (not cosmos_attempted and consecutive >= BREAKER)):
                row.update(skipped=True, reason="COSMOS_URL not set" if not cli.url else
                           f"skipped after {BREAKER} consecutive failed video requests")
                counts["skipped"] += 1
                counts["failed"] += 1
                rows.append(row)
                continue
            t0 = time.perf_counter()
            try:
                if cli.name == "cosmos":
                    if cosmos_error:
                        raise RuntimeError(cosmos_error)
                    if cosmos_frames is None:
                        raise RuntimeError("Cosmos Transfer request was not attempted")
                    out = cosmos_frames[fi]
                    params = {**cosmos_params, "frame_index": fi}
                    # The model call is per episode, so account its latency only once in aggregate metrics.
                    frame_latency = cosmos_latency_ms if selected_position == 0 else 0.0
                else:
                    out, params = cli(orig, ep, fi)
                    frame_latency = round((time.perf_counter() - t0) * 1000, 2)
                if not isinstance(out, np.ndarray):
                    raise ValueError("backend frame must be a numpy array")
                if not isinstance(params, dict):
                    raise ValueError("backend parameters must be a JSON object")
                json.dumps(params)  # sidecar parameters must be serializable
            except Exception as e:  # noqa: BLE001 - any backend error leaves the frame original
                frame_latency = cosmos_latency_ms if cli.name == "cosmos" and selected_position == 0 else round((time.perf_counter() - t0) * 1000, 2)
                row.update(latency_ms=frame_latency, reason=str(e))
                counts["failed"] += 1
                rows.append(row)
                continue
            row.update(latency_ms=frame_latency, model=cli.model, version=cli.version,
                       parameters={**row["parameters"], **params})
            chk = check_frame(orig, out, mask, edge_min, mask_max)
            row.update(edge_corr=chk["edge_corr"], mask_change=chk["mask_change"], reason=chk["reason"])
            if chk["ok"]:
                row["status"] = "ok"
                counts["ok"] += 1
                cosmos_ok += cli.name == "cosmos"
                frames[fi] = out
                changed = True
            else:
                row["status"] = "rejected"
                counts["rejected"] += 1
            rows.append(row)
        if changed:
            write_video(dst / VIDEO_DIR / key / vid.name, np.stack(frames), fps)
    label = "cosmos" if cosmos_ok else ("local_stub" if backend == "local_stub" and counts["ok"] else "none")
    (dst / "meta" / SIDECAR).write_text("".join(json.dumps(r) + "\n" for r in rows))
    backend_latency_s = round(sum(r["latency_ms"] or 0 for r in rows) / 1000, 3)
    summary = {"arrival_appearance_variation": label, "backend_requested": backend, "seed": seed,
               "fraction": fraction, "requests": counts.pop("selected"), **counts,
               "cosmos_video_requests": cosmos_video_requests if cli.name == "cosmos" else 0,
               "cosmos_video_failures": cosmos_video_failures if cli.name == "cosmos" else 0,
               "cosmos_model": cli.model if cli.name == "cosmos" and cosmos_ok else None,
               "backend_latency_s": backend_latency_s,
               "added_latency_s": round(time.perf_counter() - stage_start, 3), "cost_usd": None,
               "note": "frames not ok stay original; added latency includes copy, backend calls and video writes; "
                       "USD is unknown without endpoint billing data"}
    (dst / "meta" / SUMMARY).write_text(json.dumps(summary, indent=2))
    info["arrival_appearance_variation"] = label
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="appearance variation over a COPY of a sim dataset")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--backend", choices=["cosmos", "local_stub"], default="local_stub")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=FRACTION)
    a = ap.parse_args(argv)
    print(json.dumps(vary_dataset(a.src, a.dst, a.backend, a.seed, a.fraction), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
