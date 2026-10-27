"""L14: accept a CAP-2a RGB-D frames bundle (``frames_manifest.json`` plus the files it names, zipped) through the
upload path. The zip is unpacked into a scratch folder with zip-slip, symlink, size and count guards, then checked with
``arrival.scan.rgbd.load_manifest`` and a per-frame look at every image and depth array. Frames are stored next to
the job (``out/jobs/<id>/frames/``); the worker's realism stage records whether they were consumed.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

MANIFEST_NAME = "frames_manifest.json"
ALLOWED_SUFFIXES = (".json", ".png", ".jpg", ".jpeg", ".npy")
MAX_MEMBERS = 4000
MAX_FRAMES = 2000
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_RGB_PIXELS = 4096 * 4096
MAX_DEPTH_PIXELS = 2048 * 2048
CHUNK = 1 << 20
ARCHIVE_CODES = ("bad_zip", "bad_path", "bad_file_type", "too_many_files")  # the zip itself is unacceptable (HTTP 400)
NAME_OK = re.compile(r"^[A-Za-z0-9._()-]+(/[A-Za-z0-9._()-]+)*$")


class FramesError(ValueError):
    """A frames bundle that must not be queued. ``code`` is a rgbd ManifestError code or one of this module's."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class _Slice(io.RawIOBase):
    """Read-only view of ``length`` bytes at ``offset`` of an open file, so a zip inside a multipart body can be
    read in place instead of being copied into memory."""

    def __init__(self, f, offset: int, length: int):
        self.f, self.offset, self.length, self.pos = f, offset, length, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, pos, whence=0):
        base = {0: 0, 1: self.pos, 2: self.length}[whence]
        self.pos = max(0, min(self.length, base + pos))
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.length - self.pos)
        if n <= 0:
            return 0
        self.f.seek(self.offset + self.pos)
        data = self.f.read(n)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


def _check_name(name: str) -> None:
    if len(name) > 200 or not NAME_OK.match(name) or any(p in (".", "..") for p in name.split("/")):
        raise FramesError("bad_path", f"unsafe path in zip: {name[:80]!r}")
    if not name.lower().endswith(ALLOWED_SUFFIXES):
        raise FramesError("bad_file_type", f"{name!r}: only {', '.join(ALLOWED_SUFFIXES)} files are accepted")


def extract(src, offset: int, length: int, dest: Path) -> int:
    """Unpack the zip stored at ``src[offset:offset+length]`` into the empty folder ``dest``. Returns the number of
    files written. Entry sizes in the zip header are not trusted: every member is streamed against the caps."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    try:
        zf = zipfile.ZipFile(io.BufferedReader(_Slice(src, offset, length)))
    except zipfile.BadZipFile as e:
        raise FramesError("bad_zip", f"frames file is not a readable zip: {e}") from None
    seen: set[str] = set()
    total = written = 0
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_MEMBERS:
            raise FramesError("too_many_files", f"zip has {len(infos)} entries, limit {MAX_MEMBERS}")
        for zi in infos:
            if zi.is_dir():
                continue
            name = zi.orig_filename  # ZipInfo normalizes backslashes on Windows.
            _check_name(name)
            if (zi.external_attr >> 16) & 0o170000 == 0o120000:
                raise FramesError("bad_path", f"{name!r} is a symlink")
            if zi.flag_bits & 0x1:
                raise FramesError("bad_zip", f"{name!r} is encrypted")
            if name.lower() in seen:
                raise FramesError("bad_path", f"{name!r} appears twice in the zip")
            seen.add(name.lower())
            cap = MAX_MANIFEST_BYTES if name.lower().endswith(".json") else MAX_MEMBER_BYTES
            target = (dest / name).resolve()
            if root not in target.parents:
                raise FramesError("bad_path", f"{name!r} escapes the bundle folder")
            target.parent.mkdir(parents=True, exist_ok=True)
            size = 0
            try:
                with zf.open(zi) as fin, open(target, "wb") as fout:
                    while chunk := fin.read(CHUNK):
                        size += len(chunk)
                        total += len(chunk)
                        if size > cap or total > MAX_TOTAL_BYTES:
                            raise FramesError("too_large", f"{name!r} or the bundle exceeds the unpacked size limit")
                        fout.write(chunk)
            except (zipfile.BadZipFile, EOFError, RuntimeError, OSError) as e:
                raise FramesError("bad_zip", f"{name!r} could not be read from the zip: {e}") from None
            written += 1
    return written


def validate(root: Path) -> dict:
    """Run the CAP-2a loader over an unpacked bundle and look at every frame file. Returns the summary stored in
    the job status. Raises FramesError with the loader's error code."""
    import numpy as np
    from PIL import Image

    from arrival.scan import rgbd

    root = Path(root)
    try:
        m = rgbd.load_manifest(root)
    except rgbd.ManifestError as e:
        raise FramesError(e.code, str(e)) from None
    if len(m.frames) > MAX_FRAMES:
        raise FramesError("too_many_files", f"{len(m.frames)} frames, limit {MAX_FRAMES}")
    if m.rgb_size[0] * m.rgb_size[1] > MAX_RGB_PIXELS or m.depth_size[0] * m.depth_size[1] > MAX_DEPTH_PIXELS:
        raise FramesError("bad_size", "rgb_size or depth_size is larger than this server accepts")
    rgb_wh, depth_hw = tuple(m.rgb_size), (m.depth_size[1], m.depth_size[0])

    def open_image(path: Path, want_wh: tuple[int, int] | None, what: str):
        try:
            with Image.open(path) as im:
                if want_wh is not None and im.size != want_wh:
                    raise FramesError("bad_size", f"{what} {path.name} is {im.size}, manifest says {want_wh}")
                im.verify()
                return im.mode
        except FramesError:
            raise
        except Exception as e:  # noqa: BLE001 - PIL raises many types for a corrupt or non-image file
            raise FramesError("bad_image", f"{what} {path.name} is not a readable image: {type(e).__name__}") from None

    for f in m.frames:
        rgb_mode = open_image(f.rgb, rgb_wh, "rgb")
        if rgb_mode not in ("RGB", "RGBA"):
            raise FramesError("bad_image", f"rgb {f.rgb.name} must contain RGB channels, got mode {rgb_mode}")
        if f.confidence is not None:
            mode = open_image(f.confidence, m.depth_size, "confidence")
            if mode != "L":
                raise FramesError("bad_confidence", f"confidence {f.confidence.name} must be an 8-bit grayscale PNG")
            with Image.open(f.confidence) as im:
                if np.asarray(im).max() > 2:
                    raise FramesError("bad_confidence", f"confidence {f.confidence.name} must use ARKit levels 0, 1 or 2")
        if m.depth_unit == "m":
            try:
                head = np.load(f.depth, mmap_mode="r", allow_pickle=False)
                if not isinstance(head, np.ndarray):
                    head.close()
                    raise ValueError("expected a plain .npy array, not an archive")
            except Exception as e:  # noqa: BLE001
                raise FramesError("bad_depth", f"depth {f.depth.name} is not a plain .npy array: {type(e).__name__}") from None
            shape, dtype = head.shape, head.dtype
            del head
            if shape != depth_hw or dtype != np.dtype("float32"):
                raise FramesError("depth_shape", f"depth {f.depth.name} must be float32 {depth_hw}, got {dtype} {shape}")
        else:
            mode = open_image(f.depth, m.depth_size, "depth")
            if mode not in ("I;16", "I"):
                raise FramesError("bad_depth", f"depth {f.depth.name} must be a 16-bit PNG, got mode {mode}")
        try:
            rgbd.load_depth(m, f)
        except rgbd.ManifestError as e:
            raise FramesError(e.code, str(e)) from None
    return {
        "room_id": m.room_id,
        "count": len(m.frames),
        "train": sum(f.split == "train" for f in m.frames),
        "heldout": sum(f.split == "heldout" for f in m.frames),
        "depth_unit": m.depth_unit,
        "rgb_size": list(m.rgb_size),
        "depth_size": list(m.depth_size),
        "stored": True,
        "consumed": False,  # the worker updates this after the optional realism stage
    }
