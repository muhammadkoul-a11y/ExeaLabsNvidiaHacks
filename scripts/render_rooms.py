"""Top-down PNG of every room under scans/ for eyeballing a compiled twin (TWN-2).

    python scripts/render_rooms.py                                   # all rooms under scans/
    python scripts/render_rooms.py scans/hero_kitchen tests/fixtures/roomplan/office.json

Writes out/renders/<room>_top.png from scan.json and <room>_roomplan_top.png from roomplan.json
(a bare .json argument is treated as a RoomPlan export). Walls are dark lines, doors brown, windows
blue, openings grey, the scanned floor polygon a pale fill, fixtures rectangles rotated by yaw and
labelled with id and type, objects dots. 100 px per metre, +Y up.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrival.scan import bundle, roomplan  # noqa: E402
from arrival.twin.compile import compile_twin  # noqa: E402

MARGIN = 0.6
COLORS = {"counter": (214, 182, 128), "sink": (120, 170, 220), "stove": (220, 110, 90),
          "fridge_bottom_freezer": (160, 160, 170), "dishwasher": (150, 200, 190),
          "hinge_cabinet": (190, 160, 120), "box": (200, 180, 220)}
PORTALS = {"doors": (150, 90, 40), "windows": (70, 130, 230), "openings": (150, 150, 150)}


def _twins(paths: list[Path]) -> list[tuple[str, dict]]:
    out = []
    for p in paths:
        if p.is_dir():
            if (p / "scan.json").is_file():
                out.append((p.name, compile_twin(bundle.load(p / "scan.json"))))
            if (p / "roomplan.json").is_file():
                out.append((f"{p.name}_roomplan", compile_twin(roomplan.load(p / "roomplan.json", p.name))))
        elif p.suffix == ".json":
            out.append((p.stem, compile_twin(roomplan.load(p, p.stem))))
    return out


def _corners(f: dict) -> list[tuple[float, float]]:
    sx, sy, _ = f["size"]
    x, y, _ = f["position"]
    c, s = math.cos(f.get("yaw", 0.0)), math.sin(f.get("yaw", 0.0))
    return [(x + c * dx - s * dy, y + s * dx + c * dy)
            for dx, dy in ((sx / 2, sy / 2), (-sx / 2, sy / 2), (-sx / 2, -sy / 2), (sx / 2, -sy / 2))]


def render(name: str, twin: dict, scale: float) -> Image.Image:
    pts = [p for w in twin["walls"] for p in (w["start"], w["end"])]
    pts += [c for f in twin["fixtures"] for c in _corners(f)]
    pts += [c for fl in twin.get("floors", []) for c in fl["corners"]]
    pts += [o["position"][:2] for o in twin["objects"]]
    if not pts:
        pts = [(0, 0), (1, 1)]
    x0, x1 = min(p[0] for p in pts) - MARGIN, max(p[0] for p in pts) + MARGIN
    y0, y1 = min(p[1] for p in pts) - MARGIN, max(p[1] for p in pts) + MARGIN
    img = Image.new("RGB", (int((x1 - x0) * scale), int((y1 - y0) * scale)), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=12)
    small = ImageFont.load_default(size=10)

    def px(p):
        return ((p[0] - x0) * scale, (y1 - p[1]) * scale)

    for fl in twin.get("floors", []):
        draw.polygon([px(c) for c in fl["corners"]], fill=(236, 240, 232), outline=(190, 200, 180))
    for f in twin["fixtures"]:
        draw.polygon([px(c) for c in _corners(f)], fill=COLORS.get(f["type"], COLORS["box"]), outline=(60, 60, 60))
        label = f"{f['id'][:12]}\n{f['type']}" + ("\n(island)" if f.get("island") else "")
        draw.multiline_text(px(f["position"]), label, fill=(20, 20, 20), font=small, anchor="mm", align="center")
    for w in twin["walls"]:
        draw.line([px(w["start"]), px(w["end"])], fill=(40, 40, 40), width=max(2, int(w["thickness"] * scale)))
        draw.text(px(((w["start"][0] + w["end"][0]) / 2, (w["start"][1] + w["end"][1]) / 2)), w["id"][:12],
                  fill=(90, 90, 90), font=small, anchor="mm")
    for key, color in PORTALS.items():
        for p in twin.get(key, []):
            c, s = math.cos(p["yaw"]), math.sin(p["yaw"])
            h = p["size"][0] / 2
            a = (p["position"][0] - c * h, p["position"][1] - s * h)
            b = (p["position"][0] + c * h, p["position"][1] + s * h)
            draw.line([px(a), px(b)], fill=color, width=6)
            draw.text(px(p["position"]), key[:-1] + (" open" if p.get("is_open") else ""), fill=color, font=small,
                      anchor="md")
    for o in twin["objects"]:
        x, y = px(o["position"])
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(200, 40, 40))
        draw.text((x + 6, y), o["id"], fill=(120, 20, 20), font=small, anchor="lm")
    title = f"{name}  {len(twin['walls'])} walls, {len(twin['fixtures'])} fixtures, {len(twin['objects'])} objects"
    if twin.get("story_count", 1) > 1:
        title += f", story {twin['story']} of {twin['story_count']}"
    draw.text((6, 4), title, fill=(0, 0, 0), font=font)
    return img


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", help="room directories or RoomPlan .json files (default: scans/*)")
    ap.add_argument("--out", default="out/renders")
    ap.add_argument("--scale", type=float, default=100.0, help="pixels per metre")
    a = ap.parse_args(argv)
    missing = [p for p in a.paths if not (Path(p).is_file() and Path(p).suffix == ".json")
               and not any((Path(p) / n).is_file() for n in ("scan.json", "roomplan.json"))]
    if missing:
        ap.error(f"no scan.json or roomplan.json at {', '.join(missing)}")
    paths = [Path(p) for p in a.paths] or sorted(p for p in (ROOT / "scans").iterdir() if p.is_dir())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, twin in _twins(paths):
        path = out / f"{name}_top.png"
        render(name, twin, a.scale).save(path)
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
