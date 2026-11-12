"""Where the Classroom dolly's preflight holes are (issue #1, Code, workstation).

    .venv/bin/python docs/runs/issue-1/preflight_holes.py runs/dolly-1 [k ...]

Reads <run>/preflight.json and the preflight's own panoramas (<run>/preflight-renders/*/sanity.exr).
A hole is a panorama pixel with Object Index 0, as fov_render.record.build counts it (solid-angle
weighted, as fov_render.bl.scene.sanity_checks). For each head: the hole fraction, the share of it whose
rays point toward +X (the window wall, right of the path), the heights at which those rays cross the
plane x = 4.506 m (the scene's +X bound, tools/classroom_oracle/place_eye.py), and the hole fraction
left outside that wall's band z 0.7..3.0 m. Also writes preflight_holes.jpg next to this script: the
panoramas of the first head and of the worst one, holes in magenta (centre = +Y, the board; right = +X).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from fov_render.bl.engine import read_passes
from fov_render.sensors import PanoramaModel

X_WALL = 4.506               # place_eye.py: the +X bound of classroom_eye.blend
BAND_Z = (0.7, 3.0)


def holes(run: Path, sub: str) -> np.ndarray:
    p = read_passes(str(run / "preflight-renders" / sub / "sanity.exr"))
    return np.rint(p["object_index"]).reshape(-1) <= 0


def main(run: Path, ks: list[int]) -> None:
    pf = json.loads((run / "preflight.json").read_text())
    m = PanoramaModel(pf["width"])
    _, lat = m.lonlat()
    w = ((2 * math.pi / m.width) * (math.pi / m.height) * np.cos(lat)).reshape(-1)
    w = w / w.sum()
    f = np.asarray(pf["path"]["facing_w"], float)
    R_wh = np.stack([np.cross(f, [0, 0, 1]), [0, 0, 1], -f], axis=1)     # motion.paths.head_axes
    d = m.rays().direction.reshape(-1, 3) @ R_wh.T
    rows = {h["k"]: h for h in pf["heads"]}
    print(f"{'k':>7s} {'y_w':>7s} {'holes':>8s} {'toward +X':>10s} {'z at x=4.506':>14s} {'outside band':>13s}")
    for k in (ks or sorted(rows)) + ["control"]:
        h = pf["control"] if k == "control" else rows[k]
        o = np.asarray(h["head_origin_w_m"], float)
        hole = holes(run, "control" if k == "control" else f"k{k:04d}")
        with np.errstate(divide="ignore", invalid="ignore"):
            z = o[2] + (X_WALL - o[0]) / d[:, 0] * d[:, 2]
        px = hole & (d[:, 0] > 0)
        band = px & (z > BAND_Z[0]) & (z < BAND_Z[1])
        print(f"{str(k):>7s} {o[1]:+7.3f} {w[hole].sum():8.5f} {w[px].sum() / w[hole].sum():10.4f} "
              f"{z[px].min():6.2f}..{z[px].max():5.2f} {w[hole & ~band].sum():13.5f}")
    worst = max(rows, key=lambda k: rows[k]["hole_fraction"])
    tiles = []
    for k in (min(rows), worst):
        p = read_passes(str(run / "preflight-renders" / f"k{k:04d}" / "sanity.exr"))
        c = np.clip(p["combined"][..., :3], 0, None)
        img = (np.clip(1.6 * (c / (1 + c)) ** (1 / 2.2), 0, 1) * 255).astype(np.uint8)
        img[np.rint(p["object_index"]) <= 0] = [255, 0, 255]
        t = Image.fromarray(img)
        ImageDraw.Draw(t).text((6, 4), f"k={k}  y={rows[k]['head_origin_w_m'][1]:+.3f}  holes "
                               f"{rows[k]['hole_fraction']:.4f} (limit {pf['limits']['max_hole_fraction']})",
                               fill=(255, 255, 0))
        tiles.append(t)
    sheet = Image.new("RGB", (tiles[0].width, sum(t.height for t in tiles)))
    for i, t in enumerate(tiles):
        sheet.paste(t, (0, i * tiles[0].height))
    out = Path(__file__).with_suffix(".jpg")
    sheet.save(out, quality=85)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), [int(a) for a in sys.argv[2:]])
