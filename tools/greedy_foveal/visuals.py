"""Greedy Foveal Explorer, baseline: the overview figure (PIL + the Visual Language 1 style module).

    A  the reconstructed map seen from H0 (equirectangular) with the fixation trajectory (local solid blue, global
       dashed vermilion);
    B  the final UNSEEN / SEEN / DEPTH coverage state (1-degree grid);
    C  SEEN and DEPTH coverage vs fixation number (global saccades ticked);
    D  compact statistics (control);
    E  (free slot, right of B: f3d-vision drew its post-hoc Breadth-1 evaluation here; see docs/exercises.md)
    F  the 3-D map from above (head-frame X / Z, ceiling cut) in display RGB and in the ORACLE segmentation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "visual_language"))
sys.path.insert(0, str(HERE))

import explorer as X  # noqa: E402
import style as S  # noqa: E402  (Visual Language 1 palette and fonts)

C_LOCAL, C_GLOBAL = S.OI_BLUE, S.OI_VERM
C_UNSEEN, C_SEEN, C_DEPTH = (236, 235, 230), (166, 196, 222), (0, 92, 150)
C_COVERED, C_MISSED, C_NOHIT = (0, 120, 88), (222, 220, 214), (255, 255, 255)
CEILING_CUT_M = 0.3      # top-down view: keep head-frame Y below this (removes the ceiling)


def display_rgb(rgb: np.ndarray) -> np.ndarray:
    x = np.asarray(rgb, np.float64) * 6.0
    x = np.clip(x / (1.0 + x), 0.0, 1.0)
    return np.rint(255.0 * np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)).astype(np.uint8)


def to_xy(yaw, pitch, box) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = box
    return (x0 + (np.asarray(yaw) + 180.0) / 360.0 * (x1 - x0), y0 + (90.0 - np.asarray(pitch)) / 180.0 * (y1 - y0))


def frame_box(img, d, box, title, sub=""):
    x0, y0, x1, y1 = box
    d.rectangle([x0 - 1, y0 - 1, x1, y1], outline=S.FAINT, width=1)
    S.text(d, (x0, y0 - 34), title, size=S.T_HEAD, bold=True, outline=None)
    if sub:
        S.text(d, (x0 + d.textlength(title, font=S.font(S.T_HEAD, True)) + 14, y0 - 30), sub, size=S.T_SMALL,
               fill=S.INK2, outline=None)


def sphere_grid(d, box, labels=True):
    x0, y0, x1, y1 = box
    for yaw in range(-180, 181, 45):
        x, _ = to_xy(yaw, 0, box)
        d.line([x, y0, x, y1], fill=S.GRID if yaw else S.FAINT, width=1)
        if labels:
            S.text(d, (x, y1 + 6), f"{yaw:+d}°" if yaw else "0°", size=S.T_SMALL, fill=S.MUTED, outline=None,
                   anchor="ma")
    for pitch in range(-90, 91, 30):
        _, y = to_xy(0, pitch, box)
        d.line([x0, y, x1, y], fill=S.GRID if pitch else S.FAINT, width=1)
        if labels:
            S.text(d, (x0 - 8, y), f"{pitch:+d}°" if pitch else "0°", size=S.T_SMALL, fill=S.MUTED, outline=None,
                   anchor="rm")


def arc(a: np.ndarray, b: np.ndarray, step_deg: float = 0.5) -> np.ndarray:
    ang = math.degrees(math.acos(float(np.clip(a @ b, -1, 1))))
    n = max(2, int(ang / step_deg) + 2)
    t = np.linspace(0, 1, n)[:, None]
    om = math.radians(ang)
    if om < 1e-9:
        return np.vstack([a, b])
    return (np.sin((1 - t) * om) * a + np.sin(t * om) * b) / math.sin(om)


def dashed_polyline(d, pts, color, width=3, dash=14.0, gap=9.0):
    """Dashes measured along the whole polyline (S.dashed_line restarts per segment, so short arc pieces go solid)."""
    pos, on = 0.0, True
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        s = 0.0
        while s < L:
            left = (dash if on else gap) - pos
            e = min(L, s + left)
            if on:
                d.line([x0 + (x1 - x0) * s / L, y0 + (y1 - y0) * s / L, x0 + (x1 - x0) * e / L,
                        y0 + (y1 - y0) * e / L], fill=color, width=width)
            pos += e - s
            s = e
            if pos >= (dash if on else gap) - 1e-9:
                pos, on = 0.0, not on


def draw_path(d, box, pts_dir, color, dashed, width=3):
    yaw, pitch = X.yaw_pitch(pts_dir)
    xs, ys = to_xy(yaw, pitch, box)
    W = box[2] - box[0]
    seg = [(float(xs[0]), float(ys[0]))]
    runs = []
    for k in range(1, len(xs)):
        if abs(xs[k] - xs[k - 1]) > W / 2:
            runs.append(seg)
            seg = []
        seg.append((float(xs[k]), float(ys[k])))
    runs.append(seg)
    for r in runs:
        if len(r) < 2:
            continue
        if dashed:
            dashed_polyline(d, r, S.WHITE, width=width + 4)
            dashed_polyline(d, r, color, width=width)
        else:
            d.line(r, fill=S.WHITE, width=width + 4)
            d.line(r, fill=color, width=width)


def panel_a(img, d, box, traj, arr):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    bg = np.full((h, w, 3), C_UNSEEN, np.uint8)
    xyz = arr["xyz_h"]
    rng = np.linalg.norm(xyz, axis=1)
    yaw, pitch = X.yaw_pitch(xyz)
    px = np.clip(((yaw + 180) / 360 * w).astype(int), 0, w - 1)
    py = np.clip(((90 - pitch) / 180 * h).astype(int), 0, h - 1)
    order = np.argsort(-rng)                         # nearest written last
    bg[py[order], px[order]] = display_rgb(arr["rgb"][order])
    img.paste(Image.fromarray(bg), (x0, y0))
    sphere_grid(d, box)
    T = traj["trajectory"]
    dirs = X.direction(np.array([e["yaw_deg"] for e in T]), np.array([e["pitch_deg"] for e in T]))
    for k in range(1, len(T)):
        glob = T[k]["kind"] == "global"
        draw_path(d, box, arc(dirs[k - 1], dirs[k]), C_GLOBAL if glob else C_LOCAL, glob)
    xs, ys = to_xy([e["yaw_deg"] for e in T], [e["pitch_deg"] for e in T], box)
    for k, e in enumerate(T):
        col = C_GLOBAL if e["kind"] == "global" else C_LOCAL
        r = 5
        d.ellipse([xs[k] - r - 2, ys[k] - r - 2, xs[k] + r + 2, ys[k] + r + 2], fill=S.WHITE)
        d.ellipse([xs[k] - r, ys[k] - r, xs[k] + r, ys[k] + r], fill=col)
        if e["kind"] == "global":
            S.text(d, (xs[k] + 9, ys[k] - 9), str(e["n"]), size=S.T_SMALL, bold=True, fill=S.INK, anchor="ld")
    S.crosshair(d, float(xs[0]), float(ys[0]), r=14)
    S.text(d, (xs[0] + 16, ys[0] + 10), "start (0°, 0°)", size=S.T_SMALL, bold=True, anchor="la")
    S.square(d, float(xs[-1]), float(ys[-1]), r=7, color=S.INK)
    S.text(d, (xs[-1] + 12, ys[-1] + 10), f"end #{T[-1]['n']}", size=S.T_SMALL, bold=True, anchor="la")
    # legend
    lx, ly = x0, y1 + 40
    d.line([lx, ly + 10, lx + 46, ly + 10], fill=C_LOCAL, width=3)
    S.text(d, (lx + 56, ly), f"local saccade ({traj['local_saccades']})", size=S.T_SMALL, outline=None)
    lx += 300
    dashed_polyline(d, [(lx, ly + 10), (lx + 46, ly + 10)], C_GLOBAL, width=3)
    S.text(d, (lx + 56, ly), f"global saccade ({traj['global_saccades']}; target numbered)", size=S.T_SMALL,
           outline=None)
    lx += 470
    d.rectangle([lx, ly + 2, lx + 30, ly + 20], fill=C_UNSEEN, outline=S.FAINT)
    S.text(d, (lx + 40, ly), "no reconstructed geometry", size=S.T_SMALL, outline=None)
    lx += 330
    S.text(d, (lx, ly), "background: final H0 surfels, display RGB, nearest per pixel", size=S.T_SMALL,
           fill=S.MUTED, outline=None)


def grid_image(state: np.ndarray, box, colors: dict) -> Image.Image:
    x0, y0, x1, y1 = box
    rgb = np.zeros(state.shape + (3,), np.uint8)
    for k, c in colors.items():
        rgb[state == k] = c
    return Image.fromarray(rgb).resize((x1 - x0, y1 - y0), Image.NEAREST)


def legend_swatches(d, x, y, items):
    for label, col in items:
        d.rectangle([x, y + 2, x + 28, y + 22], fill=col, outline=S.FAINT)
        S.text(d, (x + 36, y), label, size=S.T_SMALL, outline=None)
        x += 36 + d.textlength(label, font=S.font(S.T_SMALL)) + 28


def panel_b(img, d, box, cov, traj):
    img.paste(grid_image(cov["state"], box, {X.UNSEEN: C_UNSEEN, X.SEEN: C_SEEN, X.DEPTH: C_DEPTH}), box[:2])
    sphere_grid(d, box)
    T = traj["trajectory"]
    xs, ys = to_xy([e["yaw_deg"] for e in T], [e["pitch_deg"] for e in T], box)
    for x, y in zip(xs, ys):
        d.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=S.WHITE)
    seen, depth = traj["final_seen_fraction"], traj["final_depth_fraction"]
    legend_swatches(d, box[0], box[3] + 40, [(f"UNSEEN {100 * (1 - seen):.1f}%", C_UNSEEN),
                                             (f"SEEN, no depth {100 * (seen - depth):.1f}%", C_SEEN),
                                             (f"DEPTH {100 * depth:.1f}%", C_DEPTH), ("fixation centre", S.WHITE)])


def panel_c(img, d, box, cov, traj, marks: list[tuple[int, str]] | None = None):
    x0, y0, x1, y1 = box
    sh, dh = np.asarray(cov["seen_history"]) * 100, np.asarray(cov["depth_history"]) * 100
    n = len(sh)
    ymax = max(10.0, math.ceil(sh.max() / 10) * 10)
    for v in np.arange(0, ymax + 1e-9, 10 if ymax <= 60 else 20):
        y = y1 - v / ymax * (y1 - y0)
        d.line([x0, y, x1, y], fill=S.GRID, width=1)
        S.text(d, (x0 - 8, y), f"{v:.0f}%", size=S.T_SMALL, fill=S.MUTED, outline=None, anchor="rm")
    step = 50 if n > 250 else (25 if n > 60 else (5 if n > 10 else 1))
    for k in range(0, n + 1, step):
        x = x0 + k / max(1, n) * (x1 - x0)
        S.text(d, (x, y1 + 6), str(k), size=S.T_SMALL, fill=S.MUTED, outline=None, anchor="ma")
    S.text(d, ((x0 + x1) / 2, y1 + 32), "fixation number", size=S.T_SMALL, fill=S.INK2, outline=None, anchor="ma")
    d.line([x0, y1, x1, y1], fill=S.MUTED, width=1)
    for e in traj["trajectory"]:
        if e["kind"] == "global":
            x = x0 + e["n"] / max(1, n) * (x1 - x0)
            d.line([x, y1 - 12, x, y1], fill=C_GLOBAL, width=2)
    xs = x0 + np.arange(1, n + 1) / max(1, n) * (x1 - x0)
    for i, (k, name) in enumerate(marks or []):
        if not 1 <= k <= n:
            continue
        xm = x0 + k / n * (x1 - x0)
        ym = y1 - sh[k - 1] / ymax * (y1 - y0)
        near_top = ym - 34 <= y0 + 60
        dashed_polyline(d, [(xm, y1), (xm, ym if near_top else y0 + 60)], S.INK2, width=2, dash=8.0, gap=6.0)
        d.ellipse([xm - 6, ym - 6, xm + 6, ym + 6], fill=S.WHITE, outline=S.INK, width=2)
        label = f"{name} (#{k}): SEEN {sh[k - 1]:.1f}%" if name else f"#{k}: SEEN {sh[k - 1]:.1f}%"
        ly = ym - 34 if not near_top else y1 - 50 - 32 * i     # near the top: at the foot of its guide line
        S.text(d, (xm - 10, ly), label, size=S.T_SMALL, bold=True, anchor="rm", outline=S.WHITE)
    for hist, col, lab in ((sh, S.OI_SKY, "SEEN"), (dh, C_DEPTH, "DEPTH")):
        pts = [(float(x), float(y1 - v / ymax * (y1 - y0))) for x, v in zip(xs, hist)]
        if len(pts) > 1:
            d.line(pts, fill=col, width=3)
        S.text(d, (pts[-1][0] + 8, pts[-1][1] + (-14 if lab == "SEEN" else 12)), f"{lab} {hist[-1]:.1f}%",
               size=S.T_SMALL, bold=True, anchor="lm", outline=S.WHITE)
    S.text(d, (x0 + 10, y0 + 4), "vermilion ticks: global saccades", size=S.T_SMALL, fill=S.MUTED, outline=None)
    S.text(d, (x0 + 10, y0 + 28), "(SEEN and DEPTH lines coincide where every seen cell got depth)", size=S.T_SMALL,
           fill=S.MUTED, outline=None)


def panel_d(d, x, y, traj):
    s = traj["seconds_sum"]
    rows = [("fixations", f"{traj['fixations']}  ({traj['local_saccades']} local, {traj['global_saccades']} global)"),
            ("stop", traj["stop_reason"]),
            ("angular SEEN / DEPTH", f"{100 * traj['final_seen_fraction']:.2f}% / {100 * traj['final_depth_fraction']:.2f}%"
                                     " of 4π")]
    rows += [("final surfels", f"{traj['final_surfels']:,d}"),
             ("wall time", f"{traj['wall_seconds']:.0f} s  (render {s['render']:.0f} / corr. "
                           f"{s['correspondence']:.0f} / fusion {s['fusion']:.0f} / policy {s['policy']:.0f} s)"),
             ("render", f"{traj['spp']} spp, OPTIX, 640 × 640 per eye, 12° core"),
             ("correspondence", "PERFECT / ORACLE (+ instance-0 extension)")]
    for k, v in rows:
        S.text(d, (x, y), k, size=S.T_SMALL, fill=S.INK2, outline=None)
        S.text(d, (x + 260, y), v, size=S.T_SMALL, bold=True, outline=None)
        y += 34
    return y


def top_down(arr, colors, w, h, bounds):
    xmin, xmax, zmin, zmax = bounds
    img = np.full((h, w, 3), 255, np.uint8)
    xyz = arr["xyz_h"]
    keep = xyz[:, 1] <= CEILING_CUT_M
    p, c = xyz[keep], colors[keep]
    s = min(w / (xmax - xmin), h / (zmax - zmin))
    px = np.clip(((p[:, 0] - xmin) * s).astype(int), 0, w - 1)
    pz = np.clip(((p[:, 2] - zmin) * s).astype(int), 0, h - 1)
    order = np.argsort(p[:, 1])                       # highest written last
    img[pz[order], px[order]] = c[order]
    return Image.fromarray(img), s


def oracle_colors(ids):
    ids = np.asarray(ids, np.int64)
    hsh = (ids * 2654435761) & 0xFFFFFFFF
    col = 60 + 0.75 * np.stack([(hsh >> 0) & 255, (hsh >> 8) & 255, (hsh >> 16) & 255], 1).astype(np.float64)
    col[ids == 0] = (150, 150, 150)
    return np.clip(col, 0, 255).astype(np.uint8)


OVERVIEW_TITLE = "Greedy Foveal Explorer v0 — FIXATE → RECONSTRUCT → SACCADE (Classroom, fixed head H0)"


def overview(run: Path, vis: Path, marks: list[tuple[int, str]] | None = None, title: str = OVERVIEW_TITLE) -> dict:
    vis.mkdir(parents=True, exist_ok=True)
    traj = json.loads((run / "trajectory.json").read_text())
    with np.load(run / "final-map.npz") as z:
        arr = {k: np.array(z[k]) for k in ("xyz_h", "rgb", "instance_id_oracle")}
    with np.load(run / "coverage.npz") as z:
        cov = {k: np.array(z[k]) for k in z.files}
    W_, M = 2000, 90
    img = Image.new("RGB", (W_, 4200), S.SURFACE)
    d = ImageDraw.Draw(img)
    S.text(d, (M, 30), title, size=S.T_TITLE, bold=True, outline=None)
    S.text(d, (M, 72), "PROTOTYPE. PERFECT / ORACLE correspondence; geometry from accepted spherical epipolar "
                       "triangulation; one global 12-mm surfel map.",
           size=S.T_SMALL, fill=S.INK2, outline=None)
    y = 160
    boxA = (M, y, W_ - M, y + (W_ - 2 * M) // 2)
    frame_box(img, d, boxA, "A  fixation trajectory over the reconstructed sphere",
              "equirectangular, yaw → / pitch ↑, from the head origin")
    panel_a(img, d, boxA, traj, arr)
    y = boxA[3] + 140
    half = (W_ - 2 * M - 80) // 2
    boxB = (M, y, M + half, y + half // 2)
    frame_box(img, d, boxB, "B  final coverage state", "1° cells, during control")
    panel_b(img, d, boxB, cov, traj)
    # boxE = (M + half + 80, y, W_ - M, y + half // 2) is left free (see the module docstring)
    y = boxB[3] + 150
    boxC = (M + 60, y, M + half - 120, y + 380)
    frame_box(img, d, boxC, "C  SEEN / DEPTH vs fixation", "% of 4π")
    panel_c(img, d, boxC, cov, traj, marks)
    S.text(d, (M + half + 80, y - 34), "D  statistics", size=S.T_HEAD, bold=True, outline=None)
    panel_d(d, M + half + 80, y + 6, traj)
    y = boxC[3] + 140
    xyz = arr["xyz_h"]
    lo, hi = np.percentile(xyz[:, [0, 2]], 0.5, axis=0), np.percentile(xyz[:, [0, 2]], 99.5, axis=0)
    pad = 0.2
    bounds = (lo[0] - pad, hi[0] + pad, lo[1] - pad, hi[1] + pad)
    th = int(half * (bounds[3] - bounds[2]) / (bounds[1] - bounds[0]))
    if th > 900:
        th = 900
    for k, (title, cols) in enumerate((("F  3-D map from above (display RGB)", display_rgb(arr["rgb"])),
                                       ("F  3-D map from above, ORACLE segmentation", oracle_colors(
                                           arr["instance_id_oracle"])))):
        bx = M + k * (half + 80)
        im, s = top_down(arr, cols, half, th, bounds)
        img.paste(im, (bx, y))
        frame_box(img, d, (bx, y, bx + half, y + th), title,
                  "" if k == 0 else "VISUALIZATION ONLY")
        hx, hz = (0 - bounds[0]) * s + bx, (0 - bounds[2]) * s + y
        S.crosshair(d, hx, hz, r=12)
        S.text(d, (hx + 14, hz + 6), "head", size=S.T_SMALL, bold=True)
    S.text(d, (M, y + th + 14), f"Head-frame X → right, Z ↓ toward the viewer's back (−Z = forward is up); points with "
                                f"Y > {CEILING_CUT_M} m above the head origin (ceiling) omitted; highest point per pixel.",
           size=S.T_SMALL, fill=S.MUTED, outline=None)
    img = img.crop((0, 0, W_, y + th + 60))
    out = vis / "overview.png"
    img.save(out, optimize=True)
    print(f"[gfe-visuals] wrote {out} {img.size}", flush=True)
    return {"overview": str(out)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="overview.png; --mark-fix annotates earlier stops on the coverage curve")
    ap.add_argument("--run", required=True)
    ap.add_argument("--vis", required=True)
    ap.add_argument("--mark-fix", type=int, nargs="*", default=[])
    ap.add_argument("--mark-name", nargs="*", default=None, help="one name per --mark-fix value")
    ap.add_argument("--mark-final", action="store_true", help="also mark the final fixation")
    ap.add_argument("--title", default=OVERVIEW_TITLE, help="figure title (default: the v0 title)")
    a = ap.parse_args()
    names = a.mark_name if a.mark_name is not None else (["v0 stopped here"] if len(a.mark_fix) == 1 else
                                                         [""] * len(a.mark_fix))
    if len(names) != len(a.mark_fix):
        raise SystemExit("--mark-name needs one name per --mark-fix value")
    marks = list(zip(a.mark_fix, names))
    if a.mark_final:
        n_final = len(json.loads((Path(a.run) / "trajectory.json").read_text())["trajectory"])
        marks.append((n_final, "final"))
    overview(Path(a.run).resolve(), Path(a.vis).resolve(), marks, a.title)
