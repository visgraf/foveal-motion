"""Figures for a dolly run, from its files (DERIVED: display renders, not measurements).

    <vis>/contact_sheet.png   the foveated rasters along the path, one row per gaze mode
    <vis>/flow.png            retinal flow per step against eccentricity, both modes; the flow field
                              of the middle step on the raster
    <vis>/dolly.mp4           the two modes side by side, one frame per look (needs ffmpeg)
    <vis>/depth.png           M2: range from motion, error and usable fraction by eccentricity and by angle
                              from the heading, against the precision model; one error map per mode

The rasters are the sensor's own log-polar images (the fovea at the centre fills most of the
disc), linear RGB shown with one exposure per mode (the 99th percentile of its first look) and
the sRGB curve.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from fov_render import record as R

from . import analysis


def _font(size: int):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def raster_image(run: dict, entry: dict, exposure: float) -> Image.Image:
    rec = R.SampleRecord.load(str(analysis.look_dir(run, entry) / "record_C.npz"))
    v = rec.as_raster("value", fill=0.0)
    return Image.fromarray((_srgb(v * exposure) * 255 + 0.5).astype(np.uint8))


def exposure_of(run: dict, entry: dict) -> float:
    rec = R.SampleRecord.load(str(analysis.look_dir(run, entry) / "record_C.npz"))
    lum = rec.value @ np.array([0.2126, 0.7152, 0.0722])
    return float(0.9 / max(np.quantile(lum, 0.99), 1e-6))


def _label(im: Image.Image, text: str, size: int = 14) -> Image.Image:
    out = Image.new("RGB", (im.width, im.height + size + 8), (255, 255, 255))
    out.paste(im, (0, size + 8))
    ImageDraw.Draw(out).text((4, 2), text, fill=(20, 20, 20), font=_font(size))
    return out


def contact_sheet(run: dict, vis: Path, cols: int = 8, cell: int = 220) -> Path:
    modes = list(run["looks"])
    n = len(run["looks"][modes[0]])
    ks = sorted(set(np.linspace(0, n - 1, min(cols, n)).round().astype(int).tolist()))
    target = np.asarray(run["path"]["target_w"])
    rows = []
    for m in modes:
        es = {e["k"]: e for e in run["looks"][m]}
        ex = exposure_of(run, es[0])
        tiles = []
        for k in ks:
            o = np.asarray(es[k]["head"]["head_origin_w_m"])
            im = raster_image(run, es[k], ex).resize((cell, cell), Image.NEAREST)
            tiles.append(_label(im, f"{m}  k={k}  {np.linalg.norm(target - o):.2f} m to target"))
        row = Image.new("RGB", (sum(t.width for t in tiles) + 4 * (len(tiles) - 1), tiles[0].height), "white")
        x = 0
        for t in tiles:
            row.paste(t, (x, 0))
            x += t.width + 4
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows) + 8 * len(rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 8
    p = vis / "contact_sheet.png"
    sheet.save(p)
    return p


def flow_figure(run: dict, vis: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rd = Path(run["_dir"])
    z = np.load(rd / "flow.npz")
    modes = sorted({k.split("__")[0] for k in z.files}, key=lambda m: list(run["looks"]).index(m))
    chk = json.loads((rd / "checks.json").read_text())
    summ = chk["flow_by_eccentricity"]
    step = chk["step_m"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    colors = {"target": "#1f6fb4", "ahead": "#d0602a"}
    for m in modes:
        mid = [0.5 * (r["ecc_deg"][0] + r["ecc_deg"][1]) for r in summ[m] if r["n"]]
        ax[0, 0].plot(mid, [r["median_flow_deg"] for r in summ[m] if r["n"]], "o-", color=colors.get(m), label=m)
        ax[0, 0].fill_between(mid, [0] * len(mid), [r["p90_flow_deg"] for r in summ[m] if r["n"]],
                              color=colors.get(m), alpha=0.12)
        ax[0, 1].plot(mid, [r["median_flow_px"] for r in summ[m] if r["n"]], "o-", color=colors.get(m), label=m)
    ax[0, 0].set(xlabel="eccentricity in look k+1 (deg)", ylabel="retinal flow per step (deg), median; band to p90",
                 title=f"angular flow, {step * 100:.1f} cm steps")
    ax[0, 1].set(xlabel="eccentricity in look k+1 (deg)", ylabel="flow on the foveated raster (px), median",
                 title="flow in sensor pixels")
    for a_ in ax[0]:
        a_.grid(alpha=0.3)
        a_.legend()
    for j, m in enumerate(modes[:2]):
        pair = z[f"{m}__pair"]
        if pair.size == 0:
            continue
        mid_pair = int(np.median(np.unique(pair)))
        sel = np.flatnonzero(pair == mid_pair)
        sel = sel[:: max(1, sel.size // 900)]
        rc_b, fl = z[f"{m}__rc_b"][sel], z[f"{m}__flow_px"][sel]
        a_ = ax[1, j]
        a_.quiver(rc_b[:, 1], rc_b[:, 0], fl[:, 1], -fl[:, 0], np.linalg.norm(fl, axis=-1), angles="xy",
                  scale_units="xy", scale=0.25, width=0.002, cmap="viridis")
        n = run["looks"][m][0]
        rec = R.SampleRecord.load(str(analysis.look_dir(run, n) / "record_C.npz"))
        a_.set(xlim=(0, rec.raster_shape[1]), ylim=(rec.raster_shape[0], 0), aspect="equal",
               title=f"{m}: flow field on the raster, step {mid_pair}->{mid_pair + 1} (arrows x4)")
    fig.suptitle(f"M1 dolly: {run['path']['name']}, {run['profile']} profile, {run.get('device')} (DERIVED from flow.npz)")
    fig.tight_layout()
    p = vis / "flow.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def depth_figure(run: dict, vis: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    rd = Path(run["_dir"])
    dj = json.loads((rd / "depth.json").read_text())
    ex = np.load(rd / "depth_examples.npz")
    modes = list(dj["summary"])
    colors = {"target": "#1f6fb4", "ahead": "#d0602a"}
    styles = ["-", "--", ":", "-."]
    fig = plt.figure(figsize=(15, 10))
    a1, a2, a3 = (fig.add_subplot(2, 3, i) for i in (1, 2, 3))
    for m in modes:
        for j, (key, s) in enumerate(dj["summary"][m].items()):
            lab = f"{m}, {s['stride'] * dj['step_m']:.2f} m"
            e = [r for r in s["by_eccentricity"] if r["median_rel"] is not None]
            mid = [0.5 * sum(r["bin"]) for r in e]
            a1.plot(mid, [r["median_rel"] for r in e], styles[j % 4], marker="o", ms=3, color=colors.get(m), label=lab)
            a2.plot(mid, [r["usable_fraction"] for r in e], styles[j % 4], marker="o", ms=3, color=colors.get(m), label=lab)
            if all("usable_bracket_fraction" in r for r in e):
                a2.plot(mid, [r["usable_bracket_fraction"] for r in e], styles[j % 4], color=colors.get(m), alpha=0.35,
                        lw=2.5)
            q = [r for r in s["by_psi"] if r["median_rel"] is not None]
            mq = [0.5 * sum(r["bin"]) for r in q]
            a3.plot(mq, [r["median_rel"] for r in q], styles[j % 4], marker="o", ms=3, color=colors.get(m), label=lab)
            a3.plot(mq, [r["median_pred"] for r in q], styles[j % 4], color="0.6", lw=0.8)
    a1.set(yscale="log", xlabel="eccentricity in look b (deg)", ylabel="median relative range error",
           title="range from motion, pixel-limited")
    a2.set(xlabel="eccentricity in look b (deg)", ylabel=f"usable fraction of all hits (<= {dj['usable_rel']:.0%})",
           title="usable: by error (thin), by bracket (wide, pale)", ylim=(0, 1))
    a3.set(xscale="log", yscale="log", xlabel="angle from the heading, psi (deg)", ylabel="median relative range error",
           title="against the precision model (grey)")
    for a_ in (a1, a2, a3):
        a_.grid(alpha=0.3, which="both")
    a1.legend(fontsize=8)
    keys = list(ex.files)
    for i, k in enumerate(keys[:3]):
        a_ = fig.add_subplot(2, 3, 4 + i)
        img = np.clip(ex[k], 1e-4, 10)
        im = a_.imshow(img, norm=LogNorm(1e-3, 1.0), cmap="magma_r", interpolation="nearest")
        mode, s, kk = k.split("__")
        a_.set(title=f"{mode}: relative error, looks {kk[1:]} -> +{s[1:]} on look b's raster", xticks=[], yticks=[])
        fig.colorbar(im, ax=a_, fraction=0.046)
    fig.suptitle(f"M2 range from the head's motion: {run['path']['name']}, {run['profile']}, {run.get('device')} "
                 f"(DERIVED from depth.json); M2.1 {'PASS' if dj['check']['ok'] else 'FAIL'}")
    fig.tight_layout()
    p = vis / "depth.png"
    fig.savefig(p, dpi=105)
    plt.close(fig)
    return p


def video(run: dict, vis: Path, fps: int = 6, cell: int = 506) -> Path | None:
    ff = shutil.which("ffmpeg")
    if ff is None:
        return None
    modes = list(run["looks"])
    frames = vis / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    ex = {m: exposure_of(run, run["looks"][m][0]) for m in modes}
    target = np.asarray(run["path"]["target_w"])
    n = len(run["looks"][modes[0]])
    for k in range(n):
        tiles = []
        for m in modes:
            e = run["looks"][m][k]
            o = np.asarray(e["head"]["head_origin_w_m"])
            im = raster_image(run, e, ex[m]).resize((cell, cell), Image.NEAREST)
            tiles.append(_label(im, f"{m}   k={k}   {np.linalg.norm(target - o):.2f} m to target", 18))
        f = Image.new("RGB", (sum(t.width for t in tiles) + 8 * (len(tiles) - 1), tiles[0].height), "white")
        x = 0
        for t in tiles:
            f.paste(t, (x, 0))
            x += t.width + 8
        if f.width % 2 or f.height % 2:
            f = f.crop((0, 0, f.width - f.width % 2, f.height - f.height % 2))
        f.save(frames / f"f{k:04d}.png")
    out = vis / "dolly.mp4"
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "f%04d.png"),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def main(run_dir: Path, vis: Path, fps: int = 6) -> int:
    run = analysis.load_run(run_dir)
    vis.mkdir(parents=True, exist_ok=True)
    made = [contact_sheet(run, vis)]
    if (Path(run["_dir"]) / "flow.npz").is_file():
        made.append(flow_figure(run, vis))
    else:
        print("[dolly] no flow.npz: run `checks` first for the flow figure")
    if (Path(run["_dir"]) / "depth.json").is_file():
        made.append(depth_figure(run, vis))
    v = video(run, vis, fps)
    made.append(v if v else "no ffmpeg: video skipped")
    for m in made:
        print(f"[dolly] wrote {m}")
    return 0
