"""Active Bootstrap-1b: the truth-free gaze-centered spherical epipolar geometry stage.

Contract: docs/active-bootstrap/ab1b-spherical-epipolar-geometry-contract.md, sections 6-10.  The input is exactly the
calibration and the frozen, truth-stripped correspondence product (core row / col, continuous uv_L / uv_R).  Rays come
from the accepted ``fsg_geometry.rays_h``; they are expressed in baseline-polar epipolar coordinates

    theta = atan2(sqrt(dy^2 + dz^2), dx)   (from +X, the physical baseline)
    phi   = atan2(dy, -dz)                 (the epipolar plane about X)

and triangulated directly from the angles, with an independent ray-ray cross-check.  No planar rectification, no image
resampling, no matcher: this module imports neither cv2 nor fsg_stereo, and never sees the scene truth.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent, HERE.parent / "natural_bootstrap"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ab1b_spec as SP  # noqa: E402
import fsg_geometry as FG  # noqa: E402  (accepted sensor geometry, read-only)
from nb1a_guard import OpenGuard  # noqa: E402  (accepted allowlist file-open guard, read-only)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, data) -> None:
    Path(path).write_text(json.dumps(data, indent=1, sort_keys=True, allow_nan=False) + "\n")


# ------------------------------------------------------------------ epipolar coordinates
def wrap_pi(a):
    """Canonical wrap to (-pi, pi]."""
    a = np.asarray(a, np.float64)
    return np.arctan2(np.sin(a), np.cos(a))


def theta_phi(d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baseline-polar angle from +X, epipolar-plane angle about X, and the pole-singularity flag."""
    d = np.asarray(d, np.float64)
    off_axis = np.hypot(d[..., 1], d[..., 2])
    theta = np.arctan2(off_axis, d[..., 0])
    singular = off_axis < SP.POLE_EPS
    with np.errstate(invalid="ignore"):
        phi = np.where(singular, np.nan, np.arctan2(d[..., 1], -d[..., 2]))
    return theta, phi, singular


def chart(theta, phi, theta_g: float, phi_g: float) -> tuple[np.ndarray, np.ndarray]:
    """Gaze-centered display chart (radians); display / diagnostic only."""
    return np.asarray(theta) - theta_g, np.sin(theta_g) * wrap_pi(np.asarray(phi) - phi_g)


def gaze_angles(c: dict) -> tuple[np.ndarray, float, float]:
    d_g = FG.gaze_direction(*c["gaze_yaw_pitch_deg"])
    t, p, s = theta_phi(d_g)
    if bool(s):
        raise ValueError("the cyclopean gaze lies on the baseline axis")
    return d_g, float(t), float(p)


def baseline(c: dict) -> tuple[float, np.ndarray, np.ndarray]:
    """B and the two eye origins, which must sit at -B/2 and +B/2 on head X (the +X baseline axis)."""
    b = float(c["ipd_m"])
    o_l, o_r = (np.asarray(e["centre_h_m"], np.float64) for e in c["eyes"])
    axis = np.asarray(SP.BASELINE_AXIS)
    if not (np.allclose(o_l, -b / 2 * axis, atol=1e-12) and np.allclose(o_r, b / 2 * axis, atol=1e-12)):
        raise ValueError("eye origins are not at -B/2, +B/2 on the +X baseline axis")
    return b, o_l, o_r


def triangulate_epipolar(theta_l, theta_r, phi_l, phi_r, b: float) -> dict:
    """Direct epipolar triangulation from the angles; purely numerical guard."""
    with np.errstate(divide="ignore", invalid="ignore"):
        cot_l = np.cos(theta_l) / np.sin(theta_l)
        cot_r = np.cos(theta_r) / np.sin(theta_r)
        den = cot_l - cot_r
        rho = b / den
        x = -b / 2 + rho * cot_l
        phi_bar = np.arctan2(np.sin(phi_l) + np.sin(phi_r), np.cos(phi_l) + np.cos(phi_r))
        p = np.stack([x, rho * np.sin(phi_bar), -rho * np.cos(phi_bar)], axis=-1)
    valid = (np.isfinite(theta_l) & np.isfinite(theta_r) & np.isfinite(phi_l) & np.isfinite(phi_r)
             & np.isfinite(den) & (np.abs(den) >= SP.DEN_EPS) & np.isfinite(p).all(axis=-1))
    p = np.where(valid[..., None], p, np.nan)
    return {"rho": np.where(valid, rho, np.nan), "phi_bar": phi_bar, "P_epi": p, "valid_epi": valid}


def triangulate_rays(o_l, d_l, o_r, d_r) -> dict:
    """Closest points of the two rays (least squares), their midpoint and gap."""
    w0 = o_l - o_r
    bb = np.sum(d_l * d_r, axis=-1)
    dd = d_l @ w0 if np.ndim(w0) == 1 else np.sum(d_l * w0, axis=-1)
    ee = d_r @ w0 if np.ndim(w0) == 1 else np.sum(d_r * w0, axis=-1)
    denom = 1.0 - bb * bb
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (bb * ee - dd) / denom
        t = (ee - bb * dd) / denom
    p1 = o_l + s[..., None] * d_l
    p2 = o_r + t[..., None] * d_r
    valid = (denom >= SP.PARALLEL_EPS) & np.isfinite(p1).all(axis=-1) & np.isfinite(p2).all(axis=-1)
    mid = np.where(valid[..., None], (p1 + p2) / 2, np.nan)
    gap = np.where(valid, np.linalg.norm(p1 - p2, axis=-1), np.nan)
    return {"s_L": np.where(valid, s, np.nan), "s_R": np.where(valid, t, np.nan), "P_ray": mid, "ray_gap": gap,
            "valid_ray": valid}


def conditioning(d_l, d_r, theta_l, delta_theta, b: float, f: float) -> dict:
    gamma = np.arccos(np.clip(np.sum(d_l * d_r, axis=-1), -1.0, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = 1.0 / np.abs(np.sin(gamma))
        per_px = b * np.sin(theta_l) / np.sin(delta_theta) ** 2 / f
    return {"gamma": gamma, "kappa": kappa, "range_per_px": per_px}


# ------------------------------------------------------------------ the input schema (truth-stripped product)
def validate_product(prod: dict) -> None:
    keys = set(prod)
    bad = sorted(k for k in keys if any(t in k.lower() for t in SP.FORBIDDEN_TOKENS))
    if bad:
        raise ValueError(f"correspondence product carries forbidden truth fields {bad}")
    if keys != set(SP.PRODUCT_KEYS):
        raise ValueError(f"correspondence product keys {sorted(keys)} != {list(SP.PRODUCT_KEYS)}")
    n = np.asarray(prod["left_core_row"]).shape
    if len(n) != 1:
        raise ValueError("left_core_row must be one-dimensional")
    for k in ("left_core_row", "left_core_col"):
        a = np.asarray(prod[k])
        if a.dtype.kind not in "iu" or a.shape != n or (a.size and (a.min() < 0 or a.max() >= SP.CORE_SIZE)):
            raise ValueError(f"bad {k}")
    for k in ("uv_L", "uv_R"):
        a = np.asarray(prod[k])
        if a.dtype != np.float64 or a.shape != n + (2,) or not np.isfinite(a).all():
            raise ValueError(f"bad {k}")
    want = np.stack([np.asarray(prod["left_core_col"]) + SP.CORE_ORIGIN, np.asarray(prod["left_core_row"]) + SP.CORE_ORIGIN],
                    axis=-1).astype(np.float64)
    if not np.array_equal(np.asarray(prod["uv_L"]), want):
        raise ValueError("uv_L is not the raw left pixel centre of (left_core_row, left_core_col)")


def load_product(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        prod = {k: np.asarray(z[k]) for k in z.files}
    validate_product(prod)
    return prod


# ------------------------------------------------------------------ computations
def left_core_rays(c: dict) -> dict:
    """All 65,536 left raw-core pixel rays (calibration only)."""
    rows, cols = np.mgrid[:SP.CORE_SIZE, :SP.CORE_SIZE]
    uv = np.stack([cols + SP.CORE_ORIGIN, rows + SP.CORE_ORIGIN], axis=-1).astype(np.float64)
    d = FG.rays_h(c["eyes"][0], uv)
    theta, phi, singular = theta_phi(d)
    _dg, tg, pg = gaze_angles(c)
    cu, cv = chart(theta, phi, tg, pg)
    return {"row": rows.astype(np.int32), "col": cols.astype(np.int32), "uv": uv, "direction_h": d, "theta": theta,
            "phi": phi, "singular": singular, "chart_u": cu, "chart_v": cv, "leverage": np.sin(theta)}


def pair_rays(c: dict, prod: dict) -> tuple[np.ndarray, np.ndarray]:
    """Head-frame unit rays of the matched left / right pixel coordinates (accepted fsg_geometry.rays_h)."""
    return (FG.rays_h(c["eyes"][0], np.asarray(prod["uv_L"], np.float64)),
            FG.rays_h(c["eyes"][1], np.asarray(prod["uv_R"], np.float64)))


def compute_epipolar(c: dict, prod: dict) -> dict:
    validate_product(prod)
    b, o_l, o_r = baseline(c)
    f = float(c["eyes"][0]["K"][0][0])
    d_l, d_r = pair_rays(c, prod)
    th_l, ph_l, sg_l = theta_phi(d_l)
    th_r, ph_r, sg_r = theta_phi(d_r)
    dtheta = th_r - th_l
    epi = triangulate_epipolar(th_l, th_r, ph_l, ph_r, b)
    ray = triangulate_rays(o_l, d_l, o_r, d_r)
    cond = conditioning(d_l, d_r, th_l, dtheta, b, f)
    _dg, tg, pg = gaze_angles(c)
    cul, cvl = chart(th_l, ph_l, tg, pg)
    cur, cvr = chart(th_r, ph_r, tg, pg)
    with np.errstate(invalid="ignore"):
        diff = np.linalg.norm(epi["P_epi"] - ray["P_ray"], axis=-1)
        range_l = np.linalg.norm(epi["P_epi"] - o_l, axis=-1)
    return {"left_core_row": np.asarray(prod["left_core_row"], np.int32),
            "left_core_col": np.asarray(prod["left_core_col"], np.int32),
            "d_L": d_l, "d_R": d_r, "theta_L": th_l, "theta_R": th_r, "phi_L": ph_l, "phi_R": ph_r,
            "delta_theta": dtheta, "phi_residual": wrap_pi(ph_r - ph_l), "singular_L": sg_l, "singular_R": sg_r,
            **epi, **ray, "epi_ray_diff": diff, **cond, "range_L": range_l,
            "chart_u_L": cul, "chart_v_L": cvl, "chart_u_R": cur, "chart_v_R": cvr}


def quantiles(a, mask=None) -> dict | None:
    a = np.asarray(a, np.float64)
    if mask is not None:
        a = a[np.asarray(mask, bool)]
    a = a[np.isfinite(a)].ravel()
    if a.size == 0:
        return None
    return {k: float(np.quantile(a, q)) for k, q in SP.QUANTILES.items()} | {"count": int(a.size)}


def summarize(c: dict, rays: dict, res: dict) -> dict:
    _dg, tg, pg = gaze_angles(c)
    th = rays["theta"]
    pole = np.minimum(th, np.pi - th)
    ve, vr = res["valid_epi"], res["valid_ray"]
    dth = res["delta_theta"]
    return {
        "theta_g_rad": tg, "phi_g_rad": pg, "theta_g_deg": float(np.degrees(tg)), "phi_g_deg": float(np.degrees(pg)),
        "left_core": {
            "rays": int(th.size), "finite": int((np.isfinite(rays["direction_h"]).all(axis=-1) & np.isfinite(th)
                                                 & np.isfinite(rays["phi"])).sum()),
            "pole_singular": int(rays["singular"].sum()),
            "theta_deg": {k: float(np.degrees(v)) for k, v in (("min", th.min()), ("median", np.median(th)),
                                                              ("max", th.max()))},
            "min_angle_to_baseline_pole_deg": float(np.degrees(pole.min())),
            "nearest_pole": "+X" if float(th.min()) <= float(np.pi - th.max()) else "-X",
            "phi_deg": {"min": float(np.degrees(np.nanmin(rays["phi"]))), "max": float(np.degrees(np.nanmax(rays["phi"])))},
            "chart_u_rad": {"min": float(rays["chart_u"].min()), "max": float(rays["chart_u"].max())},
            "chart_v_rad": {"min": float(np.nanmin(rays["chart_v"])), "max": float(np.nanmax(rays["chart_v"]))},
            "leverage_sin_theta": {"min": float(rays["leverage"].min()), "median": float(np.median(rays["leverage"])),
                                   "max": float(rays["leverage"].max())},
        },
        "counts": {"correspondences": int(ve.size), "triangulated_epipolar": int(ve.sum()),
                   "triangulated_ray_ray": int(vr.sum()), "singular_L": int(res["singular_L"].sum()),
                   "singular_R": int(res["singular_R"].sum()),
                   "delta_theta_positive": int((dth > 0).sum()), "delta_theta_zero": int((dth == 0).sum()),
                   "delta_theta_negative": int((dth < 0).sum())},
        "distributions": {
            "abs_phi_residual_rad": quantiles(np.abs(res["phi_residual"])),
            "delta_theta_rad": quantiles(dth),
            "ray_angle_gamma_rad": quantiles(res["gamma"]),
            "kappa": quantiles(res["kappa"]),
            "closest_ray_gap_m": quantiles(res["ray_gap"], vr),
            "epi_ray_difference_m": quantiles(res["epi_ray_diff"], ve & vr),
            "range_L_m": quantiles(res["range_L"], ve),
            "range_per_px_m": quantiles(res["range_per_px"]),
        },
    }


# ------------------------------------------------------------------ the file-level stage (guarded)
def truth_reads(events: list[dict]) -> dict:
    """Reads that the geometry must never make, counted from the guard events (allowed or refused)."""
    paths = [e["path"] for e in events if e.get("event") == "open"]
    ref = [p for p in paths if p.endswith("reference-observation.npz") or p.endswith(".exr") or "evaluation_only" in p]
    nat = [p for p in paths if "/measurement/" in p or p.endswith("stereo-result.npz")
           or (str(SP.AB1A_RUN) in p and "/evaluation/" in p)]
    return {"position_reads": len(ref), "object_index_reads": len(ref), "ab1a_natural_result_reads": len(nat)}


def run_geometry(calib_path: Path, product_path: Path, out_dir: Path) -> dict:
    """Read exactly the calibration and the frozen product; write the ray table, the result and the records."""
    if not (Path(product_path).parent / "correspondence-freeze.json").exists():
        raise RuntimeError("the correspondence product is not frozen")
    out_dir.mkdir(parents=True, exist_ok=True)
    g = OpenGuard("ab1b-geometry", [calib_path, product_path], [out_dir])
    try:
        with g:
            inputs = {"calibration": {"path": str(calib_path), "sha256": sha256(calib_path)},
                      "correspondences": {"path": str(product_path), "sha256": sha256(product_path)}}
            c = json.loads(Path(calib_path).read_text())
            FG.validate_calibration(c)
            prod = load_product(product_path)
            rays = left_core_rays(c)
            res = compute_epipolar(c, prod)
            np.savez_compressed(out_dir / "left-core-rays.npz", **rays)
            np.savez_compressed(out_dir / "epipolar-result.npz", **res)
            summary = {"schema": "AB1b-geometry-summary-v1", "truth": SP.TRUTH_DERIVED,
                       "statement": "TRUTH-FREE SPHERICAL EPIPOLAR RECONSTRUCTION: computed from the calibration and the "
                                    "frozen truth-stripped correspondence product only; no Position, Object Index or "
                                    "truth XYZ; no planar rectification, no SGBM",
                       "geometry": SP.GEOMETRY, "geometry_config_sha256": SP.config_sha256(SP.GEOMETRY),
                       "inputs": inputs, "baseline_m": float(c["ipd_m"]), "focal_px": float(c["eyes"][0]["K"][0][0]),
                       **summarize(c, rays, res)}
            write_json(out_dir / "geometry-summary.json", summary)
    finally:
        rec = g.record()
        rec["modules_loaded"] = {m: m in sys.modules for m in ("cv2", "fsg_stereo", "ab1a_stereo")}
        rec.update(truth_reads(rec["events"]))
        rec["truth_firewall_violations"] = len(rec["violations"])
        write_json(out_dir / "geometry-opened-files.json", rec)
    return summary
