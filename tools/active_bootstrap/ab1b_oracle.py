"""Active Bootstrap-1b: the perfect / oracle correspondence stage.

Contract: docs/active-bootstrap/ab1b-spherical-epipolar-geometry-contract.md, section 5.  This stage alone may use
Position and Object Index.  For every left raw-core pixel centre it projects the left Position truth through the right raw
camera and keeps the pair when the right pixel nearest to the projection sees the same instance (the accepted
Classroom-Oracle-1 visibility semantics, on the raw rasters).  It writes a truth-stripped product: core row / col and the
continuous left / right pixel coordinates only.  No SGBM, no RGB, no depth interval, no rectification: this module
imports neither cv2 nor fsg_stereo.
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


def core_grid() -> tuple[np.ndarray, np.ndarray]:
    """Core row / col of every left raw-core pixel, row-major (65,536 each)."""
    r, c = np.mgrid[:SP.CORE_SIZE, :SP.CORE_SIZE]
    return r.ravel().astype(np.int32), c.ravel().astype(np.int32)


def load_reference(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(SP.REFERENCE_KEYS):
            raise ValueError(f"reference observation arrays {sorted(z.files)}")
        return {k: np.asarray(z[k]) for k in SP.REFERENCE_KEYS}


# ------------------------------------------------------------------ the rule, one step per function
def left_hit(pos_w: np.ndarray) -> np.ndarray:
    """A finite geometric hit: every component finite and not all zero (no hit is (0, 0, 0))."""
    p = np.asarray(pos_w, np.float64)
    return np.isfinite(p).all(axis=-1) & np.any(p != 0, axis=-1)


def left_catalog(ids: np.ndarray) -> np.ndarray:
    return np.asarray(ids) > 0


def right_projection(c: dict, p_h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return FG.project_h(c["eyes"][1], p_h)


def right_projectable(uv_r: np.ndarray, z_r: np.ndarray) -> np.ndarray:
    return np.isfinite(uv_r).all(axis=-1) & np.isfinite(z_r) & (z_r > SP.RIGHT_Z_MIN)


def inside_raster(c: dict, uv_r: np.ndarray) -> np.ndarray:
    w, h = c["image_size_wh"]
    u, v = uv_r[..., 0], uv_r[..., 1]
    with np.errstate(invalid="ignore"):
        return (u >= 0.0) & (u <= w - 1.0) & (v >= 0.0) & (v <= h - 1.0)


def nearest_pixel(c: dict, uv_r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, h = c["image_size_wh"]
    u = np.where(np.isfinite(uv_r[..., 0]), np.clip(uv_r[..., 0], 0, w - 1), 0.0)
    v = np.where(np.isfinite(uv_r[..., 1]), np.clip(uv_r[..., 1], 0, h - 1), 0.0)
    return np.rint(v).astype(np.int64), np.rint(u).astype(np.int64)


def same_instance(ids_r: np.ndarray, vi: np.ndarray, ui: np.ndarray, left_ids: np.ndarray) -> np.ndarray:
    return np.asarray(ids_r)[vi, ui] == left_ids


def compute_oracle(c: dict, ref: dict) -> tuple[dict, dict]:
    """The truth-stripped product and the descriptive attrition, for the left raw nominal core."""
    FG.validate_calibration(c)
    w, h = c["image_size_wh"]
    if (w, h) != (SP.RAW_SIZE, SP.RAW_SIZE):
        raise ValueError(f"raw raster {w} x {h}")
    for k in SP.REFERENCE_KEYS:
        want = (h, w, 3) if k.startswith("position") else (h, w)
        if np.asarray(ref[k]).shape != want:
            raise ValueError(f"reference {k} shape {np.asarray(ref[k]).shape}")
    rows, cols = core_grid()
    v_raw, u_raw = rows + SP.CORE_ORIGIN, cols + SP.CORE_ORIGIN
    pos = np.asarray(ref["position_w_L"])[v_raw, u_raw].astype(np.float64)
    ids_l = np.asarray(ref["instance_L"])[v_raw, u_raw].astype(np.int64)
    hit = left_hit(pos)
    cat = hit & left_catalog(ids_l)
    p_h = FG.world_to_head(c, np.where(hit[:, None], pos, 0.0))
    uv_r, z_r = right_projection(c, p_h)
    proj = cat & right_projectable(uv_r, z_r)
    inside = proj & inside_raster(c, uv_r)
    vi, ui = nearest_pixel(c, uv_r)
    visible = inside & same_instance(ref["instance_R"], vi, ui, ids_l)
    keep = visible
    core = SP.CORE_ORIGIN, SP.CORE_ORIGIN + SP.CORE_SIZE - 1
    uv_keep = uv_r[keep]
    in_right_core = ((uv_keep[:, 0] >= core[0]) & (uv_keep[:, 0] <= core[1]) & (uv_keep[:, 1] >= core[0])
                     & (uv_keep[:, 1] <= core[1]))
    product = {
        "left_core_row": rows[keep].astype(np.int32),
        "left_core_col": cols[keep].astype(np.int32),
        "uv_L": np.stack([u_raw[keep], v_raw[keep]], axis=-1).astype(np.float64),
        "uv_R": uv_keep.astype(np.float64),
    }
    n = int(rows.size)
    attrition = [
        {"step": "left core total", "remaining": n},
        {"step": "finite left geometric hit", "remaining": int(hit.sum())},
        {"step": "positive left instance id", "remaining": int(cat.sum())},
        {"step": "right projectable (finite, z_R > 1e-9)", "remaining": int(proj.sum())},
        {"step": "inside padded right raster", "remaining": int(inside.sum())},
        {"step": "same-instance binocular-visible", "remaining": int(visible.sum())},
        {"step": "final perfect correspondences", "remaining": int(keep.sum())},
    ]
    summary = {
        "left_core_pixels": n, "correspondences": int(keep.sum()), "fraction_of_core": float(keep.sum() / n),
        "attrition": attrition,
        "excluded": {"no_finite_hit": int((~hit).sum()), "hit_with_instance_0": int((hit & ~cat).sum()),
                     "not_right_projectable": int((cat & ~proj).sum()), "outside_padded_right_raster": int((proj & ~inside).sum()),
                     "different_right_instance": int((inside & ~visible).sum())},
        "right_margin": {"inside_right_nominal_core": int(in_right_core.sum()),
                         "outside_right_nominal_core": int((~in_right_core).sum()),
                         "u_R_min": float(uv_keep[:, 0].min()) if keep.any() else None,
                         "u_R_max": float(uv_keep[:, 0].max()) if keep.any() else None,
                         "v_R_min": float(uv_keep[:, 1].min()) if keep.any() else None,
                         "v_R_max": float(uv_keep[:, 1].max()) if keep.any() else None},
    }
    return product, summary


# ------------------------------------------------------------------ the file-level stage (guarded)
def run_oracle(calib_path: Path, ref_path: Path, out_dir: Path) -> dict:
    """Read exactly the calibration and the reference observation; write the product and its records."""
    out_dir.mkdir(parents=True, exist_ok=True)
    g = OpenGuard("ab1b-oracle", [calib_path, ref_path], [out_dir])
    try:
        with g:
            inputs = {"calibration": {"path": str(calib_path), "sha256": sha256(calib_path)},
                      "reference_observation": {"path": str(ref_path), "sha256": sha256(ref_path)}}
            c = json.loads(Path(calib_path).read_text())
            ref = load_reference(ref_path)
            product, summ = compute_oracle(c, ref)
            np.savez_compressed(out_dir / "oracle-correspondences.npz", **product)
            summary = {"schema": "AB1b-oracle-summary-v1", "truth": SP.TRUTH_DERIVED,
                       "statement": "PERFECT / ORACLE CORRESPONDENCE: Position and Object Index were used here, and "
                                    "only here, to choose and project the matches; the product holds core row / col and "
                                    "continuous uv_L / uv_R only",
                       "oracle": SP.ORACLE, "oracle_config_sha256": SP.config_sha256(SP.ORACLE), "inputs": inputs,
                       "product_keys": sorted(product), **summ}
            write_json(out_dir / "oracle-summary.json", summary)
    finally:
        rec = g.record()
        rec["modules_loaded"] = {m: m in sys.modules for m in ("cv2", "fsg_stereo", "ab1a_stereo")}
        rec["truth_firewall_violations"] = len(rec["violations"])
        write_json(out_dir / "oracle-opened-files.json", rec)
    return summary
