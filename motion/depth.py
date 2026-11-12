"""Experiment M2: depth from the head's motion, monocular, on the foveated raster (numpy only).

The head moves by a known step between looks (the path), and the eye's poses are known (the run's
records), so two looks of the same point give its range by triangulation, as two eyes do in stereo,
with the path segment as the baseline. What limits the range is the correspondence: a point is seen
at a pixel, not at an exact direction. The question is how the foveated raster, whose pixels grow
with eccentricity, shares that precision across the field when the eye fixates a target or looks
along the path.

For each pair of looks (a, b = a + stride) and each surface sample of look b:

    correspondence  the oracle says where b's point P lies in look a (its projection, as M1.3), and
                    whether a sees it there (M1.3's agreement: occluded points have no match)
    pixel           the measurement: the ray through b's pixel centre and the ray through the
                    centre of the pixel of a that P projects to; triangulated (closest point of the
                    two rays), giving the range along b's ray. When the two rays are parallel there
                    is no range (error inf, not usable): the same pixel in both looks with no eye
                    rotation between them, as looking ahead
    same pixel      P lands on the same raster pixel in both looks: the flow is under half a pixel.
                    That is still a measurement. Without eye rotation it bounds the range from below
                    only (no range); under fixation the rays through one pixel cross on the surface of
                    zero flow through the fixated point, and the range is bounded on both sides
                    (Code's finding, issue #3). Reported as same_pixel_fraction
    bracket         what the measurement supports, whatever the error happens to be: the ranges along
                    b's ray whose projection into a stays inside the matched pixel (the epipolar curve
                    linearised at the estimate, the pixel a unit square in raster coordinates), as a
                    relative half-width. inf without a range. usable_bracket counts bracket <= USABLE_REL
    exact           the same with the exact directions to P: must give |P - c_b| (check M2.1)
    truth           |P - c_b|, P from b's Position pass

The precision model, to compare with: rel. error ~ range * sigma / (|T| sin psi), with psi the angle
between b's ray and the baseline T (psi = 0 at the focus of expansion) and sigma the rounding error of
look a's pixel, sqrt(omega_a / 12) (omega = pixel solid angle): b's sample is its own pixel's centre and
its Position, the pixel's mean, lies on that centre's ray to first order, so only a rounds (Code's
hypothesis, issue #3). Rounding uniform in a square pixel has variance p^2 / 12 along any direction,
and for uniform errors median |error| = 0.87 sigma: measured/model near 0.87 means rounding explains
the error; well under it, that the true points sit near the pixels' centres (as on a surface near the
zero-flow surface); over it, other error (depth edges, curvature, the linearisation).
(The model used sqrt((omega_a + omega_b) / 12) until issue #3's review: ratios from before are lower,
by 1.42-1.54 in the calibration room's runs, Chat sandbox.)

    M2.1  exact triangulation returns the true range: relative error <= 1e-6 wherever the two exact
          rays are at least 0.01 deg apart; control: look a's eye centre 1 cm along the path, which
          must give a median relative error >= 1e-3
"""
from __future__ import annotations

import numpy as np

from . import analysis

STRIDES = (1, 5, 10)
USABLE_REL = 0.10                 # a range within 10 % counts as usable depth (proposed, Chat, issue #3)
EXACT_REL_MAX = 1e-6
EXACT_MIN_PARALLAX_DEG = 0.01
CONTROL_SHIFT_M = 0.01
CONTROL_MIN_MEDIAN = 1e-3
ECC_BINS = analysis.ECC_BINS
PSI_BINS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 45.0, 90.0, 180.0)


def triangulate(ca, da, cb, db):
    """Ranges along db (from cb) of the closest points of the rays ca + s da and cb + t db, and the
    angle between the rays (deg). da, db unit (N, 3); ca, cb (3,)."""
    w0 = ca - cb
    b = np.sum(da * db, axis=-1)
    d = da @ w0
    e = db @ w0
    den = 1.0 - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (e - b * d) / den
    return t, np.degrees(np.arccos(np.clip(b, -1.0, 1.0)))


def _bracket(model, Ra, ca, cb, db, t, eps: float = 1e-4):
    """Relative half-width of the ranges along cb + r db whose projection into look a stays in the pixel
    the estimate t projects to: the epipolar curve linearised at t, the pixel a unit square centred
    on the curve. inf where t is not a range."""
    out = np.full(t.shape, np.inf)
    ok = np.isfinite(t) & (t > 0)
    if not ok.any():
        return out
    tt = t[ok][:, None]
    rp, _, _, _ = analysis._project(model, Ra, ca, cb + tt * (1 + eps) * db[ok])
    rm, _, _, _ = analysis._project(model, Ra, ca, cb + tt * (1 - eps) * db[ok])
    g = (rp - rm) / (2 * eps * tt)                       # raster px per metre along b's ray
    gn = np.linalg.norm(g, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = g / gn[:, None]
        h = 0.5 / np.max(np.abs(u), axis=-1)             # half the pixel's extent along the curve
        out[ok] = np.where(gn > 0, h / gn / tt[:, 0], np.inf)
    return out


def _rel(t, truth):
    """|t - truth| / truth; inf where the rays give no range (parallel: the two pixels see the same
    direction, or the closest point is behind b)."""
    r = np.abs(t - truth) / truth
    return np.where(np.isfinite(t) & (t > 0), r, np.inf)


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def pair_depth(la, lb, *, shift_a=None) -> dict:
    """Per hit sample of look b: truth, pixel and exact ranges, eccentricity, psi, predicted error.
    la, lb: analysis.load_look tuples. shift_a: a wrong offset of a's eye centre (the control)."""
    rec_a, Ra, ca, model = la
    rec_b, Rb, cb, model_b = lb
    hit = np.flatnonzero(np.isfinite(rec_b.distance_m))
    P = rec_b.position_w[hit]
    truth = np.linalg.norm(P - cb, axis=-1)
    agree, considered, rc_a, _ = analysis._agree(rec_a, model, Ra, ca, P)
    n = model.n
    ij = np.rint(rc_a).astype(np.int64)
    inv = np.full(n * n, -1, dtype=np.int64)
    inv[rec_a.raster_index] = np.arange(rec_a.n_samples)
    s = np.full(len(P), -1, dtype=np.int64)
    s[agree] = inv[ij[agree, 0] * n + ij[agree, 1]]
    m = agree & (s >= 0)
    ca_used = ca if shift_a is None else ca + shift_a
    db = _unit(rec_b.direction[hit[m]].astype(np.float64))
    da = _unit(rec_a.direction[s[m]].astype(np.float64))
    t_pix, par_pix = triangulate(ca_used, da, cb, db)
    # the exact rays are those through P from the true centres; the control triangulates them from a wrong one
    t_ex, par_ex = triangulate(ca_used, _unit(P[m] - ca), cb, _unit(P[m] - cb))
    T = cb - ca
    psi = np.degrees(np.arctan2(np.linalg.norm(np.cross(db, T), axis=-1), db @ T))
    sigma = np.sqrt(rec_a.footprint_sr[s[m]] / 12.0)
    same = (model_b.n == n) & (s[m] >= 0)
    same = same & (rec_a.raster_index[s[m]] == rec_b.raster_index[hit[m]])
    bracket = _bracket(model, Ra, ca_used, cb, db, t_pix)
    with np.errstate(divide="ignore"):
        pred = truth[m] * sigma / (np.linalg.norm(T) * np.sin(np.radians(psi)))
    ecc = analysis.eccentricity_deg(rec_b.direction[hit] @ Rb)
    return {"n_hits": len(hit), "ecc_all": ecc, "matched": m, "truth": truth[m], "ecc": ecc[m], "psi": psi,
            "rel_pix": _rel(t_pix, truth[m]), "rel_exact": _rel(t_ex, truth[m]),
            "parallax_exact_deg": par_ex, "parallax_pix_deg": par_pix, "pred": pred, "same": same,
            "bracket": bracket,
            "raster_b": rec_b.raster_index[hit[m]], "raster_shape": rec_b.raster_shape}


def _q(r, q):
    """A quantile of relative errors that may hold inf (no range): the nearest order statistic, so an
    inf stays inf instead of turning into nan."""
    return float(np.quantile(r, q, method="nearest")) if r.size else None


def _median_ratio(rel, pred):
    ok = np.isfinite(rel) & np.isfinite(pred) & (pred > 0)
    return float(np.median(rel[ok] / pred[ok])) if ok.any() else None


def _bins(x, edges, rows_of):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        out.append(dict({"bin": [lo, hi]}, **rows_of(sel)))
    return out


def _stats(rel, pred, bracket, same):
    def f(sel, den_sel=None):
        r = rel[sel]
        den = int(den_sel.sum()) if den_sel is not None else int(sel.sum())
        return {"n": int(sel.sum()),
                "median_rel": _q(r, 0.5), "p90_rel": _q(r, 0.9),
                "median_pred": float(np.median(pred[sel])) if r.size else None,
                "median_bracket": _q(bracket[sel], 0.5),
                "same_pixel_fraction": float(same[sel].mean()) if r.size else None,
                "usable": int((r <= USABLE_REL).sum()), "usable_bracket": int((bracket[sel] <= USABLE_REL).sum()),
                "of": den,
                "usable_fraction": float((r <= USABLE_REL).sum() / den) if den else None,
                "usable_bracket_fraction": float((bracket[sel] <= USABLE_REL).sum() / den) if den else None}
    return f


def run_depth(run: dict, strides=STRIDES, example_stride: int = 5):
    """Every pair (k, k + stride) of every mode. Returns (summary dict, example rasters, check M2.1)."""
    summary, examples = {}, {}
    ex_rel, ex_ctrl = [], []
    loads = {}
    for mode, entries in run["looks"].items():
        es = sorted(entries, key=lambda e: e["k"])
        loads = {e["k"]: analysis.load_look(run, e) for e in es}
        ks = sorted(loads)
        side = None
        for stride in strides:
            acc = {k: [] for k in ("ecc", "psi", "rel", "pred", "ecc_all", "bracket", "same")}
            mid = ks[len(ks) // 2 - stride // 2] if len(ks) > stride else None
            for ka in ks:
                kb = ka + stride
                if kb not in loads:
                    continue
                r = pair_depth(loads[ka], loads[kb])
                for k in ("ecc", "psi", "pred", "ecc_all", "bracket", "same"):
                    acc[k].append(r[k])
                acc["rel"].append(r["rel_pix"])
                if stride == strides[0]:
                    ok_par = r["parallax_exact_deg"] >= EXACT_MIN_PARALLAX_DEG
                    ex_rel.append(r["rel_exact"][ok_par])
                    if side is None:
                        T = loads[kb][2] - loads[ka][2]
                        side = CONTROL_SHIFT_M * T / np.linalg.norm(T)
                    if ka % 5 == 0:          # the control on every fifth pair is enough to show it fails
                        rc = pair_depth(loads[ka], loads[kb], shift_a=side)
                        ex_ctrl.append(rc["rel_exact"][rc["parallax_exact_deg"] >= EXACT_MIN_PARALLAX_DEG])
                if ka == mid and stride == example_stride:
                    img = np.full(int(np.prod(r["raster_shape"])), np.nan, dtype=np.float32)
                    img[r["raster_b"]] = r["rel_pix"]
                    examples[f"{mode}__s{stride}__k{ka}"] = img.reshape(r["raster_shape"])
            if not acc["rel"]:
                continue
            ecc, psi, rel, pred = (np.concatenate(acc[k]) for k in ("ecc", "psi", "rel", "pred"))
            ecc_all = np.concatenate(acc["ecc_all"])
            bracket, same = np.concatenate(acc["bracket"]), np.concatenate(acc["same"])
            f = _stats(rel, pred, bracket, same)
            by_ecc = []
            for lo, hi in zip(ECC_BINS[:-1], ECC_BINS[1:]):
                sel = (ecc >= lo) & (ecc < hi)
                row = f(sel)
                row["bin"] = [lo, hi]
                n_all = int(((ecc_all >= lo) & (ecc_all < hi)).sum())
                row["of"] = n_all                       # usable among ALL hits of b at that eccentricity
                row["usable_fraction"] = float(row["usable"] / n_all) if n_all else None
                row["usable_bracket_fraction"] = float(row["usable_bracket"] / n_all) if n_all else None
                by_ecc.append(row)
            by_psi = _bins(psi, PSI_BINS, f)
            fov = ecc < 2.0
            summary.setdefault(mode, {})[f"stride_{stride}"] = {
                "stride": stride, "pairs": len(acc["rel"]), "matched": int(rel.size), "hits": int(ecc_all.size),
                "median_rel": _q(rel, 0.5), "usable_fraction": float((rel <= USABLE_REL).sum() / ecc_all.size),
                "usable_bracket_fraction": float((bracket <= USABLE_REL).sum() / ecc_all.size),
                "median_bracket": _q(bracket, 0.5), "same_pixel_fraction": float(same.mean()),
                "fovea_2deg": {"n": int(fov.sum()), "median_rel": _q(rel[fov], 0.5),
                               "usable_fraction": float((rel[fov] <= USABLE_REL).sum() / max(int((ecc_all < 2.0).sum()), 1)),
                               "median_bracket": _q(bracket[fov], 0.5),
                               "usable_bracket_fraction": float((bracket[fov] <= USABLE_REL).sum()
                                                                / max(int((ecc_all < 2.0).sum()), 1)),
                               "same_pixel_fraction": float(same[fov].mean()) if fov.any() else None},
                "no_range_fraction": float(np.isinf(rel).sum() / max(rel.size, 1)),
                "median_measured_over_pred": _median_ratio(rel, pred),
                "by_eccentricity": by_ecc, "by_psi": by_psi}
    exact = np.concatenate(ex_rel) if ex_rel else np.zeros(0)
    ctrl = np.concatenate(ex_ctrl) if ex_ctrl else np.zeros(0)
    check = {"id": "M2.1", "what": "exact triangulation along the path returns the true range",
             "ok": bool(exact.size) and float(exact.max()) <= EXACT_REL_MAX,
             "control_fails": bool(ctrl.size) and float(np.median(ctrl)) >= CONTROL_MIN_MEDIAN,
             "control": f"look a's eye centre {CONTROL_SHIFT_M} m along the path: median relative error >= {CONTROL_MIN_MEDIAN}",
             "detail": {"samples": int(exact.size), "max_rel": float(exact.max()) if exact.size else None,
                        "control_samples": int(ctrl.size),
                        "control_median_rel": float(np.median(ctrl)) if ctrl.size else None}}
    return summary, examples, check
