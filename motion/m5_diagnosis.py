"""M5 Stage 0 (#14): what M4's depth-edge test rejects, and M4's flag as an edge detector.

M4's test (`multibaseline.match_reference`, frozen): for each candidate anchor, every other candidate in its 7 x 7
native neighbourhood is predicted as zeta_i + zeta_q,i * dq (the anchor's radial slope only), and the anchor is
rejected when one misses by more than `edge_tol_px` (0.5 px of the longest step). This module applies the same
neighbourhood to the TRUE inverse depth and classifies each rejection:

    1  discontinuity    the neighbourhood's true range jumps by more than J between adjacent pixels
    2a azimuthal slope  no jump; the true depths alone trip the test with the true radial slope, but a plane in
                        (q, phi) (radial and azimuthal slope) fits them within the tolerance
    2b curvature        no jump; even that plane misses by more than the tolerance
    3  estimation       no jump; the true depths would pass the test: the estimates tripped it

Also: the error each rejected anchor would have had if accepted, and M4's flag as a detector of class-1 edges
(precision, recall, by eccentricity band). Development data only (M-D17): calibration frames and fixtures.
"""
from __future__ import annotations

import numpy as np

from . import multibaseline as mb

BANDS_DEG = ((1, 5), (5, 15), (15, 30), (30, 44))
CLASSES = ("discontinuity", "azimuthal_slope", "curvature", "estimation")


def _shift(R, di, dj):
    n = R.shape[0]
    out = np.full_like(R, np.nan)
    out[max(-di, 0):n + min(-di, 0), max(-dj, 0):n + min(-dj, 0)] = R[max(di, 0):n + min(di, 0), max(dj, 0):n + min(dj, 0)]
    return out


def jump_raster(range_flat, n, hh):
    """Per pixel: the largest relative range jump between 4-adjacent pixels inside its (2 hh + 1)^2 block."""
    Rg = np.asarray(range_flat, float).reshape(n, n)
    J = np.zeros((n, n))
    with np.errstate(invalid="ignore", divide="ignore"):
        for di, dj in ((0, 1), (1, 0)):
            S = _shift(Rg, di, dj)
            j = np.abs(S - Rg) / np.fmin(S, Rg)
            j = np.where(np.isfinite(j), j, 0.0)
            J = np.fmax(J, j)
            J = np.fmax(J, _shift(j, -di, -dj))          # the pair belongs to both pixels
    J = np.where(np.isfinite(J), J, 0.0)
    B = J.copy()
    for di in range(-hh, hh + 1):
        for dj in range(-hh, hh + 1):
            if di or dj:
                S = _shift(J, di, dj)
                B = np.fmax(B, np.where(np.isfinite(S), S, 0.0))
    return B


def true_devs(z, zeta_true, n, hh, consider=None):
    """For every anchor: the test's statistic on the true inverse depth, (a) with the true radial slope only, as
    M4's model, (b) with a radial + azimuthal plane; both in px of the longest step. The slopes are least-squares
    fits over the neighbourhood (through the anchor's own true value). consider: anchors taken as neighbours
    (default all with a finite true value)."""
    flat = z["flat"]
    N = flat.size
    def ras(v, cond=None):
        R = np.full(n * n, np.nan)
        m = np.isfinite(v) if cond is None else (cond & np.isfinite(v))
        R[flat[m]] = v[m]
        return R.reshape(n, n)
    Zt = ras(zeta_true, consider)
    Q, P = ras(z["q"]), ras(z["phi"])
    offs = [(di, dj) for di in range(-hh, hh + 1) for dj in range(-hh, hh + 1) if di or dj]
    dZ = np.stack([_shift(Zt, di, dj) - Zt for di, dj in offs])
    dQ = np.stack([_shift(Q, di, dj) - Q for di, dj in offs])
    dP = np.stack([np.angle(np.exp(1j * (_shift(P, di, dj) - P))) for di, dj in offs])
    ok = np.isfinite(dZ) & np.isfinite(dQ) & np.isfinite(dP)
    dZ0, dQ0, dP0 = np.where(ok, dZ, 0), np.where(ok, dQ, 0), np.where(ok, dP, 0)
    # radial only: b = sum dZ dQ / sum dQ^2
    with np.errstate(invalid="ignore", divide="ignore"):
        b1 = (dZ0 * dQ0).sum(0) / (dQ0 ** 2).sum(0)
        a11, a12, a22 = (dQ0 ** 2).sum(0), (dQ0 * dP0).sum(0), (dP0 ** 2).sum(0)
        r1, r2 = (dZ0 * dQ0).sum(0), (dZ0 * dP0).sum(0)
        det = a11 * a22 - a12 ** 2
        b2 = (a22 * r1 - a12 * r2) / det
        c2 = (a11 * r2 - a12 * r1) / det
        d1 = np.where(ok, np.abs(dZ - b1 * dQ), 0).max(0)
        d2 = np.where(ok, np.abs(dZ - b2 * dQ - c2 * dP), 0).max(0)
    G = ras(z["px_per_zeta"])
    pick = lambda R: R.reshape(-1)[flat]
    return {"dev_radial_px": pick(d1 * G), "dev_plane_px": pick(d2 * G), "n_nb": pick(ok.sum(0).astype(float)),
            "zeta_q_true": pick(b1), "zeta_phi_true": pick(c2)}


def classify(z, range_flat, n, J, settings=None):
    """Classes of every anchor (meaningful for depth_edge rejections), the jump, the true statistics, the error
    if accepted (px of the longest step) and the relative range error."""
    s = dict(mb.SETTINGS, **(settings or {}))
    hh, tol = int(s["edge_half_px"]), float(s["edge_tol_px"])
    flat = z["flat"]
    rng = np.asarray(range_flat, float)[flat]
    with np.errstate(divide="ignore", invalid="ignore"):
        zt = 1.0 / (rng * np.cos(z["e"]))
    jump = jump_raster(range_flat, n, hh).reshape(-1)[flat]
    tdev = true_devs(z, zt, n, hh)
    cls = np.full(flat.size, -1, np.int8)
    disc = jump > J
    cls[disc] = 0
    cls[~disc & (tdev["dev_radial_px"] > tol) & (tdev["dev_plane_px"] <= tol)] = 1
    cls[~disc & (tdev["dev_plane_px"] > tol)] = 2
    cls[~disc & (tdev["dev_radial_px"] <= tol)] = 3
    with np.errstate(divide="ignore", invalid="ignore"):
        err_px = np.abs(z["zeta"] - zt) * z["px_per_zeta"]
        rel = np.abs(1.0 / (z["zeta"] * np.cos(z["e"])) - rng) / rng
    return {"cls": cls, "jump": jump, "zeta_true": zt, "err_px": err_px, "rel": rel, **tdev}


def _q(v, p):
    v = v[np.isfinite(v)]
    return float(np.quantile(v, p)) if v.size else None


def summarize(rows, J):
    """rows: list of (z, c) for one window. Class shares of the depth_edge rejections with their errors if
    accepted; accepted anchors for comparison; M4's flag as a detector of class-1 edges by band."""
    R = lambda k: np.concatenate([r[0][k] if k in r[0] else r[1][k] for r in rows])
    reason, cls, err, rel, e = R("reason"), R("cls"), R("err_px"), R("rel"), np.degrees(R("e"))
    jump = R("jump")
    DE, ACC = mb.REASONS.index("depth_edge"), 0
    flag, acc = reason == DE, reason == ACC
    reached = flag | acc                                   # anchors the test judged
    true_edge = jump > J
    out = {"J": J, "anchors": int(reason.size), "depth_edge": int(flag.sum()), "accepted": int(acc.sum()),
           "depth_edge_share": float(flag.mean()), "classes": {}, "accepted_err": {}, "detector": {}}
    def st(m):
        return {"n": int(m.sum()), "share_of_depth_edge": float(m.sum() / max(flag.sum(), 1)),
                "err_px_median": _q(err[m], 0.5), "err_px_p90": _q(err[m], 0.9),
                "over_1px": float(np.mean(err[m][np.isfinite(err[m])] > 1)) if np.isfinite(err[m]).any() else None,
                "rel_median": _q(rel[m], 0.5), "rel_p90": _q(rel[m], 0.9)}
    for k, name in enumerate(CLASSES):
        out["classes"][name] = st(flag & (cls == k))
    out["accepted_err"] = st(acc)
    out["accepted_err"].pop("share_of_depth_edge")
    out["accepted_true_edge_share"] = float(np.mean(true_edge[acc])) if acc.any() else None
    for lo, hi in BANDS_DEG + ((1, 44),):
        b = (e >= lo) & (e < hi)
        tp = (flag & true_edge & b).sum()
        out["detector"][f"{lo}-{hi}"] = {
            "flagged": int((flag & b).sum()), "true_edges_judged": int((true_edge & reached & b).sum()),
            "true_edges_all": int((true_edge & b).sum()),
            "precision": float(tp / max((flag & b).sum(), 1)),
            "recall_of_judged": float(tp / max((true_edge & reached & b).sum(), 1)),
            "recall_of_all": float(tp / max((true_edge & b).sum(), 1)),
            "base_rate_judged": float((true_edge & reached & b).sum() / max((reached & b).sum(), 1))}
    return out
