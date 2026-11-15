"""M5's matcher (M5a #14, Stage 1): M4's frozen multi-baseline matcher, then depth edges.

Stage 0 (#14) found that M4's depth-edge test rejects about 30 % of anchors (both5, calibration room), of which
three quarters lie on real depth discontinuities and a quarter are tripped by noisy estimates, none by slope or
curvature. M5 keeps every M4 decision before the edge test and replaces the edge test by two steps:

1. **An uncertainty-aware edge test.** A neighbour in the 7 x 7 block is inconsistent with the anchor's plane
   (zeta + zeta_q dq, as M4) only when its miss is over M4's tolerance (0.5 px) AND over edge_sigma_k times the
   combined sigma of the two estimates. Anchors that pass keep M4's fit (recovered: noisy neighbours).
2. **Shiftable windows** (Kanade & Okutomi 1994; Fusiello et al. 1997) for anchors that still fail: M4's joint
   Gauss-Newton refitted with the matching window moved wholly to one side of the anchor (outward, inward in
   q; right, left in phi; the anchor's pixel stays in the window), started from the median zeta of accepted
   neighbours on that side (a side without one is not tried: no ambiguity test exists for a side fit, the
   accepted neighbours carry M4's). A side is valid if its fit passes M4's per-anchor tests (support, texture,
   mean frame NCC, frames, move from its start, slope) and the uncertainty-aware edge test against the accepted
   neighbours on its side. Then:
     no valid side            -> depth_edge (rejected, as M4)
     valid sides agree        -> accepted, zeta of the valid side with the smallest sigma (recovered: one side)
     two valid sides disagree -> edge: the anchor lies on a depth edge. Reported: the front side (larger zeta)
                                 and its direction, the depth and sigma of the front and back sides. No single
                                 depth is given for the anchor (it is not counted as accepted).

M4's files are not changed: the side fits call M4's own `joint_fit` with its window replaced for the call
(`shifted_patch`); selftest checks that the unshifted replacement reproduces M4's fit bit for bit.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch

from . import logpolar as lp
from . import multibaseline as mb

SETTINGS = {"edge_sigma_k": 3.0, "side_agree_k": 3.0}       # fixed before measurement (#14, Stage 1)
REASONS = mb.REASONS + ("edge",)
EDGE = REASONS.index("edge")
DEPTH_EDGE = REASONS.index("depth_edge")
SIDES = ("out", "in", "right", "left", "centre")


def _window(s, side):
    r, a = int(s["r_half_px"]), int(s["a_half_px"])
    return {"out": ((0, 2 * r), (-a, a)), "in": ((-2 * r, 0), (-a, a)), "right": ((-r, r), (0, 2 * a)),
            "left": ((-r, r), (-2 * a, 0)), "centre": ((-r, r), (-a, a))}[side]


@contextlib.contextmanager
def shifted_patch(s, side):
    """M4's joint_fit with its matching window moved to one side (native px offsets, radial k and azimuthal j)."""
    (k0, k1), (j0, j1) = _window(s, side)
    orig = mb._patch

    def patch(s_, gq, gp, dtype):
        k = torch.arange(k0, k1 + 1, dtype=dtype, device=gq.device)
        j = torch.arange(j0, j1 + 1, dtype=dtype, device=gq.device)
        K, J = torch.meshgrid(k, j, indexing="ij")
        return K.reshape(1, -1) / gq[:, None], J.reshape(1, -1) / gp[:, None]
    mb._patch = patch
    try:
        yield
    finally:
        mb._patch = orig


def _shift(R, di, dj):
    n = R.shape[0]
    out = np.full_like(R, np.nan)
    out[max(-di, 0):n + min(-di, 0), max(-dj, 0):n + min(-dj, 0)] = R[max(di, 0):n + min(di, 0), max(dj, 0):n + min(dj, 0)]
    return out


def _ras(flat, n, v, cond):
    R = np.full(n * n, np.nan)
    R[flat[cond]] = v[cond]
    return R.reshape(n, n)


def edge_test(flat, n, zeta, zq, sig, q, phi, G, cand, nb, tol, k, side=None):
    """Worst inconsistency of each anchor's neighbours (in nb) with its plane, uncertainty-aware. side limits the
    neighbours to a half-block (out: dq > 0, in: dq < 0, right: dphi > 0, left: dphi < 0). Returns
    (inconsistent: bool, n_neighbours) for anchors in cand (others False, 0)."""
    hh = 3
    Zn, Sn = _ras(flat, n, zeta, nb), _ras(flat, n, sig, nb)
    Q, P = _ras(flat, n, q, np.ones_like(nb)), _ras(flat, n, phi, np.ones_like(nb))
    Z0, K0, S0, Gr = (_ras(flat, n, v, cand) for v in (zeta, zq, sig, G))
    bad = np.zeros((n, n), bool)
    cnt = np.zeros((n, n), int)
    for di in range(-hh, hh + 1):
        for dj in range(-hh, hh + 1):
            if not (di or dj):
                continue
            dq = _shift(Q, di, dj) - Q
            dp = np.angle(np.exp(1j * (_shift(P, di, dj) - P)))
            with np.errstate(invalid="ignore"):
                if side == "out":
                    on = dq > 0
                elif side == "in":
                    on = dq < 0
                elif side == "right":
                    on = dp > 0
                elif side == "left":
                    on = dp < 0
                else:
                    on = np.ones_like(dq, bool)
                zn = _shift(Zn, di, dj)
                dv = np.abs(zn - (Z0 + K0 * dq)) * Gr
                sc = np.sqrt(_shift(Sn, di, dj) ** 2 + S0 ** 2) * Gr
                use = on & np.isfinite(dv)
                bad |= use & (dv > tol) & (dv > k * sc)
                cnt += use
    pick = lambda R: R.reshape(-1)[flat]
    return pick(bad) & cand, np.where(cand, pick(cnt), 0)


def _side_neighbour_median(flat, n, zeta, q, phi, acc, idx, side):
    """Median zeta of accepted anchors in the 7 x 7 half-block on `side` of each anchor in idx."""
    hh = 3
    Za = _ras(flat, n, zeta, acc)
    Q, P = _ras(flat, n, q, np.ones_like(acc)), _ras(flat, n, phi, np.ones_like(acc))
    vals = []
    for di in range(-hh, hh + 1):
        for dj in range(-hh, hh + 1):
            if not (di or dj):
                continue
            dq = _shift(Q, di, dj) - Q
            dp = np.angle(np.exp(1j * (_shift(P, di, dj) - P)))
            with np.errstate(invalid="ignore"):
                on = {"out": dq > 0, "in": dq < 0, "right": dp > 0, "left": dp < 0}[side]
            vals.append(np.where(on, _shift(Za, di, dj), np.nan).reshape(-1)[flat[idx]])
    V = np.stack(vals, 1)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(V, 1)


def match_reference(img_r, imgs, ts, model, settings=None, settings5=None, dev=None, sides=("out", "in", "right", "left")):
    """M4's match_reference, then M5's edge steps. Returns M4's fields (prefixed m4_ where M5 changes them) and
    M5's: reason, zeta, sigma_zeta, zeta_q, side (index into SIDES; 4 = M4's centred fit), edge, front_side,
    back_side, zeta_front, sigma_front, zeta_back, sigma_back, n_valid_sides, recovered ('sigma' | 'side' | '')."""
    s = dict(mb.SETTINGS, **(settings or {}))
    s5 = dict(SETTINGS, **(settings5 or {}))
    dev = dev or mb.device()
    r = mb.match_reference(img_r, imgs, ts, model, s, dev)
    n = model.n
    flat, q, phi, e = r["flat"], r["q"], r["phi"], r["e"]
    G = r["px_per_zeta"]
    tol = float(s["edge_tol_px"])
    reason = r["reason"].astype(np.int8).copy()
    cand = (reason == 0) | (reason == DEPTH_EDGE)                       # anchors M4's edge test judged
    zeta, zq, sig = r["zeta"].copy(), r["zeta_q"].copy(), r["sigma_zeta"].copy()
    # 1. uncertainty-aware edge test among the same candidates
    bad, _ = edge_test(flat, n, zeta, zq, sig, q, phi, G, cand, cand, tol, s5["edge_sigma_k"])
    reason[cand] = np.where(bad[cand], DEPTH_EDGE, 0)
    recovered = np.full(len(flat), "", dtype=object)
    recovered[(r["reason"] == DEPTH_EDGE) & (reason == 0)] = "sigma"
    side_idx = np.full(len(flat), SIDES.index("centre"), np.int8)
    acc = reason == 0
    # 2. shiftable windows for the anchors that still fail
    fail = np.flatnonzero(reason == DEPTH_EDGE)
    edge = np.zeros(len(flat), bool)
    out5 = {k: np.full(len(flat), np.nan) for k in ("zeta_front", "sigma_front", "zeta_back", "sigma_back")}
    front_side = np.full(len(flat), -1, np.int8)
    back_side = np.full(len(flat), -1, np.int8)
    n_valid = np.zeros(len(flat), np.int8)
    if fail.size:
        dt = torch.float64
        to_t = lambda a: torch.tensor(np.asarray(a, np.float64), dtype=dt, device=dev)
        fm = mb.FovMap(model, dev)
        Ir, Is = to_t(img_r), [to_t(x) for x in imgs]
        tmax = max(abs(float(t)) for t in ts)
        gq = G / tmax
        gp = np.linalg.norm(model.pixel_of(lp.dir_of_e_phi(e, phi + 1e-5))[0] - model.pixel_of(lp.dir_of_e_phi(e, phi - 1e-5))[0],
                            axis=-1) / 2e-5
        res = {}
        for side in sides:
            z0 = _side_neighbour_median(flat, n, zeta, q, phi, acc, fail, side)
            have = np.isfinite(z0)
            idx = fail[have]
            if not idx.size:
                continue
            outs = {k: [] for k in ("zeta", "zeta_q", "sigma_zeta", "n_frames", "ok", "texture", "ncc_frames", "frames")}
            with shifted_patch(s, side):
                for a in range(0, idx.size, s["chunk"]):
                    b = min(idx.size, a + s["chunk"])
                    ii = idx[a:b]
                    m0 = torch.ones((ii.size, len(Is)), dtype=torch.bool, device=dev)
                    f = mb.joint_fit(Ir, Is, [float(t) for t in ts], fm, to_t(q[ii]), to_t(phi[ii]), to_t(gq[ii]), to_t(gp[ii]),
                                     to_t(z0[have][a:b]), m0, s)
                    for k in outs:
                        outs[k].append(f[k])
            f = {k: torch.cat(v).cpu().numpy() for k, v in outs.items()}
            with np.errstate(invalid="ignore"):
                mncc = np.where(f["frames"], f["ncc_frames"], np.nan)
                mncc = np.nansum(mncc, 1) / np.maximum(f["frames"].sum(1), 1)
                ok = (f["ok"] & (f["n_frames"] >= s["min_frames"]) & (f["texture"] >= s["texture_min"])
                      & (mncc >= s["ncc_min"]) & (np.abs(f["zeta"] - z0[have]) * G[idx] <= s["gn_max_move_px"])
                      & (np.abs(f["zeta_q"]) < s["zeta_q_max"]))
            # the side's own edge test, against accepted neighbours on its side
            zs, ks, ss = zeta.copy(), zq.copy(), sig.copy()
            zs[idx], ks[idx], ss[idx] = f["zeta"], f["zeta_q"], f["sigma_zeta"]
            c_ = np.zeros(len(flat), bool)
            c_[idx] = True
            nb = acc.copy()
            nb_bad, nb_cnt = edge_test(flat, n, zs, ks, ss, q, phi, G, c_, nb, tol, s5["edge_sigma_k"], side=side)
            valid = ok & ~nb_bad[idx] & (nb_cnt[idx] >= 1)
            res[side] = (idx, valid, f["zeta"], f["sigma_zeta"], f["zeta_q"])
        # combine per anchor
        Zs = np.full((len(flat), len(sides)), np.nan)
        Ss = np.full_like(Zs, np.nan)
        Ks = np.full_like(Zs, np.nan)
        for j, side in enumerate(sides):
            if side in res:
                idx, valid, zz, ssz, kk = res[side]
                Zs[idx[valid], j], Ss[idx[valid], j], Ks[idx[valid], j] = zz[valid], ssz[valid], kk[valid]
        V = np.isfinite(Zs)
        n_valid = V.sum(1).astype(np.int8)
        for i in fail:
            js = np.flatnonzero(V[i])
            if not js.size:
                continue
            zi, si = Zs[i, js], Ss[i, js]
            hi, lo = js[np.argmax(zi)], js[np.argmin(zi)]
            gap = (Zs[i, hi] - Zs[i, lo]) * G[i]
            comb = np.sqrt(Ss[i, hi] ** 2 + Ss[i, lo] ** 2) * G[i]
            if js.size >= 2 and gap > tol and gap > s5["side_agree_k"] * comb:
                edge[i] = True
                reason[i] = EDGE
                front_side[i] = SIDES.index(sides[hi])
                back_side[i] = SIDES.index(sides[lo])
                out5["zeta_front"][i], out5["sigma_front"][i] = Zs[i, hi], Ss[i, hi]
                out5["zeta_back"][i], out5["sigma_back"][i] = Zs[i, lo], Ss[i, lo]
            else:
                b = js[np.argmin(si)]
                reason[i] = 0
                zeta[i], sig[i], zq[i] = Zs[i, b], Ss[i, b], Ks[i, b]
                side_idx[i] = SIDES.index(sides[b])
                recovered[i] = "side"
    out = dict(r)
    out.update({"m4_reason": r["reason"], "m4_zeta": r["zeta"], "m4_sigma_zeta": r["sigma_zeta"], "m4_zeta_q": r["zeta_q"],
                "reason": reason, "zeta": zeta, "sigma_zeta": sig, "zeta_q": zq, "side": side_idx, "edge": edge,
                "front_side": front_side, "back_side": back_side, "n_valid_sides": n_valid, "recovered": recovered.astype(str), "settings5": s5,
                **out5})
    half = mb.Z95 * s["interval_scale"] * sig
    out["zeta_lo"], out["zeta_hi"], out["sigma_px"] = zeta - half, zeta + half, sig * G
    return out


def selftest(dev=None):
    """(1) M4's joint_fit through shifted_patch('centre') equals M4's own, bit for bit; (2) a shifted window
    covers the expected native offsets; (3) control: an 'out' window gives a different fit."""
    from fov_render import sensors
    from . import multibaseline_fixtures as mf
    dev = dev or mb.device()
    m = sensors.Foveated().model("full")
    sc = mf.Scene(m, "front")
    s = dict(mb.SETTINGS)
    ks = mf.WINDOWS["both5"]
    ts = [float(k * mf.STEP) for k in ks]
    dt = torch.float64
    to_t = lambda a: torch.tensor(np.asarray(a, np.float64), dtype=dt, device=dev)
    fm = mb.FovMap(m, dev)
    Ir, Is = to_t(sc.frame(0)), [to_t(sc.frame(k)) for k in ks]
    rays = m.rays()
    e_px, phi_px = lp.e_phi(rays.direction)
    sel = np.flatnonzero((rays.valid & (np.degrees(e_px) > 5) & (np.degrees(e_px) < 30)).reshape(-1))[::97][:200]
    e, ph = e_px.reshape(-1)[sel], phi_px.reshape(-1)[sel]
    q = lp.q_of_e(e)
    gq = lp.native_px_per_q(m, q, ph)
    gp = np.linalg.norm(m.pixel_of(lp.dir_of_e_phi(e, ph + 1e-5))[0] - m.pixel_of(lp.dir_of_e_phi(e, ph - 1e-5))[0], axis=-1) / 2e-5
    z0 = to_t(np.full(len(sel), 1 / 3.0))
    m0 = torch.ones((len(sel), len(Is)), dtype=torch.bool, device=dev)
    args = (Ir, Is, ts, fm, to_t(q), to_t(ph), to_t(gq), to_t(gp), z0, m0, s)
    a = mb.joint_fit(*args)
    with shifted_patch(s, "centre"):
        b = mb.joint_fit(*args)
    with shifted_patch(s, "out"):
        c = mb.joint_fit(*args)
    with shifted_patch(s, "left"):
        u, v = mb._patch(s, to_t(gq[:1]), to_t(gp[:1]), dt)
    same = all(torch.equal(torch.nan_to_num(a[k]), torch.nan_to_num(b[k])) for k in ("zeta", "zeta_q", "sigma_zeta", "n_frames"))
    offs = (np.unique(np.rint(u.cpu().numpy() * gq[0])).tolist(), np.unique(np.rint(v.cpu().numpy() * gp[0])).tolist())
    differ = bool((torch.abs(a["zeta"] - c["zeta"]) > 0).any())
    restored = mb._patch.__module__ == mb.__name__ and mb._patch.__name__ == "_patch"
    checks = [("centre window reproduces M4's joint_fit bit for bit", same),
              ("the left window spans radial -3..3, azimuthal -2..0 px", offs == ([-3, -2, -1, 0, 1, 2, 3], [-2, -1, 0])),
              ("M4's window is restored after the call", restored),
              ("control: the out window gives a different fit", differ)]
    return checks
