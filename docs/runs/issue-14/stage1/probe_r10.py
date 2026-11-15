import sys, json, time, numpy as np
sys.path.insert(0, "external/fov-render")
from motion import analysis, flow_depth, m4_pipeline as mp, m5, m5_diagnosis as dg, multibaseline as mb
run = analysis.load_run("runs/m3-calib-1024"); es = {e["k"]: e for e in run["looks"]["ahead"]}
r = int(sys.argv[1]); w = sys.argv[2]
ks = [r + o for o in mp.WINDOWS[w]]
obs = {k: flow_depth.load_observation(run, es[k]) for k in [r] + ks}
ob = obs[r]; h = ob.R @ [0, 0, -1.0]; ts = [float((obs[k].c - ob.c) @ h) for k in ks]
t0 = time.time()
res = m5.match_reference(mp._lum(ob), [mp._lum(obs[k]) for k in ks], ts, ob.model)
print(f"time {time.time() - t0:.0f} s")
n = ob.model.n; flat = res["flat"]
rec, R, c, model = analysis.load_look(run, es[r])
rng = np.full(n * n, np.nan); hit = np.isfinite(rec.distance_m); rng[rec.raster_index[hit]] = rec.distance_m[hit]
zt = 1 / (rng[flat] * np.cos(res["e"])); G = res["px_per_zeta"]
on = dg.jump_raster(rng, n, 1).reshape(-1)[flat] > 0.05
err = lambda z: np.abs(z - zt) * G
rel = lambda z: np.abs(1 / (z * np.cos(res["e"])) - rng[flat]) / rng[flat]
q = lambda v, p: float(np.nanquantile(v, p))
m4a, m5a = res["m4_reason"] == 0, res["reason"] == 0
same = np.array_equal(res["zeta"][m4a & m5a & (res["recovered"] == "")], res["m4_zeta"][m4a & m5a & (res["recovered"] == "")])
print(f"M4 accepted {m4a.mean():.3f}  M5 accepted {m5a.mean():.3f}  (M4-accepted anchors unchanged: {same}; M4-accepted now rejected: {(m4a & ~m5a).sum()})")
for lab, m in (("M4 accepted", m4a), ("recovered: sigma", res["recovered"] == "sigma"), ("recovered: side", res["recovered"] == "side")):
    e_ = err(np.where(m4a & (lab == "M4 accepted"), res["m4_zeta"], res["zeta"]))[m]
    print(f"  {lab:18s} n {m.sum():6d}  err px med {q(e_, .5):.3f} p90 {q(e_, .9):.3f} >1px {np.mean(e_ > 1):.3f}  rel med {q(rel(res['zeta'])[m], .5):.4f} p90 {q(rel(res['zeta'])[m], .9):.4f}  on-edge share {on[m].mean():.3f}")
E = res["edge"]; cand = (res["m4_reason"] == 0) | (res["m4_reason"] == m5.DEPTH_EDGE)
print(f"edges flagged {E.sum()}  precision {on[E].mean():.3f}  recall(judged) {(E & on & cand).sum() / max((on & cand).sum(), 1):.3f}  base rate {on[cand].mean():.3f}")
still = res["reason"] == m5.DEPTH_EDGE
print(f"still depth_edge {still.sum()} ({still.mean():.3f}); valid sides among them: {np.bincount(res['n_valid_sides'][still], minlength=5).tolist()}")
# front / back sides against truth: true zeta at the side's window centre, approximated by the true zeta 3 px into that side (median of the half-block 7x7)
Zt = np.full(n * n, np.nan); Zt[flat] = zt; Zt = Zt.reshape(n, n)
Q = np.full(n * n, np.nan); Q[flat] = res["q"]; Q = Q.reshape(n, n); P = np.full(n * n, np.nan); P[flat] = res["phi"]; P = P.reshape(n, n)
def side_true(side):
    vals = []
    for di in range(-3, 4):
        for dj in range(-3, 4):
            if di or dj:
                dq = m5._shift(Q, di, dj) - Q; dp = np.angle(np.exp(1j * (m5._shift(P, di, dj) - P)))
                with np.errstate(invalid="ignore"):
                    o = {"out": dq > 0, "in": dq < 0, "right": dp > 0, "left": dp < 0}[side]
                vals.append(np.where(o, m5._shift(Zt, di, dj), np.nan).reshape(-1)[flat])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); return np.nanmedian(np.stack(vals, 1), 1)
ST = {sd: side_true(sd) for sd in ("out", "in", "right", "left")}
idx = np.flatnonzero(E)
if idx.size:
    fsd = [m5.SIDES[res["front_side"][i]] for i in idx]; bsd = [m5.SIDES[res["back_side"][i]] for i in idx]
    tf = np.array([ST[a][i] for a, i in zip(fsd, idx)]); tb = np.array([ST[b][i] for b, i in zip(bsd, idx)])
    ok3 = tf > tb
    ef = np.abs(res["zeta_front"][idx] - tf) * G[idx]; eb = np.abs(res["zeta_back"][idx] - tb) * G[idx]
    print(f"front side right (true front-half median zeta > back-half): {np.nanmean(ok3):.3f} of {idx.size}")
    print(f"front depth err px med {q(ef, .5):.3f} p90 {q(ef, .9):.3f}; back depth err px med {q(eb, .5):.3f} p90 {q(eb, .9):.3f}")
    ons = on[idx]
    print(f"  (on true edges only: front right {np.nanmean(ok3[ons]):.3f}; front err med {q(ef[ons], .5):.3f}, back err med {q(eb[ons], .5):.3f})")
sr = res["recovered"] == "side"
e_ = err(res["zeta"])
for nv in range(1, 5):
    m = sr & (res["n_valid_sides"] == nv)
    if m.any(): print(f"side-recovered with {nv} valid side(s): n {m.sum()}  >1px {np.mean(e_[m] > 1):.3f}  med {q(e_[m], .5):.3f}  on-edge {on[m].mean():.3f}")
sg = res["recovered"] == "sigma"
print("sigma-recovered: M4 sigma px median", q((res["m4_sigma_zeta"] * G)[sg], .5), "vs M4 accepted", q((res["m4_sigma_zeta"] * G)[m4a], .5))
np.savez("/tmp/s1/r%04d_%s.npz" % (r, w), **{k: v for k, v in res.items() if isinstance(v, np.ndarray) and v.dtype != object}, zt=zt, on=on)
