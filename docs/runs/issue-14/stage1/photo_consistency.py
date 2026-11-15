import sys, numpy as np
sys.path.insert(0, "external/fov-render")
from motion import analysis, flow_depth, m4_pipeline as mp, matcher as m3, m5
run = analysis.load_run("runs/m3-calib-1024"); es = {e["k"]: e for e in run["looks"]["ahead"]}
r, w = 10, "both5"
ks = [r + o for o in mp.WINDOWS[w]]
obs = {k: flow_depth.load_observation(run, es[k]) for k in [r] + ks}
ob = obs[r]; h = ob.R @ [0, 0, -1.0]; ts = np.array([float((obs[k].c - ob.c) @ h) for k in ks])
L = {k: mp._lum(obs[k]) for k in [r] + ks}
z = dict(np.load("/tmp/s1/r0010_both5.npz"))
G = z["px_per_zeta"]; zt = z["zt"]
def pc(idx, zeta):
    """photo-consistency of anchor pixels idx at hypotheses zeta: mean over frames of the squared difference between
    r's value and frame k's value at q + d_k(zeta), over the anchor and its two azimuthal neighbours (1/gp), divided by
    the variance of r's 3 x 3 (texture)."""
    q, ph, e = z["q"][idx], z["phi"][idx], z["e"][idx]
    gp = np.linalg.norm(ob.model.pixel_of(lp.dir_of_e_phi(e, ph + 1e-5))[0] - ob.model.pixel_of(lp.dir_of_e_phi(e, ph - 1e-5))[0], axis=-1) / 2e-5
    v = np.array([-1, 0, 1])[None, :] / gp[:, None]
    Ir = m3.sample(L[r], ob.model, q[:, None] + 0 * v, ph[:, None] + v)
    gq = G[idx] / np.abs(ts).max()
    u = np.array([-1, 0, 1])
    T = np.stack([m3.sample(L[r], ob.model, q[:, None] + uu / gq[:, None] + 0 * v, ph[:, None] + v) for uu in u], 1)
    var = np.nanvar(T.reshape(len(idx), -1), 1)
    D = []
    for t, k in zip(ts, ks):
        with np.errstate(invalid="ignore"):
            d = -np.log1p(-t * zeta)
        Ik = m3.sample(L[k], ob.model, q[:, None] + d[:, None] + 0 * v, ph[:, None] + v)
        D.append(np.nanmean((Ik - Ir) ** 2, 1))
    return np.nanmean(np.stack(D, 1), 1) / np.maximum(var, 1e-12)
from motion import logpolar as lp
sr = np.flatnonzero(z["recovered"] == "side") if "recovered" in z else None
err = np.abs(z["zeta"] - zt) * G
good, bad = sr[err[sr] < 0.5], sr[err[sr] > 1]
pcg, pcb = pc(good, z["zeta"][good]), pc(bad, z["zeta"][bad])
acc = np.flatnonzero((z["m4_reason"] == 0))[::10]
pca = pc(acc, z["zeta"][acc])
qq = lambda v: np.round(np.nanquantile(v, [0.1, 0.5, 0.9]), 3).tolist()
print("PC at the chosen zeta: M4-accepted (sample)", qq(pca), " side-recovered good (<0.5 px)", len(good), qq(pcg), " bad (>1 px)", len(bad), qq(pcb))
x = np.concatenate([pcg, pcb]); y = np.r_[np.zeros(len(pcg)), np.ones(len(pcb))]
o = np.argsort(x); ranks = np.empty(len(x)); ranks[o] = np.arange(1, len(x) + 1)
auc = (ranks[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (len(y) - y.sum()))
print(f"AUC (PC separates bad from good): {auc:.3f}")
pct = pc(bad, zt[bad])
print("bad ones: PC at the TRUE zeta", qq(pct), " share where PC(true) < PC(chosen):", float(np.nanmean(pct < pcb)))
for thr in (0.5, 1.0, 2.0):
    print(f"  threshold PC <= {thr}: keeps good {np.nanmean(pcg <= thr):.3f}, bad {np.nanmean(pcb <= thr):.3f}, M4-accepted {np.nanmean(pca <= thr):.3f}")
