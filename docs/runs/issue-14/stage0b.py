"""M5 Stage 0, follow-up (#14), Chat sandbox: (1) the tilted fixture (50 deg plane, strong azimuthal slope), the
only development data where class 2 can show; (2) M4's flag against a finer edge truth: the anchor's own 3 x 3
block holds the jump (it touches the discontinuity), not the test's 7 x 7.

    PYTHONPATH=. python docs/runs/issue-14/stage0b.py docs/runs/issue-14/stage0b.json > docs/runs/issue-14/stage0b.log
"""
import sys, json, time
import numpy as np
sys.path.insert(0, "external/fov-render")
from pathlib import Path
from motion import analysis, m5_diagnosis as dg, multibaseline as mb, multibaseline_fixtures as mf
from fov_render import sensors

log = lambda m: print(m, flush=True)
J = 0.05
out = {"J": J}
t0 = time.time()
m = sensors.Foveated().model("full")
sc = mf.Scene(m, "tilted")
ks = mf.WINDOWS["both5"]
r = mb.match_reference(sc.frame(0), [sc.frame(k) for k in ks], list(np.array(ks, float) * mf.STEP), sc.m)
z = {k: r[k] for k in ("flat", "e", "phi", "q", "zeta", "zeta_q", "px_per_zeta", "reason")}
c = dg.classify(z, sc.truth()["range"], m.n, J)
s = dg.summarize([(z, c)], J)
acc = z["reason"] == 0
out["tilted"] = {"summary": s, "accepted_dev_radial_px_q": [float(np.quantile(c["dev_radial_px"][acc], p)) for p in (0.5, 0.9, 0.99)],
                 "accepted_dev_plane_px_q": [float(np.quantile(c["dev_plane_px"][acc], p)) for p in (0.5, 0.9, 0.99)]}
log(f"[s0b] tilted: depth_edge {s['depth_edge']} of {s['anchors']}, classes " + json.dumps({k: v['n'] for k, v in s['classes'].items()})
    + f"; true radial-only statistic over accepted anchors (px) q50/q90/q99 {np.round(out['tilted']['accepted_dev_radial_px_q'], 3).tolist()}"
    + f", with the azimuthal slope {np.round(out['tilted']['accepted_dev_plane_px_q'], 3).tolist()}  ({time.time() - t0:.0f} s)")
run = analysis.load_run("runs/m3-calib-1024")
es = {e["k"]: e for e in run["looks"]["ahead"]}
for w in ("both5", "before5"):
    fr = json.loads(Path(f"runs/m4-calib/{w}/freeze.json").read_text())
    acc_ = {b: np.zeros(4) for b in [f"{lo}-{hi}" for lo, hi in dg.BANDS_DEG + ((1, 44),)]}
    for r in fr["calibration_refs"]:
        z = dict(np.load(f"runs/m4-calib/{w}/r{r:04d}.npz"))
        rec, R, cc, model = analysis.load_look(run, es[r])
        rng = np.full(model.n * model.n, np.nan)
        hit = np.isfinite(rec.distance_m)
        rng[rec.raster_index[hit]] = rec.distance_m[hit]
        on = dg.jump_raster(rng, model.n, 1).reshape(-1)[z["flat"]] > J
        flag = z["reason"] == mb.REASONS.index("depth_edge")
        reached = flag | (z["reason"] == 0)
        e = np.degrees(z["e"])
        for lo, hi in dg.BANDS_DEG + ((1, 44),):
            b = (e >= lo) & (e < hi)
            acc_[f"{lo}-{hi}"] += [(flag & on & b).sum(), (flag & b).sum(), (on & reached & b).sum(), (reached & b).sum()]
    out[w] = {k: {"precision": v[0] / max(v[1], 1), "recall_of_judged": v[0] / max(v[2], 1), "base_rate_judged": v[2] / max(v[3], 1),
                  "flagged": int(v[1]), "true_edges_judged": int(v[2])} for k, v in acc_.items()}
    log(f"[s0b] {w}, edge truth = jump > {J} in the anchor's 3x3: " + json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in out[w].items()}))
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
