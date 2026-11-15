"""M5 Stage 0 (#14): the depth-edge diagnosis on development data only (M-D17), Chat sandbox.

    PYTHONPATH=. python docs/runs/issue-14/stage0.py docs/runs/issue-14/stage0.json > docs/runs/issue-14/stage0.log

Controls on the fixtures (both5): front (one plane, no depth edge) must give no class-1 anchors; occlusion (a depth
step) must give some. Then the calibration room's calibration frames, M4's estimates from runs/m4-calib.
"""
import sys, json, time
import numpy as np
sys.path.insert(0, "external/fov-render")
from pathlib import Path
from motion import analysis, m5_diagnosis as dg, m4_pipeline as mp, multibaseline as mb, multibaseline_fixtures as mf
from fov_render import sensors

JS = (0.01, 0.02, 0.05, 0.10)
log = lambda m: print(m, flush=True)
out = {"fixtures": {}, "calibration_room": {}}
t0 = time.time()
m = sensors.Foveated().model("full")
for name in ("front", "occlusion"):
    sc = mf.Scene(m, name)
    ks = mf.WINDOWS["both5"]
    r = mb.match_reference(sc.frame(0), [sc.frame(k) for k in ks], list(np.array(ks, float) * mf.STEP), sc.m)
    z = {k: r[k] for k in ("flat", "e", "phi", "q", "zeta", "zeta_q", "px_per_zeta", "reason")}
    rng = sc.truth()["range"]
    out["fixtures"][name] = {}
    for J in JS:
        c = dg.classify(z, rng, m.n, J)
        out["fixtures"][name][str(J)] = dg.summarize([(z, c)], J)
    s = out["fixtures"][name]["0.05"]
    log(f"[s0] fixture {name}: depth_edge {s['depth_edge']} of {s['anchors']}; classes (J 5 %) "
        + json.dumps({k: v["n"] for k, v in s["classes"].items()}) + f"  ({time.time() - t0:.0f} s)")
run = analysis.load_run("runs/m3-calib-1024")
es = {e["k"]: e for e in run["looks"]["ahead"]}
for w in ("both5", "before5"):
    fr = json.loads(Path(f"runs/m4-calib/{w}/freeze.json").read_text())
    rows = {J: [] for J in JS}
    for r in fr["calibration_refs"]:
        z = dict(np.load(f"runs/m4-calib/{w}/r{r:04d}.npz"))
        rec, R, c, model = analysis.load_look(run, es[r])
        rng = np.full(model.n * model.n, np.nan)
        hit = np.isfinite(rec.distance_m)
        rng[rec.raster_index[hit]] = np.linalg.norm(rec.position_w[hit].astype(np.float64) - c, axis=-1)
        for J in JS:
            rows[J].append((z, dg.classify(z, rng, model.n, J)))
    out["calibration_room"][w] = {"refs": fr["calibration_refs"], "by_J": {str(J): dg.summarize(rows[J], J) for J in JS}}
    jumps = np.concatenate([c["jump"] for _, c in rows[JS[0]]])
    out["calibration_room"][w]["jump_quantiles_all_anchors"] = {str(p): float(np.quantile(jumps, p)) for p in (0.5, 0.75, 0.9, 0.95, 0.99)}
    log(f"[s0] {w}: done ({time.time() - t0:.0f} s)")
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
