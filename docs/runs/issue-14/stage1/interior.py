import sys, numpy as np
sys.path.insert(0, "external/fov-render")
from motion import analysis, m4_pipeline as mp
run = analysis.load_run("runs/m3-calib-1024")
z = dict(np.load(sys.argv[1])); r = int(sys.argv[2])
inv_ = None
es = {e["k"]: e for e in run["looks"]["ahead"]}
rec, R, c, model = analysis.load_look(run, es[r])
n = model.n; inv = np.full(n * n, -1); inv[rec.raster_index] = np.arange(rec.n_samples)
z["sample"] = inv[z["flat"]]
t = mp.truth(run, r, z)
edge = t["edge"]; G = z["px_per_zeta"]; zt = z["zt"]
err = np.abs(z["zeta"] - zt) * G
rel = np.abs(1 / (z["zeta"] * np.cos(z["e"])) - t["range"]) / t["range"]
q = lambda v, p: float(np.nanquantile(v, p)) if np.isfinite(v).any() else float("nan")
for lab, m in (("M4 accepted", z["m4_reason"] == 0), ("side-recovered", z["recovered"] == "side"), ("sigma-recovered", z["recovered"] == "sigma")):
    for part, pm in (("interior", ~edge), ("edge (<= 2 px of a boundary)", edge)):
        mm = m & pm & t["hit"]
        print(f"{lab:16s} {part:30s} n {mm.sum():6d}  err px med {q(err[mm], .5):.3f} p90 {q(err[mm], .9):.3f} >1px {np.mean(err[mm] > 1):.3f}  rel med {q(rel[mm], .5):.4f} p90 {q(rel[mm], .9):.4f}")
print("-- side-recovered interior, by number of valid sides:")
for nv in (1, 2, 3, 4):
    mm = (z["recovered"] == "side") & ~edge & t["hit"] & (z["n_valid_sides"] == nv)
    print(f"   {nv}: n {mm.sum():5d}  >1px {np.mean(err[mm] > 1):.3f}  rel p90 {q(rel[mm], .9):.4f}")
for lab, extra in (("M4 alone", np.zeros_like(edge)), ("M4 + side (all)", z["recovered"] == "side"),
                   ("M4 + side (>= 2 agreeing)", (z["recovered"] == "side") & (z["n_valid_sides"] >= 2))):
    mm = ((z["m4_reason"] == 0) | extra) & ~edge & t["hit"]
    print(f"interior {lab:26s} n {mm.sum():6d} ({mm.sum() / (~edge & t['hit']).sum():.3f} of interior hits)  rel med {q(rel[mm], .5):.4f} p90 {q(rel[mm], .9):.4f}  >50% {np.mean(rel[mm] > .5):.4f}  >1px {np.mean(err[mm] > 1):.4f}")
