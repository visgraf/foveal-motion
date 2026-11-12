"""The dolly's checks and its translational flow, from the run's files only (numpy).

Every check here has a control that must fail (fov-render CLAUDE.md, rule 2):

    M1.1 heads      every record sits on its path head, exactly (record.head_origin_w, head_R_wh);
                    control: the same records against the path shifted by one step
    M1.2 target     in "target" mode the sample on the eye's axis lands on target_w, within
                    2 pixels at its range; control: target_w moved 10 cm sideways
    M1.3 reproject  look k+1's surface points, projected into look k with look k's pose, land on
                    pixels whose own Position is the same point (2 pixels at the range: incidence up
                    to ~70 deg), for at least REPROJECT_MIN of the points that fall inside look k's
                    field; control: look k's eye centre wrong by 10 cm sideways, which must lose at
                    least half of the agreeing points
    sensor          every record agrees with its Position pass (fov_render.record.sensor_consistency,
                    the scene gate's S4 limit: p99.9 within one pixel at the eccentricity)

The flow: for every point of look k+1 that M1.3 finds also in look k (not occluded), its retinal
displacement between the two looks in the eye frame, in degrees and in foveated raster pixels,
against its eccentricity in look k+1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fov_render import record as R
from fov_render.sensors import FoveatedModel

from . import paths

REPROJECT_MIN = 0.80          # proposed before any Classroom run (Chat, issue #1); not to be moved to pass
TOL_PIXELS = 2.0
CONTROL_SHIFT_M = 0.10
ECC_BINS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0)


# ----------------------------------------------------------------------------- reading a run

def load_run(run_dir) -> dict:
    run = json.loads((Path(run_dir) / "run.json").read_text())
    run["_dir"] = str(Path(run_dir).resolve())
    return run


def look_dir(run: dict, entry: dict) -> Path:
    return Path(run["_dir"]) / entry["out_dir"]


def load_look(run: dict, entry: dict):
    d = look_dir(run, entry)
    meta = json.loads((d / "meta.json").read_text())
    m = np.asarray(meta["response"]["eyes"]["C"]["pose"]["matrix_world"], dtype=np.float64)
    rec = R.SampleRecord.load(str(d / "record_C.npz"), access="oracle")
    sensor = meta["request"]["sensor"]
    model = FoveatedModel(rec.raster_shape[0], sensor["E2_deg"], sensor["e_max_deg"])
    return rec, m[:3, :3], m[:3, 3], model


def eccentricity_deg(d_local) -> np.ndarray:
    d = np.asarray(d_local, float)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(np.linalg.norm(d[..., :2], axis=-1), -d[..., 2]))


def angle_deg(a, b) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1)))


# ----------------------------------------------------------------------------- M1.1

def check_heads(run: dict) -> dict:
    spec = run["path"]
    want = paths.heads(spec)
    shifted = want[1:] + want[:1]
    out, ctrl_caught = {}, []
    for mode, entries in run["looks"].items():
        errs = []
        for e in entries:
            rec = R.SampleRecord.load(str(look_dir(run, e) / "record_C.npz"))
            h, hs = want[e["k"]], shifted[e["k"]]
            errs.append({"k": e["k"], "origin_err_m": float(np.max(np.abs(rec.head_origin_w - h.origin_w))),
                         "R_err": float(np.max(np.abs(rec.head_R_wh - h.R_wh))),
                         "eye_origin_err_m": float(np.max(np.abs(rec.origin - h.origin_w)))})
            ctrl_caught.append(bool(np.any(rec.head_origin_w != hs.origin_w)))
        out[mode] = {"looks": len(errs), "max_origin_err_m": max(x["origin_err_m"] for x in errs),
                     "max_R_err": max(x["R_err"] for x in errs),
                     "max_eye_origin_err_m": max(x["eye_origin_err_m"] for x in errs),
                     "ks": sorted(x["k"] for x in errs)}
    n = int(spec["n_looks"])
    # the record's head is the request's, bit for bit (fov-render gate head.5); the eye centre of the
    # cyclopean eye is the head origin through the Blender camera's float32 matrix
    ok = all(v["max_origin_err_m"] == 0.0 and v["max_R_err"] == 0.0 and v["ks"] == list(range(n))
             and v["max_eye_origin_err_m"] <= 1e-6 for v in out.values())
    return {"id": "M1.1", "what": "every record sits on its path head", "ok": ok,
            "control_fails": bool(ctrl_caught) and all(ctrl_caught),
            "control": "the records against the path shifted by one step: every look must differ", "detail": out}


# ----------------------------------------------------------------------------- M1.2

def _axis_sample(rec, R_wb):
    d_loc = rec.direction @ R_wb            # world -> eye (Blender camera) frame
    i = int(np.argmin(eccentricity_deg(d_loc)))
    return i


def fovea_on_target(run: dict) -> dict:
    target = np.asarray(run["path"]["target_w"], float)
    side = paths.head_axes(run["path"]["facing_w"])[:, 0]
    out = {}
    for mode, entries in run["looks"].items():
        rows = []
        for e in entries:
            rec, R_wb, c, _ = load_look(run, e)
            i = _axis_sample(rec, R_wb)
            hit = bool(np.isfinite(rec.distance_m[i]))
            p = rec.position_w[i]
            rng = float(np.linalg.norm(target - c))
            tol = TOL_PIXELS * rng * float(np.sqrt(rec.footprint_sr[i]))
            rows.append({"k": e["k"], "hit": hit, "err_m": float(np.linalg.norm(p - target)) if hit else None,
                         "err_shifted_m": float(np.linalg.norm(p - (target + CONTROL_SHIFT_M * side))) if hit else None,
                         "tol_m": tol, "range_m": rng})
        within = [r["hit"] and r["err_m"] <= r["tol_m"] for r in rows]
        within_shift = [r["hit"] and r["err_shifted_m"] <= r["tol_m"] for r in rows]
        out[mode] = {"looks": len(rows), "on_target": int(sum(within)), "on_shifted_target": int(sum(within_shift)),
                     "max_err_m": max((r["err_m"] for r in rows if r["hit"]), default=None),
                     "max_tol_m": max(r["tol_m"] for r in rows), "rows": rows}
    t = out.get("target")
    ok = t is not None and t["on_target"] == t["looks"]
    ctrl = t is not None and t["on_shifted_target"] == 0
    return {"id": "M1.2", "what": "in target mode the fovea's centre lands on target_w", "ok": ok,
            "control_fails": ctrl, "control": f"target_w moved {CONTROL_SHIFT_M} m sideways: no look may land on it",
            "detail": {m: {k: v for k, v in d.items() if k != "rows"} for m, d in out.items()},
            "rows": {m: d["rows"] for m, d in out.items()}}


# ----------------------------------------------------------------------------- M1.3 and the flow

def _project(model, R_wb, c, P):
    """World points -> (continuous (row, col), inside, range, eye-frame unit direction)."""
    v = P - c
    rng = np.linalg.norm(v, axis=-1)
    d_loc = (v / rng[:, None]) @ R_wb
    rc, inside = model.pixel_of(d_loc)
    return rc, inside & (d_loc[:, 2] < 0), rng, d_loc


def _agree(rec_a, model, R_wb, c, P):
    """Which points P are seen by look a at the pixel they project to (Position within TOL_PIXELS
    pixels at the range). Returns (agree, considered, rc, d_loc)."""
    n = model.n
    rc, inside, rng, d_loc = _project(model, R_wb, c, P)
    ij = np.rint(rc).astype(np.int64)
    inr = inside & np.all((ij >= 0) & (ij < n), axis=-1)
    inv = np.full(n * n, -1, dtype=np.int64)
    inv[rec_a.raster_index] = np.arange(rec_a.n_samples)
    s = np.full(len(P), -1, dtype=np.int64)
    s[inr] = inv[ij[inr, 0] * n + ij[inr, 1]]
    considered = s >= 0
    agree = np.zeros(len(P), dtype=bool)
    q = s[considered]
    hit_a = np.isfinite(rec_a.distance_m[q])
    err = np.linalg.norm(rec_a.position_w[q] - P[considered], axis=-1)
    tol = TOL_PIXELS * rng[considered] * np.sqrt(rec_a.footprint_sr[q])
    agree[considered] = hit_a & (err <= tol)
    return agree, considered, rc, d_loc


def reprojection(run: dict, keep_flow: bool = True) -> tuple[dict, dict]:
    side = paths.head_axes(run["path"]["facing_w"])[:, 0]
    out, flow = {}, {}
    for mode, entries in run["looks"].items():
        es = sorted(entries, key=lambda e: e["k"])
        pairs, F = [], {"pair": [], "ecc_deg": [], "flow_deg": [], "flow_px": [], "rc_b": [], "rc_a": []}
        prev = load_look(run, es[0])
        for e in es[1:]:
            cur = load_look(run, e)
            rec_a, Ra, ca, model = prev
            rec_b, Rb, cb, _ = cur
            hit_b = np.isfinite(rec_b.distance_m)
            P = rec_b.position_w[hit_b]
            agree, considered, rc_a, d_a = _agree(rec_a, model, Ra, ca, P)
            c_agree, c_considered, _, _ = _agree(rec_a, model, Ra, ca + CONTROL_SHIFT_M * side, P)
            nc = int(considered.sum())
            frac = float(agree.sum() / nc) if nc else None
            cfrac = float(c_agree.sum() / max(int(c_considered.sum()), 1))
            pairs.append({"k": [prev[0].look_id, rec_b.look_id], "ks": [es[es.index(e) - 1]["k"], e["k"]],
                          "hits_b": int(hit_b.sum()), "considered": nc, "agree": int(agree.sum()),
                          "fraction": frac, "control_fraction": cfrac})
            if keep_flow and agree.any():
                rc_b, _, _, d_b = _project(model, Rb, cb, P[agree])
                F["pair"].append(np.full(int(agree.sum()), len(pairs) - 1, dtype=np.int32))
                F["ecc_deg"].append(eccentricity_deg(d_b).astype(np.float32))
                F["flow_deg"].append(angle_deg(d_a[agree], d_b).astype(np.float32))
                F["flow_px"].append((rc_b - rc_a[agree]).astype(np.float32))
                F["rc_b"].append(rc_b.astype(np.float32))
                F["rc_a"].append(rc_a[agree].astype(np.float32))
            prev = cur
        fr = [p["fraction"] for p in pairs if p["fraction"] is not None]
        out[mode] = {"pairs": len(pairs), "min_fraction": min(fr) if fr else None,
                     "median_fraction": float(np.median(fr)) if fr else None,
                     "max_control_fraction": max(p["control_fraction"] for p in pairs),
                     "control_lost_half": all(p["control_fraction"] <= 0.5 * p["fraction"]
                                              for p in pairs if p["fraction"]),
                     "rows": pairs}
        if keep_flow:
            flow[mode] = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in F.items()}
    ok = bool(out) and all(v["min_fraction"] is not None and v["min_fraction"] >= REPROJECT_MIN for v in out.values())
    ctrl = bool(out) and all(v["control_lost_half"] for v in out.values())
    res = {"id": "M1.3", "what": f"look k+1's points reproject onto look k's surface (>= {REPROJECT_MIN})",
           "ok": ok, "control_fails": ctrl,
           "control": f"look k's eye centre {CONTROL_SHIFT_M} m sideways must lose half of the agreeing points",
           "detail": {m: {k: v for k, v in d.items() if k != "rows"} for m, d in out.items()},
           "rows": {m: d["rows"] for m, d in out.items()}}
    return res, flow


def flow_summary(flow: dict) -> dict:
    out = {}
    for mode, F in flow.items():
        rows = []
        px = np.linalg.norm(F["flow_px"], axis=-1) if len(F["flow_px"]) else np.zeros(0)
        for lo, hi in zip(ECC_BINS[:-1], ECC_BINS[1:]):
            m = (F["ecc_deg"] >= lo) & (F["ecc_deg"] < hi)
            rows.append({"ecc_deg": [lo, hi], "n": int(m.sum()),
                         "median_flow_deg": float(np.median(F["flow_deg"][m])) if m.any() else None,
                         "p90_flow_deg": float(np.quantile(F["flow_deg"][m], 0.9)) if m.any() else None,
                         "median_flow_px": float(np.median(px[m])) if m.any() else None,
                         "p90_flow_px": float(np.quantile(px[m], 0.9)) if m.any() else None})
        out[mode] = rows
    return out


def sensor_check(run: dict) -> dict:
    worst = {}
    caught = []
    for mode, entries in run["looks"].items():
        ws = []
        for e in entries:
            rec, R_wb, c, model = load_look(run, e)
            cons = R.sensor_consistency(rec)
            # one pixel at the eccentricity of the worst sample is generous at the periphery; the gate's
            # limit is the coarsest pixel, as fov-render's S4: s0 (1 + e_max / E2)
            px = model.s0_deg * (1.0 + model.emax / model.e2)
            ws.append((cons["p999_angle_deg"], px))
            # control: the record's directions turned by one coarsest pixel about the eye's up axis
            th = np.radians(px * 1.5)
            up = R_wb[:, 1]
            K = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]])
            Rot = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
            rec.direction = rec.direction @ Rot.T
            caught.append(R.sensor_consistency(rec)["p999_angle_deg"] > px)
        worst[mode] = {"max_p999_deg": max(w[0] for w in ws), "pixel_deg": ws[0][1]}
    ok = all(v["max_p999_deg"] <= v["pixel_deg"] for v in worst.values())
    return {"id": "M1.0", "what": "every record agrees with its Position pass (S4's limit)", "ok": ok,
            "control_fails": bool(caught) and all(caught),
            "control": "directions turned by 1.5 coarsest pixels must exceed the limit", "detail": worst}
