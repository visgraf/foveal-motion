"""Greedy Foveal Explorer, baseline: FIXATE -> RECONSTRUCT -> SACCADE, then freeze.

Part of foveal-playground.  Quick start: README.md.  Architecture: docs/architecture.md.  The baseline run uses
--max-fix 600 --spp 64.

    .venv/bin/python tools/greedy_foveal/run.py explore   --run RUN [--max-fix 200] [--spp 64] [--keep-raw]
    .venv/bin/python tools/greedy_foveal/run.py checks    --run RUN
    .venv/bin/python tools/greedy_foveal/run.py visualize --run RUN --vis VIS

``explore`` drives one persistent Blender render server (``render_server.py``) and ends by writing ``freeze.json``,
the sha256 of every control product.  ``checks`` runs the minimal run checks; ``visualize`` writes overview.png.

Derived from f3d-vision (tag playground-2-sparse-lattice, main bbee8ef).  The post-hoc Breadth-1 evaluation
(``evaluate``) and its file-open audit are not part of foveal-playground; see docs/provenance.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (HERE, HERE.parent, REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import explorer as X  # noqa: E402
import head_pose as HP  # noqa: E402  (the declared fixed head H0)
from visuals import display_rgb, oracle_colors  # noqa: E402  (presentation colours only)

PREFIX = "[gfe]"
BLEND = REPO / "scenes/classroom/classroom_eye.blend"
FROZEN = ("trajectory.json", "final-map.npz", "coverage.npz", "final-map.ply", "final-map-oracle-segmentation.ply")


# ------------------------------------------------------------------ small utilities
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    return x


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(jsonable(data), indent=1, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def code_state() -> dict:
    dirty = [ln for ln in git("status", "--porcelain", "--untracked-files=no").splitlines() if ln.strip()]
    return {"head": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "tracked_dirty": dirty}


# ------------------------------------------------------------------ the render server
class Server:
    def __init__(self, run: Path, spp: int) -> None:
        self.q = run / "queue"
        self.q.mkdir(parents=True)
        self.log = open(run / "blender.log", "w")
        t0 = time.perf_counter()
        self.p = subprocess.Popen(["blender", "-b", str(BLEND), "--python-exit-code", "1", "-P",
                                   str(HERE / "render_server.py"), "--", "--queue", str(self.q), "--spp", str(spp)],
                                  cwd=REPO, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        while not (self.q / "server-ready.json").exists():
            if self.p.poll() is not None:
                raise RuntimeError(f"the render server died at start (exit {self.p.returncode}); see blender.log")
            if time.perf_counter() - t0 > 600:
                raise RuntimeError("the render server did not become ready in 600 s")
            time.sleep(0.05)
        self.ready = read_json(self.q / "server-ready.json")
        self.start_seconds = time.perf_counter() - t0

    def render(self, n: int, yaw: float, pitch: float, out: Path) -> dict:
        tmp = self.q / f"req-{n:04d}.tmp"
        tmp.write_text(json.dumps({"n": n, "yaw_deg": yaw, "pitch_deg": pitch, "out_dir": str(out)}))
        os.replace(tmp, self.q / f"req-{n:04d}.json")
        done, fail = self.q / f"done-{n:04d}.json", self.q / f"fail-{n:04d}.json"
        while not done.exists():
            if fail.exists():
                raise RuntimeError("render failed:\n" + read_json(fail)["traceback"])
            if self.p.poll() is not None:
                raise RuntimeError(f"the render server died (exit {self.p.returncode}); see blender.log")
            time.sleep(0.003)
        return read_json(done)

    def stop(self) -> int:
        (self.q / "stop").touch()
        try:
            rc = self.p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.p.kill()
            rc = self.p.wait()
        self.log.close()
        return rc


# ------------------------------------------------------------------ explore (control only)
def ply_bytes(xyz: np.ndarray, rgb_u8: np.ndarray, comment: str) -> bytes:
    xyz = np.asarray(xyz, np.float32).reshape(-1, 3)
    rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]
    head = (f"ply\nformat binary_little_endian 1.0\ncomment {comment}\nelement vertex {len(xyz)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode()
    return head + rec.tobytes()


def explore(a) -> None:
    run = Path(a.run).resolve()
    if (run / "trajectory.json").exists() or (run / "queue").exists():
        raise SystemExit(f"{PREFIX} REFUSED {run} already holds a run")
    if not BLEND.is_file():
        raise SystemExit(f"{PREFIX} STOP {BLEND} is missing; build it with scripts/setup_scene.sh (README, Setup)")
    run.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    code = code_state()
    server = Server(run, a.spp)
    head_r, head_o = server.ready["eye_pose"]["head_R_wh"], server.ready["eye_pose"]["head_origin_w_m"]
    if server.ready["spp"] != a.spp or not HP.matches_h0(head_r, head_o):
        server.stop()
        raise RuntimeError("the render server's spp or head pose differs from the request / the declared head H0")
    print(f"{PREFIX} render server ready in {server.start_seconds:.1f} s ({server.ready['device']}, "
          f"{server.ready['spp']} spp, code {code['head'][:8]}{' DIRTY' if code['tracked_dirty'] else ''})", flush=True)
    gmap = X.GlobalMap()
    state = np.zeros((X.H, X.W), np.int8)
    visited = np.zeros((0, 3))
    traj, seen_hist, depth_hist = [], [], []
    yaw, pitch, kind, how = 0.0, 0.0, "initial", {}
    stop_reason, monotone = None, True
    (run / "fixations").mkdir()
    (run / "policy").mkdir()
    try:
        for n in range(1, a.max_fix + 1):
            t0 = time.perf_counter()
            fd = run / "fixations" / f"fix-{n:04d}"
            done = server.render(n, yaw, pitch, fd)
            t_render = time.perf_counter() - t0
            c = read_json(fd / "calibration.json")
            if c["head_R_wh"] != head_r or c["head_origin_w_m"] != head_o:
                raise RuntimeError(f"fixation {n}: the physical head moved")
            if c["gaze_yaw_pitch_deg"] != [yaw, pitch]:
                raise RuntimeError(f"fixation {n}: rendered gaze {c['gaze_yaw_pitch_deg']} != requested")
            t1 = time.perf_counter()
            obs = X.load_observation(fd)
            t_load = time.perf_counter() - t1
            rec = X.reconstruct(c, obs)
            if not np.isfinite(rec["xyz"]).all():
                raise RuntimeError(f"fixation {n}: non-finite XYZ")
            t2 = time.perf_counter()
            fus = gmap.fuse(f"gfe_fix_{n:04d}", rec["xyz"], rec["rgb"], rec["inst"], n)
            t_fuse = time.perf_counter() - t2
            prev = state.copy()
            cov = X.update_coverage(state, yaw, pitch, rec["xyz"])
            monotone &= bool(np.all(state >= prev))
            seen, depth = X.fractions(state)
            seen_hist.append(seen)
            depth_hist.append(depth)
            visited = np.vstack([visited, X.direction(yaw, pitch)[None]])
            np.savez_compressed(fd / "points.npz", xyz_h=rec["xyz"].astype(np.float32), rgb=rec["rgb"],
                                instance_id_oracle=rec["inst"], err_mm=(rec["err_m"] * 1e3).astype(np.float32))
            if not a.keep_raw:
                for s in ("L", "R"):
                    (fd / f"raw_{s}.exr").unlink()
            t3 = time.perf_counter()
            if seen >= X.SEEN_STOP:
                dec, stop_reason = None, "seen >= 99 % of 4 pi"
            elif n == a.max_fix:
                dec, stop_reason = None, f"fixation cap {a.max_fix}"
            else:
                dec = X.decide(state, rec["core_valid"], yaw, pitch, visited)
                np.savez_compressed(run / "policy" / f"state-{n:04d}.npz", state=state, core_valid=rec["core_valid"],
                                    gaze=np.array([yaw, pitch]), visited=visited)
                if dec["kind"] == "none":
                    stop_reason = "no valid unseen fixation"
            t_policy = time.perf_counter() - t3
            err = rec["err_m"] * 1e3
            entry = {"n": n, "kind": kind, **how, "yaw_deg": yaw, "pitch_deg": pitch,
                     "render_seconds_lr": done["render_seconds_lr"],
                     "seconds": {"render": t_render, "correspondence": t_load + rec["seconds"]["correspondence"],
                                 "exr_load": t_load, "geometry": rec["seconds"]["geometry"], "fusion": t_fuse,
                                 "policy": t_policy, "total": time.perf_counter() - t0},
                     "correspondence": rec["counts"], "fusion": fus, "coverage": cov,
                     "seen_fraction": seen, "depth_fraction": depth,
                     "err_vs_left_position_mm_diagnostic": None if not err.size else
                     {"median": float(np.median(err)), "p95": float(np.quantile(err, 0.95)), "max": float(err.max())},
                     "next": None if dec is None else {k: v for k, v in dec.items() if k != "candidates"},
                     "local_candidates": None if dec is None else dec["candidates"]}
            traj.append(entry)
            s = entry["seconds"]
            print(f"{PREFIX} fix {n:03d} {kind:7s} ({yaw:+8.2f},{pitch:+7.2f}) render {s['render']:.2f} "
                  f"corr {s['correspondence']:.2f} geom {s['geometry']:.2f} fuse {s['fusion']:.2f} "
                  f"policy {s['policy']:.2f} | pts {rec['counts']['valid']:6d} (i0 {rec['counts']['valid_instance0']:5d}) "
                  f"map {gmap.n:9,d} (+{fus['new']:6d}) | seen {100 * seen:5.2f}% depth {100 * depth:5.2f}% | "
                  f"err med {entry['err_vs_left_position_mm_diagnostic']['median'] if err.size else float('nan'):.2f} mm"
                  f" -> {dec['kind'] if dec else 'STOP'}", flush=True)
            if stop_reason:
                break
            yaw, pitch, kind = float(dec["yaw_deg"]), float(dec["pitch_deg"]), dec["kind"]
            how = {"from_dir": dec.get("dir")} if kind == "local" else {"component_sr": dec.get("component_sr"),
                                                                       "depth_deg": dec.get("depth_deg")}
    finally:
        rc = server.stop()
    wall = time.perf_counter() - t_start
    # ---------------- freeze
    gmap.save(run / "final-map.npz")
    np.savez_compressed(run / "coverage.npz", state=state, seen_history=np.array(seen_hist),
                        depth_history=np.array(depth_hist), row_weights=X.ROW_W)
    arr = gmap.arrays()
    (run / "final-map.ply").write_bytes(ply_bytes(arr["xyz_h"], display_rgb(arr["rgb"]),
                                                  "greedy foveal explorer v0 final H0 map (display RGB)"))
    (run / "final-map-oracle-segmentation.ply").write_bytes(ply_bytes(
        arr["xyz_h"], oracle_colors(arr["instance_id_oracle"]),
        "ORACLE / VISUALIZATION ONLY: Blender Object Index of the creating point; never used by the explorer"))
    sums = {k: float(sum(e["seconds"][k] for e in traj)) for k in traj[0]["seconds"]}
    summary = {"schema": "GFE-v0-trajectory", "label": "PROTOTYPE (not a scientific milestone)", "code": code,
               "spp": a.spp, "max_fix": a.max_fix, "stop_reason": stop_reason, "fixations": len(traj),
               "local_saccades": sum(e["kind"] == "local" for e in traj),
               "global_saccades": sum(e["kind"] == "global" for e in traj),
               "final_seen_fraction": seen_hist[-1], "final_depth_fraction": depth_hist[-1],
               "final_surfels": int(gmap.n), "wall_seconds": wall, "server_start_seconds": server.start_seconds,
               "seconds_sum": sums, "server_exit": rc, "monotone_cells": monotone,
               "server": {k: server.ready[k] for k in ("blender", "device", "spp", "seeds", "passes", "settings")},
               "head_pose_source": HP.SOURCE,
               "parameters": {k: getattr(X, k) for k in ("CORE_FOV_DEG", "LOCAL_STEP_DEG", "EDGE_BAND_PX", "EDGE_MIN",
                                                         "GAIN_MIN", "VISIT_TOL_DEG", "SEEN_STOP", "ASSOC_RADIUS_M",
                                                         "ASSOC_CELL_M", "ZERO_TOL_ABS_M", "ZERO_TOL_REL", "GRID_DEG")},
               "trajectory": traj}
    write_json(run / "trajectory.json", summary)
    write_json(run / "freeze.json", {
        "schema": "GFE-v0-freeze", "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code": code,
        "statement": "control finished; the products below are frozen",
        "files": {f: sha256(run / f) for f in FROZEN}})
    if rc != 0:
        raise SystemExit(f"{PREFIX} STOP the render server exited {rc}")
    print(f"{PREFIX} FROZEN {len(traj)} fixations ({summary['local_saccades']} local, {summary['global_saccades']} "
          f"global), stop: {stop_reason}; seen {100 * seen_hist[-1]:.2f}% depth {100 * depth_hist[-1]:.2f}%; "
          f"{gmap.n:,d} surfels; wall {wall:.0f} s", flush=True)


# ------------------------------------------------------------------ minimal checks
def checks(a) -> None:
    import fsg3_surface_map as SM
    run = Path(a.run).resolve()
    traj = read_json(run / "trajectory.json")
    T = traj["trajectory"]
    res = {}
    heads = [read_json(run / f"fixations/fix-{e['n']:04d}/calibration.json") for e in T]
    head_r, head_o = heads[0]["head_R_wh"], heads[0]["head_origin_w_m"]
    res["fixed_head_H0"] = bool(HP.matches_h0(head_r, head_o)
                                and all(c["head_R_wh"] == head_r and c["head_origin_w_m"] == head_o for c in heads))
    gm = X.GlobalMap.load(run / "final-map.npz")
    pts_ok = all(np.isfinite(np.load(run / f"fixations/fix-{e['n']:04d}/points.npz")["xyz_h"]).all() for e in T)
    res["finite_xyz"] = bool(np.isfinite(gm.xyz).all() and pts_ok)
    d = X.direction(np.array([e["yaw_deg"] for e in T]), np.array([e["pitch_deg"] for e in T]))
    ang = X.angle_deg(d, d) + np.eye(len(d)) * 999
    res["no_duplicate_fixation"] = {"pass": bool(ang.min() >= X.VISIT_TOL_DEG), "min_pair_deg": float(ang.min())}
    sh = np.load(run / "coverage.npz")["seen_history"]
    res["seen_monotone"] = bool(np.all(np.diff(sh) >= 0) and traj["monotone_cells"])
    with np.load(run / "final-map.npz") as z:
        saved = {k: np.array(z[k]) for k in z.files}
    back = gm.arrays()
    same = set(saved) == set(back) and all(np.array_equal(saved[k], back[k]) for k in saved)
    k = min(1000, gm.n)
    bi, bd = gm.nearest(gm.xyz[:k])
    res["map_save_load"] = bool(same and gm.n == traj["final_surfels"] and np.all(bd == 0.0))
    fz = read_json(run / "freeze.json")
    stale = [f for f, h in fz["files"].items() if sha256(run / f) != h]
    res["products_unchanged_since_freeze"] = {"pass": not stale and set(fz["files"]) == set(FROZEN), "changed": stale}
    # deterministic policy: every saved decision state re-decided (and one state twice)
    mism = []
    for e in T:
        p = run / "policy" / f"state-{e['n']:04d}.npz"
        if not p.exists():
            continue
        with np.load(p) as z:
            st, cv, g, vis = z["state"], z["core_valid"], z["gaze"], z["visited"]
        dec = X.decide(st, cv, float(g[0]), float(g[1]), vis)
        want = e["next"]
        if dec["kind"] != want["kind"] or (dec["kind"] != "none" and (dec["yaw_deg"], dec["pitch_deg"]) !=
                                           (want["yaw_deg"], want["pitch_deg"])):
            mism.append(e["n"])
    mid = T[len(T) // 2]["n"]
    with np.load(run / "policy" / f"state-{mid:04d}.npz") as z:
        args_ = (z["state"], z["core_valid"], float(z["gaze"][0]), float(z["gaze"][1]), z["visited"])
    d1, d2 = X.decide(*args_), X.decide(*args_)
    res["deterministic_policy"] = {"pass": not mism and d1 == d2, "states_rechecked": len(T) - 1, "mismatches": mism}
    # fusion-rule known answer: the vectorized global map == the accepted FSG3 fuse (its own self-test planes)
    rng = np.random.default_rng(12)

    def plane(pid, lo, hi):
        xy = np.stack(np.meshgrid(np.linspace(lo, hi, 50), np.linspace(-.10, .10, 24)), axis=-1).reshape(-1, 2)
        xyz = np.c_[xy, -2 * np.ones(len(xy))] + rng.normal(scale=.0004, size=(len(xy), 3))
        return SM.Patch(pid, xyz, rng.random((len(xy), 3)), np.full(len(xyz), 71))

    ps = [plane("A", -.28, -.02), plane("B", -.12, .12), plane("C", .02, .28)]
    m = SM.initialize(ps[0], 71)
    for q in ps[1:]:
        m, _ = SM.fuse(m, q, 71, X.ASSOC_RADIUS_M, X.ASSOC_CELL_M)
    g = X.GlobalMap()
    for i, q in enumerate(ps):
        g.fuse(q.patch_id, q.xyz_h, q.rgb, q.instance_id, i + 1)
    ga = g.arrays()
    res["fusion_rule_equals_accepted_fsg3"] = bool(np.array_equal(ga["xyz_h"], m.xyz_h)
                                                   and np.array_equal(ga["rgb"], m.rgb)
                                                   and np.array_equal(ga["support_count"], m.support_count))
    ok = all((v["pass"] if isinstance(v, dict) else v) for v in res.values())
    write_json(run / "checks.json", {"schema": "GFE-v0-checks", "all_pass": ok, "checks": res})
    for k_, v in res.items():
        print(f"{PREFIX} check {k_}: {v}", flush=True)
    print(f"{PREFIX} CHECKS {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit(1)


def visualize(a) -> None:
    import visuals
    visuals.overview(Path(a.run).resolve(), Path(a.vis).resolve())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("explore")
    e.add_argument("--run", required=True)
    e.add_argument("--max-fix", type=int, default=200)
    e.add_argument("--spp", type=int, default=64)
    e.add_argument("--keep-raw", action="store_true")
    sub.add_parser("checks").add_argument("--run", required=True)
    v = sub.add_parser("visualize")
    v.add_argument("--run", required=True)
    v.add_argument("--vis", required=True)
    a = ap.parse_args()
    {"explore": explore, "checks": checks, "visualize": visualize}[a.cmd](a)


if __name__ == "__main__":
    main()
