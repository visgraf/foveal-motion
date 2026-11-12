"""Experiment M1, the dolly: the head moves straight along a path, one monocular foveated look per step.

The cyclopean eye ("C", at the head origin) of fov-render's rig, the Foveated sensor, the head given
per look (fov-render #20, PLAN D19). One Blender session renders both gaze modes.

    python -m motion.dolly selftest                                      numpy only, seconds
    python -m motion.dolly preflight --path paths/classroom_dolly.json --run runs/dolly-1 --device OPTIX
    python -m motion.dolly run       --path paths/classroom_dolly.json --run runs/dolly-1 --device OPTIX
    python -m motion.dolly checks    --run runs/dolly-1
    python -m motion.dolly visualize --run runs/dolly-1 --vis visuals/dolly-1
    python -m motion.dolly depth     --run runs/dolly-1 [--strides 1 5 10]   M2: range from the head's motion
    python -m motion.dolly sandbox   --run runs/calib-dolly            the calibration room on CPU, end to end

`preflight` writes <run>/preflight.json, `run` refuses a run folder that already holds a run.json,
records the commit of this repository and of fov-render and whether both trees were clean. Exit
status: 0 done and every check passed with its control failing, 1 otherwise, 2 refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FOV_RENDER = REPO / "external" / "fov-render"
if (FOV_RENDER / "fov_render").is_dir() and str(FOV_RENDER) not in sys.path:
    sys.path.insert(0, str(FOV_RENDER))

from fov_render import RenderConfig, Rig, Session, launch, sensors  # noqa: E402

from . import analysis, paths  # noqa: E402

PREFIX = "[dolly]"


def _git(cwd: Path) -> dict:
    def g(*a):
        r = subprocess.run(["git", "-C", str(cwd), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = g("status", "--porcelain", "--ignore-submodules=dirty")
    return {"commit": g("rev-parse", "HEAD"), "clean": status == "" if status is not None else None}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _blend(spec: dict, given) -> Path:
    b = Path(given) if given else REPO / spec["scene"]
    if not b.is_file():
        raise SystemExit(f"{PREFIX} refused: no scene at {b} (scripts/setup_scene.sh builds the Classroom)")
    return b.resolve()


def _load_spec(p) -> dict:
    try:
        return paths.load(p)
    except paths.PathError as e:
        for r in e.reasons:
            print(f"{PREFIX} refused: {p}: {r}", file=sys.stderr)
        raise SystemExit(2)


# ----------------------------------------------------------------------------- preflight

def preflight(a) -> int:
    spec = _load_spec(a.path)
    blend = _blend(spec, a.blend)
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    out = run / "preflight.json"
    exe = launch.find(a.launcher)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(FOV_RENDER)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    args = ["--blend", str(blend), "--path", str(Path(a.path).resolve()), "--out", str(out.resolve()),
            "--device", a.device, "--width", str(a.width), "--spp", str(a.spp)]
    script = REPO / "motion" / "bl_preflight.py"
    cmd = ([exe, "-b", "--python-exit-code", "1", "-P", str(script), "--", *args] if launch.is_blender(exe)
           else [exe, "-m", "motion.bl_preflight", *args])
    print(f"{PREFIX} preflight: {len(paths.heads(spec))} heads + 1 control, {a.width} px, {a.spp} spp, {a.device}",
          flush=True)
    with open(run / "preflight.log", "w") as log:
        rc = subprocess.run(cmd, env=env, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT).returncode
    tail = [ln for ln in (run / "preflight.log").read_text().splitlines() if ln.startswith("[preflight]")]
    print("\n".join(tail[-3:]))
    if not out.is_file():
        print(f"{PREFIX} preflight wrote no {out}; see {run / 'preflight.log'}", file=sys.stderr)
        return 1
    return rc


# ----------------------------------------------------------------------------- run

def run(a) -> int:
    spec = _load_spec(a.path)
    blend = _blend(spec, a.blend)
    rd = Path(a.run)
    if (rd / "run.json").exists():
        print(f"{PREFIX} refused: {rd} already holds a run", file=sys.stderr)
        return 2
    modes = list(dict.fromkeys(a.modes))
    bad = [m for m in modes if m not in paths.MODES]
    if bad:
        print(f"{PREFIX} refused: unknown modes {bad}; expected {paths.MODES}", file=sys.stderr)
        return 2
    rd.mkdir(parents=True, exist_ok=True)
    pre = rd / "preflight.json"
    pre_ok = json.loads(pre.read_text()) if pre.is_file() else None
    if pre_ok is None and not a.no_preflight:
        print(f"{PREFIX} refused: no {pre}; run `preflight` first (or --no-preflight for a quick look)", file=sys.stderr)
        return 2
    if pre_ok is not None and not (pre_ok["ok"] and pre_ok["control_fails"]) and not a.no_preflight:
        print(f"{PREFIX} refused: the preflight failed (heads {pre_ok['summary']['failing_heads']}, control fails "
              f"{pre_ok['control_fails']}); see {pre}", file=sys.stderr)
        return 2
    try:
        spec = paths.resolved(spec, pre_ok)
    except paths.PathError as e:
        print(f"{PREFIX} refused: {e}", file=sys.stderr)
        return 2
    heads = paths.heads(spec)
    cfg = RenderConfig(profile=a.profile, device=a.device)
    record = {"schema": "foveal-motion-run-v1", "experiment": "M1 dolly", "path": spec,
              "target_source": "target_ray, found by the preflight" if "target_ray" in spec else "target_w",
              "path_file": str(Path(a.path)), "blend": str(blend), "blend_sha256": _sha256(blend),
              "profile": a.profile, "spp": a.spp, "device_requested": a.device, "modes": modes,
              "sensor": {"type": "foveated", "E2_deg": a.e2, "e_max_deg": a.emax}, "eye": "C",
              "git": {"foveal-motion": _git(REPO), "fov-render": _git(FOV_RENDER)},
              "command": sys.argv, "preflight": None if pre_ok is None else
              {k: pre_ok[k] for k in ("ok", "control_fails", "summary")}, "looks": {m: [] for m in modes}}
    t0 = time.perf_counter()
    with Session(blend, cfg, Rig(), access="oracle", launcher=a.launcher, workdir=rd / "session",
                 prepare=sensors.Foveated(E2_deg=a.e2, e_max_deg=a.emax)) as s:
        record["session"] = {"start_seconds": round(s.start_seconds, 2), "launcher": s.launcher,
                             "device": s.ready.get("device"), "session_head": s.ready.get("head")}
        for mode in modes:
            g = paths.gaze(spec, mode)
            for k, h in enumerate(heads):
                out = rd / mode / f"k{k:04d}"
                t1 = time.perf_counter()
                look = s.look(sensors.Foveated(E2_deg=a.e2, e_max_deg=a.emax), ("C",), g, head=h, spp=a.spp,
                              out_dir=out, outputs=("exr", "records"))
                record["looks"][mode].append({"k": k, "out_dir": str(out.relative_to(rd)), "head": h.as_dict(),
                                              "head_source": look.response.get("head_source"),
                                              "render_seconds": look.response["eyes"]["C"]["render_seconds"],
                                              "wall_seconds": round(time.perf_counter() - t1, 3)})
                print(f"{PREFIX} {mode} k={k:3d} head {np.round(h.origin_w, 3).tolist()} "
                      f"{record['looks'][mode][-1]['wall_seconds']:.2f} s", flush=True)
        record["device"] = s.ready.get("device")
    record["seconds"] = round(time.perf_counter() - t0, 2)
    (rd / "run.json").write_text(json.dumps(record, indent=1))
    print(f"{PREFIX} {sum(len(v) for v in record['looks'].values())} looks in {record['seconds']} s -> {rd}")
    return 0


# ----------------------------------------------------------------------------- checks

def checks(a) -> int:
    r = analysis.load_run(a.run)
    res = [analysis.sensor_check(r), analysis.check_heads(r), analysis.fovea_on_target(r)]
    rep, flow = analysis.reprojection(r)
    res.append(rep)
    summary = analysis.flow_summary(flow)
    rd = Path(a.run)
    np.savez_compressed(rd / "flow.npz", **{f"{m}__{k}": v for m, F in flow.items() for k, v in F.items()})
    out = {"schema": "foveal-motion-checks-v1", "run": str(rd), "git": r["git"], "device": r.get("device"),
           "step_m": paths.step_m(r["path"]), "checks": res, "flow_by_eccentricity": summary}
    (rd / "checks.json").write_text(json.dumps(out, indent=1))
    allok = True
    for c in res:
        good = c["ok"] and c["control_fails"]
        allok &= good
        print(f"{PREFIX} {c['id']:5s} {'PASS' if c['ok'] else 'FAIL'}  control "
              f"{'fails' if c['control_fails'] else 'DOES NOT FAIL'}  {c['what']}")
        for m, d in c["detail"].items():
            print(f"        {m}: {json.dumps(d)}")
    print(f"{PREFIX} flow per step ({out['step_m']:.4f} m), median deg | px, by eccentricity:")
    for m, rows in summary.items():
        cells = [f"{x['ecc_deg'][0]:g}-{x['ecc_deg'][1]:g}: "
                 + ("-" if x["median_flow_deg"] is None else f"{x['median_flow_deg']:.3f}|{x['median_flow_px']:.2f}")
                 for x in rows]
        print(f"        {m:6s} " + "  ".join(cells))
    print(f"{PREFIX} {'ALL PASS' if allok else 'NOT ALL PASS'} -> {rd / 'checks.json'}")
    return 0 if allok else 1


# ----------------------------------------------------------------------------- depth (M2)

def depth_cmd(a) -> int:
    from . import depth
    r = analysis.load_run(a.run)
    rd = Path(a.run)
    summary, examples, check = depth.run_depth(r, tuple(a.strides))
    np.savez_compressed(rd / "depth_examples.npz", **examples)
    step = paths.step_m(r["path"])
    out = {"schema": "foveal-motion-depth-v1", "run": str(rd), "git": r["git"], "device": r.get("device"),
           "step_m": step, "usable_rel": depth.USABLE_REL, "strides": list(a.strides), "check": check,
           "target_w": r["path"]["target_w"], "summary": summary}
    (rd / "depth.json").write_text(json.dumps(out, indent=1))
    good = check["ok"] and check["control_fails"]
    print(f"{PREFIX} {check['id']} {'PASS' if check['ok'] else 'FAIL'}  control "
          f"{'fails' if check['control_fails'] else 'DOES NOT FAIL'}  {check['what']}: {json.dumps(check['detail'])}")
    print(f"{PREFIX} range from motion, pixel-limited: median relative error | usable (<= {depth.USABLE_REL:.0%}) "
          f"by eccentricity")
    for mode, d in summary.items():
        for key, s in d.items():
            def g(v):
                return "-" if v is None else f"{v:.3f}"
            cells = [f"{x['bin'][0]:g}-{x['bin'][1]:g}: {g(x['median_rel'])}|{g(x['usable_fraction'])}"
                     for x in s["by_eccentricity"]]
            print(f"        {mode:6s} {s['stride'] * step:4.2f} m  " + "  ".join(cells)
                  + f"   fovea<2deg {g(s['fovea_2deg']['median_rel'])}|{g(s['fovea_2deg']['usable_fraction'])}"
                  + f"   no range {s['no_range_fraction']:.2f}   measured/model {g(s['median_measured_over_pred'])}")
    print(f"{PREFIX} {'PASS' if good else 'NOT PASS'} -> {rd / 'depth.json'}")
    return 0 if good else 1


# ----------------------------------------------------------------------------- visualize

def visualize(a) -> int:
    from . import visuals
    return visuals.main(Path(a.run), Path(a.vis), fps=a.fps)


# ----------------------------------------------------------------------------- selftest and sandbox

def selftest(_a) -> int:
    res = list(paths.selftest())
    # the inverse warp round-trips the forward warp: the projection M1.3 relies on
    m = sensors.Foveated().model("small")
    rays = m.rays()
    d = rays.direction.reshape(-1, 3)[rays.valid.reshape(-1)]
    rc, inside = m.pixel_of(d)
    j, i = np.meshgrid(np.arange(m.n), np.arange(m.n))
    want = np.stack([i.reshape(-1), j.reshape(-1)], -1)[rays.valid.reshape(-1)]
    res.append(("pixel_of inverts the warp (max 1e-6 px)", bool(inside.all() and np.max(np.abs(rc - want)) < 1e-6)))
    c = int(np.argmin(analysis.eccentricity_deg(d)))          # a central ray, turned by one foveal pixel
    off = d[c:c + 1] + [[np.radians(m.s0_deg), 0.0, 0.0]]
    res.append(("control: a central ray turned by one foveal pixel moves >= 0.5 px",
                bool(np.max(np.abs(m.pixel_of(off)[0] - want[c:c + 1])) >= 0.5)))
    for name, ok in res:
        print(f"{PREFIX} selftest {'PASS' if ok else 'FAIL'} {name}")
    return 0 if all(ok for _, ok in res) else 1


def sandbox(a) -> int:
    """The calibration room (fov-render's fixture) on CPU, small profile: preflight, run, checks, visualize."""
    from fov_render.checks.step1 import calib_room
    rd = Path(a.run)
    blend = calib_room({"work": str(REPO / "runs" / "calib-scene"), "device": "CPU"})
    common = dict(path=str(REPO / "paths" / a.path), run=str(rd), blend=blend, launcher=a.launcher)
    ns = argparse.Namespace
    steps = [("preflight", preflight, ns(device="CPU", width=360, spp=2, **common)),
             ("run", run, ns(device="CPU", profile="small", spp=a.spp, modes=a.modes, e2=2.0, emax=45.0,
                             no_preflight=False, **common)),
             ("checks", checks, ns(run=str(rd))),
             ("depth", depth_cmd, ns(run=str(rd), strides=[1, 5, 10])),
             ("visualize", visualize, ns(run=str(rd), vis=str(rd / "visuals"), fps=6))]
    rc = 0
    for name, fn, args in steps:
        print(f"{PREFIX} sandbox: {name}", flush=True)
        r = fn(args)
        rc = rc or r
        if r and name in ("preflight", "run"):
            return r
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m motion.dolly", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "run"):
        p = sub.add_parser(name)
        p.add_argument("--path", required=True)
        p.add_argument("--run", required=True)
        p.add_argument("--blend", default=None, help="override the spec's scene")
        p.add_argument("--device", default="OPTIX")
        p.add_argument("--launcher", default="auto")
    sub.choices["preflight"].add_argument("--width", type=int, default=720)
    sub.choices["preflight"].add_argument("--spp", type=int, default=4)
    r = sub.choices["run"]
    r.add_argument("--profile", default="full", choices=("small", "full"))
    r.add_argument("--spp", type=int, default=64)
    r.add_argument("--modes", nargs="+", default=list(paths.MODES))
    r.add_argument("--e2", type=float, default=2.0)
    r.add_argument("--emax", type=float, default=45.0)
    r.add_argument("--no-preflight", action="store_true")
    p = sub.add_parser("checks")
    p.add_argument("--run", required=True)
    p = sub.add_parser("visualize")
    p.add_argument("--run", required=True)
    p.add_argument("--vis", required=True)
    p.add_argument("--fps", type=int, default=6)
    p = sub.add_parser("depth")
    p.add_argument("--run", required=True)
    p.add_argument("--strides", type=int, nargs="+", default=[1, 5, 10])
    sub.add_parser("selftest")
    p = sub.add_parser("sandbox")
    p.add_argument("--run", required=True)
    p.add_argument("--spp", type=int, default=16)
    p.add_argument("--launcher", default="auto")
    p.add_argument("--path", default="calib_dolly.json", help="a spec in paths/")
    p.add_argument("--modes", nargs="+", default=list(paths.MODES))
    a = ap.parse_args(argv)
    return {"preflight": preflight, "run": run, "checks": checks, "visualize": visualize, "selftest": selftest,
            "sandbox": sandbox, "depth": depth_cmd}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
