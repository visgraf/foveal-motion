"""Blender side: the scene gate's S2 (holes, back-faces, clearance) from every head of a path.

fov-render's scene gate runs S2 from the session's head only, so a moved head inside or against
geometry is not detected (fov-render docs/PLAN.md, "Head motion (#20)"). This runs the same
sanity panorama (fov_render.bl.scene.sanity_checks) from each head of the path, and from the
path's control head, which must fail.

    blender -b --python-exit-code 1 -P motion/bl_preflight.py -- --blend B --path P --out J [--device OPTIX]
    python -m motion.bl_preflight --blend B --path P --out J --device CPU        (the bpy wheel)

Normally started by `python -m motion.dolly preflight`. Exit 0 when every path head passes and the
control head fails; 1 otherwise. Imports only bpy, numpy, the standard library and fov_render.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "external" / "fov-render"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def verdict(sanity: dict, pf: dict) -> dict:
    from fov_render.bl.scene import backface_verdict
    v = backface_verdict(sanity, pf["backface_allowed"], pf["max_backface_fraction_other"])
    near = sanity["nearest_surface_m"]
    out = {"hole_fraction": sanity["hole_fraction"], "nearest_surface_m": near,
           "backface_closed_fraction": v["closed_fraction"], "backface_open_fraction": v["open_fraction"],
           "backface_closed_not_allowed": v["closed_not_allowed"],
           "consistency_p999_deg": sanity["consistency"]["p999_angle_deg"],
           "render_seconds": sanity["render_seconds"], "device": sanity["device"]}
    out["fails"] = [k for k, bad in (("holes", sanity["hole_fraction"] > pf["max_hole_fraction"]),
                                     ("clearance", near is None or near < pf["min_clearance_m"]),
                                     ("backfaces", not v["ok"])) if bad]
    out["ok"] = not out["fails"]
    return out


def main(argv) -> int:
    ap = argparse.ArgumentParser(prog="bl_preflight")
    ap.add_argument("--blend", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="OPTIX")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--spp", type=int, default=4)
    a = ap.parse_args(argv)

    from fov_render.bl import config, env
    from fov_render.bl.scene import open_scene, sanity_checks
    from motion import paths
    env.prepare()
    spec = paths.load(a.path)
    pf = spec["preflight"]
    t0 = time.perf_counter()
    h = open_scene(a.blend)
    session_head = h.head
    cfg = config.RenderConfig(profile="small", device=a.device)
    work = Path(a.out).resolve().parent / "preflight-renders"
    rows = []
    for k, head in enumerate(paths.heads(spec)):
        h.head = head
        s = sanity_checks(h, cfg, a.width, a.spp, True, str(work / f"k{k:04d}"))
        rows.append(dict(k=k, head_origin_w_m=head.origin_w.tolist(), **verdict(s, pf)))
        print(f"[preflight] k={k:3d} {head.origin_w.round(3).tolist()} holes {rows[-1]['hole_fraction']:.5f} "
              f"nearest {rows[-1]['nearest_surface_m']} {'ok' if rows[-1]['ok'] else rows[-1]['fails']}", flush=True)
    h.head = paths.control_head(spec)
    c = verdict(sanity_checks(h, cfg, a.width, a.spp, True, str(work / "control")), pf)
    c["head_origin_w_m"] = h.head.origin_w.tolist()
    h.head = session_head
    ok, ctrl = all(r["ok"] for r in rows), not c["ok"]
    out = {"schema": "foveal-motion-preflight-v1", "blend": str(Path(a.blend).resolve()), "path": spec,
           "device_requested": a.device, "width": a.width, "spp": a.spp, "limits": pf,
           "ok": ok, "control_fails": ctrl, "control": c, "heads": rows,
           "summary": {"max_hole_fraction": max(r["hole_fraction"] for r in rows),
                       "min_nearest_surface_m": min((r["nearest_surface_m"] or 0.0) for r in rows),
                       "max_backface_closed_fraction": max(r["backface_closed_fraction"] for r in rows),
                       "failing_heads": [r["k"] for r in rows if not r["ok"]]},
           "seconds": round(time.perf_counter() - t0, 2)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"[preflight] {'PASS' if ok else 'FAIL'} over {len(rows)} heads; control "
          f"{'fails as it must' if ctrl else 'PASSED (the check cannot fail)'}: {c['fails']}", flush=True)
    return 0 if ok and ctrl else 1


if __name__ == "__main__":
    from fov_render.bl.env import loud, script_args
    args = script_args() if "--" in sys.argv else sys.argv[1:]
    rc = loud(lambda: main(args), "[preflight]")
    sys.exit(rc or 0)
