"""Blender side: the scene gate's S2 (holes, back-faces, clearance) from every head of a path.

fov-render's scene gate runs S2 from the session's head only, so a moved head inside or against
geometry is not detected (fov-render docs/PLAN.md, "Head motion (#20)"). This runs the same
sanity panorama (fov_render.bl.scene.sanity_checks) from each head of the path, and from the
path's control head, which must fail.

    blender -b --python-exit-code 1 -P motion/bl_preflight.py -- --blend B --path P --out J [--device OPTIX]
    python -m motion.bl_preflight --blend B --path P --out J --device CPU        (the bpy wheel)

It also finds the target when the spec gives a target_ray (the first opaque surface along it), and
requires the target visible from every head: the first opaque surface on the segment from the head
toward it lies within paths.TARGET_TOL_M of it. Objects the scene's material audit flags as
non-opaque (glass, fog) are passed through, as fov-render's gate S5 does. Its control: a point 0.5 m
behind the target, seen from the same heads, must be hidden from every one of them.

Normally started by `python -m motion.dolly preflight`. Exit 0 when every path head passes, the
control head fails and the hidden point is hidden; 1 otherwise. Imports only bpy, numpy, the standard library and fov_render.
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


def first_opaque(scene, dg, flagged, origin, d, steps=16):
    """(distance, object name) of the first surface along origin + t d that the audit does not flag,
    or (None, None) when the ray leaves the scene."""
    import numpy as np
    o0 = np.asarray(origin, float)
    o = o0.copy()
    for _ in range(steps):
        hit, loc, _n, _f, obj, _m = scene.ray_cast(dg, o.tolist(), d.tolist())
        if not hit:
            return None, None
        obj = getattr(obj, "original", obj)
        if obj.name_full not in flagged:
            return float(np.dot(np.asarray(loc) - o0, d)), obj.name_full
        o = np.asarray(loc) + 1e-4 * d
    return None, None


def target_block(h, spec: dict) -> dict:
    """Resolve target_ray, then the target's visibility from every head and the hidden control."""
    import bpy
    import numpy as np
    from motion import paths
    dg = bpy.context.evaluated_depsgraph_get()
    flagged = set(h.audit.get("flagged_objects", []))
    heads = paths.heads(spec)
    out = {"flagged_passed_through": sorted(flagged), "tol_m": paths.TARGET_TOL_M}
    if "target_w" in spec:
        target = np.asarray(spec["target_w"], float)
        out["source"] = "target_w"
    else:
        tr = spec["target_ray"]
        o = heads[tr["from_k"]].origin_w
        d = np.asarray(tr["toward_w"], float) - o
        d /= np.linalg.norm(d)
        dist, name = first_opaque(h.scene, dg, flagged, o, d)
        out.update(source="target_ray", ray=tr, hit_object=name)
        if dist is None:
            out.update(target_w=None, ok=False, control_hidden=False, rows=[])
            return out
        target = o + dist * d
    out["target_w"] = target.tolist()

    def seen(head_o, p):
        v = p - head_o
        r = float(np.linalg.norm(v))
        dist, name = first_opaque(h.scene, dg, flagged, head_o, v / r)
        return dist is not None and abs(dist - r) <= paths.TARGET_TOL_M, dist, name, r

    rows, hidden = [], []
    away = target - heads[0].origin_w
    behind = target + 0.5 * away / np.linalg.norm(away)
    for k, hd in enumerate(heads):
        ok, dist, name, r = seen(hd.origin_w, target)
        rows.append({"k": k, "visible": ok, "range_m": r, "first_hit_m": dist, "first_hit_object": name})
        hidden.append(not seen(hd.origin_w, behind)[0])
    out.update(rows=rows, ok=all(x["visible"] for x in rows), occluded_heads=[x["k"] for x in rows if not x["visible"]],
               control_point_w=behind.tolist(), control_hidden=all(hidden))
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
    tgt = target_block(h, spec)
    print(f"[preflight] target {tgt.get('source')} {tgt.get('target_w')} ({tgt.get('hit_object', '')}): visible from "
          f"every head {tgt['ok']}, occluded {tgt.get('occluded_heads')}; hidden control hidden {tgt['control_hidden']}",
          flush=True)
    ok, ctrl = all(r["ok"] for r in rows) and tgt["ok"], (not c["ok"]) and tgt["control_hidden"]
    out = {"schema": "foveal-motion-preflight-v1", "blend": str(Path(a.blend).resolve()), "path": spec,
           "device_requested": a.device, "width": a.width, "spp": a.spp, "limits": pf,
           "ok": ok, "control_fails": ctrl, "control": c, "heads": rows, "target": tgt,
           "summary": {"max_hole_fraction": max(r["hole_fraction"] for r in rows),
                       "min_nearest_surface_m": min((r["nearest_surface_m"] or 0.0) for r in rows),
                       "max_backface_closed_fraction": max(r["backface_closed_fraction"] for r in rows),
                       "failing_heads": [r["k"] for r in rows if not r["ok"]],
                       "target_occluded_heads": tgt.get("occluded_heads"), "target_w": tgt.get("target_w")},
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
