"""Blender side: check that the Classroom scene is ready for the explorer, without rendering.

    blender -b scenes/classroom/classroom_eye.blend --python-exit-code 1 -P tools/greedy_foveal/scene_preflight.py

Checks: the EYE camera exists and sits at the declared head H0 (head_pose.py, 1e-6); every file the scene links
(libraries, images; a point-cache directory only if some cache is stored on disk) resolves to an existing file;
Cycles can see an OPTIX device (the explorer requests OPTIX and has
no fallback; see docs/exercises.md for a CPU path).  Exits nonzero on any failure, since ``blender -b -P`` alone may
exit 0 after a failed script.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent, HERE):
    sys.path.insert(0, str(_p))

PREFIX = "[scene-preflight]"


def optix_devices() -> list[str]:
    import bpy
    prefs = bpy.context.preferences.addons["cycles"].preferences
    try:
        prefs.compute_device_type = "OPTIX"
    except TypeError:
        return []
    prefs.refresh_devices()
    return [d.name for d in prefs.devices if d.type == "OPTIX"]


def point_caches() -> list:
    import bpy
    caches = []
    for ob in bpy.data.objects:
        caches += [ps.point_cache for ps in ob.particle_systems]
        for m in ob.modifiers:
            if getattr(m, "point_cache", None) is not None:                  # cloth, soft body
                caches.append(m.point_cache)
            if m.type == "DYNAMIC_PAINT" and m.canvas_settings is not None:
                caches += [s.point_cache for s in m.canvas_settings.canvas_surfaces]
    caches += [sc.rigidbody_world.point_cache for sc in bpy.data.scenes if sc.rigidbody_world is not None]
    return caches


def main() -> int:
    import bpy
    import head_pose as HP
    ok = True
    try:
        pose = HP.eye_pose(strict=True)
        print(f"{PREFIX} PASS EYE at H0: {pose['eye_note']}; max |dR| {pose['max_abs_diff_from_h0']['R']:.2e}, "
              f"|do| {pose['max_abs_diff_from_h0']['origin']:.2e} m", flush=True)
    except Exception as exc:                                         # noqa: BLE001  (report, then fail)
        ok = False
        print(f"{PREFIX} FAIL EYE: {exc}", flush=True)
    missing = sorted({p for p in bpy.utils.blend_paths(absolute=True, local=False) if not os.path.exists(p)})
    # blend_paths lists the default point-cache directory //blendcache_<file>/ for every point cache, even one baked
    # into the .blend (the Classroom's dust particles).  It is needed only when some cache stores frames on disk.
    caches = point_caches()
    in_blend = bool(caches) and not any(pc.use_disk_cache or pc.use_external for pc in caches)
    if in_blend:
        cache_dir = os.path.normpath(bpy.path.abspath(f"//blendcache_{Path(bpy.data.filepath).stem}"))
        missing = [p for p in missing if os.path.normpath(p) != cache_dir]
    print(f"{PREFIX} {'PASS' if not missing else 'FAIL'} linked files: {len(missing)} missing"
          + (f" ({len(caches)} point cache(s), all inside the .blend)" if in_blend else "")
          + "".join(f"\n{PREFIX}     missing {p}" for p in missing[:20]), flush=True)
    ok &= not missing
    dev = optix_devices()
    print(f"{PREFIX} {'PASS' if dev else 'FAIL'} OPTIX devices: {dev or 'none (the baseline needs an NVIDIA RTX GPU)'}",
          flush=True)
    ok &= bool(dev)
    print(f"{PREFIX} Blender {bpy.app.version_string} (the baseline was measured with 5.2.1)", flush=True)
    print(f"{PREFIX} {'READY' if ok else 'NOT READY'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        rc = main()
    except BaseException:
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    os._exit(rc)
