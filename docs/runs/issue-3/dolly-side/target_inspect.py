"""What the side path's target_ray found (issue #3, Code, workstation): the object, what lies behind it, two views.

Blender side (facts and two 24-degree views from heads 0 and 46, centred on the target, 32 spp, OPTIX):

    blender -b scenes/classroom/classroom_eye.blend --python-exit-code 1 \
        -P docs/runs/issue-3/dolly-side/target_inspect.py -- runs/dolly-side/preflight.json OUT_DIR

Host side (marks the target and a 2-degree circle, writes target_views.jpg next to this script):

    .venv/bin/python docs/runs/issue-3/dolly-side/target_inspect.py --mark OUT_DIR
"""
import json
import math
import sys
from pathlib import Path

VIEW_DEG, SIZE, KS = 24.0, 480, (0, 46)


def blender_side(pre: Path, out: Path) -> None:
    import bpy
    import numpy as np
    from mathutils import Vector
    pf = json.loads(pre.read_text())
    tgt = pf["target"]
    T = np.asarray(tgt["target_w"], float)
    heads = [np.asarray(h["head_origin_w_m"], float) for h in pf["heads"]]
    sc = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    o = bpy.data.objects[tgt["hit_object"]]
    bb = np.array([o.matrix_world @ Vector(c) for c in o.bound_box])
    print(f"[target] hit_object {o.name_full} ({o.type}), world bbox {bb.min(0).round(3).tolist()} .. "
          f"{bb.max(0).round(3).tolist()}, size {(bb.max(0) - bb.min(0)).round(3).tolist()} m")
    print(f"[target] target_w {T.round(4).tolist()}: {T[2] - bb.min(0)[2]:.3f} m above its lower edge, "
          f"{T[0] - bb.min(0)[0]:.3f} m right of its left edge")
    org = heads[tgt["ray"]["from_k"]]
    d = np.asarray(tgt["ray"]["toward_w"], float) - org
    d /= np.linalg.norm(d)
    p = org.copy()
    for i in range(4):                       # the same ray, through every surface (flagged or not)
        hit, loc, nrm, _f, ob, _m = sc.ray_cast(dg, Vector(p.tolist()), Vector(d.tolist()))
        if not hit:
            print(f"[target] ray surface {i}: leaves the scene")
            break
        loc = np.asarray(loc)
        print(f"[target] ray surface {i}: {getattr(ob, 'original', ob).name_full} at {loc.round(3).tolist()}, "
              f"{float(np.dot(loc - org, d)):.3f} m from head {tgt['ray']['from_k']}, normal {np.round(list(nrm), 3).tolist()}")
        p = loc + 1e-4 * d
    pr = bpy.context.preferences.addons["cycles"].preferences
    pr.compute_device_type = "OPTIX"
    pr.get_devices()
    for dev in pr.devices:
        dev.use = dev.type == "OPTIX"
    sc.render.engine, sc.cycles.device, sc.cycles.samples = "CYCLES", "GPU", 32
    sc.render.resolution_x = sc.render.resolution_y = SIZE
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    cam = bpy.data.objects.new("TARGET_VIEW", bpy.data.cameras.new("TARGET_VIEW"))
    sc.collection.objects.link(cam)
    cam.data.sensor_fit, cam.data.angle, sc.camera = "HORIZONTAL", math.radians(VIEW_DEG), cam
    out.mkdir(parents=True, exist_ok=True)
    for k in KS:
        cam.location = Vector(heads[k].tolist())
        cam.rotation_euler = (Vector(T.tolist()) - cam.location).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = str(out / f"view_k{k:02d}.png")
        bpy.ops.render.render(write_still=True)
        v = T - heads[k]
        print(f"[target] k={k}: {math.degrees(math.atan2(-v[0], v[1])):.1f} deg left of the heading, "
              f"{np.linalg.norm(v):.2f} m; wrote {sc.render.filepath}")


def mark(out: Path) -> None:
    from PIL import Image, ImageDraw
    pf = json.loads((Path("runs/dolly-side") / "preflight.json").read_text())
    rows = {r["k"]: r for r in pf["target"]["rows"]}
    sheet = Image.new("RGB", (SIZE * len(KS), SIZE))
    for i, k in enumerate(KS):
        im = Image.open(out / f"view_k{k:02d}.png").convert("RGB")
        d, c = ImageDraw.Draw(im), SIZE / 2
        for a, b in ((-14, -4), (4, 14)):
            d.line([c + a, c, c + b, c], fill=(0, 255, 0), width=2)
            d.line([c, c + a, c, c + b], fill=(0, 255, 0), width=2)
        r = SIZE / VIEW_DEG * 2.0
        d.ellipse([c - r, c - r, c + r, c + r], outline=(255, 255, 0), width=1)
        d.text((6, 4), f"k={k}, {rows[k]['range_m']:.2f} m ({VIEW_DEG:g} deg view; yellow: 2 deg around the target)",
               fill=(255, 255, 0))
        sheet.paste(im, (SIZE * i, 0))
    dst = Path(__file__).with_name("target_views.jpg")
    sheet.save(dst, quality=90)
    print(f"wrote {dst}")


if __name__ == "__main__":
    if "--mark" in sys.argv:
        mark(Path(sys.argv[sys.argv.index("--mark") + 1]))
    else:
        a = sys.argv[sys.argv.index("--") + 1:]
        blender_side(Path(a[0]), Path(a[1]))
