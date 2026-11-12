"""Where M2's foveal range comes from when the flow is sub-pixel (issue #3, Code, workstation).

    PYTHONPATH=. .venv/bin/python docs/runs/issue-3/same_pixel.py runs/dolly-side 1
    PYTHONPATH=. .venv/bin/python docs/runs/issue-3/same_pixel.py runs/dolly-1 10

For the pairs (k, k + stride) of every mode, as motion.depth.pair_depth matches them: among the foveal
(< 2 deg) samples of look b, the share whose point lands, in look a, on the SAME raster pixel as in b.
For those, the pixel-limited triangulation intersects the rays through one pixel from the two eye
centres: no flow was measured, and the range is where those rays cross, which the eye's rotation
between the looks decides (in `ahead` they are parallel: no range; fixating, they cross on the
surface of zero flow through the fixated point). The error of same-pixel and other samples is split.
"""
import sys

import numpy as np

from motion import analysis, depth


def main(run_dir: str, stride: int) -> None:
    run = analysis.load_run(run_dir)
    for mode, entries in run["looks"].items():
        L = {e["k"]: analysis.load_look(run, e) for e in entries}
        same, rel, fov = [], [], []
        for ka in sorted(L):
            if ka + stride not in L:
                continue
            la, lb = L[ka], L[ka + stride]
            rec_a, Ra, ca, model = la
            rec_b = lb[0]
            r = depth.pair_depth(la, lb)
            hit = np.flatnonzero(np.isfinite(rec_b.distance_m))
            _, _, rc_a, _ = analysis._agree(rec_a, model, Ra, ca, rec_b.position_w[hit])
            ij = np.rint(rc_a).astype(np.int64)
            m = r["matched"]
            same.append((ij[:, 0] * model.n + ij[:, 1])[m] == rec_b.raster_index[hit[m]])
            rel.append(r["rel_pix"])
            fov.append(r["ecc"] < 2.0)
        same, rel, fov = (np.concatenate(x) for x in (same, rel, fov))

        def med(x):
            return "-" if not x.size else f"{np.median(x):.4f}"

        def usable(x):
            return "-" if not x.size else f"{np.mean(x <= depth.USABLE_REL):.3f}"

        s, o = rel[fov & same], rel[fov & ~same]
        print(f"{run_dir} {mode:6s} stride {stride:2d}: fovea {int(fov.sum())} samples, same pixel in a and b "
              f"{same[fov].mean():.3f}; median error same-pixel {med(s)}, other {med(o)}; usable same-pixel "
              f"{usable(s)}, other {usable(o)} | whole field same pixel {same.mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
