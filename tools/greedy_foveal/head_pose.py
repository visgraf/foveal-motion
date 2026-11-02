"""The fixed head H0: its declared pose, and the check that the scene's EYE camera still matches it.

foveal-playground replacement for the f3d-vision head-pose source.  In f3d-vision the render server took the head pose
from the EYE object of ``scenes/classroom/classroom_eye.blend`` and required it equal, within 1e-6, to an accepted
calibration record on the project workstation (``ns1a_render.eye_pose(strict=True)``).  That record is not
distributed.  Here the same pose is declared as constants, and the same tolerance is applied.

The pose the explorer *uses* is still the one Blender reads from the EYE object, exactly as before, so the calibrations
it builds are unchanged.  The constants only guard against a wrong or moved EYE: an EYE placed by

    blender -b classroom.blend -P tools/classroom_oracle/place_eye.py -- --location -0.6 -1.0 1.2 --yaw 0 ...

gives head origin (-0.6, -1.0, 1.2) m in world coordinates and head axes +X right (world +X), +Y up (world +Z),
-Z forward (world +Y).  ``p_H = (p_W - o) . R_wh`` (``fsg_geometry.world_to_head``), so the columns of R_wh are the
head axes expressed in world coordinates.

Standard library only (plus numpy inside ``eye_pose``), so both interpreters can import it: Blender's and the host
``.venv``.

A project that moves the head (see docs/exercises.md) replaces this module; it is the one place the fixed head is
declared.
"""
from __future__ import annotations

H0_ORIGIN_W_M = (-0.6, -1.0, 1.2)
H0_R_WH = ((1.0, 0.0, 0.0),
           (0.0, 0.0, -1.0),
           (0.0, 1.0, 0.0))
EYE_POSE_TOL = 1e-6          # as in f3d-vision (ab1a_spec / ns1a_spec)
SOURCE = ("EYE object of scenes/classroom/classroom_eye.blend, "
          "checked against tools/greedy_foveal/head_pose.py H0 within 1e-6")


def max_abs_diff(r, o) -> dict:
    """Largest absolute deviation of a head pose (R_wh, origin) from the declared H0."""
    dr = max(abs(float(r[i][j]) - H0_R_WH[i][j]) for i in range(3) for j in range(3))
    do = max(abs(float(o[i]) - H0_ORIGIN_W_M[i]) for i in range(3))
    return {"R": dr, "origin": do}


def matches_h0(r, o) -> bool:
    d = max_abs_diff(r, o)
    return max(d["R"], d["origin"]) <= EYE_POSE_TOL


def eye_pose(strict: bool = True) -> dict:
    """Blender side: the EYE pose of the open scene, required equal to H0 within EYE_POSE_TOL."""
    import classroom_oracle1_render as CO
    _eye, note, o, r = CO._head_pose()
    d = max_abs_diff(r, o)
    if strict and max(d["R"], d["origin"]) > EYE_POSE_TOL:
        raise RuntimeError(f"STOP the EYE pose differs from the declared head H0: |dR| {d['R']}, |do| {d['origin']}")
    return {"eye_note": note, "head_R_wh": r.tolist(), "head_origin_w_m": o.tolist(),
            "head_pose_source": SOURCE, "max_abs_diff_from_h0": d}


if __name__ == "__main__":
    # self-check: the declared R_wh is a rotation, and the check can fail
    import itertools
    R = H0_R_WH
    ortho = all(abs(sum(R[k][i] * R[k][j] for k in range(3)) - (1.0 if i == j else 0.0)) < 1e-15
                for i, j in itertools.product(range(3), repeat=2))
    det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
           + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
    ok = [ortho, abs(det - 1.0) < 1e-15, matches_h0(R, H0_ORIGIN_W_M),
          not matches_h0(R, (H0_ORIGIN_W_M[0] + 1e-5, H0_ORIGIN_W_M[1], H0_ORIGIN_W_M[2])),
          not matches_h0(((1, 0, 0), (0, 1, 0), (0, 0, 1)), H0_ORIGIN_W_M)]
    print(f"[head-pose] orthonormal {ok[0]} det+1 {ok[1]} H0 accepted {ok[2]} "
          f"moved origin rejected {ok[3]} wrong axes rejected {ok[4]}")
    raise SystemExit(0 if all(ok) else 1)
