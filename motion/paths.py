"""A head path: where the head goes, which way it faces, and what the eye looks at (numpy only).

A path spec is a JSON file (`paths/*.json`, schema foveal-motion-path-v1):

    scene            the .blend, relative to the repository root (the CLI's --blend overrides it)
    start_w, end_w   the head origin's first and last positions, world metres
    n_looks          looks along the straight segment, both ends included (n_looks - 1 equal steps)
    facing_w         the head's forward direction, horizontal; up is world +Z
    target_w         the world point the "target" gaze fixates
    preflight        the scene gate's limits along the path: min_clearance_m, max_hole_fraction,
                     max_backface_fraction_other, backface_allowed, and control_head_w, a head that
                     must FAIL the preflight (rule 2)
    source, notes    where every number comes from

Gaze modes (the same path is run once per mode):

    target   Gaze.verged_point(target_w): the cyclopean eye stays on a world point while the head moves
    ahead    Gaze.yaw_pitch(0, 0): the eye looks along the head's forward axis (the focus of expansion
             of a straight path sits at the fovea's centre)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from fov_render import Gaze, HeadPose

SCHEMA = "foveal-motion-path-v1"
MODES = ("target", "ahead")
REQUIRED = ("name", "scene", "start_w", "end_w", "n_looks", "facing_w", "target_w", "preflight", "source")
PREFLIGHT = ("min_clearance_m", "max_hole_fraction", "max_backface_fraction_other", "backface_allowed",
             "control_head_w")


class PathError(ValueError):
    def __init__(self, reasons):
        self.reasons = list(reasons)
        super().__init__("; ".join(self.reasons))


def _vec3(x) -> bool:
    return isinstance(x, (list, tuple)) and len(x) == 3 and all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in x)


def validate(spec: dict) -> list[str]:
    """Every reason the spec is refused (empty when it is accepted)."""
    out = []
    if spec.get("schema") != SCHEMA:
        out.append(f"schema must be {SCHEMA!r}")
    out += [f"missing {k}" for k in REQUIRED if k not in spec]
    for k in ("start_w", "end_w", "facing_w", "target_w"):
        if k in spec and not _vec3(spec[k]):
            out.append(f"{k}: three finite numbers")
    n = spec.get("n_looks")
    if "n_looks" in spec and not (isinstance(n, int) and n >= 2):
        out.append("n_looks: an integer >= 2")
    if _vec3(spec.get("facing_w")):
        f = np.asarray(spec["facing_w"], float)
        if abs(f[2]) > 1e-12 or np.linalg.norm(f[:2]) == 0:
            out.append("facing_w: horizontal and nonzero (the head stays level)")
    if _vec3(spec.get("start_w")) and _vec3(spec.get("end_w")) and spec["start_w"] == spec["end_w"]:
        out.append("start_w equals end_w: no motion")
    pf = spec.get("preflight", {})
    if "preflight" in spec:
        out += [f"preflight.{k} missing" for k in PREFLIGHT if k not in pf]
        if "control_head_w" in pf and not _vec3(pf["control_head_w"]):
            out.append("preflight.control_head_w: three finite numbers")
    return out


def load(path) -> dict:
    spec = json.loads(Path(path).read_text())
    bad = validate(spec)
    if bad:
        raise PathError(bad)
    return spec


def head_axes(facing_w) -> np.ndarray:
    """R_wh for a level head facing `facing_w`: columns +X right, +Y up (world +Z), +Z backward."""
    f = np.asarray(facing_w, dtype=np.float64)
    f = f / np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    return np.stack([right, up, -f], axis=1)


def positions(spec: dict) -> np.ndarray:
    """(n_looks, 3) head origins, world metres, evenly spaced from start_w to end_w."""
    a, b = np.asarray(spec["start_w"], float), np.asarray(spec["end_w"], float)
    t = np.linspace(0.0, 1.0, int(spec["n_looks"]))
    return a[None, :] + t[:, None] * (b - a)[None, :]


def heads(spec: dict) -> list[HeadPose]:
    r = head_axes(spec["facing_w"])
    return [HeadPose(r, p) for p in positions(spec)]


def control_head(spec: dict) -> HeadPose:
    return HeadPose(head_axes(spec["facing_w"]), spec["preflight"]["control_head_w"])


def step_m(spec: dict) -> float:
    a, b = np.asarray(spec["start_w"], float), np.asarray(spec["end_w"], float)
    return float(np.linalg.norm(b - a) / (int(spec["n_looks"]) - 1))


def gaze(spec: dict, mode: str) -> Gaze:
    if mode == "target":
        return Gaze.verged_point(spec["target_w"])
    if mode == "ahead":
        return Gaze.yaw_pitch(0.0, 0.0)
    raise ValueError(f"gaze mode must be one of {MODES}, got {mode!r}")


def selftest() -> list[tuple[str, bool]]:
    """Checks that need nothing but numpy; each has a case that must be refused or differ."""
    spec = {"schema": SCHEMA, "name": "t", "scene": "x.blend", "start_w": [0.897, -3.793, 1.6],
            "end_w": [0.897, 0.759, 1.6], "n_looks": 47, "facing_w": [0, 1, 0], "target_w": [0.897, 3.259, 1.334],
            "source": "selftest", "preflight": {"min_clearance_m": 0.25, "max_hole_fraction": 0.02,
                                                "max_backface_fraction_other": 0.001, "backface_allowed": [],
                                                "control_head_w": [0.897, 3.159, 1.6]}}
    h = heads(spec)
    h0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # foveal-playground's H0 axes (facing +Y)
    fwd = [hh.dir_to_world([0.0, 0.0, -1.0]) for hh in h]
    res = [
        ("accepted spec", validate(spec) == []),
        ("facing +Y gives the playground's H0 axes", bool(np.array_equal(head_axes([0, 1, 0]), h0))),
        ("first and last heads are start_w and end_w", bool(np.allclose(h[0].origin_w, spec["start_w"], 0, 1e-15)
                                                            and np.allclose(h[-1].origin_w, spec["end_w"], 0, 1e-12))),
        ("equal steps", bool(np.allclose(np.linalg.norm(np.diff(positions(spec), axis=0), axis=1), step_m(spec)))),
        ("every head faces facing_w", all(np.allclose(f, [0, 1, 0]) for f in fwd)),
        ("refused: tilted facing", bool(validate(dict(spec, facing_w=[0, 1, 0.1])))),
        ("refused: n_looks 1", bool(validate(dict(spec, n_looks=1)))),
        ("refused: no control head", bool(validate(dict(spec, preflight={k: v for k, v in spec["preflight"].items()
                                                                           if k != "control_head_w"})))),
        ("refused: no motion", bool(validate(dict(spec, end_w=spec["start_w"])))),
        ("modes differ", gaze(spec, "target") != gaze(spec, "ahead")),
    ]
    return res
