"""Active Bootstrap-1b: declared constants, the pinned AB1a source and the spherical epipolar configuration.

Contract: docs/active-bootstrap/ab1b-spherical-epipolar-geometry-contract.md.  Standard library only.  The checker keeps
its own literal copies.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
REPO = TOOLS.parent
SHARED = REPO   # foveal-playground: was the f3d-vision workstation checkout; the source records
                #   below belong to the original AB1a/AB1b experiments and the playground never reads them
BASE_COMMIT = "cbdc4eb4d5cf005805ab26ad5a1863a825ad48a5"   # accepted main after the AB1a closure
EXPERIMENT = "Active Bootstrap-1b: Gaze-Centered Spherical Epipolar Geometry"
CONTRACT = "docs/active-bootstrap/ab1b-spherical-epipolar-geometry-contract.md"
MARKER = "ACTIVE_BOOTSTRAP1B_SPHERICAL_EPIPOLAR_GEOMETRY_COMPLETE"

TRUTH_ORACLE, TRUTH_DERIVED, TRUTH_REFERENCE = "ORACLE INPUT", "DERIVED", "REFERENCE / EVALUATION"

# ---- section 2: the accepted AB1a observation, re-analysed (read-only; full hashes from the accepted AB1a run)
AB1A_RUN = SHARED / "previews/active-bootstrap/ab1a-first-natural-stereo-look"
AB1A_VISUALS_MANIFEST = (SHARED / "visuals/active-bootstrap/ab1a-first-natural-stereo-look/visuals-manifest.json",
                         "d38bfdc2b9158739cc635f3cbc5ba4a764e6234c5397ba50ca99780987b22da0")
AB1A_PINS = {
    "manifest.json": "74eba4d6112f2de69fd797f9f41ff987e128ce1ecbcb5d48f05f597e9a2731fe",
    "process-log.jsonl": "4ff1056857455329386fb639e79271d0b2678ef9c3d7c29590e10508b58f7ae7",
    "acquisition/calibration.json": "9960c86e0fda1c002ae67e702d34ea3399bff3180d61bf71ac535563f8fc3913",
    "acquisition/rgb-observation.npz": "eb84831aee9676e011651821197812c81e7d51d76ab2cecfea609d3a8dd4268d",
    "evaluation_only/reference-observation.npz": "f23a7a53cdb7816ee4161337551d90825026a4a25fd8a57ba4981abaa99cdb08",
    "source/nb1c-action-manifest.json": "9fc65ef731601f44c67c141467f613a90c2d742007c79fc8090e9b530f0afe15",
    "measurement/measurement-freeze.json": "53dc5ff2985cfda4ed0ef0ed13f10be12a33e1c7693942dcf9101a2d1448243e",
    "evaluation_only/instance-catalog.json": "3924b5b4bde3e694d99e3c86b351be0b7f23d447ae6eda917619462b92732c1c",
    "measurement/stereo-summary.json": "166955c0e01f0f3e86f221e7e842e33f0162432949d3e05e42370e09bd7dbaf0",
    "evaluation/evaluation-summary.json": "5ed406369a4df597d943a01c8be9c393ef44ec60061bbae461b460643bab8e23",
    "prelook/prelook-geometry.json": "e65a85b433eb74e5d0592cc6236ccede43335966916026d6bc882b80eb783b8a",
}
# hashed by ``source`` (before the geometry freeze); the others are only checked against the AB1a manifest then
PRE_FREEZE_HASHED = ("manifest.json", "process-log.jsonl", "acquisition/calibration.json",
                     "acquisition/rgb-observation.npz", "evaluation_only/reference-observation.npz",
                     "source/nb1c-action-manifest.json", "measurement/measurement-freeze.json")
POST_FREEZE_COMPARISON = ("measurement/stereo-summary.json", "evaluation/evaluation-summary.json",
                          "prelook/prelook-geometry.json")
CATALOG_REL = "evaluation_only/instance-catalog.json"
CALIBRATION_REL = "acquisition/calibration.json"
REFERENCE_REL = "evaluation_only/reference-observation.npz"
RGB_REL = "acquisition/rgb-observation.npz"

GAZE_RANK, GAZE_ROW, GAZE_COL = 1, 164, 513
GAZE_YAW_DEG, GAZE_PITCH_DEG = 76.75, 7.75
ACTION_SOURCE = "frozen NB1c RGB gaze #1 (executed once by AB1a; re-analysed, not re-executed)"

# ---- section 3: the measurement domain (the original raw left nominal core)
RAW_SIZE = 640
CORE_SIZE = 256
CORE_ORIGIN = (RAW_SIZE - CORE_SIZE) // 2          # 192: raw x, y = 192..447
FOCAL_PX = 1217.8386501404907
IPD_M = 0.063
BASELINE_AXIS = (1.0, 0.0, 0.0)
TANGENT_FRAME = "baseline_projected"
PROFILE = "full"

# ---- section 5: the oracle correspondence stage
REFERENCE_KEYS = ("instance_L", "instance_R", "position_w_L", "position_w_R")
PRODUCT_KEYS = ("left_core_row", "left_core_col", "uv_L", "uv_R")
FORBIDDEN_TOKENS = ("xyz", "position", "range", "depth", "instance", "object", "normal", "semantic", "label", "truth",
                    "world")
RIGHT_Z_MIN = 1e-9
ORACLE = {
    "name": "AB1b perfect / oracle correspondence (accepted Classroom-Oracle-1 visibility semantics, raw rasters)",
    "domain": "left raw nominal core: raw x, y = 192..447 (65,536 pixel centres); right: the full padded 640 x 640 raster",
    "left_rule": "Position finite and not all zero (a geometric hit) and instance_L > 0",
    "projection": "P_h = fsg_geometry.world_to_head(P_w); fsg_geometry.project_h(right eye, P_h)",
    "right_projectable": "finite uv_R and z_R > 1e-9",
    "inside": "0 <= u_R <= 639 and 0 <= v_R <= 639 (closed, continuous)",
    "visibility": "instance_R[rint(v_R), rint(u_R)] == instance_L",
    "correspondence": "uv_L = raw left pixel centre; uv_R = exact continuous right projection (not rounded)",
    "not_used": ["SGBM", "RGB search", "depth thresholds", "the [0.75, 4.5] m interval", "planar rectification"],
    "product_keys": list(PRODUCT_KEYS),
}

# ---- sections 6-9: the spherical geometry stage
POLE_EPS = 1e-12            # sqrt(dy^2 + dz^2) below this: the ray lies on the baseline axis; phi undefined
DEN_EPS = 1e-12             # |cot theta_L - cot theta_R| below this: no epipolar triangulation (numerical only)
PARALLEL_EPS = 1e-15        # 1 - (d_L . d_R)^2 below this: no ray-ray triangulation (numerical only)
GEOMETRY = {
    "name": "AB1b gaze-centered spherical epipolar geometry (truth-free)",
    "inputs": ["AB1a acquisition/calibration.json", "oracle/oracle-correspondences.npz"],
    "rays": "fsg_geometry.rays_h (accepted, read-only)",
    "baseline_axis": list(BASELINE_AXIS),
    "baseline_m": "calibration ipd_m (eye centres at -B/2, +B/2 on head X)",
    "theta": "atan2(sqrt(dy^2 + dz^2), dx), [0, pi]",
    "phi": "atan2(dy, -dz), (-pi, pi]; NaN when sqrt(dy^2 + dz^2) < pole_eps",
    "wrap": "atan2(sin a, cos a), (-pi, pi]",
    "delta_theta": "theta_R - theta_L",
    "phi_residual": "wrap(phi_R - phi_L)",
    "epipolar": "rho = B / (cot theta_L - cot theta_R); x = -B/2 + rho cot theta_L; "
                "phi_bar = atan2(sin phi_L + sin phi_R, cos phi_L + cos phi_R); y = rho sin phi_bar; z = -rho cos phi_bar",
    "ray_ray": "closest points of o_L + s d_L and o_R + t d_R (least squares); midpoint; gap",
    "conditioning": "gamma = acos(clamp(d_L . d_R)); kappa = 1 / |sin gamma|; "
                    "range_per_px = B sin(theta_L) / sin(delta_theta)^2 / f",
    "chart": "chart_u = theta - theta_g; chart_v = sin(theta_g) wrap(phi - phi_g); d_g = gaze_direction(calibration gaze)",
    "pole_eps": POLE_EPS, "den_eps": DEN_EPS, "parallel_eps": PARALLEL_EPS,
    "thresholds": "none (no range, disparity, sign or conditioning threshold)",
    "not_used": ["Position", "Object Index", "truth XYZ", "cv2", "fsg_stereo", "planar rectification", "SGBM"],
}

# ---- sections 9 and 13: statistics (descriptive)
QUANTILES = {"min": 0.0, "p05": 0.05, "median": 0.5, "p90": 0.90, "p95": 0.95, "p99": 0.99, "max": 1.0}
ERROR_FRACTIONS_M = (0.001, 0.005, 0.010, 0.025)
MIN_INSTANCE_PAIRS = 50

# ---- section 11: synthetic known-answer tolerances (synthetic data only)
SYN_EXACT_M = 1e-9
SYN_PHI_EXACT = 1e-12
SYN_NEAR_POLE_M = 1e-6
SYN_SENSITIVITY_REL = 0.01
SYN_E2E_MEDIAN_M = 1e-4
SYN_E2E_MAX_M = 5e-4
SYN_E2E_PHI = 1e-6

# ---- accepted sources reused read-only (sha256 at the AB1b base); a mismatch is a STOP
SOURCE_PINS = {
    "tools/fsg_geometry.py": "ae3779bc5cbe8f9702227ae9f123c6be1abe394a0a3c617627f38f171c556c54",
    "tools/natural_bootstrap/nb1a_guard.py": "29ebca57be9e977def7a94ea45a7b0cff0f22a8b1d86b40f740f132491ebe99d",
    "tools/visual_language/style.py": "c379a2a31c9112d22bbe8989324d2ff74608c669fe84d58905509b4d62264053",
    "tools/classroom_oracle/breadth1_visuals.py": "61683aea17993da09bfddce4ed13d48da47d710d1630e40c6bdcc7a6f6d2144c",
}

# ---- section 18: run layout
RUN_DEFAULT = SHARED / "previews/active-bootstrap/ab1b-spherical-epipolar-geometry"
VIS_DEFAULT = SHARED / "visuals/active-bootstrap/ab1b-spherical-epipolar-geometry"
ORACLE_CODE = ["tools/active_bootstrap/ab1b_oracle.py", "tools/active_bootstrap/ab1b_spec.py", "tools/fsg_geometry.py",
               "tools/natural_bootstrap/nb1a_guard.py"]
GEOMETRY_CODE = ["tools/active_bootstrap/ab1b_geometry.py", "tools/active_bootstrap/ab1b_spec.py",
                 "tools/fsg_geometry.py", "tools/natural_bootstrap/nb1a_guard.py"]
FROZEN_CORRESPONDENCE = ["source/ab1a-source-manifest.json", "oracle/oracle-correspondences.npz",
                         "oracle/oracle-summary.json", "oracle/oracle-opened-files.json"]
FROZEN_GEOMETRY = ["oracle/correspondence-freeze.json", "oracle/oracle-correspondences.npz",
                   "geometry/left-core-rays.npz", "geometry/epipolar-result.npz", "geometry/geometry-summary.json",
                   "geometry/geometry-opened-files.json"]
COMMANDS = ("synthetic", "source", "oracle", "freeze-correspondence", "geometry", "freeze-geometry", "evaluate",
            "visualize")


def config_sha256(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
