"""Greedy Foveal Explorer, baseline: the host-side pieces of FIXATE -> RECONSTRUCT -> SACCADE.

Part of foveal-playground.  Architecture: docs/architecture.md.  This file is unchanged in behavior from the
f3d-vision repository, tag playground-2-sparse-lattice (see docs/provenance.md).  Contract paths (docs/playground/...)
and tag names below refer to f3d-vision.

- the 1-degree whole-sphere coverage grid (UNSEEN / SEEN / DEPTH, exact cell solid angles, longitude wrap);
- one fixation's reconstruction: the accepted AB1b PERFECT / ORACLE correspondence (``ab1b_oracle.compute_oracle``,
  unchanged, positive Object Index), EXTENDED to Object Index 0 geometry (see ``perfect_correspondence``), then the
  accepted AB1b spherical epipolar geometry (``ab1b_geometry.compute_epipolar``) in canonical fixed-head H0;
- one global H0 surfel map with the accepted FSG3 12-mm association / fusion rule (``fsg3_surface_map.fuse``)
  re-implemented with a vectorized spatial hash, WITHOUT the instance filter and without the 63-patch provenance cap;
- the greedy policy, restricted to a fixed gaze lattice (Playground-1, docs/playground/gaze-lattice-contract.md).  The
  lattice is Playground-2's (docs/playground/sparse-lattice-contract.md): 17 rings about the eye baseline (+X), whose
  12-deg footprints sit square-on to them and cover every 1-deg cell with 356 gazes; LATTICE_KIND = "pitch" rebuilds
  Playground-1's 496-gaze pitch rings.  Local: the unvisited lattice neighbours within
  16 deg, scored EDGE_SUPPORT * UNSEEN_GAIN, eligible at any UNSEEN_GAIN > 0; else a global saccade to the unvisited
  lattice gaze with the largest UNSEEN_GAIN; else stop (every cell SEEN).  The official baseline's 8 tangent-frame
  neighbours and deepest-cell jump are at tag greedy-foveal-playground-baseline.

No object identity enters gaze, fusion, switching or stopping.  The oracle Object Index is carried on the map for an
ORACLE / VISUALIZATION-ONLY segmentation PLY.
"""
from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "active_bootstrap", HERE.parent / "natural_bootstrap", HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ab1b_geometry as BG  # noqa: E402  (accepted AB1b spherical geometry, read-only)
import ab1b_oracle as O  # noqa: E402  (accepted AB1b perfect / oracle correspondence, read-only)
import fsg_geometry as FG  # noqa: E402  (accepted sensor geometry, read-only)
from exr_lite import read_uncompressed_exr  # noqa: E402

# ------------------------------------------------------------------ parameters (chosen once, before any run)
CORE_FOV_DEG = FG.CORE_FOV_DEG                 # 12 deg nominal square core (accepted sensor)
CORE_SIZE, CORE_ORIGIN = O.SP.CORE_SIZE, O.SP.CORE_ORIGIN
LOCAL_STEP_DEG = 0.6 * CORE_FOV_DEG            # 7.2 deg: the baseline's neighbour offset (recorded, unused here)
EDGE_BAND_PX = CORE_SIZE // 4                  # the outer quarter of the core in a candidate's direction
EDGE_MIN = 0.25                                # "nontrivial" valid-geometry fraction of that band
GAIN_MIN = 0.0                                 # local eligibility: UNSEEN gain strictly above this (baseline 0.25)
VISIT_TOL_DEG = 2.0                            # an "effectively identical" fixation
SEEN_STOP = 2.0                                # unreachable on purpose: the lattice run stops when decide() is
                                               # "none" (every cell SEEN), so run.py records that reason
LATTICE_KIND = "baseline"                      # Playground-2 baseline rings; "pitch" = Playground-1's pitch rings
LATTICE_RING_DEG = 10.0                        # Playground-1: pitch-ring spacing
LATTICE_CAP_DEG = 6.0                          # Playground-2: first ring's angle from the baseline axis (the envelope)
LATTICE_AXIS_STEP_DEG = 10.5                   # Playground-2: spacing of the baseline rings (6, 16.5, ..., 90, ..., 174)
LATTICE_NEIGHBOUR_DEG = 16.0                   # local candidates: unvisited lattice gazes this close to the gaze
LATTICE_VISIT_DEG = 0.5                        # a lattice gaze is visited when a fixation lies this close to it
ASSOC_RADIUS_M = 0.012                         # the accepted 12-mm association radius ...
ASSOC_CELL_M = 0.012                           # ... and hash cell (FSG3 / NS1a persistence)
ZERO_TOL_ABS_M, ZERO_TOL_REL = 0.005, 0.01     # instance-0 extension: right pixel sees the same surface point
GRID_DEG = 1.0
H, W = 180, 360
UNSEEN, SEEN, DEPTH = 0, 1, 2
NEIGHBOURS = (("E", 1, 0), ("W", -1, 0), ("N", 0, -1), ("S", 0, 1),        # image +x = right, +y = down
              ("NE", 1, -1), ("NW", -1, -1), ("SE", 1, 1), ("SW", -1, 1))


# ------------------------------------------------------------------ the sphere grid
def row_weights() -> np.ndarray:
    """Exact solid angle of one 1-degree cell in each row (row 0 at the top, +90 deg)."""
    dlam = math.radians(GRID_DEG)
    j = np.arange(H, dtype=np.float64)
    hi = np.radians(90.0 - j * GRID_DEG)
    lo = np.radians(90.0 - (j + 1) * GRID_DEG)
    return dlam * (np.sin(hi) - np.sin(lo))


ROW_W = row_weights()
CELL_W = np.repeat(ROW_W[:, None], W, axis=1)          # (H, W) solid angle per cell
SPHERE_SR = float(CELL_W.sum())                         # 4 pi


def direction(yaw_deg, pitch_deg) -> np.ndarray:
    """Head-frame unit direction of (yaw, pitch) (accepted fsg_geometry.gaze_direction, vectorized)."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.stack([np.sin(y) * np.cos(p), np.sin(p), -np.cos(y) * np.cos(p)], axis=-1)


def yaw_pitch(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(d, np.float64)
    n = np.linalg.norm(d, axis=-1)
    return (np.degrees(np.arctan2(d[..., 0], -d[..., 2])),
            np.degrees(np.arcsin(np.clip(d[..., 1] / n, -1.0, 1.0))))


def cell_of(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yaw, pitch = yaw_pitch(d)
    r = np.clip(np.floor((90.0 - pitch) / GRID_DEG), 0, H - 1).astype(np.int64)
    c = (np.floor((yaw + 180.0) / GRID_DEG).astype(np.int64)) % W
    return r, c


_rr, _cc = np.mgrid[:H, :W]
CELL_YAW = -180.0 + (_cc + 0.5) * GRID_DEG
CELL_PITCH = 90.0 - (_rr + 0.5) * GRID_DEG
CELL_DIR = direction(CELL_YAW, CELL_PITCH).reshape(-1, 3)   # (H*W, 3)


def frame(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Columns (x, y, z) of the gaze's local tangent frame: the accepted baseline-projected camera convention
    (x = head +X projected into the tangent plane, y = z cross x = image down, z = gaze)."""
    return FG.camera_rotation_h(FG.gaze_direction(yaw_deg, pitch_deg), np.array([1.0, 0.0, 0.0]))


def footprint(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Flat cell indices whose centres fall inside the nominal 12-deg square core of the gaze (cyclopean, H0)."""
    r = frame(yaw_deg, pitch_deg)
    t = math.tan(math.radians(CORE_FOV_DEG / 2))
    dz = CELL_DIR @ r[:, 2]
    near = np.flatnonzero(dz > math.cos(math.radians(CORE_FOV_DEG)))
    d = CELL_DIR[near]
    u, v = (d @ r[:, 0]) / dz[near], (d @ r[:, 1]) / dz[near]
    return near[(np.abs(u) <= t) & (np.abs(v) <= t)]


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.asarray(a) @ np.asarray(b).T, -1.0, 1.0)))


# ------------------------------------------------------------------ one observation -> local H0 geometry
def load_observation(fix_dir: Path) -> dict:
    """The rendered pair split into the ORACLE AID (Position, Object Index) and the sensory left RGB."""
    out = {}
    for s in ("L", "R"):
        ch = read_uncompressed_exr(str(fix_dir / f"raw_{s}.exr"))

        def one(suffix: str) -> np.ndarray:
            keys = [k for k in ch if k.endswith(suffix)]
            if len(keys) != 1:
                raise RuntimeError(f"expected one EXR channel ending {suffix!r}, got {keys}")
            return np.asarray(ch[keys[0]])

        out[f"position_w_{s}"] = np.stack([one(f"Position.{c}") for c in "XYZ"], -1).astype(np.float32)
        idx = np.rint(one("Object Index.X")).astype(np.int32)
        idx[idx < 0] = 0
        out[f"instance_{s}"] = idx
        if s == "L":
            out["rgb_L"] = np.stack([one(f"Combined.{c}") for c in "RGB"], -1).astype(np.float32)
    return out


def perfect_correspondence(c: dict, ref: dict) -> tuple[dict, np.ndarray, dict]:
    """PERFECT / ORACLE correspondence over ALL first-hit geometry of the left raw core.

    Positive Object Index: the accepted AB1b rule, unchanged (``compute_oracle``).  Object Index 0 (non-catalog
    geometry), which the accepted rule excludes: the same projection of the left Position into the right raw camera,
    kept when the nearest right pixel is a finite hit with Object Index 0 whose Position lies within
    5 mm + 1 % of range of the left point (the right eye sees the same surface point; occlusion otherwise).
    Returns the truth-stripped product (accepted keys), a per-pair source flag (1 catalog, 0 instance-0) and counts.
    """
    ref_acc = {k: ref[k] for k in O.SP.REFERENCE_KEYS}
    prod_cat, summ = O.compute_oracle(c, ref_acc)
    rows, cols = O.core_grid()
    n = rows.size
    v_raw, u_raw = rows + CORE_ORIGIN, cols + CORE_ORIGIN
    pos = np.asarray(ref["position_w_L"])[v_raw, u_raw].astype(np.float64)
    ids_l = np.asarray(ref["instance_L"])[v_raw, u_raw].astype(np.int64)
    hit = O.left_hit(pos)
    zero = hit & (ids_l == 0)
    p_h = FG.world_to_head(c, np.where(hit[:, None], pos, 0.0))
    uv_r, z_r = O.right_projection(c, p_h)
    ok = zero & O.right_projectable(uv_r, z_r) & O.inside_raster(c, uv_r)
    vi, ui = O.nearest_pixel(c, uv_r)
    pr = np.asarray(ref["position_w_R"])[vi, ui].astype(np.float64)
    rng = np.linalg.norm(p_h - np.asarray(c["eyes"][0]["centre_h_m"]), axis=-1)
    same = (O.left_hit(pr) & (np.asarray(ref["instance_R"])[vi, ui] == 0)
            & (np.linalg.norm(pr - pos, axis=-1) <= ZERO_TOL_ABS_M + ZERO_TOL_REL * rng))
    keep0 = ok & same
    flat_cat = prod_cat["left_core_row"].astype(np.int64) * CORE_SIZE + prod_cat["left_core_col"]
    keep_cat = np.zeros(n, bool)
    keep_cat[flat_cat] = True
    if np.any(keep_cat & keep0):
        raise RuntimeError("catalog and instance-0 correspondences overlap")
    uv_full = np.full((n, 2), np.nan)
    uv_full[flat_cat] = prod_cat["uv_R"]
    uv_full[keep0] = uv_r[keep0]
    keep = keep_cat | keep0
    prod = {"left_core_row": rows[keep].astype(np.int32), "left_core_col": cols[keep].astype(np.int32),
            "uv_L": np.stack([u_raw[keep], v_raw[keep]], axis=-1).astype(np.float64),
            "uv_R": uv_full[keep].astype(np.float64)}
    counts = {"core_pixels": int(n), "left_hits": int(hit.sum()), "catalog": int(keep_cat.sum()),
              "instance0": int(keep0.sum()), "instance0_hits": int(zero.sum()), "total": int(keep.sum())}
    return prod, keep_cat[keep].astype(np.int8), counts


def reconstruct(c: dict, obs: dict) -> dict:
    """Correspondence + accepted spherical geometry -> valid local H0 points (all instance ids)."""
    t0 = time.perf_counter()
    prod, source, counts = perfect_correspondence(c, obs)
    t1 = time.perf_counter()
    res = BG.compute_epipolar(c, prod)
    valid = np.asarray(res["valid_epi"], bool) & np.isfinite(res["P_epi"]).all(axis=-1)
    xyz = np.asarray(res["P_epi"], np.float64)[valid]
    u = prod["uv_L"][valid].astype(np.int64)
    rgb = np.asarray(obs["rgb_L"])[u[:, 1], u[:, 0]].astype(np.float32)
    inst = np.asarray(obs["instance_L"])[u[:, 1], u[:, 0]].astype(np.int32)    # ORACLE label, visualization only
    core_valid = np.zeros((CORE_SIZE, CORE_SIZE), bool)
    core_valid[prod["left_core_row"][valid], prod["left_core_col"][valid]] = True
    # diagnostic only (never a control input): triangulated point vs the left Position oracle aid
    truth = FG.world_to_head(c, np.asarray(obs["position_w_L"])[u[:, 1], u[:, 0]].astype(np.float64))
    err = np.linalg.norm(xyz - truth, axis=-1)
    counts.update(valid=int(valid.sum()), valid_catalog=int((source[valid] == 1).sum()),
                  valid_instance0=int((source[valid] == 0).sum()))
    return {"xyz": xyz, "rgb": rgb, "inst": inst, "core_valid": core_valid, "counts": counts,
            "err_m": err, "kappa": np.asarray(res["kappa"])[valid],
            "seconds": {"correspondence": t1 - t0, "geometry": time.perf_counter() - t1}}


# ------------------------------------------------------------------ the one global H0 surfel map
_OFF = np.int64(1 << 20)
OFFSETS = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], np.int64)


def _pack(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, np.int64) + _OFF
    return (k[..., 0] << 42) | (k[..., 1] << 21) | k[..., 2]


class GlobalMap:
    """One persistent H0 surfel map.  Association and fusion follow ``fsg3_surface_map.fuse`` exactly (snapshot
    association; nearest existing surfel strictly within the radius over the 27 neighbouring hash cells; all patch
    points on one surfel reduce to one patch contribution, averaged in with weight support; unmatched points become
    new surfels) except: no instance filter, no 63-patch cap (support counts only), and a vectorized hash."""

    def __init__(self, radius: float = ASSOC_RADIUS_M, cell: float = ASSOC_CELL_M) -> None:
        if radius > cell:
            raise ValueError("the 27-cell search is exact only for radius <= cell")
        self.radius, self.cell, self.n = float(radius), float(cell), 0
        self.xyz = np.zeros((0, 3), np.float64)
        self.rgb = np.zeros((0, 3), np.float64)
        self.inst = np.zeros(0, np.int32)
        self.support = np.zeros(0, np.int32)
        self.first_fix = np.zeros(0, np.int32)
        self.patch_ids: list[str] = []

    def _grow(self, k: int) -> None:
        need = self.n + k
        if need <= len(self.xyz):
            return
        cap = max(need, 2 * len(self.xyz), 1 << 16)
        for name, shape in (("xyz", (cap, 3)), ("rgb", (cap, 3)), ("inst", (cap,)), ("support", (cap,)),
                            ("first_fix", (cap,))):
            old = getattr(self, name)
            new = np.zeros(shape, old.dtype)
            new[:self.n] = old[:self.n]
            setattr(self, name, new)

    def nearest(self, p: np.ndarray, budget: int = 4_000_000) -> tuple[np.ndarray, np.ndarray]:
        """For each point: the index of the nearest surfel (smallest index on ties) and its distance; -1 / inf
        when no surfel lies within the 27 neighbouring cells.  Exact (two stages: own cell, then the 26 others only
        where the own-cell distance exceeds the distance to the cell boundary)."""
        p = np.asarray(p, np.float64)
        best_i = np.full(len(p), -1, np.int64)
        best_d = np.full(len(p), np.inf)
        if self.n == 0 or len(p) == 0:
            return best_i, best_d
        cell = self.cell
        kp = np.floor(p / cell).astype(np.int64)
        lo, hi = (kp.min(0) - 2) * cell, (kp.max(0) + 3) * cell      # one spare cell against rounding
        m = self.xyz[:self.n]
        idx = np.flatnonzero(np.all((m >= lo) & (m < hi), axis=1))
        if idx.size == 0:
            return best_i, best_d
        keys = _pack(np.floor(m[idx] / cell).astype(np.int64))
        order = np.argsort(keys, kind="stable")
        sk, cand = keys[order], idx[order]

        def search(js: np.ndarray, offsets: np.ndarray) -> None:
            for o in offsets:
                q = _pack(kp[js] + o)
                a = np.searchsorted(sk, q, "left")
                cnt = np.searchsorted(sk, q, "right") - a
                nz = cnt > 0
                jn, an, cn = js[nz], a[nz], cnt[nz]
                if jn.size == 0:
                    continue
                cs = np.cumsum(cn)
                bounds = np.searchsorted(cs, np.arange(budget, cs[-1] + budget, budget), "left") + 1
                start = 0
                for stop in np.unique(np.minimum(bounds, len(jn))):
                    if stop <= start:
                        continue
                    jj, aa, cc = jn[start:stop], an[start:stop], cn[start:stop]
                    tot = int(cc.sum())
                    seg = np.cumsum(cc) - cc
                    within = np.arange(tot) - np.repeat(seg, cc)
                    mi = cand[np.repeat(aa, cc) + within]
                    pj = np.repeat(jj, cc)
                    d = np.linalg.norm(p[pj] - m[mi], axis=1)
                    segmin = np.minimum.reduceat(d, seg)
                    # first (smallest-index) candidate attaining the segment minimum
                    hitmask = d == np.repeat(segmin, cc)
                    grp = np.repeat(np.arange(len(jj)), cc)[hitmask]
                    first = np.unique(grp, return_index=True)[1]
                    arg_i = mi[hitmask][first]
                    better = segmin < best_d[jj]
                    tie = (segmin == best_d[jj]) & (arg_i < best_i[jj])
                    upd = better | tie
                    best_d[jj[upd]] = segmin[upd]
                    best_i[jj[upd]] = arg_i[upd]
                    start = stop

        allj = np.arange(len(p))
        search(allj, OFFSETS[13:14])                         # own cell (0, 0, 0)
        frac = p / cell - kp
        bdist = cell * np.minimum(frac.min(axis=1), (1.0 - frac).min(axis=1))
        rest = allj[~(best_d <= bdist)]
        if rest.size:
            search(rest, np.delete(OFFSETS, 13, axis=0))
        return best_i, best_d

    def fuse(self, patch_id: str, xyz: np.ndarray, rgb: np.ndarray, inst: np.ndarray, fix: int) -> dict:
        if patch_id in self.patch_ids:
            return {"duplicate_patch": True, "matched": 0, "new": 0, "affected_surfels": 0, "input_points": 0}
        xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
        rgb = np.asarray(rgb, np.float64).reshape(-1, 3)
        inst = np.asarray(inst, np.int32).reshape(-1)
        keep = np.isfinite(xyz).all(1) & np.isfinite(rgb).all(1)
        xyz, rgb, inst = xyz[keep], rgb[keep], inst[keep]
        bi, bd = self.nearest(xyz)
        matched = (bi >= 0) & (bd < self.radius)
        aff, inv = np.unique(bi[matched], return_inverse=True)
        if aff.size:
            cnt = np.bincount(inv).astype(np.float64)
            mean_xyz = np.stack([np.bincount(inv, xyz[matched, k]) for k in range(3)], 1) / cnt[:, None]
            mean_rgb = np.stack([np.bincount(inv, rgb[matched, k]) for k in range(3)], 1) / cnt[:, None]
            w = self.support[aff].astype(np.float64)[:, None]
            self.xyz[aff] = (self.xyz[aff] * w + mean_xyz) / (w + 1.0)
            self.rgb[aff] = (self.rgb[aff] * w + mean_rgb) / (w + 1.0)
            self.support[aff] = np.minimum(self.support[aff] + 1, np.iinfo(np.int32).max)
        un = ~matched
        k = int(un.sum())
        self._grow(k)
        s = slice(self.n, self.n + k)
        self.xyz[s], self.rgb[s], self.inst[s] = xyz[un], rgb[un], inst[un]
        self.support[s], self.first_fix[s] = 1, int(fix)
        self.n += k
        self.patch_ids.append(patch_id)
        d = bd[matched]
        return {"duplicate_patch": False, "input_points": int(len(xyz)), "matched": int(matched.sum()), "new": k,
                "affected_surfels": int(aff.size), "map_after": int(self.n),
                "matched_distance_mm": None if not d.size else
                {"median": float(np.median(d) * 1e3), "p95": float(np.quantile(d, 0.95) * 1e3)}}

    def arrays(self) -> dict:
        s = slice(0, self.n)
        return {"xyz_h": self.xyz[s].copy(), "rgb": self.rgb[s].copy(), "instance_id_oracle": self.inst[s].copy(),
                "support_count": self.support[s].copy(), "first_fixation": self.first_fix[s].copy(),
                "patch_ids": np.array(self.patch_ids, dtype="U32"),
                "radius_cell_m": np.array([self.radius, self.cell])}

    def save(self, path: Path) -> None:
        np.savez(path, **self.arrays())

    @classmethod
    def load(cls, path: Path) -> "GlobalMap":
        with np.load(path, allow_pickle=False) as z:
            r, c = (float(v) for v in z["radius_cell_m"])
            g = cls(r, c)
            g.xyz, g.rgb = np.array(z["xyz_h"]), np.array(z["rgb"])
            g.inst, g.support, g.first_fix = np.array(z["instance_id_oracle"]), np.array(z["support_count"]), \
                np.array(z["first_fixation"])
            g.patch_ids = [str(x) for x in z["patch_ids"]]
            g.n = len(g.xyz)
        return g


# ------------------------------------------------------------------ coverage
def update_coverage(state: np.ndarray, yaw: float, pitch: float, xyz: np.ndarray) -> dict:
    """SEEN: the gaze's nominal footprint.  DEPTH: footprint cells holding a valid reconstructed point (its
    direction from the H0 origin).  States only increase."""
    flat = state.reshape(-1)
    fp = footprint(yaw, pitch)
    flat[fp] = np.maximum(flat[fp], SEEN)
    if len(xyz):
        r, c = cell_of(xyz)
        cells = np.unique(r * W + c)
        cells = np.intersect1d(cells, fp, assume_unique=True)
        flat[cells] = DEPTH
    else:
        cells = np.zeros(0, np.int64)
    return {"footprint_cells": int(fp.size), "depth_cells": int(cells.size)}


def fractions(state: np.ndarray) -> tuple[float, float]:
    return (float(CELL_W[state >= SEEN].sum() / SPHERE_SR), float(CELL_W[state == DEPTH].sum() / SPHERE_SR))


# ------------------------------------------------------------------ the greedy policy
def edge_support(core_valid: np.ndarray, a: int, b: int) -> float:
    """Valid-geometry fraction of the outer core band (side) or corner block (diagonal) toward (a, b)."""
    band = np.ones_like(core_valid, bool)
    e = EDGE_BAND_PX
    if a > 0:
        band[:, :CORE_SIZE - e] = False
    elif a < 0:
        band[:, e:] = False
    if b > 0:
        band[:CORE_SIZE - e, :] = False
    elif b < 0:
        band[e:, :] = False
    return float(core_valid[band].mean())


def unseen_gain(state: np.ndarray, yaw: float, pitch: float) -> float:
    """UNSEEN solid-angle fraction of the candidate's nominal footprint."""
    fp = footprint(yaw, pitch)
    w = CELL_W.reshape(-1)[fp]
    return float(w[state.reshape(-1)[fp] == UNSEEN].sum() / w.sum()) if fp.size else 0.0


def visited_near(d: np.ndarray, visited: np.ndarray) -> bool:
    return bool(len(visited)) and float(angle_deg(d[None], visited).min()) < VISIT_TOL_DEG


# ------------------------------------------------------------------ the gaze lattice (Playground-1 / Playground-2)
_LATTICE: dict | None = None


def ring_yaws(n: int) -> np.ndarray:
    """n equally spaced yaws, shifted half a step when n mod 4 == 2 so that yaw +-90 deg (degenerate at pitch 0) is
    never on the ring."""
    return -180.0 + (np.arange(n) + 0.5) * 360.0 / n + (180.0 / n if n % 4 == 2 else 0.0)


def build_lattice(ring_deg: float = LATTICE_RING_DEG) -> dict:
    """Pitch rings every ring_deg from -90 to +90 deg (one gaze at each pole); on each other ring the fewest
    (>= 4) equally spaced yaws whose nominal footprints contain the centre of every cell whose centre pitch lies within
    half a ring spacing of the ring.  Returns yaw / pitch / dir arrays and each gaze's footprint (flat cell indices)."""
    k = int(math.ceil(90.0 / ring_deg))
    half = 90.0 / k / 2.0
    pitch_c = CELL_PITCH.reshape(-1)
    yaws: list[float] = []
    pitches: list[float] = []
    fps: list[np.ndarray] = []
    for p in np.linspace(-90.0, 90.0, 2 * k + 1):
        p = float(p)
        if abs(p) >= 90.0:
            yaws.append(0.0), pitches.append(p), fps.append(footprint(0.0, p))
            continue
        band = np.flatnonzero((pitch_c >= p - half) & (pitch_c <= p + half))
        for n in range(4, 1000):
            ring = [(float(y), footprint(float(y), p)) for y in ring_yaws(n)]
            cov = np.zeros(H * W, bool)
            for _, f in ring:
                cov[f] = True
            if cov[band].all():
                break
        else:
            raise RuntimeError(f"no ring of <= 999 gazes covers the band at pitch {p}")
        for y, f in ring:
            yaws.append(y), pitches.append(p), fps.append(f)
    yaw, pitch = np.array(yaws), np.array(pitches)
    return {"yaw_deg": yaw, "pitch_deg": pitch, "dir": direction(yaw, pitch), "footprints": fps,
            "ring_deg": float(ring_deg)}


def baseline_direction(theta_deg: float, phi_deg: float) -> np.ndarray:
    """Head-frame unit direction at angle theta from the baseline axis (+X) and rotation phi about it, phi = 0 forward
    (-Z): (cos theta, sin theta sin phi, -sin theta cos phi), the epipolar convention of CLAUDE.md."""
    t, p = math.radians(theta_deg), math.radians(phi_deg)
    return np.array([math.cos(t), math.sin(t) * math.sin(p), -math.sin(t) * math.cos(p)])


def build_baseline_lattice(cap_deg: float = LATTICE_CAP_DEG, step_deg: float = LATTICE_AXIS_STEP_DEG) -> dict:
    """Playground-2: rings at theta = cap, cap + step, ..., 180 - cap from the baseline axis (a ring must land on
    theta = 90).  Ring i carries the fewest (>= 1) gazes at phi = j * 360 / n whose nominal footprints contain the
    centre of every cell with theta_c in [theta_i - step / 2, theta_i + step / 2]; the two cap rings reach the poles.
    Every ring starts at phi = 0, so the start gaze (yaw 0, pitch 0) is on the theta = 90 ring."""
    k = (90.0 - cap_deg) / step_deg
    if abs(k - round(k)) > 1e-9:
        raise ValueError("the baseline rings must land on theta = 90 deg")
    thetas = [cap_deg + i * step_deg for i in range(2 * int(round(k)) + 1)]
    theta_c = np.degrees(np.arccos(np.clip(CELL_DIR[:, 0], -1.0, 1.0)))
    yaws: list[float] = []
    pitches: list[float] = []
    fps: list[np.ndarray] = []
    for i, t in enumerate(thetas):
        lo = 0.0 if i == 0 else t - step_deg / 2.0
        hi = 180.0 if i == len(thetas) - 1 else t + step_deg / 2.0
        band = np.flatnonzero((theta_c >= lo) & (theta_c <= hi))
        for n in range(1, 1000):
            ring = []
            for j in range(n):
                y, p = (float(v) for v in yaw_pitch(baseline_direction(t, j * 360.0 / n)))
                ring.append((y, p, footprint(y, p)))
            cov = np.zeros(H * W, bool)
            for *_, f in ring:
                cov[f] = True
            if cov[band].all():
                break
        else:
            raise RuntimeError(f"no ring of <= 999 gazes covers the band at theta {t}")
        for y, p, f in ring:
            yaws.append(y), pitches.append(p), fps.append(f)
    yaw, pitch = np.array(yaws), np.array(pitches)
    return {"yaw_deg": yaw, "pitch_deg": pitch, "dir": direction(yaw, pitch), "footprints": fps,
            "ring_deg": float(step_deg), "kind": "baseline"}


def lattice() -> dict:
    """The gaze lattice selected by LATTICE_KIND, built once per process (deterministic)."""
    global _LATTICE
    if _LATTICE is None:
        if LATTICE_KIND == "baseline":
            _LATTICE = build_baseline_lattice()
        elif LATTICE_KIND == "pitch":
            _LATTICE = build_lattice()
        else:
            raise ValueError(f"unknown LATTICE_KIND {LATTICE_KIND!r}")
    return _LATTICE


def lattice_gains(state: np.ndarray) -> np.ndarray:
    """UNSEEN solid-angle fraction of every lattice gaze's footprint (``unseen_gain``, vectorized over the lattice)."""
    w, un = CELL_W.reshape(-1), (state.reshape(-1) == UNSEEN)
    return np.array([float(w[f][un[f]].sum() / w[f].sum()) for f in lattice()["footprints"]])


def lattice_visited(visited: np.ndarray) -> np.ndarray:
    """Lattice gazes within LATTICE_VISIT_DEG of any fixation direction."""
    L = lattice()
    if not len(visited):
        return np.zeros(len(L["dir"]), bool)
    return angle_deg(L["dir"], np.asarray(visited)).min(axis=1) < LATTICE_VISIT_DEG


_OCTANTS = (("E", 1, 0), ("SE", 1, 1), ("S", 0, 1), ("SW", -1, 1),       # angle atan2(v, u) in 45-deg steps;
            ("W", -1, 0), ("NW", -1, -1), ("N", 0, -1), ("NE", 1, -1))  # image +x = right, +y = down


def octant(r: np.ndarray, d: np.ndarray) -> tuple[str, int, int]:
    """The neighbour direction (name, a, b) of the octant that holds d's offset in the tangent frame r."""
    u, v = float(d @ r[:, 0]), float(d @ r[:, 1])
    return _OCTANTS[int(round(math.degrees(math.atan2(v, u)) / 45.0)) % 8]


def local_candidates(state, core_valid, yaw, pitch, visited) -> list[dict]:
    """The lattice gazes within LATTICE_NEIGHBOUR_DEG of the gaze (the gaze itself excluded), in lattice order."""
    L = lattice()
    r = frame(yaw, pitch)
    a_deg = angle_deg(direction(yaw, pitch)[None], L["dir"])[0]
    near = np.flatnonzero((a_deg < LATTICE_NEIGHBOUR_DEG) & (a_deg >= LATTICE_VISIT_DEG))
    gains, seen_before = lattice_gains(state), lattice_visited(visited)
    out = []
    for i in near:
        name, a, b = octant(r, L["dir"][i])
        e, g, vis = edge_support(core_valid, a, b), float(gains[i]), bool(seen_before[i])
        out.append({"dir": name, "lattice_index": int(i), "yaw_deg": float(L["yaw_deg"][i]),
                    "pitch_deg": float(L["pitch_deg"][i]), "edge_support": e, "unseen_gain": g, "score": e * g,
                    "visited": vis, "eligible": bool(e >= EDGE_MIN and g > GAIN_MIN and not vis)})
    return out


def unseen_components(state: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """4-connected UNSEEN components with longitude wrap and across-pole links in the polar rows."""
    lab = np.full((H, W), -1, np.int32)
    un = state == UNSEEN
    sizes: list[float] = []
    for r0, c0 in zip(*np.nonzero(un)):
        if lab[r0, c0] >= 0:
            continue
        k = len(sizes)
        lab[r0, c0] = k
        dq, sr = deque([(r0, c0)]), 0.0
        while dq:
            r, c = dq.popleft()
            sr += ROW_W[r]
            nb = [(r, (c + 1) % W), (r, (c - 1) % W)]
            if r > 0:
                nb.append((r - 1, c))
            else:
                nb.append((0, (c + W // 2) % W))
            if r < H - 1:
                nb.append((r + 1, c))
            else:
                nb.append((H - 1, (c + W // 2) % W))
            for rr, cc in nb:
                if un[rr, cc] and lab[rr, cc] < 0:
                    lab[rr, cc] = k
                    dq.append((rr, cc))
        sizes.append(sr)
    return lab, sizes


def global_target(state: np.ndarray, visited: np.ndarray) -> dict | None:
    """The unvisited lattice gaze with the largest UNSEEN gain (lowest index on ties); None when none adds area."""
    L = lattice()
    gains = lattice_gains(state)
    gains[lattice_visited(visited)] = 0.0
    i = int(np.argmax(gains))
    if gains[i] <= 0.0:
        return None
    return {"yaw_deg": float(L["yaw_deg"][i]), "pitch_deg": float(L["pitch_deg"][i]), "lattice_index": i,
            "unseen_gain": float(gains[i]), "lattice_open": int((gains > 0).sum())}


def decide(state: np.ndarray, core_valid: np.ndarray, yaw: float, pitch: float, visited: np.ndarray) -> dict:
    """LOCAL if some lattice neighbour has nontrivial edge support and any unseen gain (highest score, first in lattice
    order on ties); else GLOBAL (the max-gain unvisited lattice gaze); else NONE (no lattice gaze adds area)."""
    cands = local_candidates(state, core_valid, yaw, pitch, visited)
    elig = [c for c in cands if c["eligible"]]
    if elig:
        best = max(elig, key=lambda c: c["score"])          # max() keeps the first of equal scores
        return {"kind": "local", "yaw_deg": best["yaw_deg"], "pitch_deg": best["pitch_deg"], "dir": best["dir"],
                "lattice_index": best["lattice_index"], "candidates": cands}
    g = global_target(state, visited)
    if g is None:
        return {"kind": "none", "candidates": cands}
    return {"kind": "global", **g, "candidates": cands}
