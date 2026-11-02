"""Persistent multi-look head-frame surfel map for FSG3.

No truth access. A new patch associates against a snapshot of the existing map.
Multiple samples from the same patch may choose one old surfel; they are reduced
to one patch contribution, then fused using the number of distinct prior patch
supports. Replaying a patch id is exactly idempotent.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class Patch:
    patch_id: str
    xyz_h: np.ndarray
    rgb: np.ndarray
    instance_id: np.ndarray

@dataclass
class SurfaceMap:
    xyz_h: np.ndarray
    rgb: np.ndarray
    instance_id: np.ndarray
    support_count: np.ndarray
    provenance_mask: np.ndarray
    patch_ids: list[str]

def _clean_patch(p: Patch, wanted_instance: int) -> Patch:
    xyz = np.asarray(p.xyz_h, float).reshape(-1,3)
    rgb = np.asarray(p.rgb, float).reshape(-1,3)
    ids = np.asarray(p.instance_id).reshape(-1)
    keep = np.isfinite(xyz).all(1) & np.isfinite(rgb).all(1) & (ids == wanted_instance)
    return Patch(p.patch_id, xyz[keep], rgb[keep], ids[keep].astype(np.int32))

def initialize(p: Patch, wanted_instance: int) -> SurfaceMap:
    q = _clean_patch(p, wanted_instance)
    if len(q.xyz_h) < 100:
        raise ValueError("too few object points to initialize map")
    return SurfaceMap(q.xyz_h.copy(), q.rgb.copy(), q.instance_id.copy(),
                      np.ones(len(q.xyz_h), np.int16),
                      np.ones(len(q.xyz_h), np.uint64), [q.patch_id])

def _key(x: np.ndarray, cell: float) -> tuple[int,int,int]:
    return tuple(np.floor(x/cell).astype(np.int64).tolist())

def _copy(m: SurfaceMap) -> SurfaceMap:
    return SurfaceMap(m.xyz_h.copy(), m.rgb.copy(), m.instance_id.copy(),
                      m.support_count.copy(), m.provenance_mask.copy(), list(m.patch_ids))

def fuse(m: SurfaceMap, p: Patch, wanted_instance: int, radius: float, cell: float) -> tuple[SurfaceMap, dict]:
    if p.patch_id in m.patch_ids:
        return _copy(m), {"duplicate_patch": True, "matched": 0, "new": 0,
                          "affected_surfels": 0, "distances_m": np.empty(0),
                          "input_points": 0, "base_points": len(m.xyz_h)}
    if len(m.patch_ids) >= 63:
        raise ValueError("uint64 provenance supports at most 63 fused patch ids")
    q = _clean_patch(p, wanted_instance)
    if len(q.xyz_h) < 100:
        raise ValueError("too few object points in patch")
    bins: dict[tuple[int,int,int], list[int]] = {}
    for i, x in enumerate(m.xyz_h):
        bins.setdefault(_key(x,cell), []).append(i)
    matched_idx = np.full(len(q.xyz_h), -1, np.int64)
    dist = np.full(len(q.xyz_h), np.inf, float)
    offsets = [(a,b,c) for a in (-1,0,1) for b in (-1,0,1) for c in (-1,0,1)]
    for j, x in enumerate(q.xyz_h):
        k = _key(x,cell); best = -1; bd = radius
        for o in offsets:
            for i in bins.get((k[0]+o[0], k[1]+o[1], k[2]+o[2]), ()):
                if m.instance_id[i] != wanted_instance:
                    continue
                d = float(np.linalg.norm(x-m.xyz_h[i]))
                if d < bd:
                    bd = d; best = i
        if best >= 0:
            matched_idx[j] = best; dist[j] = bd
    out = _copy(m)
    out.patch_ids.append(p.patch_id)
    bit = np.uint64(1 << (len(out.patch_ids)-1))
    affected = np.unique(matched_idx[matched_idx >= 0])
    for i in affected:
        js = np.flatnonzero(matched_idx == i)
        patch_xyz = q.xyz_h[js].mean(axis=0)
        patch_rgb = q.rgb[js].mean(axis=0)
        old_w = float(out.support_count[i])
        out.xyz_h[i] = (out.xyz_h[i]*old_w + patch_xyz) / (old_w + 1.0)
        out.rgb[i] = (out.rgb[i]*old_w + patch_rgb) / (old_w + 1.0)
        out.support_count[i] = np.int16(min(32767, int(out.support_count[i])+1))
        out.provenance_mask[i] |= bit
    un = matched_idx < 0
    if np.any(un):
        n = int(un.sum())
        out.xyz_h = np.vstack((out.xyz_h, q.xyz_h[un]))
        out.rgb = np.vstack((out.rgb, q.rgb[un]))
        out.instance_id = np.concatenate((out.instance_id, q.instance_id[un]))
        out.support_count = np.concatenate((out.support_count, np.ones(n, np.int16)))
        out.provenance_mask = np.concatenate((out.provenance_mask, np.full(n, bit, np.uint64)))
    return out, {
        "duplicate_patch": False,
        "matched": int((~un).sum()),
        "new": int(un.sum()),
        "affected_surfels": int(len(affected)),
        "distances_m": dist[~un],
        "input_points": int(len(q.xyz_h)),
        "base_points": int(len(m.xyz_h)),
    }

def save_map(path: Path, m: SurfaceMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, xyz_h=m.xyz_h.astype(np.float32), rgb=m.rgb.astype(np.float32),
                        instance_id=m.instance_id, support_count=m.support_count,
                        provenance_mask=m.provenance_mask,
                        patch_ids=np.array(m.patch_ids, dtype="U64"))

def load_map(path: Path) -> SurfaceMap:
    with np.load(path, allow_pickle=False) as f:
        return SurfaceMap(f["xyz_h"], f["rgb"], f["instance_id"], f["support_count"],
                          f["provenance_mask"], f["patch_ids"].astype(str).tolist())

def self_test() -> None:
    rng = np.random.default_rng(12)
    def plane(pid: str, lo: float, hi: float) -> Patch:
        xy = np.stack(np.meshgrid(np.linspace(lo,hi,50), np.linspace(-.10,.10,24)), axis=-1).reshape(-1,2)
        xyz = np.c_[xy, -2*np.ones(len(xy))] + rng.normal(scale=.0004, size=(len(xy),3))
        return Patch(pid, xyz, np.ones_like(xyz)*(.2+.1*len(pid)), np.full(len(xyz),71))
    a,b,c = plane("A",-.28,-.02), plane("B",-.12,.12), plane("C",.02,.28)
    m = initialize(a,71)
    m2,s2 = fuse(m,b,71,.010,.010)
    m3,s3 = fuse(m2,c,71,.010,.010)
    if s2["matched"] < 300 or s2["new"] < 300 or s3["matched"] < 300 or s3["new"] < 300:
        raise AssertionError("three-patch overlap/new support not recovered")
    if int(m3.support_count.max()) < 2:
        raise AssertionError("multi-look support not recorded")
    before = (m3.xyz_h.copy(), m3.rgb.copy(), m3.support_count.copy(), m3.provenance_mask.copy())
    m4, dup = fuse(m3,c,71,.010,.010)
    if not dup["duplicate_patch"] or not all(np.array_equal(x,y) for x,y in zip(before,(m4.xyz_h,m4.rgb,m4.support_count,m4.provenance_mask))):
        raise AssertionError("duplicate replay changed map")
    print(f"[fsg3-map] PASS B={s2['matched']}/{s2['new']} C={s3['matched']}/{s3['new']} idempotent=true")

if __name__ == "__main__":
    self_test()
