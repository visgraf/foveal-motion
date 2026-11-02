"""Playground-1 / Playground-2 (gaze lattice): fail-capable host-side checks.  No Blender, no rendering, no Breadth-1.

    .venv/bin/python tools/greedy_foveal/check_lattice.py        # exit 1 on any failed check

Contracts (in f3d-vision): docs/playground/gaze-lattice-contract.md and sparse-lattice-contract.md, section 4 of each.
The lattice checked is the one explorer.lattice() returns (Playground-2's baseline rings).

K1 the lattice footprints contain every 1-deg cell centre.
K2 every lattice gaze has a valid tangent frame, and every pair is >= VISIT_TOL_DEG apart.
K3 known answer behind the change: the baseline's first-step UNSEEN gain is 0.825 for its diagonal neighbours and 0.58
   for its side neighbours (the diagonal bias).
K4 a coverage-only replay of the run.py control loop (start at gaze (0, 0), full edge support, no reconstructed
   points) through explorer.decide ends with "none", 0 UNSEEN cells, at most one fixation per lattice gaze (plus the
   start), and no two fixations closer than VISIT_TOL_DEG.
K5 a mid-replay state re-decided twice gives the same decision as the live one.
K6 the lattice has 356 gazes, and the Playground-1 pitch lattice (LATTICE_KIND "pitch") still has 496.
K7 no gaze is redundant (each covers a cell no other gaze covers), every gaze is >= 6 deg from the baseline axis, and
   the start gaze (0, 0) is a lattice gaze.
N1 negative control: K1 with the start gaze removed must FAIL.
N2 negative control: K7's 6-deg envelope on the 2-deg cap design must FAIL.

The replay numbers are a Chat-sandbox model of the workstation run, not a measurement of it.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import explorer as X  # noqa: E402

PREFIX = "[lattice]"
LATTICE_SIZE = 356       # Playground-2 known answer (K6)
PITCH_LATTICE_SIZE = 496 # Playground-1 known answer (K6)
ENVELOPE_DEG = 6.0       # no gaze closer to the baseline axis (K7)
checked = failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global checked, failed
    checked += 1
    failed += not ok
    print(f"{PREFIX} {'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)


def covers_all(footprints: list[np.ndarray]) -> tuple[bool, int]:
    cov = np.zeros(X.H * X.W, bool)
    for f in footprints:
        cov[f] = True
    return bool(cov.all()), int((~cov).sum())


def replay(max_fix: int = 1000) -> dict:
    """The run.py explore loop with the renderer and reconstruction replaced by full edge support and no points."""
    cv = np.ones((X.CORE_SIZE, X.CORE_SIZE), bool)
    state = np.zeros((X.H, X.W), np.int8)
    visited = np.zeros((0, 3))
    yaw, pitch, kind = 0.0, 0.0, "initial"
    kinds, seen_hist, mid, stop = [], [], None, None
    for n in range(1, max_fix + 1):
        X.update_coverage(state, yaw, pitch, np.zeros((0, 3)))
        seen, _ = X.fractions(state)
        seen_hist.append(seen)
        visited = np.vstack([visited, X.direction(yaw, pitch)[None]])
        kinds.append(kind)
        if seen >= X.SEEN_STOP:
            stop = "seen"
            break
        dec = X.decide(state, cv, yaw, pitch, visited)
        if n == 150:
            mid = (state.copy(), cv, yaw, pitch, visited.copy(), dec)
        if dec["kind"] == "none":
            stop = "none"
            break
        yaw, pitch, kind = dec["yaw_deg"], dec["pitch_deg"], dec["kind"]
    last_local = max((i for i, k in enumerate(kinds) if k == "local"), default=-1)
    ang = X.angle_deg(visited, visited) + np.eye(len(visited)) * 999
    sh = np.array(seen_hist)
    return {"stop": stop, "fixations": len(kinds), "local": kinds.count("local"), "global": kinds.count("global"),
            "tail": len(kinds) - 1 - last_local, "unseen_cells": int((state == X.UNSEEN).sum()),
            "min_pair_deg": float(ang.min()), "to_99": int(np.argmax(sh >= 0.99)) + 1 if sh[-1] >= 0.99 else None,
            "seen": float(sh[-1]), "holes": len(X.unseen_components(state)[1]), "mid": mid}


def k4_ok(r: dict, n_lattice: int) -> bool:
    return (r["stop"] == "none" and r["unseen_cells"] == 0 and r["fixations"] <= n_lattice
            and r["min_pair_deg"] >= X.VISIT_TOL_DEG)


def axis_angle_deg(L: dict) -> np.ndarray:
    """Each gaze's angle from the baseline axis (either direction)."""
    return np.degrees(np.arccos(np.clip(np.abs(L["dir"][:, 0]), 0.0, 1.0)))


def redundant(footprints: list[np.ndarray]) -> int:
    cnt = np.zeros(X.H * X.W, np.int32)
    for f in footprints:
        cnt[f] += 1
    return sum(1 for f in footprints if cnt[f].min() >= 2)


def main() -> None:
    t0 = time.perf_counter()
    L = X.lattice()
    n = len(L["dir"])
    ring_key = (np.round(np.degrees(np.arccos(np.clip(L["dir"][:, 0], -1.0, 1.0))), 6)
                if L.get("kind") == "baseline" else L["pitch_deg"])           # baseline rings: theta from +X
    rings = {float(v): int((ring_key == v).sum()) for v in np.unique(ring_key)}
    print(f"{PREFIX} lattice ({L.get('kind', 'pitch')}): {n} gazes on {len(rings)} rings of {L['ring_deg']:g} deg "
          f"({time.perf_counter() - t0:.1f} s); per ring {list(rings.values())}", flush=True)

    ok, missing = covers_all(L["footprints"])
    check("K1 the lattice covers every 1-deg cell", ok, f"{X.H * X.W - missing} / {X.H * X.W} cells")

    frames_ok = True
    for y, p in zip(L["yaw_deg"], L["pitch_deg"]):
        try:
            X.frame(float(y), float(p))
        except ValueError:
            frames_ok = False
    ang = X.angle_deg(L["dir"], L["dir"]) + np.eye(n) * 999
    check("K2 every gaze has a frame; every pair >= VISIT_TOL_DEG apart",
          frames_ok and float(ang.min()) >= X.VISIT_TOL_DEG, f"frames {frames_ok}, min pair {ang.min():.2f} deg")

    st = np.zeros((X.H, X.W), np.int8)
    X.update_coverage(st, 0.0, 0.0, np.zeros((0, 3)))
    r = X.frame(0.0, 0.0)
    t = math.tan(math.radians(X.LOCAL_STEP_DEG))
    gains = {}
    for name, a, b in X.NEIGHBOURS:
        d = r[:, 2] + t * (a * r[:, 0] + b * r[:, 1])
        y2, p2 = (float(v) for v in X.yaw_pitch(d / np.linalg.norm(d)))
        gains[name] = X.unseen_gain(st, y2, p2)
    diag = [gains[k] for k in ("NE", "NW", "SE", "SW")]
    side = [gains[k] for k in ("E", "W", "N", "S")]
    check("K3 baseline first-step gain: diagonals 0.825, sides 0.58 (+-0.005)",
          all(abs(g - 0.825) <= 0.005 for g in diag) and all(abs(g - 0.58) <= 0.005 for g in side),
          f"diagonal {[round(g, 3) for g in diag]}, side {[round(g, 3) for g in side]}")

    t1 = time.perf_counter()
    rep = replay()
    check("K4 replay: stops with none, 0 UNSEEN cells, <= one fixation per gaze, no duplicate",
          k4_ok(rep, n), f"stop {rep['stop']}, {rep['fixations']} fixations ({rep['local']} local / {rep['global']} "
          f"global, tail {rep['tail']}), UNSEEN cells {rep['unseen_cells']}, min pair {rep['min_pair_deg']:.2f} deg, "
          f"99 % at #{rep['to_99']} ({time.perf_counter() - t1:.0f} s)")

    st_, cv_, y_, p_, vis_, live = rep["mid"]
    d1, d2 = X.decide(st_, cv_, y_, p_, vis_), X.decide(st_, cv_, y_, p_, vis_)
    check("K5 a mid-replay state re-decides identically", d1 == d2 == live, f"{live['kind']} -> "
          f"({live.get('yaw_deg', float('nan')):.2f}, {live.get('pitch_deg', float('nan')):.2f})")

    kind = X.LATTICE_KIND
    X.LATTICE_KIND, X._LATTICE = "pitch", None
    try:
        n_pitch = len(X.lattice()["dir"])
    finally:
        X.LATTICE_KIND, X._LATTICE = kind, L
    check(f"K6 lattice size {LATTICE_SIZE}; Playground-1 pitch lattice {PITCH_LATTICE_SIZE}",
          n == LATTICE_SIZE and n_pitch == PITCH_LATTICE_SIZE, f"{n} gazes; pitch lattice {n_pitch}")

    red, env = redundant(L["footprints"]), float(axis_angle_deg(L).min())
    start = float(X.angle_deg(X.direction(0.0, 0.0)[None], L["dir"]).min())
    check(f"K7 no redundant gaze; every gaze >= {ENVELOPE_DEG:g} deg from the baseline axis; start gaze on the lattice",
          red == 0 and env >= ENVELOPE_DEG - 1e-9 and start < X.LATTICE_VISIT_DEG,
          f"redundant {red}, closest to axis {env:.2f} deg, start offset {start:.2e} deg")

    s0 = int(np.argmin(X.angle_deg(X.direction(0.0, 0.0)[None], L["dir"])[0]))
    ok_n1, miss_n1 = covers_all(L["footprints"][:s0] + L["footprints"][s0 + 1:])
    check("N1 negative control: removing the start gaze breaks the cover", not ok_n1, f"{miss_n1} cells uncovered")

    near = X.build_baseline_lattice(cap_deg=2.0, step_deg=11.0)
    env2 = float(axis_angle_deg(near).min())
    check(f"N2 negative control: the 2-deg cap design breaks the {ENVELOPE_DEG:g}-deg envelope", env2 < ENVELOPE_DEG,
          f"{len(near['dir'])} gazes, closest to axis {env2:.2f} deg")

    print(f"{PREFIX} SUMMARY checked={checked} failed={failed} ({time.perf_counter() - t0:.0f} s)", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
