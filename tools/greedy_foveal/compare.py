"""Compare a run with the baseline: a compact signature of the frozen Playground-2 run, plus its literal numbers.

    .venv/bin/python tools/greedy_foveal/compare.py check --run RUN            # full run or a shorter (prefix) run
    .venv/bin/python tools/greedy_foveal/compare.py selftest                   # every comparator can fail
    .venv/bin/python tools/greedy_foveal/compare.py signature --run BASE --out FILE   # maintainers only, once

This replaces f3d-vision's ``check_equivalence.py run``, which needs the 2.2 GB frozen baseline run on disk.  A
signature keeps, for every fixation, the sha256 of each control-path product instead of the product itself, so the same
comparisons fit in a small committed file (``tests/known-answer/playground-2-signature.json``):

- trajectory: the run-level control fields; per fixation the whole record (gaze, kind, decision, candidates,
  correspondence counts, fusion, coverage, SEEN / DEPTH) except wall-clock timings;
- per fixation: calibration.json bytes; points.npz xyz_h / instance_id_oracle / err_mm bitwise;
- per decision: the saved policy state (state / core_valid / gaze / visited) bitwise;
- coverage: SEEN and DEPTH histories and the final state bitwise;
- full runs only: the final map arrays bitwise, the segmentation PLY bytes, the display PLY header and XYZ.

RGB is never compared: it is not reproducible run to run under Cycles OPTIX and is never a control input.  Bitwise means
same dtype, same shape, same bytes.  ``--prefix`` (automatic when RUN has fewer fixations than the baseline) compares a
shorter run with the first N baseline fixations, as the smoke test does with N = 75.  Only a run cut short by
``--max-fix`` (stop reason "fixation cap N") is a prefix run; a run that stopped by itself is compared as a full run.

A full run is also checked against the literal Playground-2 numbers (KNOWN_ANSWER), which need no signature file.
A prefix run without a signature file is checked only for its LOCAL / GLOBAL sequence (fixation 1 initial, 2-75 local).

``check`` exits 0 on PASS and 1 on FAIL.  A real experiment is expected to FAIL ``check``: use it to show that a
refactor did NOT change behavior, not to grade a new idea.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PREFIX = "[compare]"
SIGNATURE = REPO / "tests/known-answer/playground-2-signature.json"
SCHEMA = "foveal-playground-signature-v1"

# trajectory.json run-level fields compared exactly.  head_pose_source and code are metadata (paths, commits).
SUMMARY_FULL = ("schema", "label", "spp", "max_fix", "stop_reason", "fixations", "local_saccades", "global_saccades",
                "final_seen_fraction", "final_depth_fraction", "final_surfels", "monotone_cells", "server_exit",
                "parameters", "server")
SUMMARY_PREFIX = ("schema", "label", "spp", "monotone_cells", "server_exit", "parameters", "server")
ENTRY_METADATA = ("seconds", "render_seconds_lr")
LAST_OF_PREFIX = ("next", "local_candidates")       # the baseline continued past a prefix run's last fixation
POINTS_BITWISE = ("xyz_h", "instance_id_oracle", "err_mm")
POLICY_BITWISE = ("state", "core_valid", "gaze", "visited")
MAP_BITWISE = ("xyz_h", "instance_id_oracle", "support_count", "first_fixation", "patch_ids", "radius_cell_m")
FROZEN = ("trajectory.json", "final-map.npz", "coverage.npz", "final-map.ply", "final-map-oracle-segmentation.ply")
KNOWN_ANSWER = {"fixations": 356, "stop_reason": "no valid unseen fixation", "local_saccades": 315,
                "global_saccades": 40, "seen_pct": "100.00", "depth_pct": "99.26", "final_surfels": 16943073}
#   MEASURED: f3d-vision docs/playground/sparse-lattice-report.md and docs/engineering/sparse-lattice-baseline-report.md
SMOKE_LOCAL_THROUGH = 75     # fixations 2..75 of the baseline are local saccades; the first global jump is at 168
MAX_LISTED = 10


# ------------------------------------------------------------------ digests
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(a: np.ndarray) -> str:
    """dtype, shape and bytes: equal digests <=> bitwise-equal arrays (NaN-safe; -0.0 differs from +0.0)."""
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def npz_digests(path: Path, keys: tuple[str, ...]) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {"keys": sorted(z.files), "bitwise": {k: array_digest(z[k]) for k in keys if k in z.files}}


def entry_digest(e: dict, ignore: tuple[str, ...]) -> str:
    kept = {k: v for k, v in e.items() if k not in ignore}
    return sha256_bytes(json.dumps(kept, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def float_hex(a: np.ndarray) -> dict:
    a = np.asarray(a)
    return {"dtype": a.dtype.str, "hex": [float(x).hex() for x in a.ravel()]}


def from_hex(d: dict, n: int | None = None) -> np.ndarray:
    v = d["hex"] if n is None else d["hex"][:n]
    return np.array([float.fromhex(x) for x in v], dtype=np.dtype(d["dtype"]))


def ply_header_and_xyz(path: Path) -> dict:
    data = Path(path).read_bytes()
    if b"end_header\n" not in data:
        return {"header": None, "xyz": "malformed: no end_header"}
    end = data.index(b"end_header\n") + len(b"end_header\n")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    if (len(data) - end) % dt.itemsize:
        return {"header": sha256_bytes(data[:end]), "xyz": "malformed: body is not a whole number of vertices"}
    rec = np.frombuffer(data[end:], dtype=dt)
    return {"header": sha256_bytes(data[:end]), "xyz": {c: array_digest(rec[c]) for c in "xyz"}}


def read_json(path: Path):
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------ signature
def make_signature(run: Path) -> dict:
    t = read_json(run / "trajectory.json")
    T = t["trajectory"]
    fixes = []
    for e in T:
        k = int(e["n"])
        fd = run / f"fixations/fix-{k:04d}"
        rec = {"n": k, "kind": e["kind"], "yaw_deg": e["yaw_deg"], "pitch_deg": e["pitch_deg"],
               "entry": entry_digest(e, ENTRY_METADATA), "entry_without_next": entry_digest(e, ENTRY_METADATA +
                                                                                            LAST_OF_PREFIX),
               "calibration": sha256_file(fd / "calibration.json"), "points": npz_digests(fd / "points.npz",
                                                                                         POINTS_BITWISE)}
        ps = run / f"policy/state-{k:04d}.npz"
        rec["policy"] = npz_digests(ps, POLICY_BITWISE) if ps.is_file() else None
        fixes.append(rec)
    with np.load(run / "coverage.npz", allow_pickle=False) as z:
        cov = {"keys": sorted(z.files), "state": array_digest(z["state"]), "row_weights": array_digest(z["row_weights"]),
               "seen_history": float_hex(z["seen_history"]), "depth_history": float_hex(z["depth_history"])}
    return {"schema": SCHEMA, "source_run": run.name,
            "note": "sha256 digests of the control-path products of one frozen run; see tools/greedy_foveal/compare.py",
            "summary": {k: t.get(k) for k in SUMMARY_FULL}, "fixations": fixes, "coverage": cov,
            "final_map": npz_digests(run / "final-map.npz", MAP_BITWISE),
            "segmentation_ply": sha256_file(run / "final-map-oracle-segmentation.ply"),
            "display_ply": ply_header_and_xyz(run / "final-map.ply")}


# ------------------------------------------------------------------ comparison
class Ledger:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def check(self, name: str, ok: bool, detail=None) -> bool:
        self.rows.append({"check": name, "pass": bool(ok), "detail": detail})
        return ok

    def report(self, name: str, value) -> None:
        self.rows.append({"report": name, "value": value})

    def all_pass(self) -> bool:
        return all(r["pass"] for r in self.rows if "check" in r)


def compare(run: Path, sig: dict | None, known: bool = True) -> Ledger:
    L = Ledger()
    t = read_json(run / "trajectory.json")
    T = t["trajectory"]
    n_r = len(T)
    n_b = len(sig["fixations"]) if sig else KNOWN_ANSWER["fixations"]
    # a prefix run is one cut short by --max-fix; a run that stopped by itself is compared as a full run
    prefix = n_r < n_b and str(t.get("stop_reason", "")).startswith("fixation cap")
    L.report("fixations", {"run": n_r, "baseline": n_b, "mode": "prefix" if prefix else "full"})

    # ---- integrity of the run itself
    fz = read_json(run / "freeze.json")
    stale = [f for f, h in fz["files"].items() if sha256_file(run / f) != h]
    L.check("run products unchanged since its freeze", not stale and set(fz["files"]) == set(FROZEN), stale or None)
    ck = run / "checks.json"
    L.check("run.py checks recorded and all pass", ck.is_file() and read_json(ck).get("all_pass") is True,
            None if ck.is_file() else "checks.json missing: run `run.py checks --run RUN` first")

    # ---- literal known answer (no signature needed)
    kinds = [e["kind"] for e in T]
    if not prefix and known:
        got = {"fixations": t["fixations"], "stop_reason": t["stop_reason"], "local_saccades": t["local_saccades"],
               "global_saccades": t["global_saccades"], "seen_pct": f"{100 * t['final_seen_fraction']:.2f}",
               "depth_pct": f"{100 * t['final_depth_fraction']:.2f}", "final_surfels": t["final_surfels"]}
        wrong = {k: [got.get(k), v] for k, v in KNOWN_ANSWER.items() if got.get(k) != v}
        ka = KNOWN_ANSWER
        L.check(f"known answer: {ka['fixations']} fixations / {ka['stop_reason']} / {ka['local_saccades']} local / "
                f"{ka['global_saccades']} global / SEEN {ka['seen_pct']} % / DEPTH {ka['depth_pct']} % / "
                f"{ka['final_surfels']:,d} surfels", not wrong, wrong or None)
    if prefix and known:
        m = min(n_r, SMOKE_LOCAL_THROUGH)
        want = ["initial"] + ["local"] * (m - 1)
        L.check(f"known answer: fixation 1 initial, 2..{m} local", kinds[:m] == want)

    if sig is None:
        L.report("signature", f"NOT RUN: {SIGNATURE.relative_to(REPO)} is not installed; only the literal checks ran")
        return L
    if sig.get("schema") != SCHEMA:
        L.check("signature schema", False, sig.get("schema"))
        return L
    S = sig["fixations"]

    # ---- run-level fields
    fields = SUMMARY_PREFIX if prefix else SUMMARY_FULL
    diff = [k for k in fields if t.get(k, KeyError) != sig["summary"].get(k, KeyError)]
    L.check(f"trajectory run-level fields ({'prefix set' if prefix else 'all control fields'})", not diff,
            {k: [t.get(k), sig["summary"].get(k)] for k in diff[:MAX_LISTED]} or None)
    if not prefix and n_r != n_b:
        L.check("fixation count", False, f"{n_r} vs {n_b}")
        return L

    # ---- per fixation
    bad_entry, bad_calib, bad_pts, missing = [], [], {}, []
    for i in range(n_r):
        e, s = T[i], S[i]
        k = int(s["n"])
        last = prefix and i == n_r - 1
        if entry_digest(e, ENTRY_METADATA + (LAST_OF_PREFIX if last else ())) != \
                s["entry_without_next" if last else "entry"]:
            bad_entry.append(k)
        fd = run / f"fixations/fix-{k:04d}"
        if not ((fd / "calibration.json").is_file() and (fd / "points.npz").is_file()):
            missing.append(k)
            continue
        if sha256_file(fd / "calibration.json") != s["calibration"]:
            bad_calib.append(k)
        p = npz_digests(fd / "points.npz", POINTS_BITWISE)
        bad = [key for key in POINTS_BITWISE if p["bitwise"].get(key) != s["points"]["bitwise"].get(key)]
        if p["keys"] != s["points"]["keys"]:
            bad.append("archive keys")
        if bad:
            bad_pts[k] = bad
    L.check(f"per-fixation records 1..{n_r} (gaze, kind, decision, candidates, correspondence, fusion, coverage, "
            "SEEN / DEPTH)", not bad_entry, bad_entry[:MAX_LISTED] or None)
    L.check(f"LOCAL / GLOBAL sequence 1..{n_r}", kinds == [s["kind"] for s in S[:n_r]])
    L.check(f"fixation directions in order 1..{n_r}",
            [(e["yaw_deg"], e["pitch_deg"]) for e in T] == [(s["yaw_deg"], s["pitch_deg"]) for s in S[:n_r]])
    L.check(f"per-fixation files present 1..{n_r}", not missing, missing[:MAX_LISTED] or None)
    L.check(f"calibration.json byte-identical 1..{n_r}", not bad_calib, bad_calib[:MAX_LISTED] or None)
    L.check(f"points.npz {'/'.join(POINTS_BITWISE)} bitwise 1..{n_r}", not bad_pts,
            dict(list(bad_pts.items())[:MAX_LISTED]) or None)

    # ---- policy decision states
    n_dec = n_r - 1 if prefix else n_r
    bad_pol, missing_pol = [], []
    for s in S[:n_dec]:
        k = int(s["n"])
        p = run / f"policy/state-{k:04d}.npz"
        if p.is_file() != (s["policy"] is not None):
            missing_pol.append(k)
            continue
        if p.is_file() and npz_digests(p, POLICY_BITWISE) != s["policy"]:
            bad_pol.append(k)
    if not prefix:
        n_run_states = len(list((run / "policy").glob("state-*.npz")))
        n_sig_states = sum(s["policy"] is not None for s in S)
        if n_run_states != n_sig_states:
            missing_pol.append(f"{n_run_states} states vs {n_sig_states}")
    L.check(f"policy decision states present 1..{n_dec}", not missing_pol, missing_pol[:MAX_LISTED] or None)
    L.check(f"policy decision states bitwise 1..{n_dec}", not bad_pol, bad_pol[:MAX_LISTED] or None)

    # ---- coverage
    c = sig["coverage"]
    with np.load(run / "coverage.npz", allow_pickle=False) as z:
        sh, dh, st, rw, keys = (np.array(z["seen_history"]), np.array(z["depth_history"]), np.array(z["state"]),
                                np.array(z["row_weights"]), sorted(z.files))
    L.check("coverage.npz keys and row weights bitwise", keys == c["keys"] and array_digest(rw) == c["row_weights"])
    L.check(f"SEEN history 1..{n_r} bitwise", array_digest(sh) == array_digest(from_hex(c["seen_history"], n_r)))
    L.check(f"DEPTH history 1..{n_r} bitwise", array_digest(dh) == array_digest(from_hex(c["depth_history"], n_r)))
    if prefix:
        ref = S[n_r - 1]["policy"]
        want = ref["bitwise"].get("state") if ref else None
    else:
        want = c["state"]
    L.check(f"coverage state after fixation {n_r} bitwise", array_digest(st) == want)
    if prefix:
        return L

    # ---- final map
    m = npz_digests(run / "final-map.npz", MAP_BITWISE)
    L.check("final-map.npz keys", m["keys"] == sig["final_map"]["keys"], [m["keys"], sig["final_map"]["keys"]])
    bad = [k for k in MAP_BITWISE if m["bitwise"].get(k) != sig["final_map"]["bitwise"].get(k)]
    L.check(f"final-map.npz {' / '.join(MAP_BITWISE)} bitwise", not bad, bad or None)
    L.check("final-map-oracle-segmentation.ply byte-identical",
            sha256_file(run / "final-map-oracle-segmentation.ply") == sig["segmentation_ply"])
    L.check("final-map.ply header and XYZ bitwise", ply_header_and_xyz(run / "final-map.ply") == sig["display_ply"])
    return L


def finish(L: Ledger, out: Path | None) -> int:
    for r in L.rows:
        if "check" in r:
            tail = "" if r["pass"] or r["detail"] is None else f"  {json.dumps(r['detail'])[:300]}"
            print(f"{PREFIX} {'PASS' if r['pass'] else 'FAIL'}  {r['check']}{tail}", flush=True)
        else:
            print(f"{PREFIX} ....  {r['report']}: {json.dumps(r['value'])[:300]}", flush=True)
    ok = L.all_pass()
    if out is not None:
        out.write_text(json.dumps({"schema": "foveal-playground-compare-v1", "all_pass": ok, "rows": L.rows},
                                  indent=1) + "\n")
    print(f"{PREFIX} {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------ selftest: a synthetic run, then perturbations
def _write_run(d: Path, n: int, rng_seed: int = 7) -> None:
    """A tiny synthetic run with the real product layout (not the real scene; no Blender)."""
    rng = np.random.default_rng(rng_seed)
    d.mkdir(parents=True)
    T = []
    for k in range(1, n + 1):
        fd = d / f"fixations/fix-{k:04d}"
        fd.mkdir(parents=True)
        (fd / "calibration.json").write_text(json.dumps({"gaze_yaw_pitch_deg": [k * 1.5, -k * 0.5]}) + "\n")
        np.savez_compressed(fd / "points.npz", xyz_h=rng.normal(size=(50, 3)).astype(np.float32),
                            rgb=rng.random((50, 3)).astype(np.float32), instance_id_oracle=rng.integers(0, 9, 50),
                            err_mm=rng.random(50).astype(np.float32))
        if k < n:
            (d / "policy").mkdir(exist_ok=True)
            np.savez_compressed(d / f"policy/state-{k:04d}.npz", state=rng.integers(0, 3, (18, 36)).astype(np.int8),
                                core_valid=rng.random((16, 16)) > 0.5, gaze=np.array([k * 1.5, -k * 0.5]),
                                visited=rng.normal(size=(k, 3)))
        T.append({"n": k, "kind": "initial" if k == 1 else "local", "yaw_deg": k * 1.5, "pitch_deg": -k * 0.5,
                  "seconds": {"total": rng.random()}, "render_seconds_lr": {"L": 1.0, "R": 1.0},
                  "seen_fraction": k / n, "depth_fraction": k / (2 * n),
                  "next": None if k == n else {"kind": "local", "yaw_deg": (k + 1) * 1.5},
                  "local_candidates": None if k == n else [[k, 0.5]]})
    t = {"schema": "GFE-v0-trajectory", "label": "SELFTEST", "spp": 64, "max_fix": 600, "stop_reason": "selftest",
         "fixations": n, "local_saccades": n - 1, "global_saccades": 0, "final_seen_fraction": 1.0,
         "final_depth_fraction": 0.5, "final_surfels": 100, "monotone_cells": True, "server_exit": 0,
         "parameters": {"GRID_DEG": 1.0}, "server": {"device": "OPTIX"}, "trajectory": T}
    (d / "trajectory.json").write_text(json.dumps(t, indent=1) + "\n")
    last_state = rng.integers(0, 3, (18, 36)).astype(np.int8)
    np.savez_compressed(d / "coverage.npz", state=last_state, seen_history=np.linspace(0.1, 1.0, n),
                        depth_history=np.linspace(0.05, 0.5, n), row_weights=np.ones(18))
    np.savez_compressed(d / "final-map.npz", **{k: rng.normal(size=(100, 3)) for k in MAP_BITWISE}, rgb=rng.random(5))
    rec = np.zeros(100, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"] = rng.normal(size=100)
    (d / "final-map.ply").write_bytes(b"ply\nend_header\n" + rec.tobytes())
    (d / "final-map-oracle-segmentation.ply").write_bytes(b"ply\nend_header\n" + rec.tobytes())
    (d / "checks.json").write_text(json.dumps({"all_pass": True}) + "\n")
    _refreeze(d)


def _refreeze(d: Path) -> None:
    (d / "freeze.json").write_text(json.dumps({"files": {f: sha256_file(d / f) for f in FROZEN}}) + "\n")


def _prefix_of(base: Path, d: Path, n: int) -> None:
    """A shorter run cut from a synthetic base, the way a real smoke run relates to the full run."""
    shutil.copytree(base, d)
    t = read_json(d / "trajectory.json")
    t["trajectory"] = t["trajectory"][:n]
    t["fixations"] = n
    t["stop_reason"] = f"fixation cap {n}"
    (d / "trajectory.json").write_text(json.dumps(t, indent=1) + "\n")
    for k in range(n + 1, len(read_json(base / "trajectory.json")["trajectory"]) + 1):
        shutil.rmtree(d / f"fixations/fix-{k:04d}")
    for k in range(n, 100):
        (d / f"policy/state-{k:04d}.npz").unlink(missing_ok=True)
    with np.load(base / f"policy/state-{n:04d}.npz") as z:
        st = np.array(z["state"])
    with np.load(base / "coverage.npz") as z:
        c = {k: np.array(z[k]) for k in z.files}
    np.savez_compressed(d / "coverage.npz", state=st, seen_history=c["seen_history"][:n],
                        depth_history=c["depth_history"][:n], row_weights=c["row_weights"])
    _refreeze(d)


def selftest(_a) -> int:
    results = []

    def expect(name: str, ledger: Ledger, want_pass: bool) -> None:
        got = ledger.all_pass()
        results.append(got == want_pass)
        fails = [r["check"] for r in ledger.rows if "check" in r and not r["pass"]]
        print(f"{PREFIX} selftest {'ok ' if got == want_pass else 'BAD'} {name}: comparison "
              f"{'PASS' if got else 'FAIL'} (expected {'PASS' if want_pass else 'FAIL'})"
              f"{'' if not fails else '  <- ' + '; '.join(fails)[:160]}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = tmp / "base"
        _write_run(base, 6)
        sig = json.loads(json.dumps(make_signature(base)))          # through JSON, as when read from disk
        expect("identical full run", compare(base, sig, known=False), True)
        _prefix_of(base, tmp / "prefix", 4)
        expect("prefix run (first 4 of 6)", compare(tmp / "prefix", sig, known=False), True)

        def perturbed(name: str, edit, want_pass: bool = False) -> None:
            d = tmp / name.replace(" ", "-")
            shutil.copytree(base, d)
            edit(d)
            _refreeze(d)
            expect(name, compare(d, sig, known=False), want_pass)

        def flip_point(d: Path) -> None:
            p = d / "fixations/fix-0003/points.npz"
            with np.load(p) as z:
                a = {k: np.array(z[k]) for k in z.files}
            a["xyz_h"].view(np.uint32)[0, 0] ^= 1                    # one ULP in one coordinate
            np.savez_compressed(p, **a)

        def change_rgb(d: Path) -> None:
            p = d / "fixations/fix-0003/points.npz"
            with np.load(p) as z:
                a = {k: np.array(z[k]) for k in z.files}
            a["rgb"] += 0.01
            np.savez_compressed(p, **a)

        def edit_traj(d: Path, fn) -> None:
            t = read_json(d / "trajectory.json")
            fn(t)
            (d / "trajectory.json").write_text(json.dumps(t, indent=1) + "\n")

        def edit_npz(rel: str, key: str, fn):
            def go(d: Path) -> None:
                p = d / rel
                with np.load(p) as z:
                    a = {k: np.array(z[k]) for k in z.files}
                a[key] = fn(a[key])
                np.savez_compressed(p, **a)
            return go

        perturbed("one ULP in a reconstructed point", flip_point)
        perturbed("a different gaze", lambda d: edit_traj(d, lambda t: t["trajectory"][4].update(yaw_deg=7.6)))
        perturbed("a different candidate list", lambda d: edit_traj(
            d, lambda t: t["trajectory"][1].update(local_candidates=[[2, 0.6]])))
        perturbed("a run-level parameter", lambda d: edit_traj(d, lambda t: t.update(parameters={"GRID_DEG": 2.0})))
        perturbed("one policy-state cell", edit_npz("policy/state-0002.npz", "state",
                                                    lambda s: np.where(np.arange(s.size).reshape(s.shape) == 0,
                                                                       (s + 1) % 3, s).astype(s.dtype)))
        perturbed("SEEN history stored as float32", edit_npz("coverage.npz", "seen_history",
                                                             lambda s: s.astype(np.float32)))
        perturbed("a final-map value", edit_npz("final-map.npz", "support_count", lambda s: s + 1e-12))
        perturbed("a calibration byte", lambda d: (d / "fixations/fix-0002/calibration.json").write_text(
            (d / "fixations/fix-0002/calibration.json").read_text() + " "))
        perturbed("a missing policy state", lambda d: (d / "policy/state-0004.npz").unlink())
        d = tmp / "stopped-early-by-itself"
        _prefix_of(base, d, 4)
        edit_traj(d, lambda t: t.update(stop_reason="no valid unseen fixation"))
        _refreeze(d)
        expect("a shorter run that stopped by itself (compared as full)", compare(d, sig, known=False), False)
        perturbed("RGB only (presentation; must still PASS)", change_rgb, want_pass=True)
        perturbed("timings only (metadata; must still PASS)", lambda d: edit_traj(
            d, lambda t: t["trajectory"][2].update(seconds={"total": 99.0})), want_pass=True)
        d = tmp / "tampered-after-freeze"
        shutil.copytree(base, d)
        (d / "final-map.ply").write_bytes((d / "final-map.ply").read_bytes() + b"x")
        expect("a product changed after the freeze", compare(d, sig, known=False), False)
        # literal known answer: the synthetic run is not the Playground-2 run, so it must FAIL
        expect("literal known answer on a non-baseline full run", compare(base, None, known=True), False)
        expect("literal smoke sequence on a prefix run without a signature", compare(tmp / "prefix", None), True)
        d = tmp / "prefix-with-a-global"
        shutil.copytree(tmp / "prefix", d)
        edit_traj(d, lambda t: t["trajectory"][2].update(kind="global"))
        _refreeze(d)
        expect("literal smoke sequence with a global jump at 3", compare(d, None), False)
    ok = all(results)
    print(f"{PREFIX} SELFTEST {'PASS' if ok else 'FAIL'} ({sum(results)}/{len(results)})", flush=True)
    return 0 if ok else 1


# ------------------------------------------------------------------ CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="compare RUN with the baseline signature and literal numbers")
    c.add_argument("--run", required=True)
    c.add_argument("--signature", default=str(SIGNATURE))
    c.add_argument("--out", default=None, help="record path (default: RUN/compare.json)")
    s = sub.add_parser("signature", help="maintainers: write the signature of a frozen baseline run")
    s.add_argument("--run", required=True)
    s.add_argument("--out", required=True)
    sub.add_parser("selftest", help="show that every comparator can fail (synthetic runs; no Blender)")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest(a)
    run = Path(a.run).resolve()
    missing = [f for f in ("trajectory.json", "freeze.json") if not (run / f).is_file()]
    if missing:
        raise SystemExit(f"{PREFIX} STOP {run} is not a finished run (no {', '.join(missing)})")
    if a.cmd == "signature":
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(make_signature(run), indent=1, sort_keys=True) + "\n")
        print(f"{PREFIX} wrote {out} ({out.stat().st_size:,d} bytes)", flush=True)
        return 0
    sp = Path(a.signature)
    sig = read_json(sp) if sp.is_file() else None
    return finish(compare(run, sig), Path(a.out) if a.out else run / "compare.json")


if __name__ == "__main__":
    sys.exit(main())
