# Foveal Playground — architecture

This is the architecture reference for the **Greedy Foveal Explorer**, the baseline of foveal-playground. It is adapted
from f3d-vision's canonical architecture document (`docs/architecture/greedy-foveal-playground.md` there); see
[provenance](provenance.md). Code: [`tools/greedy_foveal/`](../tools/greedy_foveal/).

Each number is labelled as MEASURED (with its run) or as a parameter. Numbers measured in f3d-vision are labelled so;
this repository reproduces the same control path (section 16).

---

## 1. Purpose

The playground answers one question with as little machinery as possible:

> Can a pair of foveated eyes on a fixed head reconstruct a whole 360° scene by looking around, one fixation at a time,
> using only what it has measured so far to decide where to look next?

The baseline does this in the Blender **Classroom** scene. It is deliberately small, a handful of source files, so that each
component can be swapped out and compared against a frozen, reproducible reference run.

What it is: a research / engineering playground, and a known-good baseline for comparisons.
What it is not: a general vision system, a framework, or an optimal exploration strategy.

## 2. Design principle

    FIXATE -> RECONSTRUCT EVERYTHING LOCALLY -> UPDATE GLOBAL H0 MAP + CYCLOPEAN COVERAGE
           -> LOCAL OR GLOBAL SACCADE -> REPEAT

1. **Fixate.** Point both eyes at a gaze direction. The head never moves.
2. **Reconstruct everything locally.** Every pixel of the 12° foveal core that can be matched between the eyes becomes
   a metric 3-D point. There is no object selection and no region of interest inside the fovea.
3. **Update memory.** Fuse the points into **one** global surfel map in the fixed head frame H0. Mark the gaze's
   footprint on a coarse sphere as SEEN, and as DEPTH where points landed.
4. **Saccade.** If the fovea's edge still shows geometry in the direction of an unexplored neighbouring lattice view,
   step there (local). Otherwise jump to the unvisited lattice view that adds the most unseen area (global).
5. **Repeat** until no lattice view adds unseen area (section 11).

Only four pieces of state cross from one fixation to the next: the map, the coverage sphere, the list of visited gaze
directions and the current gaze. There is no object scheduler, no per-object state and no controller state machine.

## 3. System context

```mermaid
flowchart TD
    B["Blender render server<br/>(observation service)"]
    V["local binocular view<br/>L / R raw rasters, 12° foveal core"]
    R["metric reconstruction<br/>PERFECT correspondence + spherical triangulation<br/>-> points in H0"]
    M["persistent H0 surfel map<br/>(12-mm fusion)"]
    C["cyclopean coverage<br/>1° sphere: UNSEEN / SEEN / DEPTH"]
    E["local edge evidence<br/>valid-geometry mask of the core"]
    P["saccade policy<br/>LOCAL or GLOBAL"]
    B -->|"RGB, Position, Object Index"| V
    V --> R
    R -->|"fuse"| M
    R -->|"point directions"| C
    R -->|"core_valid"| E
    C --> P
    E --> P
    P -->|"next gaze (yaw, pitch)"| B
```

How to read the diagram:

- The **render server** is the only thing that touches the scene. Given a gaze, it returns a binocular observation:
  two 640 × 640 raw images per fixation. Each one holds RGB plus the Blender Position and Object Index passes.
- **Reconstruction** turns the central 256 × 256 foveal core of the left eye into metric points in H0. It uses
  PERFECT / ORACLE correspondence (section 7) and the accepted spherical epipolar triangulation.
- The points feed three things. The **persistent map** keeps them, fused at 12 mm. The **coverage sphere** records
  which directions have been looked at and which got depth. The **edge evidence** is a 256 × 256 mask of the core
  pixels that produced a valid point.
- The **policy** reads only the coverage sphere, the current edge evidence and the visited list. It chooses the next
  gaze.

**An architectural fact to keep in mind.** In this baseline the global map is **written but never read during
control**. Coverage and edge evidence are computed from the *current fixation's* points, not by querying the map. The
map is the product of the run; the coverage sphere is the controller's memory. A future experiment can close the loop
through the map (section 15).

## 4. Persistent state

| state | code | type / shape | meaning | updated |
|---|---|---|---|---|
| global surfel map **M** | `explorer.GlobalMap` | arrays: `xyz` (N, 3) float64 in H0 (m), `rgb` (N, 3) float64, `inst` int32, `support` int32, `first_fix` int32, `patch_ids` list | every reconstructed surface element so far | `GlobalMap.fuse`, once per fixation |
| coverage **S** | `state` in `run.explore` | int8 array (180, 360), values `UNSEEN 0`, `SEEN 1`, `DEPTH 2` | `S_seen = state >= 1`, `S_depth = state == 2` on the 1° cyclopean sphere | `explorer.update_coverage`; values never decrease |
| visited gazes | `visited` | (n, 3) unit vectors in H0 | every fixation direction so far | appended once per fixation |
| fixation trajectory | `traj` | list of per-fixation records | gaze, how it was chosen, counts, timings | appended once per fixation |
| current gaze | `yaw, pitch` | degrees in H0 | where the eyes point now | set by the decision |

Nothing else persists. The eye images, the correspondences, the triangulated points (except as saved
`points.npz` records), the core-valid mask and the tangent frame are rebuilt at every fixation and then discarded.

## 5. One fixation

```mermaid
sequenceDiagram
    participant H as run.explore (host)
    participant S as render_server (Blender)
    participant X as explorer (host)
    H->>S: QUEUE/req-NNNN.json {n, yaw, pitch, out_dir}
    S->>S: eye calibration for (yaw, pitch), fixed head; render L and R (Cycles, 64 spp)
    S-->>H: calibration.json, raw_L.exr, raw_R.exr, then QUEUE/done-NNNN.json
    H->>H: check: head pose unchanged, rendered gaze == requested gaze
    H->>X: load_observation -> reconstruct (PERFECT correspondence + spherical triangulation)
    X-->>H: xyz (H0), rgb, oracle id, core_valid, counts
    H->>X: GlobalMap.fuse (12 mm)
    H->>X: update_coverage, fractions
    H->>X: decide (unless a stop condition holds)
    X-->>H: LOCAL / GLOBAL next gaze, or NONE
```

Step by step, for fixation *n* at gaze (yaw, pitch):

1. **Gaze.** The cyclopean gaze direction in H0 is
   `d = (sin yaw · cos pitch, sin pitch, −cos yaw · cos pitch)`. The first gaze is (0°, 0°), straight ahead.
2. **Binocular render.** `run.Server.render` writes a request file. `render_server.py` builds the accepted eye
   calibration. The two eyes sit at ±31.5 mm on the head X axis (63 mm baseline) and verge on the point 2.1 m along
   the gaze. Each eye's raw raster is 640 × 640 px, focal 1217.84 px (29.4° field of view). Both eyes are rendered
   (Cycles OPTIX, 64 spp, box filter, no denoising, fixed seeds L 2111 / R 2112) into uncompressed multilayer EXRs with
   Combined, Position and Object Index.
3. **PERFECT correspondence.** `explorer.perfect_correspondence` is applied to the left eye's nominal core: raw pixels
   192..447 in x and y, which is 65,536 pixel centres covering about 12°. For each pixel it finds the exact matching
   position in the right image (section 7). The output is a list of `(left pixel, right sub-pixel position)` pairs.
4. **Spherical triangulation.** The accepted AB1b `compute_epipolar` turns each pair of rays into a 3-D point in H0.
   The pairs are expressed in baseline-polar epipolar coordinates (θ measured from the baseline +X; φ the rotation
   about it). Points that are not valid or not finite are dropped.
5. **H0 XYZ.** The result is typically close to the 65,536 core pixels; MEASURED in f3d-vision (Engineering-1 smoke run):
   63,528 – 65,536 points at fixations 66–75. Each point carries the left-eye RGB and the oracle Object Index of its
   pixel. The diagnostic error against the left Position pass (median 0.02 – 0.28 mm at those fixations) is recorded
   but never used.
6. **12-mm fusion.** `GlobalMap.fuse` merges the points into M (section 8).
7. **Coverage update.** `update_coverage` sets the gaze's 12° footprint to at least SEEN, and to DEPTH the footprint
   cells that contain a reconstructed point's direction. `fractions` returns the SEEN and DEPTH shares of 4π.
8. **Next gaze.** Unless a stop rule fires (section 11), `decide` returns LOCAL, GLOBAL or NONE (section 10). The
   decision inputs (coverage, core mask, gaze, visited list) are saved to `policy/state-NNNN.npz` so that every
   decision can be replayed.

Per-fixation cost, MEASURED in f3d-vision on the frozen v0 grow600 run (`trajectory.json` `seconds_sum`, 542 fixations, 729 s wall):
render 412 s, fusion 255 s, correspondence 18 s (EXR loading 9 s of it), geometry 13 s, policy 3 s. Rendering one
64-spp pair takes about 0.75 s. Fusion grows with the map, from about 0.1 s early to up to 2.9 s late.

## 6. Coordinate systems

| frame | definition | lifetime |
|---|---|---|
| **world W** | Blender world coordinates, +Z up, metres | persistent (scene only) |
| **fixed head H0** | origin at `head_origin_w_m` = (−0.6, −1.0, 1.2) in W; `p_H = (p_W − o) · R_wh` (`fsg_geometry.world_to_head`). Axes: **+X right** (the eye baseline), **+Y up** (= world +Z), **−Z forward** (= world +Y). Metres. | **persistent**: the map, the gazes and the coverage directions all live here |
| **eye cameras L / R** | centres at (∓0.0315, 0, 0) in H0; rotation `R_hc` per eye (OpenCV convention: x right, y down, z forward). Each eye rotates about its own centre. The two verge on the point 2.1 m along the gaze; the image +x stays aligned with the baseline projected into the image plane. Pixel `(u, v)` with integer pixel centres and rows top-down. | temporary (one fixation; stored in `calibration.json`) |
| **current tangent frame** | `explorer.frame(yaw, pitch)`, with columns x = head +X projected onto the plane orthogonal to the gaze, y = z × x (image down), z = gaze | temporary: the footprint and the 8 local neighbours are defined in it |
| **epipolar (θ, φ)** | for a head-frame direction d: θ = angle from +X (the baseline); φ = atan2(d_y, −d_z), the rotation about X. Corresponding directions share φ, and the parallax θ_R − θ_L > 0. | temporary (inside triangulation) |
| **cyclopean sphere** | directions from the H0 **origin** (the midpoint between the eyes): yaw = atan2(x, −z), pitch = asin(y); a 1° equirectangular grid, 180 rows × 360 columns. Row 0 is the top (pitch +89.5° centre); column 0 is yaw −180°. | **persistent**: the coverage state |

The rule of thumb: **anything stored across fixations is in H0** (points) or **on the cyclopean sphere about the
H0 origin** (coverage, gazes). Eye frames, pixel coordinates, the tangent frame and the epipolar coordinates exist only
inside one fixation.

Because the eyes are only ±31.5 mm from the cyclopean origin, the cyclopean footprint and the actual left-eye view
differ by a small parallax. DEPTH is therefore always intersected with the cyclopean footprint.

## 7. Observation / measurement service

The baseline uses two services, and both are replaceable.

**Rendering** (`render_server.py`, driven by `run.Server`). It runs as a persistent Blender process: Classroom is loaded
once and only the eye cameras move, which gives about 0.75 s per 64-spp pair. The protocol is file based and
atomic:

- the host writes `QUEUE/req-NNNN.json` `{n, yaw_deg, pitch_deg, out_dir}`;
- the server writes `OUT/calibration.json`, `OUT/raw_L.exr`, `OUT/raw_R.exr`, then `QUEUE/done-NNNN.json`;
- a failure writes `QUEUE/fail-NNNN.json` and exits nonzero; `QUEUE/stop` ends the loop.

At start-up the server checks that the scene's EYE pose equals the declared fixed head H0 (`head_pose.py`, within
1e-6).
It assigns Object Index labels and keeps only the Combined, Position and Object Index passes.

**Correspondence** (`explorer.perfect_correspondence`). This is the **PERFECT / ORACLE** matcher.

- *Positive Object Index* (catalogued objects): the accepted AB1b rule, unchanged. The left pixel's Position is
  projected into the right raw camera. The pair is kept when the nearest right pixel sees the same Object Index (the
  visibility test). `uv_R` is the exact continuous projection.
- *Object Index 0* (geometry without a catalogue Object Index, which the AB1b rule excludes): the same projection. The
  pair is kept when the nearest right pixel is a finite hit with Object Index 0 whose Position lies within
  5 mm + 1 % of range of the left point (the right eye sees the same surface point; otherwise it is occluded).

**The interface is the important part.** The matcher returns a *truth-stripped* product: `left_core_row`,
`left_core_col`, `uv_L` and `uv_R`. Geometry consumes only this product plus the calibration. It never sees Position
or Object Index.

**Natural stereo plugs in here.** A natural matcher must produce the same four keys from the left and right RGB. It
also needs `load_observation` to return `rgb_R` (today only `rgb_L` is loaded). Triangulation, fusion, coverage and the
policy then run unchanged. f3d-vision's earlier natural-stereo studies (AB1d – AB1d3, `tools/active_bootstrap/` there)
are a starting point.

## 8. Reconstruction / memory

**Local reconstruction** (`explorer.reconstruct`): correspondence, then `ab1b_geometry.compute_epipolar`, keeping
points that are `valid_epi` and finite. Each point keeps:

- the left-eye RGB at its pixel (sensory, used for display);
- the oracle Object Index (ORACLE / VISUALIZATION ONLY);
- the core position (for `core_valid`).

**The one global map** (`explorer.GlobalMap`) follows the accepted FSG3 12-mm association / fusion rule
(`fsg3_surface_map.fuse`) exactly, with three changes: a vectorized spatial hash; no instance filter (all geometry is
one map); no 63-patch provenance cap (support counts only).

For each new patch (one fixation's points), in order:

1. **Snapshot association.** Each point is matched to the nearest *existing* surfel strictly within
   **12 mm**, searched over the 27 neighbouring 12-mm hash cells. The search is exact; ties go to the smallest surfel
   index. Points of the same patch never match each other.
2. **One contribution per surfel.** All patch points matched to the same surfel are averaged into one contribution
   (mean XYZ, mean RGB). That contribution is blended in with weight `support`:
   `x <- (x · support + mean) / (support + 1)`, then `support += 1`.
3. **New surfels.** Every unmatched point becomes a new surfel (support 1, `first_fix = n`, `inst` = the point's oracle
   id).
4. A repeated patch id is refused.

| surfel attribute | meaning |
|---|---|
| `xyz_h` | position in H0 (m), float64 |
| `rgb` | blended linear RGB (display only) |
| `instance_id_oracle` | oracle Object Index of the creating point (visualization only) |
| `support_count` | number of patches that contributed |
| `first_fixation` | fixation that created the surfel |
| `patch_ids`, `radius_cell_m` | provenance and the fusion parameters |

**Current storage behavior.** The map is unoptimized on purpose. Unmatched points are not de-duplicated against each
other, so each first-time point becomes a surfel. Arrays are float64 and grow by doubling. MEASURED in f3d-vision on the v0 grow600 run:
16,517,811 surfels; `final-map.npz` 991 MB; the full run directory 2.2 GB, mostly the per-fixation `points.npz`
records. Fusion is the largest host cost (255 of 729 s).

## 9. Cyclopean coverage representation

- **Grid.** 1° cells on the cyclopean sphere: 180 rows (pitch +90° → −90°) × 360 columns (yaw −180° → +180°).
- **States.** `UNSEEN 0`, `SEEN 1` (looked at), `DEPTH 2` (looked at and reconstructed). The update uses `max` for SEEN
  and sets DEPTH directly, so a cell's state never decreases. `run.checks` verifies this.
- **SEEN footprint** (`footprint`). A fixation sees the cells whose centre falls inside the nominal 12° square core in
  the gaze's tangent plane, that is `|u|, |v| ≤ tan 6°` after projecting the cell direction onto the tangent frame.
- **DEPTH.** Footprint cells that contain at least one reconstructed point's direction from the H0 origin.
- **Solid-angle accounting.** Every fraction is weighted by exact cell solid angle,
  `Ω_row = Δλ · (sin φ_top − sin φ_bottom)` (`row_weights`). The weights sum to 4π, so polar cells count for what they
  cover, not one unit each. `fractions` returns SEEN and DEPTH as fractions of 4π.
- **Seams.** Longitude wraps: column indices are taken modulo 360, and the neighbour tests use `np.roll`. In the
  UNSEEN component labelling, the polar rows link across the pole (column c to column c + 180). Pitch is clipped to the
  grid. The footprint itself is computed from direction dot products, so it has no seam.

## 10. Greedy gaze policy

`explorer.decide(state, core_valid, yaw, pitch, visited)` is a pure function of the coverage, the current core mask,
the current gaze and the visited directions. It is deterministic: `run.checks` re-decides every saved state.

**The gaze lattice.** Every gaze is one of a fixed set of 356 views, built once by `build_baseline_lattice()`
(Playground-2; its contract is f3d-vision's `docs/playground/sparse-lattice-contract.md`).

- A footprint is a 12° square in its gaze's tangent plane, and its x axis is the eye baseline (head +X) projected into
  that plane (`frame`). The squares are therefore aligned with circles about the baseline axis.
- The lattice puts 17 rings about that axis at θ = 6°, 16.5°, …, 174° from +X, with gazes at φ = j · 360°/n about it
  (φ = 0 forward). Each ring has the fewest gazes that cover its band, so every square sits square-on to its ring.
- MEASURED (`check_lattice.py`): the ring sizes are 6, 11, 16, 20, 24, 27, 29, 30, 30, 30, 29, 27, 24, 20, 16, 11 and
  6 gazes. The footprints contain every 1° cell centre, and no gaze is redundant.
- No gaze is closer than 6° to the baseline axis, and the start gaze (0°, 0°) is a lattice gaze.
- `LATTICE_KIND = "pitch"` rebuilds Playground-1's 496-gaze pitch-ring lattice for comparison.

A lattice gaze is visited when a fixation lies within 0.5° of it.

**LOCAL.** The candidates are the unvisited lattice gazes within 16° of the current gaze.

- `EDGE_SUPPORT` is the fraction of valid-geometry pixels in the outer quarter (64 px) of the 256-px core, on the side
  (or in the corner block) of the octant that holds the candidate's offset in the current tangent frame. It asks
  whether the fovea's own measurement shows surface continuing that way.
- `UNSEEN_GAIN` is the UNSEEN solid-angle fraction of the candidate's footprint.
- A candidate is eligible when `EDGE_SUPPORT ≥ 0.25` and `UNSEEN_GAIN > 0`.
- The highest `EDGE_SUPPORT × UNSEEN_GAIN` wins; ties go to the lower lattice index.

**GLOBAL.** Used when no local candidate is eligible: the unvisited lattice gaze with the largest `UNSEEN_GAIN`
(lowest index on ties). If no unvisited lattice gaze has any unseen gain: NONE, and the run stops.

**No object scheduler.** No object identity, segmentation or per-object state enters the decision.

| parameter (`explorer.py`) | value | role |
|---|---|---|
| `CORE_FOV_DEG` | 12.0 | nominal square fovea |
| `LATTICE_KIND` | `"baseline"` | the lattice construction (`"pitch"` = Playground-1) |
| `LATTICE_CAP_DEG`, `LATTICE_AXIS_STEP_DEG` | 6.0, 10.5 | first ring's angle from the baseline axis; ring spacing |
| `LATTICE_NEIGHBOUR_DEG` | 16.0 | local candidates: unvisited lattice gazes this close |
| `LATTICE_VISIT_DEG` | 0.5 | a lattice gaze counts as visited within this angle |
| `EDGE_BAND_PX` | 64 | edge band width (core / 4) |
| `EDGE_MIN`, `GAIN_MIN` | 0.25, 0.0 | eligibility: edge support ≥ 0.25, unseen gain > 0 |
| `VISIT_TOL_DEG` | 2.0 | "same fixation" tolerance of `run.checks` |
| `SEEN_STOP` | 2.0 | unreachable on purpose: the stop comes from `decide` returning NONE |
| `LOCAL_STEP_DEG`, `LATTICE_RING_DEG` | 7.2, 10.0 | v0's neighbour offset (recorded, unused); Playground-1's pitch-ring spacing |
| `ASSOC_RADIUS_M`, `ASSOC_CELL_M` | 0.012, 0.012 | fusion radius and hash cell |
| `ZERO_TOL_ABS_M`, `ZERO_TOL_REL` | 0.005, 0.01 | instance-0 correspondence tolerance |
| `GRID_DEG` (with `H`, `W`) | 1.0 (180, 360) | coverage resolution |

The v0 policy (8 tangent-frame neighbours 7.2° away with `GAIN_MIN` 0.25, and a jump to the deepest cell of the largest
UNSEEN component) is at f3d-vision tag `greedy-foveal-playground-baseline`.

Every run records these in `trajectory.json` under `parameters`.

## 11. Stop rule

After each fixation's coverage update, `run.explore` checks, in this order:

```mermaid
flowchart TD
    A["fixation n: map fused, coverage updated"] --> S{"SEEN >= SEEN_STOP? (2.0: never)"}
    S -- yes --> X1["STOP: seen >= 99 % of 4 pi (unreachable)"]
    S -- no --> K{"n == max-fix?"}
    K -- yes --> X2["STOP: fixation cap"]
    K -- no --> L{"an eligible local lattice neighbour?"}
    L -- yes --> LOC["LOCAL saccade"]
    L -- no --> G{"an unvisited lattice gaze with unseen gain?"}
    G -- yes --> GLO["GLOBAL saccade"]
    G -- no --> X3["STOP: no valid unseen fixation"]
    LOC --> N["fixation n + 1"]
    GLO --> N
```

1. **SEEN ≥ `SEEN_STOP`**: set to 2.0, so it never fires. `run.py` is unchanged, and its fixed text "seen >= 99 % of
   4 pi" would misreport the stop. v0 used 0.99 here.
2. **The fixation cap** (`--max-fix`): a safety cap. The baseline uses 600 and stops at 356 without reaching
   it.
3. **No candidate** (`decide` returns NONE): no lattice gaze adds unseen area. Since the lattice covers every cell,
   this is the scientific stop: 100 % SEEN.

## 12. What a run measures (and does not)

`explore` ends by writing `freeze.json`: the sha256 of `trajectory.json`, `final-map.npz`, `coverage.npz` and both
PLYs. `checks` and `compare.py` verify that the products did not change after the freeze.

What a run reports about itself, during control: the fixation count and the LOCAL / GLOBAL split; the SEEN and DEPTH
fractions of 4π after every fixation (`coverage.npz`); the surfel count; and the wall time with its breakdown.

What it does not report: how *good* the reconstruction is. f3d-vision measured that post hoc, after the freeze, against
a full-sphere first-hit render from the head origin (Breadth-1): 98.82 % of first-hit solid angle had a surfel within
12 mm. That evaluation, its reference file and the replay video that used it are not part of this repository.
Building an evaluation is exercise E3 in [docs/exercises.md](exercises.md). Whatever you build, keep the f3d-vision
rule: the reference is opened only after the run is frozen, so that nothing in control can be tuned against it.

## 13. Truth / oracle boundaries

| category | what | where | may it influence gaze, fusion or stopping? |
|---|---|---|---|
| **CONTROLLER-TIME** | coverage state, core mask, visited list, gaze, the map XYZ | `explorer.decide`, `update_coverage`, `GlobalMap` | yes: this *is* the control state |
| **PERFECT / ORACLE CORRESPONDENCE** | Blender Position and Object Index, used **only** to decide which left and right pixels correspond | `explorer.perfect_correspondence` | indirectly, as the measurement service. The output is truth-stripped, and the points come from triangulating pixel pairs, not from copying Position. |
| **DIAGNOSTIC** | the error of each point against the left Position | `reconstruct` → `err_mm` | **no** (recorded only) |
| **ORACLE SEGMENTATION / VISUALIZATION ONLY** | Object Index carried on points and surfels (`instance_id_oracle`) | segmentation PLY | **no** |
| **DERIVED** | figures and PLYs computed from run products | `visuals.py` | no |

## 14. Data products and software map

A run directory (`run.py explore --run RUN`):

| path | written by | content | frozen |
|---|---|---|---|
| `trajectory.json` | `explore` | run summary (code commit, parameters, server settings, stop reason, counts, timings) and one record per fixation (gaze, kind, how it was chosen, the full decision and its local candidates, correspondence / fusion / coverage counts, SEEN / DEPTH, the diagnostic error, timings) | yes |
| `coverage.npz` | `explore` | final `state`, `seen_history`, `depth_history`, `row_weights` | yes |
| `final-map.npz` | `explore` | the surfel map (section 8) | yes |
| `final-map.ply` | `explore` | map XYZ (float32) with display RGB | yes |
| `final-map-oracle-segmentation.ply` | `explore` | map XYZ coloured by oracle Object Index (VISUALIZATION ONLY) | yes |
| `fixations/fix-NNNN/calibration.json` | render server | the eye calibration of that fixation | — |
| `fixations/fix-NNNN/points.npz` | `explore` | that fixation's points: `xyz_h`, `rgb`, `instance_id_oracle`, `err_mm` | — |
| `fixations/fix-NNNN/raw_{L,R}.exr` | render server | raw observation; deleted unless `--keep-raw` | — |
| `policy/state-NNNN.npz` | `explore` | the inputs of decision n: `state`, `core_valid`, `gaze`, `visited` | — |
| `freeze.json` | `explore` | sha256 of the frozen products | — |
| `blender.log`, `queue/` | render server | server log, request / done files, `server-ready.json` | — |
| `checks.json` | `checks` | the minimal run checks | — |
| `compare.json` | `compare.py check` | the comparison with the baseline signature | — |

`run.py visualize --run RUN --vis VIS` writes `VIS/overview.png`.

The source files (`tools/greedy_foveal/`):

| file | runs in | key contents |
|---|---|---|
| `explorer.py` | host `.venv` | the parameters; the sphere grid; the observation (`load_observation`, `perfect_correspondence`, `reconstruct`); the map (`GlobalMap`); coverage (`update_coverage`, `fractions`); the lattice; the policy (`edge_support`, `unseen_gain`, `local_candidates`, `global_target`, `decide`) |
| `run.py` | host `.venv` | `Server` (starts and drives the render server); `explore` (the control loop and the freeze); `checks`; `visualize` |
| `render_server.py` | Blender's Python | `serve` (the request loop); `trim_passes` |
| `head_pose.py` | both | the declared fixed head H0 and its check (`eye_pose`, `matches_h0`) |
| `visuals.py` | host `.venv` | `overview` (the one-page figure); `display_rgb`, `oracle_colors` (also used for the PLYs) |
| `check_lattice.py` | host `.venv` | the lattice's fail-capable checks K1–K7, N1–N2, with a coverage-only replay of the policy; no Blender |
| `compare.py` | host `.venv` | `check` a run against the baseline signature and literal numbers; `selftest`; `signature` (maintainers) |
| `scene_preflight.py` | Blender's Python | EYE at H0, linked files present, an OPTIX device |

The explorer reuses f3d-vision modules, copied unchanged except where [provenance](provenance.md) says otherwise:

| dependency | module(s) | used by | for |
|---|---|---|---|
| sensor geometry | `tools/fsg_geometry.py` | `explorer`, AB1a helpers | `CORE_FOV_DEG`, `gaze_direction`, `camera_rotation_h`, `world_to_head`, `project_h`, `make_calibration` |
| spherical epipolar geometry | `tools/active_bootstrap/ab1b_geometry.py`, `ab1b_spec.py` | `explorer.reconstruct` | `compute_epipolar`; core size / origin |
| FSG3 fusion semantics | `tools/fsg3_surface_map.py` | `run.checks` only | the rule that `GlobalMap.fuse` re-implements; its bitwise known answer |
| PERFECT correspondence | `tools/active_bootstrap/ab1b_oracle.py` | `explorer.perfect_correspondence` | `compute_oracle`, `core_grid`, `left_hit`, `right_projection`, … |
| open guard | `tools/natural_bootstrap/nb1a_guard.py` | imported by `ab1b_oracle` / `ab1b_geometry` | import-time only; not called by the explorer |
| EXR reading | `tools/exr_lite.py` | `explorer.load_observation` | pure-Python uncompressed EXR reader |
| Blender acquisition | `tools/active_bootstrap/ab1a_render.py`, `ab1a_spec.py` | `render_server` | `calibration`, `configure`, `readback`, seeds, `DEVICE` |
| Classroom oracle render | `tools/classroom_oracle1_render.py`, `classroom_oracle1_public.py`, `fsg6f_public.py` | `render_server`, `head_pose` | `_head_pose`, `_assign_instance_ids`, `_eye_matrix`, `_prepare_perspective_pair` |
| foveated render | `tools/render_foveated.py`, `bl_common.py`, `warp.py` | `render_server` | `render_fixation` (Cycles device, seed, EXR output) |
| Visual Language 1 | `tools/visual_language/style.py` | `visuals` (and `run` at import, via `visuals`) | palette, fonts (DejaVu), text helpers |
| EYE placement | `tools/classroom_oracle/place_eye.py` | `scripts/setup_scene.sh` | builds `classroom_eye.blend` from the public `classroom.blend` |

The spec modules (`ab1a_spec.py`, `ab1b_spec.py`) still name f3d-vision's original experiment records (contracts,
commits, run paths). The playground does not read those records; they are kept so that the modules stay identical to
their accepted versions apart from one line (provenance).

## 15. Extension points — the playground map

Each row is one component that a single experiment can replace. "Consumes" and "produces" give the interface that the
rest of the loop relies on.

| component | current implementation | consumes | produces | likely future experiment |
|---|---|---|---|---|
| **renderer** | `render_server.py` + `run.Server`: Blender Cycles, persistent process, file queue | gaze (yaw, pitch), fixed head pose, spp | `calibration.json`; L / R raw rasters with RGB (+ Position, Object Index for the oracle) | another scene (Tabletop); 4096 spp; a rasterizer for speed; a real camera rig |
| **stereo matcher** | `explorer.perfect_correspondence` (AB1b oracle + instance-0 extension) | calibration; L / R observation | `{left_core_row, left_core_col, uv_L, uv_R}` | natural stereo (SGBM, learned matching) from `rgb_L` / `rgb_R`; also load `rgb_R` |
| **local measurement** | `explorer.reconstruct` → `ab1b_geometry.compute_epipolar` | calibration; correspondence product | valid points: `xyz` (H0), `rgb`, `inst`, `core_valid`, counts | per-point uncertainty (`kappa` is already computed); surface normals; a larger or variable fovea |
| **map representation** | `explorer.GlobalMap`: flat growable float64 arrays | — | the surfel set with attributes; `nearest` queries | voxel / TSDF / octree; float32; within-patch de-duplication |
| **fusion** | `GlobalMap.fuse` + `nearest` (FSG3 12-mm rule) | one patch: `xyz`, `rgb`, `inst`, fixation number | updated map; matched / new / affected counts | confidence-weighted fusion; outlier rejection; a different radius |
| **coverage resolution** | `explorer` grid (`GRID_DEG`, `H`, `W`, `ROW_W`, `footprint`, `update_coverage`) | gaze; the fixation's points | the UNSEEN / SEEN / DEPTH sphere; SEEN / DEPTH fractions | 0.5° or HEALPix grids; graded confidence states. Change `GRID_DEG`, `H` and `W` together. |
| **gaze lattice** | `build_baseline_lattice`, `lattice` (356 gazes on 17 baseline rings) | the footprint and frame conventions | the candidate gazes with their footprints | a sparser lattice (e.g. a 2° cap, 346 gazes); an adaptive or scene-dependent lattice |
| **local policy** | `local_candidates`, `edge_support`, `unseen_gain` (lattice gazes within 16°, `EDGE_SUPPORT × UNSEEN_GAIN`, gain > 0) | coverage, `core_valid`, gaze, visited | the best eligible lattice neighbour, or none | 3-D frontier edges from the map; information gain |
| **global policy** | `global_target` (the unvisited lattice gaze with the largest unseen gain) | coverage, visited | a target gaze, or none | the nearest useful gaze (shorter jumps); tour ordering |
| **stopping rule** | `decide` returning NONE + the `run.explore` loop (`SEEN_STOP` unreachable) | lattice gains, fixation number | stop reason | stop on DEPTH, on map growth or on a time budget |
| **segmentation / semantics** | none in control; the oracle Object Index is carried for visualization only | — | — | natural segmentation feeding identity; object-aware attention |
| **head motion** | none: the head is fixed (`head_pose.py`) and checked every fixation | — | — | head pose as an action; observations transformed into H0; well-conditioned stereo by recentering the head |
| **visualization** | `visuals.overview` | run products | figures, PLYs | an interactive viewer; a web page; a replay video (f3d-vision has one, `demo.py`) |

Two cautions:

- `run.py checks` contains known answers for the baseline. If you change fusion, its FSG3 bitwise check fails by
  design: replace it with a known answer for your rule.
- `explore` records the Git commit and whether the tracked tree was clean. Commit before you run.

- There is no quality evaluation in this repository: SEEN / DEPTH say where the eyes looked and where they got
  depth, not whether the depth is right. A project that changes the matcher, the measurement or fusion needs one
  (docs/exercises.md, E3).
## 16. Reproducibility baseline

| item | value |
|---|---|
| baseline | the sparse covering lattice of f3d-vision Playground-2, tag `playground-2-sparse-lattice` (`1f50bd6`), from f3d-vision `main` at `bbee8ef` |
| frozen run | f3d-vision `previews/playground-sparse-lattice/` (workstation only); its signature is `tests/known-answer/playground-2-signature.json` |
| command | `run.py explore --run RUN --max-fix 600 --spp 64`, then `checks`, then `compare.py check` |
| environment | Blender 5.2.1 LTS, Cycles OPTIX (RTX 4090), 64 spp, seeds L 2111 / R 2112 |

MEASURED result of the frozen run (f3d-vision Playground-2 and Engineering-2 reports):

- it stops at **fixation 356** (`no valid unseen fixation`) with **100.00 %** SEEN (0 UNSEEN cells) and **99.26 %**
  DEPTH;
- **315** local and **40** global saccades;
- **16,943,073** surfels, in 413–416 s on an RTX 4090.

**Determinism.** In f3d-vision the Position and Object Index passes, and therefore every control decision, point,
fusion and coverage value, reproduced bitwise run to run on the same workstation (MEASURED, Engineering-1 and -2:
23 / 23 equivalence checks). Cycles OPTIX RGB does not; RGB is display-only and `compare.py` never compares it.
Whether the control path is also bitwise across different GPU models, drivers or Blender builds has **not been
measured**. If `compare.py check` fails on your machine with an unmodified checkout, that is a finding worth reporting:
say which checks failed and how far the gazes and SEEN curve drift.

## 17. Limitations

- Fixed head; static synthetic Classroom; 64-spp synthetic rendering.
- PERFECT / ORACLE correspondence: Blender Position and Object Index decide the matches. Natural stereo is deferred.
- Oracle identity is used only for visualization. There is no natural segmentation.
- No reconstruction-quality evaluation (section 12).
- The fixation count is the lattice size: every one of the 356 lattice gazes is fixated once, and each still adds
  area. A smaller lattice is the way to fewer fixations. The 6° envelope around the baseline axis is a choice, not a
  necessity; v0 fixated as close as 0.71° (f3d-vision Playground-2 report, F1).
- The v0 trajectory, kept at its f3d-vision tag, has a diagonal bias, triangular holes and a long tail of single global jumps;
  the lattice removed both (Playground-1).
- Storage and speed are not optimized (section 8).
- The renderer requires an NVIDIA RTX (OPTIX) GPU; there is no CPU fallback (docs/exercises.md, E1).
- The map is not read by the policy (section 3).


## 18. How to start a project

1. **Branch** from `main`: `git switch -c project/<topic>`.
2. **Change one component** from section 15. Write down the question, the single variable and what you will compare,
   before you run.
3. **Smoke test** with `--max-fix 75` (about 1.5 min on an RTX 4090). If the change claims to preserve behavior (a
   refactor, a speed-up), `compare.py check` must pass. A real experiment is expected to diverge.
4. **Full run** into a new directory under `runs/`, then `checks` (adapt the known answers you deliberately changed)
   and `visualize` into `visuals/`.
5. **Compare with the baseline**: fixations to the stop; the SEEN / DEPTH curves (`coverage.npz`); the local / global
   split; surfels; wall time and its breakdown; and, if you built one, your evaluation.
6. **Report** the commit, the commands, the numbers (MEASURED, with their run) and what changed.
