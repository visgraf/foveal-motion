# Active vision under head motion

## Why this repository

foveal-playground explores a static scene from a **fixed** head with two eyes. fov-render can now move the head
between looks (fov-render #20: a per-look `head` in the request, D19; no body frame, D20). foveal-motion is where the
moving head is tried, starting monocular: one foveated eye, the cyclopean `C` at the head origin. The expectation
(Luiz, 2026-11-09) is that the foveated sensor helps under translation through the optical flow it sees.

Decisions (Luiz, 2026-11-12, in Chat):

| # | Question | Decision |
|---|---|---|
| M-D1 | Where the work lives | an independent repository, `visgraf/foveal-motion`, carrying foveal-playground's history, with fov-render as a submodule |
| M-D2 | The first setting | monocular, the cyclopean head, the Foveated sensor |
| M-D3 | The first test | the head moves straight from the back of the Classroom to the front, near the board |
| M-D4 | Where the eye looks during it | both, as two runs of the same path: fixating the board centre (`target`) and straight ahead (`ahead`) |
| M-D5 | Path and height | yarbus-agent's measured placements, back_of_room → near_board, at standing height 1.6 m |
| M-D6 | The dolly's hole limit (Luiz, 2026-11-12, issue #1) | 0.046 in `paths/classroom_dolly.json`: the scene's 0.025 failed at heads 6–46 (0.0250–0.0383), every hole on the right-hand window band; the measured maximum plus about a fifth, as D18 |
| M-D7 | The next step after M1 (Luiz, 2026-11-12, in Chat) | M2: range from the head's motion on the foveated raster, with M1's path run again fixating a target far off the path's line |
| M-D8 | The side target (Luiz, 2026-11-12, issue #3, option A) | accept what `target_ray` finds, `worldMap` at (−2.116, 3.540, 1.323): a hanging map 3.3 cm proud of the front wall, visible from every head, 22°→47° off the heading; the target sits 6.5 cm above the map's lower edge, so the fovea holds a small depth step. The spec is unchanged |

## Experiment M1, the dolly

**Path** ([`paths/classroom_dolly.json`](../paths/classroom_dolly.json)): from (0.897, −3.793, 1.6) to (0.897, 0.759,
1.6) m, 4.552 m along +Y toward the board, in 46 steps of 0.0990 m (47 looks), the head level and facing +Y (the
playground's H0 axes). The board centre (0.897, 3.259, 1.334) is 7.06 m away at the start and 2.51 m at the end.
**Looks:** fov-render's `Foveated()` (E2 2°, e_max 45°), eye `C`, profile `full` (253 × 253 raster, 50,269 rays),
64 spp, one `Session` for both modes, oracle access (Position, Object Index).

**Gaze modes.** `target`: `Gaze.verged_point(board centre)`; the eye turns to keep the board centre on the fovea, so
the retinal flow is translation plus the eye's compensating rotation, zero at the fixated point. `ahead`:
`Gaze.yaw_pitch(0, 0)`; the eye looks along the path, so the focus of expansion is the fovea's centre and the flow is
pure translation (looming).

**Preflight** (`motion.dolly preflight`, [`motion/bl_preflight.py`](../motion/bl_preflight.py)). fov-render's scene
gate checks holes, back-faces and clearance from the session's head only (its PLAN, "Head motion"). The preflight runs
the same sanity panorama (`fov_render.bl.scene.sanity_checks`, 720 px, 4 spp) from **every** head of the path, with the
scene's own limits (clearance 0.25 m, holes 0.025 for the Classroom's windows, D18), and from a control head 10 cm in
front of the board, which must fail. `run` refuses to start without a passing preflight.

**Checks** (`motion.dolly checks`, [`motion/analysis.py`](../motion/analysis.py)), each with a control that must fail:

| id | what | passes when | control (must fail) |
|---|---|---|---|
| M1.0 | each record agrees with its Position pass | p99.9 angle ≤ the coarsest foveated pixel (fov-render S4's limit) | directions turned by 1.5 coarsest pixels |
| M1.1 | each record sits on its path head | `head_origin_w`, `head_R_wh` equal the path's, exactly; eye centre within 1 µm | the path shifted by one step |
| M1.2 | in `target` mode the fovea's centre is on the board centre | the sample on the eye axis is within 2 pixels at its range of the target, every look | the target 10 cm sideways: no look on it |
| M1.3 | look k+1's surface reprojects onto look k | ≥ 0.80 of look k+1's hits that fall in look k's field land on a pixel whose own Position is within 2 pixels at the range (incidence up to ~70°), every pair | look k's eye centre 10 cm sideways must lose half of the agreeing points |

The 0.80 of M1.3 was set from first principles before any Classroom run (what is lost is occlusion, depth edges and
grazing surfaces); the calibration room gives 0.92–0.95 (below). If the Classroom falls under it, the reason is found
before anything moves (rule 2).

**The flow.** For each point that M1.3 finds in both looks (not occluded), its retinal displacement between looks k and
k+1 in the eye frame, in degrees and in raster pixels, against its eccentricity in look k+1 (`flow.npz`, and binned
medians in `checks.json`). `visualize` writes the contact sheet, the flow figure (curves by eccentricity, the flow field
of the middle step on the raster) and a side-by-side video.

## Results

### Chat sandbox, calibration room (CPU, `small` profile)

`python -m motion.dolly sandbox --run runs/calib-dolly` at `aa15748` (clean), fov-render `0f6fa7d`, bpy 5.2.1 wheel,
numpy 2.2.6, CPU, 16 spp, 126 × 126 raster. Path [`paths/calib_dolly.json`](../paths/calib_dolly.json): 2.1 m in 21
steps of 0.10 m toward the Siemens-star rings, 0.3 m left of and 0.1 m below the EYE's line. Chat sandbox numbers:

- Preflight: 22 heads pass (holes 0, nearest surface ≥ 0.711 m, no back-faces); the control head (0.10 m from the far
  wall) fails on clearance.
- M1.0–M1.3 all PASS, every control fails. M1.1 errors are exactly 0. M1.2: `target` within 4.4 mm of the target
  (tolerance up to 12.6 mm); `ahead` never on it. M1.3: agreeing fraction 0.922–0.934 (`target`), 0.937–0.949
  (`ahead`); with the control at most 0.341 and 0.335.
- 44 looks in 29.2 s, the whole sandbox in 47 s.

Median flow per 10 cm step, degrees | raster pixels, by eccentricity (Chat sandbox, calibration room):

| eccentricity (deg) | 0–1 | 1–2 | 2–5 | 5–10 | 10–20 | 20–30 | 30–45 |
|---|---|---|---|---|---|---|---|
| `target` | 0.095 \| 0.75 | 0.134 \| 0.89 | 0.157 \| 0.65 | 0.234 \| 0.61 | 0.348 \| 0.52 | 0.594 \| 0.49 | 0.998 \| 0.55 |
| `ahead` | 0.012 \| 0.09 | 0.028 \| 0.16 | 0.074 \| 0.25 | 0.146 \| 0.30 | 0.285 \| 0.34 | 0.508 \| 0.39 | 0.962 \| 0.52 |

A first reading, to be checked on the Classroom: in degrees the flow grows with eccentricity in both modes, as
translational flow does; on the log-polar raster it is nearly flat, a few tenths of a pixel per step from 5° to 45°,
because the sensor's pixels grow with eccentricity about as the flow does. Looking ahead leaves the fovea almost still
(the focus of expansion); fixating the target moves the near-foveal background most in pixels, the parallax around the
fixated point.

### Classroom (workstation)

Measured by Code (issue #1): `python -m motion.dolly preflight | run | checks | visualize --path
paths/classroom_dolly.json --run runs/dolly-1 --device OPTIX` at `8c1856a` (clean), fov-render `0f6fa7d` (clean),
Blender 5.2.1, OPTIX, RTX 4090, profile `full` (253 × 253 raster), 64 spp. Records: [`docs/runs/issue-1/`](runs/issue-1/)
(`preflight.json`, `run.json`, `checks.json`, the figures).

- Preflight: all 47 heads pass with M-D6's hole limit 0.046 (holes 0.0205–0.0383, every hole on the right-hand window
  band; nearest surface ≥ 0.793 m; back-faces on closed meshes ≤ 5.8e-6); the control head fails on clearance
  (0.100 m). Under the scene's 0.025 heads 6–46 had failed ([`preflight-limit-0.025.json`](runs/issue-1/preflight-limit-0.025.json)).
- M1.0–M1.3 all PASS, every control fails. M1.0: p99.9 ≤ 0.00082° (limit 1.17°). M1.1 errors are exactly 0. M1.2:
  `target` within 7.7 µm of the board centre (tolerance 4.4–12.4 mm); `ahead` never on it (its fovea is 0.266 m
  higher, the eye height above the board centre). M1.3: agreeing fraction 0.900–0.989 (`target`), 0.904–0.990
  (`ahead`), lowest at the first pair and rising toward the board; with the control at most 0.040 and 0.038.
- 94 looks in 19.5 s (session start 0.5 s; the first look 0.82 s, then 0.18–0.21 s a look, of which 0.06 s render);
  the preflight's 48 panoramas (720 px, 4 spp) in 46–56 s.

Median flow per 9.9 cm step, degrees | raster pixels, by eccentricity (Code, workstation, Classroom):

| eccentricity (deg) | 0–1 | 1–2 | 2–5 | 5–10 | 10–20 | 20–30 | 30–45 |
|---|---|---|---|---|---|---|---|
| `target` | 0.013 \| 0.19 | 0.031 \| 0.35 | 0.071 \| 0.51 | 0.150 \| 0.64 | 0.311 \| 0.74 | 0.600 \| 0.93 | 1.062 \| 1.18 |
| `ahead` | 0.013 \| 0.20 | 0.031 \| 0.35 | 0.072 \| 0.52 | 0.151 \| 0.64 | 0.312 \| 0.75 | 0.606 \| 0.94 | 1.066 \| 1.16 |

A first reading (Code, to be checked): the two modes give the same flow to within 0.01°. Here the board centre is
only 2.2°–6.1° below the path's forward direction, and the fovea sees the board, a plane facing the head, in both
modes. The calibration room's target, off the path line, separated them. Unlike the calibration room, the flow in
raster pixels is not flat: it rises from 0.2 px near the fovea to 1.2 px at 30–45°, where the periphery sees desks,
floor and ceiling close by.

Chat's review (2026-11-12, from `checks.json` and the geometry; nothing re-rendered):

1. **Same flow in both modes: confirmed, and expected for this path.** In `target` mode the eye pitches down to keep
   the board centre, from atan(0.266 / 7.052) = 2.2° to atan(0.266 / 2.5) = 6.1°, so it turns 0.03° per step at the
   back and 0.24° at the end (computed). For a plane facing the head, a translation plus the rotation that holds a
   point of the plane still is, to first order, an expansion about that point, with the same rate as the expansion
   about the focus of expansion in `ahead` mode. So the board, which fills the fovea and most of the field, flows the
   same in both modes, only centred on a different point (the focus of expansion is 2–6° above the fovea in
   `target` mode, about 29–56 raster pixels). The modes differ only by the parallax of points off the board's
   plane, which scales with that 2–6° offset and is lost in the medians. To separate them, the target must be far
   off the path's line (M2).
2. **The rise in raster pixels: partly a correction to Chat's own sandbox reading.** Chat wrote that the flow on the
   raster is "nearly flat" in the calibration room. That holds for `target` there (0.61 → 0.55 px from 5–10° to
   30–45°), not for `ahead` (0.30 → 0.52 px, ×1.7), which rises about as the Classroom does (0.64 → 1.16 px, ×1.8).
   The Classroom's pixel values are about twice the sandbox's mainly because the `full` raster has twice the
   `small` raster's resolution (s0 0.05° against 0.1°): in degrees the two scenes agree within about 10 % at 30–45°
   (1.07° against 0.96°). On a log-polar raster the pixel grows as (E2 + e), the translational flow as
   sin(e) / range(e); the raster flow is flat only where the range is about constant, and in both rooms the
   periphery sees nearer surfaces (floor, ceiling, desks) than the centre.

## Experiment M2, range from the head's motion

**Question.** The head's step between looks is known (the path) and so are the eye's poses (the records). Two looks
of a point therefore give its range by triangulation, with the path segment as the baseline, like stereo with the
eyes one behind the other. The precision is limited by the correspondence: a point is seen at a pixel, not at an exact
direction, and the foveated raster's pixels grow with eccentricity (E2 + e). The parallax, on the other hand, is zero
at the focus of expansion and grows with the angle psi from the heading. Where the eye looks decides which pixels get
which parallax: looking ahead puts the finest pixels where the parallax vanishes; fixating a target off the path's
line puts them where it is large. M2 measures how much that matters.

**Measurement** ([`motion/depth.py`](../motion/depth.py), `python -m motion.dolly depth --run R`): for every pair of
looks (k, k + stride), strides 1, 5 and 10 (about 0.1, 0.5 and 1 m), and every surface sample of the later look:

- correspondence by the oracle, as M1.3 (projection into the earlier look; occluded points have no match);
- **pixel-limited**: triangulate the ray through the later sample's pixel centre with the ray through the centre of the
  earlier look's pixel that the point projects to; when the two pixels see the same direction (the flow is under
  about half a pixel) there is no range;
- error: |range − truth| / truth, truth from the Position pass; a range within 10 % is counted **usable** (a reporting
  threshold proposed by Chat, not a pass/fail), as a fraction of all the later look's hits at that eccentricity;
- the precision model to compare with: range · sqrt((omega_a + omega_b)/12) / (|T| sin psi);
- the depth figure (`visualize`): error and usable fraction by eccentricity, error by psi against the model, and an
  error map on the raster for the middle pair at stride 5.

**Check** M2.1: with the exact rays through the point, triangulation returns its true range (relative error
≤ 1e-6 where the rays are ≥ 0.01° apart); control: the earlier eye centre 1 cm along the path, which must give a median
relative error ≥ 1e-3.

**Off-axis target** (M-D7). [`paths/classroom_dolly_side.json`](../paths/classroom_dolly_side.json): M1's path, limits and
control head, with the target given as a ray (`target_ray`) from the first head toward the front wall's plane 2.9 m left
of the board centre, at its height: about 22° left of the heading at the back and 49° at the front (computed for that
point). The preflight now finds the target (the first opaque surface along the ray; non-opaque objects passed through as
in fov-render's S5) and requires it **visible from every head**; its control, a point 0.5 m behind the target, must be
hidden from all of them. M1's own run (`runs/dolly-1`) is reused for M2 as it is: both its modes, no new rendering.

### Chat sandbox, calibration room (CPU, `small` profile)

`python -m motion.dolly sandbox --run runs/calib-m2` (M1's calibration path, both modes) and
`... sandbox --run runs/calib-m2-side --path calib_dolly_side.json --modes target` (the same path fixating the room's
left wall toward (−3, 3, 1): 31° left of the heading at the start, 48° at the end, computed). A first side target on
the floor was hidden from heads 12–17 by the card `ring_e40_m225`, which the new preflight caught. M2.1 PASS in both
runs (max relative error 2.1e-8 and 1.1e-8), control fails (median 0.100). Chat sandbox numbers, at `4d9ef0a` (clean), fov-render
`0f6fa7d`, bpy 5.2.1 wheel, CPU:

| run, stride | fovea (< 2°): median error \| usable | whole field: usable | no range |
|---|---|---|---|
| `ahead`, 0.1 m | — (no range) \| 0.00 | 0.03 | 0.78 |
| `ahead`, 0.5 m | 0.391 \| 0.06 | 0.36 | 0.02 |
| `ahead`, 1.0 m | 0.223 \| 0.21 | 0.60 | 0.00 |
| `target` near the path line (5–12° off), 0.1 m | 0.168 \| 0.31 | 0.10 | 0.15 |
| `target` near the path line, 0.5 m | 0.043 \| 0.68 | 0.48 | 0.00 |
| `target` near the path line, 1.0 m | 0.025 \| 0.80 | 0.67 | 0.00 |
| `target` off the path line (31–48° off), 0.1 m | 0.030 \| 0.96 | 0.26 | 0.02 |
| `target` off the path line, 0.5 m | 0.010 \| 0.97 | 0.86 | 0.00 |
| `target` off the path line, 1.0 m | 0.005 \| 0.97 | 0.90 | 0.00 |

The measured median error is 0.70–1.13 times the model's. A first reading, to be checked on the Classroom: looking
along the path, the fovea gets no range at all from 10 cm steps and little from 1 m; fixating a point far off the
path's line, the fovea gives 1–3 % range from a single 10 cm step for nearly every sample it sees, and the whole field
improves with it. The finest pixels are worth most where the parallax is, and the eye can put them there.

### Classroom (workstation)

Measured by Code (issue #3), workstation, RTX 4090, Blender 5.2.1, OPTIX, profile `full`, 64 spp, fov-render `0f6fa7d`.
Records: [`docs/runs/issue-3/`](runs/issue-3/).

- **`dolly-1`**: M1's run (#1, rendered at `8c1856a`), both modes, reused as it is; `depth` at `780b050` (clean).
  M2.1 PASS (max relative error 2.8e-8), control fails (median 0.101).
- **`dolly-side`**: `paths/classroom_dolly_side.json`, `target` only, at `0de439a` (clean). The target that
  `target_ray` finds is on `worldMap` (M-D8) at (−2.116, 3.540, 1.323): 22.3° left of the heading at k = 0 (7.93 m), 47.3°
  at k = 46 (4.11 m). Preflight: 47 heads pass, the target visible from every head, the hidden point hidden from all;
  the control head fails on clearance. M1.0–M1.3 PASS, every control fails (M1.2 within 5.7 µm, tolerance 7.2–13.9 mm;
  M1.3 0.905–0.977, control ≤ 0.068). M2.1 PASS (max 2.0e-8), control fails (median 0.101).
- Times: `dolly-side` 47 looks in 10.8 s (the first 0.98 s, then 0.19–0.21 s a look); `depth` 3.6–6.8 s; the side
  preflight 46.7 s.

| run, stride | fovea (< 2°): median error \| usable | whole field: usable | no range | measured / model |
|---|---|---|---|---|
| `ahead`, 0.10 m | — (no range) \| 0.00 | 0.16 | 0.22 | 0.75 |
| `ahead`, 0.49 m | 0.164 \| 0.31 | 0.69 | 0.00 | 1.00 |
| `ahead`, 0.99 m | 0.094 \| 0.53 | 0.87 | 0.00 | 1.14 |
| `target` the board centre (2–6° off the heading), 0.10 m | 0.201 \| 0.26 | 0.18 | 0.06 | 0.78 |
| `target` the board centre, 0.49 m | 0.046 \| 0.81 | 0.73 | 0.00 | 0.96 |
| `target` the board centre, 0.99 m | 0.026 \| 0.96 | 0.89 | 0.00 | 1.10 |
| `target` on `worldMap` (22–47° off), 0.10 m | 0.010 \| 0.996 | 0.57 | 0.02 | 0.53 |
| `target` on `worldMap`, 0.49 m | 0.008 \| 0.992 | 0.87 | 0.00 | 0.67 |
| `target` on `worldMap`, 0.99 m | 0.004 \| 0.995 | 0.87 | 0.00 | 0.78 |

**Does the sandbox reading hold?** (Code.) *The fovea gets no range looking ahead*: yes. At 10 cm it gets none, and at
1 m it is still the worst part of the field (0.094 | 0.53, against 0.83–0.94 usable beyond 5°). *Fixating off the
path's line, the fovea gives 1–3 % range from a single 10 cm step*: yes in number (1.0 % at 10 cm, 0.8 % at 0.5 m,
0.4 % at 1 m, 99 % usable), but at 10 cm this is not parallax the fovea measured.
[`same_pixel.py`](runs/issue-3/same_pixel.py) ([output](runs/issue-3/same_pixel.txt)) shows that at a 10 cm step **every**
foveal point lands on the same pixel in both looks (its flow is under half a pixel). Its range is then where the rays
through that one pixel from the two eye centres cross, which the eye's known rotation between the looks decides: the
surface of zero flow through the fixated point. Near the target, the map and its wall lie close to that surface, so the
range comes out right. At 1 m only 5 % of foveal points are same-pixel, and the range of the others (0.40 %) is measured
flow. Fixating the board centre, 2–6° off the heading, the eye turns too little per step for this: 92.5 % of foveal points
are same-pixel at 10 cm, with 21 % median error. Looking ahead the eye does not turn, so the rays are parallel and there is
no range. The model counts pixel rounding only, so it does not describe the same-pixel range; that is why the measured /
model ratio drops to 0.53 at 10 cm on the side run.

**Chat's review of #3** (2026-11-12, from the records and the sandbox; nothing re-rendered).

1. *Same pixel.* Code is right, and `motion/depth.py`'s docstring was wrong: rays through one pixel are parallel, so
   give no range, only when the eye does not turn between the looks. A same-pixel match is itself a measurement, though:
   the flow is under half a pixel. In inverse range it is an interval of the same width, about p / (|T| sin psi),
   whether or not the eye turns. Looking ahead the interval is centred on 1/range = 0, so it bounds the range only from
   below: no range. Fixating, it is centred on the zero-flow surface through the fixated point, so it bounds the range
   on both sides. This is the active-vision argument that fixation makes depth well posed, relative to the fixated
   point. The fixating estimate is therefore legitimate. What was missing is how much the measurement supports,
   whatever the error happened to be. On the side run's fovea at 10 cm the error came out under what rounding allows
   (measured/model 0.53 with the old model, about 0.75 with the corrected one, against 0.87 for rounding alone):
   the map lies near the zero-flow surface, so its points sit near the pixels' centres. That is the geometry's luck,
   not the sensor's precision.
2. *What changed in `motion/depth.py`*. Each range now carries a **bracket**: the relative half-width of the ranges along
   b's ray that project into the same pixel of a, the precision the measurement supports. `depth.json` reports its
   median and `usable_bracket_fraction` (bracket ≤ 10 %) next to the error-based ones, and the **same-pixel fraction**.
   The error-based numbers are unchanged: in the two sandbox runs all 882 compared values (medians, p90s, usable
   counts and fractions, by eccentricity and by psi) are identical, as is M2.1. The model's sigma is now
   sqrt(omega_a / 12), Code's hypothesis (b's Position is its pixel's mean, on the centre's ray to first order).
   Uniform rounding then gives a measured/model ratio of 0.87; the sandbox gives 1.0–1.7, so beyond 10 cm steps the
   errors are somewhat larger than rounding alone. `depth.json` and `checks.json` now record the code that computed
   them (`computed_by`), next to the run's commits (`git`) (Code's provenance note).
3. *The reading, with the bracket* (Chat sandbox, calibration room; fovea < 2°, bracket median | usable by bracket):

   | run | 0.1 m | 0.5 m | 1.0 m |
   |---|---|---|---|
   | `ahead` | no range \| 0.00 | 0.594 \| 0.00 | 0.556 \| 0.00 |
   | `target` near the path line (5–12° off) | 0.421 \| 0.02 | 0.106 \| 0.35 | 0.069 \| 0.80 |
   | `target` off the path line (31–48° off) | 0.089 \| 0.64 | 0.019 \| 0.97 | 0.010 \| 0.97 |

   The conclusion holds with the honest measure, but it is weaker at 10 cm. Fixating off the path line, the fovea
   supports about 9 % from one 10 cm step, not 1–3 %, and about 2 % from 0.5 m. Looking ahead, the fovea supports
   nothing usable at any stride. The Classroom's brackets are for Code to measure (`depth` again on both runs, no
   rendering).
