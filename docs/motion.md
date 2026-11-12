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
