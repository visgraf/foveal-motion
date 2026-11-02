# Exercises and project directions

Each direction changes **one component** of the loop (docs/architecture.md, section 15) and compares with the
baseline. The first three are infrastructure that other projects may need; the rest are research directions.

Numbers below are MEASURED in f3d-vision unless marked ESTIMATE.

---

## E1. Render on the CPU

The baseline needs an NVIDIA RTX GPU: Cycles renders through OPTIX, and the code refuses any other device. Making the
device a choice is a self-contained exercise.

**Where the device is fixed.**

- `tools/active_bootstrap/ab1a_spec.py`: `DEVICE = "OPTIX"`.
- `tools/active_bootstrap/ab1a_render.py`, `configure()`: calls
  `classroom_oracle1_render._prepare_perspective_pair(c, SP.DEVICE, spp)` and raises unless the backend it got equals
  `SP.DEVICE` ("no fallback").
- `tools/classroom_oracle1_render.py`, `_prepare_perspective_pair()`: calls `bl_common.setup_device`, and refuses a CPU
  backend that was not requested explicitly.
- `tools/bl_common.py`, `setup_device()`: tries the preferred GPU type, then CUDA, else sets `scene.cycles.device =
  "CPU"`.
- `tools/greedy_foveal/scene_preflight.py` reports FAIL without an OPTIX device.
- `tools/greedy_foveal/visuals.py`, `panel_d`, prints "OPTIX" as a literal.

**Guidelines.**

1. Make the device a parameter rather than editing the constant: an `explore --device {OPTIX,CUDA,CPU}` option in
   `run.py`, passed to `render_server.py` on its command line (as `--spp` is) and on to `configure`. Keep OPTIX the
   default, so the baseline commands do not change.
2. Record the device actually used. `server-ready.json` already holds `device` and `trajectory.json` copies it under
   `server`, so `compare.py` will notice a different device.
3. Time it before a full run: `--max-fix 5`, and look at `seconds.render` per fixation in `trajectory.json`. One
   64-spp pair (two 640 × 640 renders) takes about 0.75 s on an RTX 4090. On a CPU, ESTIMATE one to two orders of
   magnitude slower; measure it, and report the CPU model and thread count.
4. Expect the control path to change. The Position and Object Index passes are box-filtered averages of the samples
   in each pixel, so a different device, and certainly a different sample count, can change them, and with them the
   correspondence, the points and possibly the gazes. `compare.py check` will then FAIL, which is correct. Report what
   changed: the first fixation whose points differ, whether the LOCAL / GLOBAL sequence and the stop (fixation 356)
   still hold, and how far the SEEN / DEPTH curves move.
5. To make CPU runs practical, look at the sample count. RGB is display-only; the control path depends only on the
   geometry passes. How low can `--spp` go before the trajectory changes? That is a measurable question (start with a
   smoke run at each value).
6. Keep the guard: a run should still refuse to start when it cannot get the device it asked for.

## E2. Faster iteration without Blender

`check_lattice.py` replays the policy on coverage alone (no rendering, no points) in seconds. Every `policy/
state-NNNN.npz` of a run holds the exact inputs of one decision, so a new `decide` can be tested on a frozen run's
states without rendering anything: one step at a time, i.e. what it would have chosen at each state of the baseline
trajectory (after its first different choice, the real trajectory would diverge). Extend `check_lattice.py`, or write
a small replay tool, so that a policy project can be developed on a laptop and only the final runs need the GPU.

## E3. Build an evaluation

A run reports where the eyes looked (SEEN) and where they got depth (DEPTH), not whether the depth is right. f3d-vision
measured that post hoc: 98.82 % of the first-hit solid angle seen from the head origin had a surfel within 12 mm.

**Guidelines.**

1. **Reference.** Render the scene once from the head origin with an equirectangular panoramic camera, with the
   Position pass (and Object Index, to report per object). 720 × 360 (0.5°) was f3d-vision's resolution. Keep the
   camera orientation equal to the head frame H0 (`head_pose.py`), or transform the reference points into H0 with
   `fsg_geometry.world_to_head` and test the orientation on a few known points.
2. **Firewall.** The reference is opened only after the run is frozen. A good pattern: `evaluate` first checks
   `freeze.json` against the run's files, then opens the reference.
3. **Metric.** For every reference cell with a first hit, the cell is covered if a map surfel lies within 12 mm
   (`GlobalMap.nearest`). Weight cells by solid angle (`explorer.row_weights` shows how).
4. **A check that can fail.** Build a tiny synthetic case with a known answer (a plane, a map with a known hole) and
   show the metric gets it right and that a perturbed map gets it wrong.
5. **Limitations to state.** First hit only: surfaces hidden from the head origin are not targets. A project that
   moves the head needs a different reference.

## E4. Natural stereo

Replace `explorer.perfect_correspondence` (the oracle) with a matcher that sees only the left and right RGB. The
interface is in docs/architecture.md, section 7: produce `left_core_row`, `left_core_col`, `uv_L`, `uv_R`; also load
`rgb_R` in `load_observation`. The rest of the loop is unchanged. Without E3, the effect can only be read in SEEN /
DEPTH, the surfel count and the diagnostic error `err_mm` (which uses the oracle Position and must stay diagnostic).

## E5. The map in the loop

The policy never reads the 3-D map (docs/architecture.md, section 3). Use it: frontier edges in 3-D, surfaces with low
support, or expected information gain, in `local_candidates` / `global_target`.

## E6. Lattice and policy

- A sparser lattice: f3d-vision tried a 2° cap around the baseline axis (346 gazes, Playground-3).
- Shorter global jumps: the nearest useful gaze instead of the one with the largest gain.
- Other stop rules: on DEPTH, on map growth, on a fixation budget.

## E7. Head motion

The head is fixed. `tools/greedy_foveal/head_pose.py` is the one place the head pose is declared, and `run.explore`
checks it every fixation. Making head pose an action means moving the EYE between fixations in `render_server.py`,
transforming each observation into one common frame, and replacing the fixed-head checks with the new invariant.
