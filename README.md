# foveal-motion

Active foveal vision **under head motion**: an independent experimental repository, forked from
[visgraf/foveal-playground](https://github.com/visgraf/foveal-playground) (its history is kept here) and rendering through
[visgraf/fov-render](https://github.com/visgraf/fov-render) (a submodule, `external/fov-render`), whose head may move between
looks (fov-render #20, PLAN D19–D20).

The first setting is **monocular**: the cyclopean eye `C` of fov-render's rig (at the head origin), the **Foveated**
sensor (log-polar, E2 = 2°, e_max = 45°), and a head that moves. The playground's binocular explorer below is kept
unchanged as the reference.

## Experiment M1, the dolly

The head walks straight from the back of the Classroom to the board, level, at standing height (1.6 m), 46 steps of
about 10 cm, one foveated look per step, twice: the eye **fixating the board centre** (`target`) and **looking straight
ahead** (`ahead`). Path, limits and where every number comes from: [`paths/classroom_dolly.json`](paths/classroom_dolly.json);
design and results: [`docs/motion.md`](docs/motion.md).

    git clone --recurse-submodules https://github.com/visgraf/foveal-motion.git && cd foveal-motion
    python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt  # Python 3.11+ (fov-render); numpy, Pillow,
                                                                           # matplotlib, fov-render
    scripts/setup_scene.sh                                                 # the Classroom, as in the playground

    .venv/bin/python -m motion.dolly selftest                                       # numpy only, seconds
    .venv/bin/python -m motion.dolly preflight --path paths/classroom_dolly.json --run runs/dolly-1 --device OPTIX
    .venv/bin/python -m motion.dolly run       --path paths/classroom_dolly.json --run runs/dolly-1 --device OPTIX
    .venv/bin/python -m motion.dolly checks    --run runs/dolly-1
    .venv/bin/python -m motion.dolly visualize --run runs/dolly-1 --vis visuals/dolly-1

    .venv/bin/python -m motion.dolly sandbox --run runs/calib-dolly    # no GPU, no Classroom: fov-render's calibration
                                                                       # room on CPU with the bpy wheel (Python 3.13), < 1 min

`preflight` runs fov-render's scene sanity (holes, back-faces, nearest surface) from **every head of the path**, which
fov-render's own scene gate does only from the session's head, plus a control head against the board that must fail.
**Experiment M2** (`python -m motion.dolly depth --run R`) measures the range the head's own motion gives, on the
foveated raster, for M1's run and for the same path fixating a target off the path's line
([`paths/classroom_dolly_side.json`](paths/classroom_dolly_side.json)).

`checks` (M1.0–M1.3, each with a control that must fail) and the translational flow are described in
[`docs/motion.md`](docs/motion.md). Blender is found as fov-render finds it (`$FOVRENDER_BLENDER`, `blender` on PATH, or the
`bpy` wheel).

| file | role |
|---|---|
| [`motion/paths.py`](motion/paths.py) | the path spec, its heads and the two gaze modes |
| [`motion/dolly.py`](motion/dolly.py) | the command line: preflight, run (one fov-render `Session`), checks, visualize, sandbox |
| [`motion/bl_preflight.py`](motion/bl_preflight.py) | Blender side: the scene sanity from every head of the path |
| [`motion/analysis.py`](motion/analysis.py) | the checks and the flow, from the run's files |
| [`motion/visuals.py`](motion/visuals.py) | contact sheet, flow figure, side-by-side video, depth figure |
| [`motion/depth.py`](motion/depth.py) | M2: range from the head's motion, pixel-limited, and check M2.1 |

How we work on it (Luiz, Chat, Code, GitHub Issues): [`CLAUDE.md`](CLAUDE.md).

---

# The playground baseline (unchanged from foveal-playground)

A small playground for **active foveal 3-D vision**. A binocular head sits in a static synthetic scene (the Blender
Classroom). Its eyes explore the scene one fixation at a time, reconstruct what the fovea sees, remember it globally
and decide where to look next:

    FIXATE
        ↓
    RECONSTRUCT          everything the 12° fovea can match, as metric 3-D points
        ↓
    UPDATE GLOBAL MEMORY one persistent 3-D surfel map + a sphere of where it has looked
        ↓
    SACCADE              to a neighbouring view, or jump to an unexplored region
        ↓
    repeat

The baseline, the **Greedy Foveal Explorer**, is intentionally simple: a reproducible starting point for experiments,
not a finished vision system. It is extracted from the research repository
[visgraf/f3d-vision](https://github.com/visgraf/f3d-vision) (tag `playground-2-sparse-lattice`); see
[docs/provenance.md](docs/provenance.md).

## The baseline

- a **fixed head** with two eyes that rotate about their own centres;
- **binocular 12° foveal observations**, rendered by Blender (Cycles, 64 spp);
- **PERFECT / ORACLE correspondence**: the renderer's geometry decides which left and right pixels match;
- **spherical metric triangulation** of every matched foveal ray;
- **one persistent surfel map** in the head frame, fused at 12 mm;
- a **1° coverage sphere** with each cell UNSEEN, SEEN or DEPTH;
- a fixed **gaze lattice** of 356 views on 17 rings about the eye baseline, whose 12° footprints cover every 1° cell;
- **greedy local saccades** to a neighbouring lattice view that still adds unseen area, preferring views where the
  fovea's edge shows surface; **global jumps** to the unvisited view that adds the most; a stop when none adds any.

In the Classroom, the baseline run stops by itself after **356 fixations** (315 local, 40 global saccades) with
**100 %** of the sphere SEEN, **99.26 %** with reconstructed DEPTH and **16,943,073** surfels, in about 7 minutes on an
RTX 4090 (MEASURED in f3d-vision). These numbers describe this one setting.

> **The policy does not read the 3-D map.** Gaze selection uses the coverage sphere and the current fixation's
> evidence; the map is memory and output. This is deliberate, and one of the clearest extension points.

## Requirements

- Linux with an **NVIDIA RTX GPU** (Cycles OPTIX). There is no CPU path yet; see [exercise E1](docs/exercises.md).
- **Blender 5.2.1** on `PATH` as `blender` (or set `BLENDER=/path/to/blender` for the setup script).
- Python 3.10+, git, unzip, curl; the DejaVu fonts for the overview figure (Debian/Ubuntu: `fonts-dejavu-core`).
- About 160 MB for the scene and 2.2 GB per full run.

## Setup

    git clone https://github.com/visgraf/foveal-playground.git && cd foveal-playground
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    scripts/setup_scene.sh            # downloads the Classroom demo (CC0), places the EYE, checks the scene

`setup_scene.sh` takes an already-downloaded `classroom.zip` as its argument, too. It ends with a Blender-side
preflight: the EYE at the declared head pose, no missing linked files, an OPTIX device.

Checks that need no GPU and no Blender (seconds):

    .venv/bin/python tools/greedy_foveal/check_lattice.py     # the gaze lattice and a coverage-only policy replay
    .venv/bin/python tools/greedy_foveal/compare.py selftest  # every comparator can fail
    .venv/bin/python tools/greedy_foveal/head_pose.py         # the declared head pose

## Run

Commands run from the repository root. `explore` refuses an existing run directory, and records the git commit and
whether the tree was clean, so commit before you run.

**Smoke test** (75 fixations, about 1.5 min):

    .venv/bin/python tools/greedy_foveal/run.py explore --run runs/smoke --max-fix 75
    .venv/bin/python tools/greedy_foveal/run.py checks  --run runs/smoke
    .venv/bin/python tools/greedy_foveal/compare.py check --run runs/smoke

Fixations 1–75 must equal the first 75 fixations of the baseline run (fixations 2–75 are local saccades; the first
global jump is at 168).

**Full baseline** (about 7 min):

    .venv/bin/python tools/greedy_foveal/run.py explore   --run runs/baseline --max-fix 600 --spp 64
    .venv/bin/python tools/greedy_foveal/run.py checks    --run runs/baseline
    .venv/bin/python tools/greedy_foveal/compare.py check --run runs/baseline
    .venv/bin/python tools/greedy_foveal/run.py visualize --run runs/baseline --vis visuals/baseline

Expected: `stop: no valid unseen fixation` at fixation 356; 315 local and 40 global saccades. `visualize` writes a
one-page `overview.png`: the trajectory over the reconstructed sphere, the coverage state and curves, statistics, and
the map from above. `final-map.ply` opens in MeshLab or CloudCompare.

`compare.py check` tells you whether a run reproduces the baseline. Use it to show that a refactor or a speed-up did
**not** change behavior. A real experiment is expected to fail it.

## Code map

| file | role |
|---|---|
| [`tools/greedy_foveal/explorer.py`](tools/greedy_foveal/explorer.py) | geometry, coverage, the map, the lattice, the greedy policy |
| [`tools/greedy_foveal/run.py`](tools/greedy_foveal/run.py) | the active loop and freeze (`explore`), `checks`, `visualize` |
| [`tools/greedy_foveal/render_server.py`](tools/greedy_foveal/render_server.py) | the persistent Blender observation service |
| [`tools/greedy_foveal/head_pose.py`](tools/greedy_foveal/head_pose.py) | the fixed head H0 |
| [`tools/greedy_foveal/visuals.py`](tools/greedy_foveal/visuals.py) | the overview figure |
| [`tools/greedy_foveal/check_lattice.py`](tools/greedy_foveal/check_lattice.py) | lattice checks and a policy replay, without rendering |
| [`tools/greedy_foveal/compare.py`](tools/greedy_foveal/compare.py) | comparison with the baseline |
| `tools/…` (other modules) | sensor geometry, oracle correspondence, triangulation, Blender acquisition, reused from f3d-vision |

## Starting a project

**Change one component at a time and compare against the baseline.** The components (renderer, matcher,
measurement, map, fusion, coverage, lattice, local and global policy, stop rule, head motion, visualization) are listed
with their interfaces in [docs/architecture.md](docs/architecture.md), section 15. Directions and exercises, including
CPU rendering and building a reconstruction-quality evaluation, are in [docs/exercises.md](docs/exercises.md).

1. Branch: `git switch -c project/<topic>`.
2. Write down the question, the single thing you change and what you will compare, before you run.
3. Smoke test, then a full run into its own directory under `runs/`.
4. Compare with the baseline: fixations to the stop, SEEN / DEPTH curves (`coverage.npz`), the local / global split,
   surfels, wall time. `run.py checks` holds known answers for the baseline: when you change a component on purpose,
   replace its known answer with one for your version, never just delete it.
5. Report the commit, the commands and the numbers, with the run each came from.

## Limitations

- synthetic, static Classroom scene; 64-spp rendering; fixed head;
- PERFECT / ORACLE correspondence, no natural stereo yet; oracle object identity used only for visualization;
- no measure of reconstruction quality (SEEN / DEPTH only; [exercise E3](docs/exercises.md));
- NVIDIA RTX (OPTIX) only; map and storage not optimized (about 1 GB map, 2.2 GB per run).

## License

CC0 1.0 (see `LICENSE`), as f3d-vision. The Classroom scene is CC0 (Christophe Seux) and is downloaded, not
redistributed.
