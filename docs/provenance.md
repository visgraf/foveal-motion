# Provenance

foveal-playground is extracted from the research repository
[visgraf/f3d-vision](https://github.com/visgraf/f3d-vision), `main` at commit `bbee8ef` (2026-10-12, "Governance-1
follow-up"). The explorer's behavior is that of tag **`playground-2-sparse-lattice`** (`1f50bd6`), the sparse covering
lattice that Engineering-2 promoted to f3d-vision's official baseline (tag `greedy-foveal-playground-baseline-2`).
Between that tag and `bbee8ef`, `explorer.py` changed only in its docstring.

The extraction follows f3d-vision's own rule: *migrate behavior first, redesign structure second*. The directory
layout and the `sys.path` setup are kept, so each file can be diffed against its original.

## What was copied

The host and Blender import closure of `tools/greedy_foveal/` at `bbee8ef`, minus the evaluation, the demo and the
equivalence checker (below): 22 Python modules, about 5,100 lines. Also `tools/classroom_oracle/place_eye.py` (to
build the scene), `scenes/classroom/ASSET_MANIFEST.json` and `LICENSE` (CC0 1.0).

**Byte-identical to f3d-vision** (16): `LICENSE`, `tools/bl_common.py`,
`tools/classroom_oracle1_public.py`, `tools/classroom_oracle1_render.py`, `tools/exr_lite.py`,
`tools/fsg3_surface_map.py`, `tools/fsg6f_public.py`, `tools/fsg_geometry.py`, `tools/render_foveated.py`,
`tools/warp.py`, `tools/active_bootstrap/ab1a_render.py`, `tools/active_bootstrap/ab1b_geometry.py`,
`tools/active_bootstrap/ab1b_oracle.py`, `tools/natural_bootstrap/nb1a_guard.py`,
`tools/visual_language/style.py`, `tools/classroom_oracle/place_eye.py`.

**Modified** (8):

| file | change | effect on the control path |
|---|---|---|
| `tools/greedy_foveal/explorer.py` | docstring header only | none (executable code identical) |
| `tools/greedy_foveal/check_lattice.py` | one docstring line | none |
| `scenes/classroom/ASSET_MANIFEST.json` | the legacy-checkout fields removed (`legacy_workstation_path`, `legacy_identical`, `missing_legacy_reference`, `correction_to_consolidation_1`); provenance notes point at `setup_scene.sh` | none: paths, sha256s, `required` and the accepted `classroom_eye.blend` sha256 unchanged |
| `tools/active_bootstrap/ab1a_spec.py`, `ab1b_spec.py` | `SHARED = Path("<f3d-vision checkout>")` → `SHARED = REPO`. The constants built from it name f3d-vision's original AB1a / AB1b experiment records, which the playground never reads. | none |
| `tools/greedy_foveal/render_server.py` | the fixed-head check uses `head_pose.eye_pose` instead of `ns1a_render.eye_pose` (below) | none: the pose used is still the EYE pose read from the scene |
| `tools/greedy_foveal/run.py` | `evaluate` removed; the Breadth-1 open audit removed; the head pose taken from the render server and checked against `head_pose.py`; `checks` adapted (below); workstation paths removed | none intended; to be confirmed by `compare.py check` (below) |
| `tools/greedy_foveal/visuals.py` | the evaluation panel (E) and statistics rows removed; one subtitle | presentation only |

**New**: `tools/greedy_foveal/head_pose.py`, `compare.py`, `scene_preflight.py`; `scripts/setup_scene.sh`;
`requirements.txt` (numpy and Pillow only; f3d-vision's OpenEXR and OpenCV pins are not needed by this closure);
`tests/known-answer/`; these docs.

## The fixed head

In f3d-vision the render server reads the head pose from the EYE object of `classroom_eye.blend` and requires it equal,
within 1e-6, to an accepted AB1a `calibration.json` that exists only on the project workstation
(`ns1a_render.eye_pose(strict=True)`; `run.py` also read that file). Here the same check is made against declared
constants, `head_pose.H0_ORIGIN_W_M = (-0.6, -1.0, 1.2)` and the axis matrix `H0_R_WH`, with the same tolerance. The
values come from f3d-vision's architecture document (section 6) and from the `place_eye.py` arguments recorded in the
asset manifest. The calibrations the explorer writes are built from the EYE pose itself, as before, so
`calibration.json` stays byte-identical.

`run.py checks` changed accordingly: `fixed_head_H0` now requires every fixation's head equal to the first fixation's
and the first within 1e-6 of H0 (it used to compare with the workstation file); `breadth1_not_opened_before_freeze` is
replaced by `products_unchanged_since_freeze`. The other checks are unchanged.

## What was left out, and why

| f3d-vision | why it is not here |
|---|---|
| `run.py evaluate`, `tools/classroom_oracle/breadth1_*.py`, `tools/classroom_oracle1_eval.py` | the post-hoc Breadth-1 evaluation needs a reference render that exists only on the workstation, and regenerating it needs Controller-01 records; decision of 2026-11-02 to leave evaluation out (docs/exercises.md, E3) |
| `tools/greedy_foveal/demo.py` | its reference stage and headline numbers read the evaluation |
| `tools/greedy_foveal/check_equivalence.py` | it compares against the 2.2 GB frozen run and pins f3d-vision commits; replaced by `compare.py`, which compares against a committed signature of that run |
| `tools/north_star/ns1a_render.py`, `ns1a_spec.py` | used only for `eye_pose`; replaced by `head_pose.py` |
| `CLAUDE.md`, `docs/process/`, `docs/chat-handoff.md` | f3d-vision's research-governance process |
| `fov3d/`, the controller, partition, natural-bootstrap and north-star lines | not in the playground's import closure |

## Confirming the extraction

`compare.py check` compares a run of this repository with the frozen f3d-vision run through its signature. The
extraction is confirmed when, on the f3d-vision workstation (same GPU, driver and Blender), a 75-fixation smoke run and
a full run of this repository both pass, against the signature committed in `tests/known-answer/`. The commands are in
`tests/known-answer/README.md`.

## License

f3d-vision is released under CC0 1.0, and so is this repository (`LICENSE`). The Classroom scene is CC0 (Christophe
Seux, via download.blender.org); it is not redistributed here.
