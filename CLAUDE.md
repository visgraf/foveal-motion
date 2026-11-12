# How we work on foveal-motion

The same light N-Surface method as [fov-render](https://github.com/visgraf/fov-render/blob/main/CLAUDE.md): one file
of rules, GitHub Issues between Chat and Code. If something here gets in the way, say so in an issue.

## Who does what

- **Luiz** decides: direction, what gets built, what is accepted. Only Luiz changes this file.
- **Chat** (Claude in claude.ai) plans, writes code that runs without a GPU, runs what its sandbox can (the `bpy`
  5.2.1 wheel on CPU, fov-render's calibration room; no GPU, no Classroom), opens issues for Code, reviews what
  comes back.
- **Code** (Claude Code on the RTX 4090 workstation) runs what needs the GPU or the Classroom, measures, fixes what
  the issue allows, and reports.

## Issues are the channel

1. **Open** an issue: goal, branch, what to run, what "done" means, what not to touch.
2. **Whose turn** is a label: `for-code`, `for-chat` or `for-luiz`. One label at a time; switch it when you hand over.
3. **Report** in a comment: the commit, the commands, the numbers (where each comes from, rule 1), what failed. Long
   output goes in a file in the repo; link it.
4. **Decisions** go to Luiz (`for-luiz`): the question, the options, a recommendation. His answer in the issue is the
   record; decisions that matter go to `docs/motion.md`.
5. **Done** when Luiz closes the issue. Commits and PRs say `Refs #n` or `Fixes #n`.

## Rules

1. **Say where a number comes from.** Measured (command, commit, machine), estimated, or proposed. A Chat sandbox
   number says "Chat sandbox". GPU, timing and Classroom numbers count only when Code measures them.
2. **Every check can fail**, and ships a control that must fail. Never weaken a check or move a threshold to make
   something pass; find out why first.
3. **If it isn't in the repo or an issue, it didn't happen.**
4. **Fix forward.** No rewriting pushed history.
5. **When unsure, ask** (`for-luiz`).
6. **fov-render is a dependency, not a workspace.** It is pinned as the submodule `external/fov-render`. Something
   fov-render should do differently is an issue in fov-render; this repository moves its pin when Luiz accepts.
7. **The playground stays the reference.** `tools/` and the playground's docs are left as imported; new work goes in
   `motion/`, `paths/` and `docs/motion.md`.

## Project facts

- Metres, world +Z up. Heads come from a path spec (`paths/*.json`): level, facing `facing_w`; the head axes follow
  fov-render (`HeadPose`: +X right, +Y up, −Z forward). A look's head is its request's, the scene's `EYE` is only the
  session's default.
- The eye is fov-render's cyclopean `C`, at the head origin. The sensor is `Foveated` (E2 2°, e_max 45°).
- Runs and figures are not committed (`runs/`, `visuals/` are ignored); the command and the run record are.
- Cost: interactive under ~10 s, batch under ~5 min, overnight only when the issue says so.
