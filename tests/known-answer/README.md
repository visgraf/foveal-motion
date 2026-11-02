# Known answer

`playground-2-signature.json` is the signature of the frozen Playground-2 baseline run (f3d-vision
`previews/playground-sparse-lattice`): the sha256 of every control-path product of that run, per fixation, in a small
file. It lets `tools/greedy_foveal/compare.py check` tell, bitwise, whether a run reproduces the baseline, without the
2.2 GB run itself. See the docstring of `compare.py` for exactly what is compared.

It was checked against fresh runs of this repository on the workstation that holds the frozen run:

    .venv/bin/python tools/greedy_foveal/run.py explore --run runs/smoke --max-fix 75
    .venv/bin/python tools/greedy_foveal/run.py checks  --run runs/smoke
    .venv/bin/python tools/greedy_foveal/compare.py check --run runs/smoke

Without the file (or with `--signature` pointing at a missing file), `compare.py check` runs only the literal checks
(the full-run numbers 356 / 315 / 40 / 100.00 % / 99.26 % / 16,943,073 surfels, or, for a smoke run, fixation 1
initial and 2–75 local) and says that the signature comparison did not run.
