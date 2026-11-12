"""Every value of an earlier depth.json must be in the new one, and equal, except the model's (issue #3, Code).

Chat's review of #3 added the bracket and the same-pixel fraction to motion/depth.py and changed the
precision model (sigma from look a only). Everything else must come out bit for bit as before.

    git show 81d0a7e:docs/runs/issue-3/dolly-1/depth.json    > OLD1.json
    git show 81d0a7e:docs/runs/issue-3/dolly-side/depth.json > OLD2.json
    .venv/bin/python docs/runs/issue-3/compare_depth.py OLD1.json runs/dolly-1/depth.json OLD2.json runs/dolly-side/depth.json

Skipped on purpose: median_pred and median_measured_over_pred (the model), and git (compared on its own).
Values the new file adds (bracket, same pixel, computed_by) are not in the old one, so not compared.
Exit 1 on any difference or missing value. Control: an old file against the other run's new file, or one
value moved by 1e-12, must be reported (both were, at 5f68240).
"""
import json
import sys

MODEL = {"median_pred", "median_measured_over_pred"}


def walk(o, n, path, out):
    if isinstance(o, dict):
        for k, v in o.items():
            p = f"{path}.{k}"
            if k in MODEL:
                out["model"] += 1
            elif not isinstance(n, dict) or k not in n:
                out["missing"].append(p)
            else:
                walk(v, n[k], p, out)
    elif isinstance(o, list):
        if not isinstance(n, list) or len(n) != len(o):
            out["diff"].append((path, "list length"))
            return
        for i, (a, b) in enumerate(zip(o, n)):
            walk(a, b, f"{path}[{i}]", out)
    else:
        out["compared"] += 1
        if not (o == n or (o != o and n != n)):
            out["diff"].append((path, o, n))


def main(args) -> int:
    bad = False
    for old, new in zip(args[::2], args[1::2]):
        O, N = json.load(open(old)), json.load(open(new))
        out = {"compared": 0, "diff": [], "missing": [], "model": 0}
        for top in O:
            if top != "git":
                walk(O[top], N.get(top), top, out)
        same_git = O["git"] == N["git"]
        print(f"{new}: compared {out['compared']} values, differ {len(out['diff'])}, missing {len(out['missing'])}, "
              f"model values skipped {out['model']}; git equal {same_git}")
        for d in out["diff"][:10] + out["missing"][:10]:
            print("    ", d)
        bad |= bool(out["diff"] or out["missing"]) or not same_git
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
