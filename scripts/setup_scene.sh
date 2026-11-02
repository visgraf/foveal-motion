#!/usr/bin/env bash
# Build scenes/classroom/ from the public Blender Classroom demo (CC0, Christophe Seux), then check it.
#
#     scripts/setup_scene.sh                    # download classroom.zip from download.blender.org
#     scripts/setup_scene.sh path/to/classroom.zip
#
# Steps: verify the zip's md5; unpack it into scenes/classroom/; place the EYE camera with
# tools/classroom_oracle/place_eye.py (head origin (-0.6, -1.0, 1.2) m, yaw 0) into classroom_eye.blend;
# compare every unpacked file with scenes/classroom/ASSET_MANIFEST.json; run the Blender-side preflight
# (EYE at H0, no missing linked files, an OPTIX device).  Re-running refuses an existing classroom_eye.blend.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE="$REPO/scenes/classroom"
URL="https://download.blender.org/demo/test/classroom.zip"
MD5="3adbb7114b514bfc6fc724ce20f86b4e"
BLENDER="${BLENDER:-blender}"

die() { echo "[setup-scene] STOP $*" >&2; exit 1; }

command -v "$BLENDER" >/dev/null || die "Blender not found (set BLENDER=/path/to/blender, or put blender on PATH)"
command -v unzip >/dev/null || die "unzip not found"
[ -e "$SCENE/classroom_eye.blend" ] && die "$SCENE/classroom_eye.blend exists; delete scenes/classroom/* to rebuild"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

ZIP="${1:-}"
if [ -z "$ZIP" ]; then
  ZIP="$TMP/classroom.zip"
  echo "[setup-scene] downloading $URL"
  curl -fL --retry 3 -o "$ZIP" "$URL" || die "download failed; download it by hand and pass its path"
fi
[ -f "$ZIP" ] || die "no such file: $ZIP"

if command -v md5sum >/dev/null; then got="$(md5sum "$ZIP" | cut -d' ' -f1)"; else got="$(md5 -q "$ZIP")"; fi
[ "$got" = "$MD5" ] || die "classroom.zip md5 is $got, expected $MD5 (a different version of the demo?)"
echo "[setup-scene] classroom.zip md5 OK"

unzip -q "$ZIP" -d "$TMP/unpacked"
SRC_BLEND="$(find "$TMP/unpacked" -name classroom.blend -type f | head -n 1)"
[ -n "$SRC_BLEND" ] || die "classroom.blend not found inside the zip"
mkdir -p "$SCENE"
cp -R "$(dirname "$SRC_BLEND")"/. "$SCENE"/
echo "[setup-scene] unpacked into scenes/classroom/"

"$BLENDER" -b "$SCENE/classroom.blend" --python-exit-code 1 -P "$REPO/tools/classroom_oracle/place_eye.py" -- \
  --location -0.6 -1.0 1.2 --yaw 0 --out "$SCENE/classroom_eye.blend" > "$TMP/place_eye.log" 2>&1 \
  || { cat "$TMP/place_eye.log"; die "place_eye.py failed"; }
[ -f "$SCENE/classroom_eye.blend" ] || { cat "$TMP/place_eye.log"; die "place_eye.py wrote no classroom_eye.blend"; }
grep "\[place_eye\]" "$TMP/place_eye.log" || true

python3 - "$SCENE" <<'EOF'
import hashlib, json, sys
from pathlib import Path
scene = Path(sys.argv[1])
m = json.loads((scene / "ASSET_MANIFEST.json").read_text())
want = m["accepted_classroom_eye_blend_sha256"]
bad, missing = [], []
for f in m["files"]:
    p = scene / f["path"]
    if f["path"] == "classroom_eye.blend":
        continue
    if not p.is_file():
        (missing if f.get("required") else bad).append(f["path"])
        continue
    if hashlib.sha256(p.read_bytes()).hexdigest() != f["sha256"]:
        bad.append(f["path"])
eye = hashlib.sha256((scene / "classroom_eye.blend").read_bytes()).hexdigest()
print(f"[setup-scene] {'PASS' if not missing else 'FAIL'} required scene files present ({len(missing)} missing)")
print(f"[setup-scene] {'PASS' if not bad else 'NOTE'} unpacked files byte-identical to the manifest"
      + ("" if not bad else f" ({len(bad)} differ: {bad[:5]})"))
print(f"[setup-scene] {'SAME' if eye == want else 'NOTE'} classroom_eye.blend sha256 "
      + ("equals the f3d-vision file" if eye == want else
         "differs from the f3d-vision file (expected when Blender versions differ; the explorer checks the EYE pose, "
         "not the file hash, and compare.py tells you whether the runs agree)"))
sys.exit(1 if missing else 0)
EOF

"$BLENDER" -b "$SCENE/classroom_eye.blend" --python-exit-code 1 -P "$REPO/tools/greedy_foveal/scene_preflight.py" \
  2>&1 | grep "\[scene-preflight\]" || die "the scene preflight failed (see above)"
echo "[setup-scene] done"
