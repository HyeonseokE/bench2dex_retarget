#!/bin/bash
# Stage 5: render recorded episodes the way Bench2Dex built its released replay-generalization data.
#
#   bash tools/retarget/stage5_render.sh <origin-generalization dir> [episode file names...]
#
# For each origin episode: Bench2Dex replay.py --restore-generalization (the released replay files
# carry _replay_generalization_mode=restored) with RGB on the six collect-config cameras and TacMap
# tactile, written to the sibling replay-generalization/ directory; then the GT labels
# (occupancy + box3d/box2d) the released data has, per file (tools/retarget/label_episode.py).
#
# PY (default: python) selects the interpreter.
set -e
ORIGIN="$(cd "$1" && pwd)"; shift
REPLAY="$(dirname "$ORIGIN")/replay-generalization"
PY="${PY:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.."                              # Bench2Dex root
mkdir -p "$REPLAY"
files=("$@")
[ ${#files[@]} -gt 0 ] || files=($(cd "$ORIGIN" && ls episode_??????.hdf5))
for f in "${files[@]}"; do
  [ -f "$REPLAY/$f" ] && { echo "skip $f (rendered)"; continue; }
  "$PY" replay.py --hdf5 "$ORIGIN/$f" --output "$REPLAY/$f.partial" \
    --restore-generalization --enable-rgb --enable-tactile --headless
  "$PY" "$HERE/label_episode.py" "$REPLAY/$f.partial"
  mv "$REPLAY/$f.partial" "$REPLAY/$f"
done
echo "RESULT stage5 $REPLAY"
