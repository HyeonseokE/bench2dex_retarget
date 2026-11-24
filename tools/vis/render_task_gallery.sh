#!/usr/bin/env bash
# Render every scene in scenes/ (or the given YAMLs) one process at a time.
#   bash tools/vis/render_task_gallery.sh <out_dir> [scenes/XX.yaml ...]
# Skips tasks whose <task>.json already exists in <out_dir>. Logs: <out_dir>/../logs/<task>.log
set -u
OUT="${1:?out_dir}"; shift
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
mkdir -p "$OUT" "$OUT/../logs"
if [ $# -eq 0 ]; then set -- scenes/*.yaml; fi
for y in "$@"; do
  t="$(basename "$y" .yaml)"
  if [ -f "$OUT/$t.json" ]; then echo "[skip] $t"; continue; fi
  echo "[start] $t $(date -u +%FT%TZ)"
  timeout 1800 python tools/vis/render_task_gallery.py --task "$y" --out-dir "$OUT" --headless \
    > "$OUT/../logs/$t.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$OUT/$t.json" ]; then echo "[ok] $t"; else echo "[fail] $t rc=$rc"; fi
done
echo "[all done] $(date -u +%FT%TZ)"
