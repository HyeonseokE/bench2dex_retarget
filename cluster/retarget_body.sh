# One episode, every target robot: stage 1 once, then per target
#   2 kinematic retarget -> 3 SPIDER plan -> 4 open-loop re-execution recorded as a Bench2Dex
#   origin episode -> 5 (if 4 succeeded) RGB + tactile render and GT labels, as the released data.
# Sourced by retarget_array.sbatch with TASK and EPISODE set; no #SBATCH here.
#
# Idempotent: each stage skips when its output exists, so a resubmitted or walltime-killed item
# picks up where it stopped (a SPIDER run that was interrupted restarts that episode-target).
#
#   TARGETS       space-separated robots (default: every robot but the episode's source)
#   PACK          targets optimised at the same time on this GPU (default 2)
#   SPIDER_ARGS   extra stage-3 arguments, e.g. "--num_samples 1024 --iters 5"
#   RECORD_ARGS   extra stage-4 arguments, e.g. --keep_failed
#   RENDER=0      skip stage 5 (record only)

set +e        # failures are handled per stage; one target failing must not kill the others
RUN="$WS/b2dr_runs"
ALL_ROBOTS="rh56dfx rh5dg2 shadow schunk wuji"
PACK="${PACK:-2}"
SPIDER_ARGS="${SPIDER_ARGS:-}"
RECORD_ARGS="${RECORD_ARGS:-}"
RENDER="${RENDER:-1}"
cd "$REPO_DIR"

stage() {   # stage LOG TIMEOUT_S script args...
  local log="$1" tmo="$2"; shift 2
  mkdir -p "$(dirname "$log")"
  timeout --kill-after=60 "$tmo" "${ISAAC[@]}" "$PY" "$@" --headless >> "$log" 2>&1
  local rc=$?
  grep -h "^RESULT\|^skip" "$log" | tail -1
  return $rc
}

echo "=== $TASK ep$EPISODE: stage 1 (reference) $(date +%T) ==="
S1LOG="$RUN/logs/${TASK}_ep${EPISODE}_stage1.log"
stage "$S1LOG" 3600 scripts/stage1_reference.py --task "$TASK" --episode "$EPISODE" \
  || { echo "FATAL: stage 1 failed, see $S1LOG"; tail -30 "$S1LOG"; exit 1; }
TASKNAME="$(basename "$(ls -d "$WS/b2d_origin/dataset/${TASK}"*/ | head -1)")"
HOST_EPDIR="$RUN/$TASKNAME/ep$(printf %03d "$EPISODE")"
SRC="$(grep -o '"source": "[a-z0-9]*"' "$HOST_EPDIR/reference.json" | cut -d'"' -f4)"
[ -n "$SRC" ] || { echo "FATAL: could not read the source robot from $HOST_EPDIR/reference.json"; exit 1; }
TARGETS="${TARGETS:-$(for r in $ALL_ROBOTS; do [ "$r" = "$SRC" ] || printf '%s ' "$r"; done)}"
echo "=== source $SRC -> targets: $TARGETS (PACK=$PACK) ==="

one_target() {
  local tgt="$1" log="$HOST_EPDIR/$1/run.log"
  mkdir -p "$HOST_EPDIR/$tgt"
  echo "--- $tgt: stage 2 $(date +%T)" >> "$log"
  stage "$log" 3600 scripts/stage2_kinematic.py --task "$TASK" --episode "$EPISODE" --target "$tgt" || return 1
  echo "--- $tgt: stage 3 $(date +%T)" >> "$log"
  # shellcheck disable=SC2086
  stage "$log" 64800 scripts/stage3_spider.py --task "$TASK" --episode "$EPISODE" --target "$tgt" $SPIDER_ARGS || return 1
  echo "--- $tgt: stage 4 $(date +%T)" >> "$log"
  # shellcheck disable=SC2086
  stage "$log" 3600 scripts/stage4_record.py --task "$TASK" --episode "$EPISODE" --target "$tgt" $RECORD_ARGS || return 1
  local rec="$HOST_EPDIR/$tgt/spider_record.json"
  if [ "$RENDER" = "1" ] && grep -q '"success": true' "$rec"; then
    local h5; h5="$(grep -o '"episode_hdf5": "[^"]*"' "$rec" | cut -d'"' -f4)"
    echo "--- $tgt: stage 5 $(date +%T) $h5" >> "$log"
    timeout --kill-after=60 7200 "${ISAAC[@]}" bash -c \
      "PY=$PY bash scripts/stage5_render.sh $(dirname "$h5") $(basename "$h5")" >> "$log" 2>&1 || return 1
  fi
}

pids=() rc=0
for tgt in $TARGETS; do
  ( one_target "$tgt" && echo "[$tgt] done" || echo "[$tgt] FAILED, see $HOST_EPDIR/$tgt/run.log" ) &
  pids+=("$!")
  sleep 20                                   # stagger Kit start-ups
  while [ "$(jobs -rp | wc -l)" -ge "$PACK" ]; do sleep 30; done
done
for p in "${pids[@]}"; do wait "$p" || rc=1; done

echo "=== $TASK ep$EPISODE summary $(date +%T) ==="
for tgt in $TARGETS; do
  f="$HOST_EPDIR/$tgt/spider_record.json"
  if [ -f "$f" ]; then
    printf '  %-8s %s\n' "$tgt" "$(grep -o '"success": [a-z]*' "$f") $(grep -o '"max": [0-9.]*' "$f") (drift cm)"
  else
    printf '  %-8s %s\n' "$tgt" "no result (see $HOST_EPDIR/$tgt/run.log)"
  fi
done
exit "$rc"
