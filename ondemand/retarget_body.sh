# One episode, every target robot: stage 1 once, then per target tools/retarget/run_target.py --
#   stage 2 kinematic retarget -> SPIDER + Bench2Dex record, retried with new seeds / more search until
#   MetricTracker calls it a success (MAX_ATTEMPTS) -> render like the released replay data -> push the
#   episode to <ns>/b2d-<scene>-<target>-retargeting on Hugging Face.
# Sourced by retarget_array.sbatch with TASK and EPISODE set; no #SBATCH here. Every step skips what
# is already done, so a resubmitted or walltime-killed item resumes.
#
#   TARGETS        space-separated robots (default: every robot but the episode's source)
#   PACK           targets processed at the same time on this GPU (default 2)
#   MAX_ATTEMPTS   SPIDER attempts per target before the episode is left out (default 5)
#   SPIDER_ARGS    extra stage-3 arguments for every attempt
#   UPLOAD=0       keep results local; HF_NAMESPACE / HF_STAGES are read by tools/retarget/upload_hf.py

set +e        # failures are handled per stage; one target failing must not kill the others
RUN="$WS/b2dr_runs"
ALL_ROBOTS="rh56dfx rh5dg2 shadow schunk wuji"
PACK="${PACK:-2}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
SPIDER_ARGS="${SPIDER_ARGS:-}"
UPLOAD="${UPLOAD:-1}"
export APPTAINERENV_HF_NAMESPACE="${HF_NAMESPACE:-}" APPTAINERENV_HF_STAGES="${HF_STAGES:-origin replay}"
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
stage "$S1LOG" 3600 tools/retarget/stage1_reference.py --task "$TASK" --episode "$EPISODE" \
  || { echo "FATAL: stage 1 failed, see $S1LOG"; tail -30 "$S1LOG"; exit 1; }
TASKNAME="$(basename "$(ls -d "$WS/b2d_origin/dataset/${TASK}"*/ | head -1)")"
HOST_EPDIR="$RUN/$TASKNAME/ep$(printf %03d "$EPISODE")"
SRC="$(grep -o '"source": "[a-z0-9]*"' "$HOST_EPDIR/reference.json" | cut -d'"' -f4)"
[ -n "$SRC" ] || { echo "FATAL: could not read the source robot from $HOST_EPDIR/reference.json"; exit 1; }
TARGETS="${TARGETS:-$(for r in $ALL_ROBOTS; do [ "$r" = "$SRC" ] || printf '%s ' "$r"; done)}"
echo "=== source $SRC -> targets: $TARGETS (PACK=$PACK) ==="

one_target() {
  local tgt="$1"
  mkdir -p "$HOST_EPDIR/$tgt"
  local up=""; [ "$UPLOAD" = "1" ] || up="--no_upload"
  timeout --kill-after=60 172000 "${ISAAC[@]}" env PY="$PY" "$PY" tools/retarget/run_target.py \
    --task "$TASK" --episode "$EPISODE" --target "$tgt" --max_attempts "$MAX_ATTEMPTS" \
    --spider_args "$SPIDER_ARGS" $up 2>&1 | grep "^RESULT"
  grep -q '"state": "\(uploaded\|rendered\)"' "$HOST_EPDIR/$tgt/status.json"
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
  f="$HOST_EPDIR/$tgt/status.json"
  printf '  %-8s %s\n' "$tgt" "$( [ -f "$f" ] && grep -o '"state": "[a-z_0-9]*"' "$f" || echo 'no status' )"
done
exit "$rc"
