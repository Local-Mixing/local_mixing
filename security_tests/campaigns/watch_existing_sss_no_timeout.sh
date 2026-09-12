#!/usr/bin/env bash
set -euo pipefail

if (( $# < 12 )); then
  echo "usage: $0 SERVER_NAME RUN_ID TAG PID ROUNDS M X COLLISION_ROUNDS START_EPOCH DEST LOG SUMMARY" >&2
  exit 2
fi

SERVER_NAME="$1"
RUN_ID="$2"
TAG="$3"
PID="$4"
ROUNDS="$5"
M="$6"
X="$7"
COLLISION_ROUNDS="$8"
START_EPOCH="$9"
DEST="${10}"
LOG="${11}"
SUMMARY="${12}"

now_epoch="$(date +%s)"
if (( START_EPOCH > now_epoch )); then
  START_EPOCH="$now_epoch"
fi

ROOT="${ROOT:-/home/cc/local_mixing}"
SOLVER_DIR="${SOLVER_DIR:-$ROOT/mixing_tests/sat_testing}"
PY="${PY:-$SOLVER_DIR/.venv/bin/python}"
SOLVER="${SOLVER:-$SOLVER_DIR/solve_feistal_preimage.py}"
RANDOM_TARGET_SEED="${RANDOM_TARGET_SEED:-20260613}"

RUN_DIR="$(dirname "$SUMMARY")"
PHASE="$RUN_DIR/${TAG}.phase"

last_number_for_pattern() {
  local pattern="$1"
  local file="$2"
  awk -v pat="$pattern" '
    $0 ~ pat {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^[0-9]+$/) value = $i
      }
    }
    END { if (value != "") print value }
  ' "$file" 2>/dev/null || true
}

if [[ ! -s "$SUMMARY" ]]; then
  printf 'status\tserver\ttag\trounds\tm\tx\tcollision_rounds\telapsed_seconds\treached_compression\tafter_collision_gates\tfinal_gates\texit_code\tdest\tlog\n' > "$SUMMARY"
fi

while kill -0 "$PID" 2>/dev/null; do
  now="$(date +%s)"
  elapsed=$((now - START_EPOCH))
  printf 'compression tag=%s elapsed=%s no_timeout=1 watched_pid=%s\n' "$TAG" "$elapsed" "$PID" > "$PHASE"
  sleep 60
done

now="$(date +%s)"
elapsed=$((now - START_EPOCH))
after_collision="$(last_number_for_pattern 'After collision game:' "$LOG")"
final_gates="$(last_number_for_pattern 'Final len:' "$LOG")"
if [[ -z "$final_gates" ]]; then
  final_gates="$(last_number_for_pattern 'After compression:' "$LOG")"
fi

status="failed_no_final"
if [[ -n "$final_gates" && -s "$DEST" ]]; then
  status="ok"
fi

if ! grep -q $'\t'"$TAG"$'\t' "$SUMMARY" 2>/dev/null; then
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t1\t%s\t%s\tTBD\t%s\t%s\n' \
    "$status" "$SERVER_NAME" "$TAG" "$ROUNDS" "$M" "$X" "$COLLISION_ROUNDS" \
    "$elapsed" "${after_collision:-TBD}" "${final_gates:-TBD}" "$DEST" "$LOG" >> "$SUMMARY"
fi

if [[ "$status" == "ok" && -x "$PY" && -r "$SOLVER" ]]; then
  sat_log="$RUN_DIR/${TAG}.sat.log"
  sat_pid_file="$RUN_DIR/${TAG}.sat.pid"
  if [[ ! -s "$sat_pid_file" ]] && ! pgrep -af "solve_feistal_preimage.py.*$DEST" >/dev/null 2>&1; then
    if command -v kissat >/dev/null 2>&1; then
      kissat_path="$(command -v kissat)"
      nohup "$PY" "$SOLVER" "$DEST" \
        --n 128 \
        --backend kissat \
        --kissat "$kissat_path" \
        --random-target-seed "$RANDOM_TARGET_SEED" \
        --progress-every 100000 \
        > "$sat_log" 2>&1 < /dev/null &
    else
      nohup "$PY" "$SOLVER" "$DEST" \
        --n 128 \
        --backend pysat \
        --solver kissat404 \
        --random-target-seed "$RANDOM_TARGET_SEED" \
        --progress-every 100000 \
        > "$sat_log" 2>&1 < /dev/null &
    fi
    sat_pid="$!"
    printf '%s\n' "$sat_pid" > "$sat_pid_file"
    printf 'sat_started tag=%s pid=%s log=%s\n' "$TAG" "$sat_pid" "$sat_log" > "$RUN_DIR/${TAG}.sat.phase"
  fi
fi

printf '%s tag=%s elapsed=%s exit_code=TBD reached=1 final_gates=%s no_timeout=1\n' \
  "$status" "$TAG" "$elapsed" "${final_gates:-TBD}" > "$PHASE"
