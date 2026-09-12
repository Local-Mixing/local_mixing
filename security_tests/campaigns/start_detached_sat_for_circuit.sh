#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 )); then
  echo "usage: $0 SERVER_NAME TAG CIRCUIT_PATH RUN_DIR" >&2
  exit 2
fi

SERVER_NAME="$1"
TAG="$2"
CIRCUIT_PATH="$3"
RUN_DIR="$4"

ROOT="${ROOT:-/home/cc/local_mixing}"
SOLVER_DIR="${SOLVER_DIR:-$ROOT/mixing_tests/sat_testing}"
PY="${PY:-$SOLVER_DIR/.venv/bin/python}"
SOLVER="${SOLVER:-$SOLVER_DIR/solve_feistal_preimage.py}"
N="${N:-128}"
RANDOM_TARGET_SEED="${RANDOM_TARGET_SEED:-20260613}"

mkdir -p "$RUN_DIR"

if [[ ! -s "$CIRCUIT_PATH" ]]; then
  echo "missing circuit: $CIRCUIT_PATH" >&2
  exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "missing python venv: $PY" >&2
  exit 1
fi
if [[ ! -r "$SOLVER" ]]; then
  echo "missing solver wrapper: $SOLVER" >&2
  exit 1
fi

sat_log="$RUN_DIR/${TAG}.sat.log"
sat_pid_file="$RUN_DIR/${TAG}.sat.pid"
sat_phase="$RUN_DIR/${TAG}.sat.phase"
sat_jobs="$RUN_DIR/sat_jobs.tsv"

if [[ -s "$sat_pid_file" ]]; then
  pid="$(cat "$sat_pid_file")"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "already running pid=$pid log=$sat_log"
    exit 0
  fi
fi

if pgrep -af "solve_feistal_preimage.py.*$CIRCUIT_PATH" >/dev/null 2>&1; then
  echo "matching SAT process already running for $CIRCUIT_PATH"
  exit 0
fi

if [[ ! -s "$sat_jobs" ]]; then
  printf 'server\ttag\tpid\tbackend\tcircuit\tlog\tstarted_at\n' > "$sat_jobs"
fi

existing_log="$(
  awk -F '\t' -v circuit="$CIRCUIT_PATH" '
    NR > 1 && $5 == circuit && $6 != "" { print $6; exit }
  ' "$sat_jobs" 2>/dev/null || true
)"
if [[ -n "$existing_log" && -s "$existing_log" ]]; then
  echo "already has SAT log for circuit=$CIRCUIT_PATH log=$existing_log"
  exit 0
fi

backend="pysat-kissat404"
if command -v kissat >/dev/null 2>&1; then
  kissat_path="$(command -v kissat)"
  backend="kissat:${kissat_path}"
  nohup "$PY" "$SOLVER" "$CIRCUIT_PATH" \
    --n "$N" \
    --backend kissat \
    --kissat "$kissat_path" \
    --random-target-seed "$RANDOM_TARGET_SEED" \
    --progress-every 100000 \
    > "$sat_log" 2>&1 < /dev/null &
else
  nohup "$PY" "$SOLVER" "$CIRCUIT_PATH" \
    --n "$N" \
    --backend pysat \
    --solver kissat404 \
    --random-target-seed "$RANDOM_TARGET_SEED" \
    --progress-every 100000 \
    > "$sat_log" 2>&1 < /dev/null &
fi

sat_pid="$!"
printf '%s\n' "$sat_pid" > "$sat_pid_file"
started_at="$(date -Is)"
printf 'sat_started tag=%s pid=%s backend=%s log=%s\n' "$TAG" "$sat_pid" "$backend" "$sat_log" > "$sat_phase"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$SERVER_NAME" "$TAG" "$sat_pid" "$backend" "$CIRCUIT_PATH" "$sat_log" "$started_at" >> "$sat_jobs"
echo "started SAT pid=$sat_pid backend=$backend log=$sat_log"
