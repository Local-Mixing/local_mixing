#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "usage: $0 SERVER_NAME TAG RUN_DIR" >&2
  exit 2
fi

SERVER_NAME="$1"
TAG="$2"
RUN_DIR="$3"

ROOT="${ROOT:-/home/cc/local_mixing}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
POLL_SECONDS="${POLL_SECONDS:-5}"
TEMP="$ROOT/temp_compression.txt"
WRITE_NOW="$ROOT/write_now"
START_SAT="$ROOT/security_tests/campaigns/start_detached_sat_for_circuit.sh"

mkdir -p "$RUN_DIR"

start_epoch="$(date +%s)"
touch "$WRITE_NOW"

waited=0
while (( waited <= WAIT_SECONDS )); do
  temp_epoch="$(stat -c '%Y' "$TEMP" 2>/dev/null || echo 0)"
  if [[ -s "$TEMP" && "$temp_epoch" -ge "$start_epoch" ]]; then
    sleep 2
    stamp="$(date -u +%Y%m%d_%H%M%S)"
    snapshot="$RUN_DIR/${TAG}_snapshot_${stamp}.txt"
    cp "$TEMP" "$snapshot"
    sat_tag="${TAG}_snapshot_${stamp}"
    if [[ -x "$START_SAT" ]]; then
      "$START_SAT" "$SERVER_NAME" "$sat_tag" "$snapshot" "$RUN_DIR"
    fi
    printf 'snapshot=%s\n' "$snapshot"
    exit 0
  fi
  sleep "$POLL_SECONDS"
  waited=$((waited + POLL_SECONDS))
done

echo "timed out waiting for fresh temp_compression after ${WAIT_SECONDS}s" >&2
exit 1
