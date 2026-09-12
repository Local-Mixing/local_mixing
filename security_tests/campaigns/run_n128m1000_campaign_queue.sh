#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "usage: $0 SERVER_NAME RUN_ID CONFIG_FILE" >&2
  exit 2
fi

SERVER_NAME="$1"
RUN_ID="$2"
CONFIG_FILE="$3"

ROOT="${ROOT:-/home/cc/local_mixing}"
TEST_ROOT="${TEST_ROOT:-$ROOT/mixing_tests/n128m1000_campaign}"
RUN_DIR="$TEST_ROOT/$RUN_ID/$SERVER_NAME"
SOURCE="${SOURCE:-$TEST_ROOT/n128m1000.txt}"
BIN="${BIN:-${SECURITY_LEGACY_BIN:-$ROOT/target/release/legacy_mixing}}"
START_SAT="${START_SAT:-$ROOT/security_tests/campaigns/start_detached_sat_for_circuit.sh}"
POLL_SECONDS="${POLL_SECONDS:-10}"
SUMMARY="$RUN_DIR/summary.tsv"

mkdir -p "$RUN_DIR"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

stop_process_group() {
  local pid="$1"
  [[ -z "$pid" ]] && return 0
  kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  sleep 5
  kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
}

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

if [[ ! -x "$BIN" ]]; then
  echo "missing binary: $BIN" >&2
  exit 1
fi

if [[ ! -s "$SOURCE" ]]; then
  echo "missing source: $SOURCE" >&2
  exit 1
fi

if [[ ! -s "$CONFIG_FILE" ]]; then
  echo "missing config file: $CONFIG_FILE" >&2
  exit 1
fi

cd "$ROOT"

if [[ ! -s "$SUMMARY" ]]; then
  printf 'status\tserver\ttag\trounds\tm\tx\tcollision_rounds\telapsed_seconds\treached_compression\tafter_collision_gates\tfinal_gates\texit_code\tdest\tlog\n' > "$SUMMARY"
fi

run_one() {
  local tag="$1"
  local rounds="$2"
  local m="$3"
  local x="$4"
  local collision_rounds="$5"
  local pre_timeout="$6"
  local total_timeout="$7"

  local dest="$RUN_DIR/${tag}_n128m1000_r${rounds}_m${m}_x${x}_cr${collision_rounds}.txt"
  local gadget="$RUN_DIR/${tag}_gadget.txt"
  local out_log="$RUN_DIR/${tag}.sss.log"
  local phase="$RUN_DIR/${tag}.phase"
  local status=""
  local exit_code=""
  local reached=0
  local start_epoch
  local now
  local elapsed=0
  local pid

  rm -f "$ROOT/write_now"
  printf 'running tag=%s rounds=%s m=%s x=%s collision_rounds=%s\n' \
    "$tag" "$rounds" "$m" "$x" "$collision_rounds" > "$phase"

  log "starting $tag rounds=$rounds m=$m x=$x collision_rounds=$collision_rounds"
  start_epoch="$(date +%s)"
  set +e
  setsid "$BIN" sss \
      -n 128 \
      -m "$m" \
      -x "$x" \
      -s "$SOURCE" \
      -d "$dest" \
      -r "$rounds" \
      --feistalize \
      --slice_zero_random \
      --gates_ahead_expand 3 \
      --gates_ahead_samf 3 \
      --type_attempts 4 \
      --shooting_times 2 \
      --collision_rounds "$collision_rounds" \
      --rg-frequency 2 \
      --expansion_game \
      --gadget_path "$gadget" \
      > "$out_log" 2>&1 &
  pid=$!
  set -e

  while kill -0 "$pid" 2>/dev/null; do
    now="$(date +%s)"
    elapsed=$((now - start_epoch))
    if (( reached == 0 )) && grep -q 'After collision game:' "$out_log" 2>/dev/null; then
      reached=1
      printf 'compression tag=%s elapsed=%s\n' "$tag" "$elapsed" > "$phase"
      log "$tag reached compression after ${elapsed}s"
    fi
    if (( reached == 0 && elapsed >= pre_timeout )); then
      status="skipped_precompression_timeout"
      printf 'skipped tag=%s elapsed=%s reason=precompression_timeout\n' "$tag" "$elapsed" > "$phase"
      log "$tag did not reach compression in ${pre_timeout}s; stopping"
      stop_process_group "$pid"
      break
    fi
    if (( reached == 0 && elapsed >= total_timeout )); then
      status="timeout_precompression"
      printf 'timeout tag=%s elapsed=%s reached=%s\n' "$tag" "$elapsed" "$reached" > "$phase"
      log "$tag hit total timeout ${total_timeout}s; stopping"
      stop_process_group "$pid"
      break
    fi
    sleep "$POLL_SECONDS"
  done

  set +e
  wait "$pid" 2>/dev/null
  exit_code=$?
  set -e

  now="$(date +%s)"
  elapsed=$((now - start_epoch))
  if grep -q 'After collision game:' "$out_log" 2>/dev/null; then
    reached=1
  fi
  if [[ -z "$status" ]]; then
    if [[ "$exit_code" -eq 0 ]]; then
      status="ok"
    else
      status="failed"
    fi
  fi

  local after_collision
  local final_gates
  after_collision="$(last_number_for_pattern 'After collision game:' "$out_log")"
  final_gates="$(last_number_for_pattern 'Final len:' "$out_log")"
  if [[ -z "$final_gates" ]]; then
    final_gates="$(last_number_for_pattern 'After compression:' "$out_log")"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$status" "$SERVER_NAME" "$tag" "$rounds" "$m" "$x" "$collision_rounds" \
    "$elapsed" "$reached" "${after_collision:-TBD}" "${final_gates:-TBD}" \
    "$exit_code" "$dest" "$out_log" >> "$SUMMARY"

  printf '%s tag=%s elapsed=%s exit_code=%s reached=%s final_gates=%s\n' \
    "$status" "$tag" "$elapsed" "$exit_code" "$reached" "${final_gates:-TBD}" > "$phase"
  if [[ "$status" == "ok" && -s "$dest" && -x "$START_SAT" ]]; then
    "$START_SAT" "$SERVER_NAME" "$tag" "$dest" "$RUN_DIR" || \
      log "warning: failed to start SAT for $tag"
  fi
  log "finished $tag status=$status elapsed=${elapsed}s final_gates=${final_gates:-TBD}"
}

while IFS=$'\t' read -r tag rounds m x collision_rounds pre_timeout total_timeout; do
  [[ -z "${tag:-}" || "${tag:0:1}" == "#" ]] && continue
  run_one "$tag" "$rounds" "$m" "$x" "$collision_rounds" "$pre_timeout" "$total_timeout"
done < "$CONFIG_FILE"

log "queue complete summary=$SUMMARY"
