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
RUN_QUEUE="${RUN_QUEUE:-$ROOT/security_tests/campaigns/run_n128m1000_campaign_queue.sh}"
START_SAT="${START_SAT:-$ROOT/security_tests/campaigns/start_detached_sat_for_circuit.sh}"
CAP="${CAP:-8}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MIN_MEM_AVAILABLE_KIB="${MIN_MEM_AVAILABLE_KIB:-67108864}"
STOP_WHEN_QUEUE_EMPTY="${STOP_WHEN_QUEUE_EMPTY:-0}"

RUN_DIR="$TEST_ROOT/$RUN_ID/$SERVER_NAME"
LOG="$RUN_DIR/autofill.log"
STATE="$RUN_DIR/autofill_launched.tsv"
LOCK="$RUN_DIR/autofill.lock"

mkdir -p "$RUN_DIR"
touch "$LOG"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG"
}

if [[ ! -s "$CONFIG_FILE" ]]; then
  echo "missing config file: $CONFIG_FILE" >&2
  exit 1
fi
if [[ ! -x "$RUN_QUEUE" ]]; then
  echo "missing run queue script: $RUN_QUEUE" >&2
  exit 1
fi
if [[ ! -x "$START_SAT" ]]; then
  echo "missing SAT starter script: $START_SAT" >&2
  exit 1
fi

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "autofill monitor already running for $SERVER_NAME $RUN_ID"
  exit 0
fi

if [[ ! -s "$STATE" ]]; then
  printf 'tag\tpid\tconfig\tlauncher_log\tstarted_at\n' > "$STATE"
fi

active_sss() {
  { ps -C local_mixing_bi --no-headers 2>/dev/null || true; } |
    wc -l |
    awk '{ print $1 + 0 }'
}

active_sat() {
  { pgrep -af solve_feistal_preimage.py 2>/dev/null || true; } |
    awk '{ count++ } END { print count + 0 }'
}

available_kib() {
  awk '/MemAvailable:/ { print $2; found=1 } END { if (!found) print 0 }' /proc/meminfo
}

tag_launched() {
  local tag="$1"
  awk -F '\t' -v tag="$tag" 'NR > 1 && $1 == tag { found=1 } END { exit found ? 0 : 1 }' "$STATE"
}

sweep_sat() {
  find "$TEST_ROOT" -path "*/$SERVER_NAME/*round*.txt" -type f -print 2>/dev/null |
    sort |
    while IFS= read -r circuit; do
      case "$circuit" in
        *gadget*|*snapshot*|*.sat.*) continue ;;
      esac
      local run_dir
      local tag
      run_dir="$(dirname "$circuit")"
      tag="$(basename "$circuit" .txt)"
      "$START_SAT" "$SERVER_NAME" "$tag" "$circuit" "$run_dir" >> "$LOG" 2>&1 || \
        log "warning: failed SAT sweep for $circuit"
    done
}

launch_one() {
  local tag="$1"
  local rounds="$2"
  local m="$3"
  local x="$4"
  local collision_rounds="$5"
  local pre_timeout="$6"
  local total_timeout="$7"
  local one_config="$RUN_DIR/${tag}.config.tsv"
  local launcher_log="$RUN_DIR/${tag}.launcher.log"
  local pid

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$tag" "$rounds" "$m" "$x" "$collision_rounds" "$pre_timeout" "$total_timeout" \
    > "$one_config"

  nohup "$RUN_QUEUE" "$SERVER_NAME" "$RUN_ID" "$one_config" \
    > "$launcher_log" 2>&1 < /dev/null &
  pid="$!"
  printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "$pid" "$one_config" "$launcher_log" "$(date -Is)" >> "$STATE"
  log "launched $tag pid=$pid rounds=$rounds m=$m x=$x collision_rounds=$collision_rounds"
}

queue_empty=0
log "autofill starting server=$SERVER_NAME run=$RUN_ID cap=$CAP config=$CONFIG_FILE"

while true; do
  sweep_sat

  sss_count="$(active_sss)"
  sat_count="$(active_sat)"
  mem_kib="$(available_kib)"
  log "status sss=$sss_count sat=$sat_count mem_available_kib=$mem_kib"

  if (( mem_kib < MIN_MEM_AVAILABLE_KIB )); then
    log "memory below threshold; skipping launches"
    sleep "$POLL_SECONDS"
    continue
  fi

  queue_empty=1
  while (( sss_count < CAP )); do
    launched_this_round=0
    while IFS=$'\t' read -r tag rounds m x collision_rounds pre_timeout total_timeout extra; do
      [[ -z "${tag:-}" || "${tag:0:1}" == "#" ]] && continue
      if tag_launched "$tag"; then
        continue
      fi
      queue_empty=0
      launch_one "$tag" "$rounds" "$m" "$x" "$collision_rounds" "$pre_timeout" "$total_timeout"
      sss_count=$((sss_count + 1))
      launched_this_round=1
      break
    done < "$CONFIG_FILE"

    if (( launched_this_round == 0 )); then
      break
    fi
  done

  if (( queue_empty == 1 && STOP_WHEN_QUEUE_EMPTY == 1 )); then
    log "queue empty; exiting"
    exit 0
  fi

  sleep "$POLL_SECONDS"
done
