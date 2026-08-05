#!/bin/bash
# b>3 probe (2026-08-05): does the b=3/c=1 equalization keep improving at
# b=3.5/4, or does over-damping stall growth (lower equilibrium, later
# arrival)? Same recipe as the frontier arms; snapshots every 2M for the
# arrival-peak read. Note w2 splits pass w.p. 1/b: expect slower arrival —
# budgets stay at the frontier arms' generous marks.
set -u
cd "$(dirname "$0")"
BIN=$HOME/local_mixing_sd/target/release/fmix

run() {
  local TAG=$1; shift
  export FMIX_STOP_FLAG=$PWD/$TAG.stop FMIX_DUMP_FLAG=$PWD/$TAG.dump
  "$BIN" "$@" --resume "$PWD/nR20_k2p_split.state" \
    --split-damp 1 --p-twist 0 --p-db 0 --p-comp 0 --p-any 0 --k-max 12 \
    --report-every 500000 --snap-every-moves 2000000 \
    --state-out "$PWD/$TAG.state" --output "$PWD/$TAG.mpmct1" \
    > "$PWD/$TAG.log" 2>&1
  echo "$TAG exit=$? $(date +%H:%M:%S)" >> panel_progress.txt
}

run bh_r2_b3.5_c1  --split-base 3.5 --target-size 725990 --temp 29040 --moves 21785058 &
run bh_r2_b4_c1    --split-base 4   --target-size 725990 --temp 29040 --moves 21785058 &
run bh_r2.5_b3.5_c1 --split-base 3.5 --target-size 907488 --temp 36300 --moves 36305000 &
run bh_r2.5_b4_c1   --split-base 4   --target-size 907488 --temp 36300 --moves 36305000 &

wait
echo "BHEAVY ARMS COMPLETE $(date)" >> panel_progress.txt
