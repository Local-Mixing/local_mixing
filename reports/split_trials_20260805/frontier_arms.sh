#!/bin/bash
# Pareto-frontier sweep (2026-08-05): find, per realized size, the move count
# at which frac(inputs with >=3 descendants) crosses 2/3. Arms at b=3 c=1
# t25 (the equalizing recipe) across targets, snapshots every 2M moves;
# post-hoc analysis finds the crossing. Budgets 40x target (moves ABSOLUTE,
# +5,358 for the split state's counter).
set -u
cd "$(dirname "$0")"
BIN=$HOME/local_mixing_sd/target/release/fmix

run() {
  local TAG=$1; shift
  export FMIX_STOP_FLAG=$PWD/$TAG.stop FMIX_DUMP_FLAG=$PWD/$TAG.dump
  "$BIN" "$@" --resume "$PWD/nR20_k2p_split.state" \
    --split-base 3 --split-damp 1 \
    --p-twist 0 --p-db 0 --p-comp 0 --p-any 0 --k-max 12 \
    --report-every 500000 --snap-every-moves 2000000 \
    --state-out "$PWD/$TAG.state" --output "$PWD/$TAG.mpmct1" \
    > "$PWD/$TAG.log" 2>&1
  echo "$TAG exit=$? $(date +%H:%M:%S)" >> panel_progress.txt
}

run fr_r1.75_b3_c1 --target-size 635241 --temp 25410 --moves 25415000 &
run fr_r2.25_b3_c1 --target-size 816739 --temp 32670 --moves 32675000 &
run fr_r2.5_b3_c1  --target-size 907488 --temp 36300 --moves 36305000 &

wait
echo "FRONTIER ARMS COMPLETE $(date)" >> panel_progress.txt
