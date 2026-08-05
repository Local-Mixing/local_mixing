#!/bin/bash
# xpanel_20260805: crossing-parameter calibration, resumed from the
# split-stage output of nR20_mixed (nR20_k2p_split.state, 362,995 gates).
# 81 arms: r{1.5,2,3} x b{1.2,1.5,1.8} x c{2,3,4} x temp{target/100,/25,/8},
# no twists, no DB. <=16 concurrent (the OOM ceiling is ~18).
set -u
cd "$(dirname "$0")"
BIN=$HOME/local_mixing_sd/target/release/fmix
STATE=$PWD/nR20_k2p_split.state

python3 xpanel_gen.py > arms.txt
echo "PANEL START $(date) — $(wc -l < arms.txt) arms" >> panel_progress.txt

run_arm() {
  local TAG TGT TEMP MOVES B C
  read -r TAG TGT TEMP MOVES B C <<< "$1"
  export FMIX_STOP_FLAG=$PWD/$TAG.stop FMIX_DUMP_FLAG=$PWD/$TAG.dump
  "$BIN" --resume "$STATE" --output "$PWD/$TAG.mpmct1" --state-out "$PWD/$TAG.state" \
    --moves "$MOVES" --target-size "$TGT" --temp "$TEMP" \
    --split-base "$B" --split-damp "$C" \
    --p-twist 0 --p-db 0 --p-comp 0 --p-any 0 --k-max 12 \
    --report-every 50000 > "$PWD/$TAG.log" 2>&1
  echo "$TAG exit=$? $(date +%H:%M:%S)" >> panel_progress.txt
}
export -f run_arm
export BIN STATE PWD

xargs -a arms.txt -d '\n' -P 16 -I{} bash -c 'run_arm "{}"'
echo "PANEL COMPLETE $(date)" >> panel_progress.txt
