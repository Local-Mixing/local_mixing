#!/bin/bash
# Extension arms for the spread-vs-expansion question (2026-08-05):
#  - linger test: a finished r=1.5 arm continued 4x its budget at CONSTANT
#    target — does hold-time substitute for growth-leg transport?
#  - knee mini-panel: r 1.6 / 1.75 / 1.9 at the second-order-fixed knobs
#    (b=1.5 c=3 temp=target/25), same recipe as the main panel.
set -u
cd "$(dirname "$0")"
BIN=$HOME/local_mixing_sd/target/release/fmix

run() {
  local TAG=$1; shift
  export FMIX_STOP_FLAG=$PWD/$TAG.stop FMIX_DUMP_FLAG=$PWD/$TAG.dump
  "$BIN" "$@" --state-out "$PWD/$TAG.state" --output "$PWD/$TAG.mpmct1" \
      > "$PWD/$TAG.log" 2>&1
  echo "$TAG exit=$? $(date +%H:%M:%S)" >> panel_progress.txt
}

# Linger extension: state finished at absolute move 6,005,358; +24M more.
run xp_ext_r1.5_4x --resume "$PWD/xp_r1.5_b1.5_c3_t25.state" \
  --target-size 544492 --temp 21780 --split-base 1.5 --split-damp 3 \
  --p-twist 0 --p-db 0 --p-comp 0 --p-any 0 --k-max 12 \
  --moves 30005358 --report-every 500000 &

# Knee mini-panel (resumes the SPLIT state like the main panel; --moves is
# absolute = 12 x target + 5,358).
run xp_r1.6_b1.5_c3_t25 --resume "$PWD/nR20_k2p_split.state" \
  --target-size 580792 --temp 23232 --split-base 1.5 --split-damp 3 \
  --p-twist 0 --p-db 0 --p-comp 0 --p-any 0 --k-max 12 \
  --moves 6974862 --report-every 500000 &

run xp_r1.75_b1.5_c3_t25 --resume "$PWD/nR20_k2p_split.state" \
  --target-size 635241 --temp 25410 --split-base 1.5 --split-damp 3 \
  --p-twist 0 --p-db 0 --p-comp 0 --p-any 0 --k-max 12 \
  --moves 7628250 --report-every 500000 &

run xp_r1.9_b1.5_c3_t25 --resume "$PWD/nR20_k2p_split.state" \
  --target-size 689691 --temp 27588 --split-base 1.5 --split-damp 3 \
  --p-twist 0 --p-db 0 --p-comp 0 --p-any 0 --k-max 12 \
  --moves 8281650 --report-every 500000 &

wait
echo "EXT ARMS COMPLETE $(date)" >> panel_progress.txt
