#!/bin/bash
# Heavy-damping arms (2026-08-05, user request): c=1 with b in {2,3} — the
# indirect "prioritize not-yet-crossed gates" lever: a width-w split passes
# with prob b^-(w-1), so reaching the same target spreads the crossing work
# over many more distinct shot gates. r in {1.5,2} at the second-order-fixed
# t25; budgets stretched (b=2: 20x target, b=3: 30x target) because damped
# growth is slower and runtime is declared expendable.
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

# r=1.5: target 544,492 temp 21,780 | b=2: 20x = 10,895,198 abs; b=3: 30x = 16,340,118 abs
run xp_r1.5_b2_c1_t25 --resume "$PWD/nR20_k2p_split.state" \
  --target-size 544492 --temp 21780 --split-base 2 --split-damp 1 \
  --p-twist 0 --p-db 0 --p-comp 0 --p-any 0 --k-max 12 \
  --moves 10895198 --report-every 500000 &

run xp_r1.5_b3_c1_t25 --resume "$PWD/nR20_k2p_split.state" \
  --target-size 544492 --temp 21780 --split-base 3 --split-damp 1 \
  --p-twist 0 --p-db 0 --p-comp 0 --p-any 0 --k-max 12 \
  --moves 16340118 --report-every 500000 &

# r=2: target 725,990 temp 29,040 | b=2: 20x = 14,525,158 abs; b=3: 30x = 21,785,058 abs
run xp_r2_b2_c1_t25 --resume "$PWD/nR20_k2p_split.state" \
  --target-size 725990 --temp 29040 --split-base 2 --split-damp 1 \
  --p-twist 0 --p-db 0 --p-comp 0 --p-any 0 --k-max 12 \
  --moves 14525158 --report-every 500000 &

run xp_r2_b3_c1_t25 --resume "$PWD/nR20_k2p_split.state" \
  --target-size 725990 --temp 29040 --split-base 3 --split-damp 1 \
  --p-twist 0 --p-db 0 --p-comp 0 --p-any 0 --k-max 12 \
  --moves 21785058 --report-every 500000 &

wait
echo "DAMP ARMS COMPLETE $(date)" >> panel_progress.txt
