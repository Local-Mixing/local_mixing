#!/bin/bash
# gssmix4 (2026-08-05): four full GSS-MIX pipeline arms at n=128, sharing
# ONE gadget (same C, same GSS — phase-A knobs are the only variable):
#   expansion {1.5, 2} x hold {10, 30} effs.
# Stages 4-6 run at the calibrated defaults inside each pipeline invocation.
set -u
export FROZEN_DB_DIR=$HOME/frozen_m1_m11
export FROZEN_CURATED_DIR=$HOME/frozen_curated_m1_m11
export FROZEN_CURATED_VALUE_CONVENTION=legacy-swapped-controls
GM=$HOME/local_mixing_sd/scripts/gss_mix.sh
BASE=$HOME/tds/gssmix128_m3000_20260805
mkdir -p "$BASE"
echo "GSSMIX4 START $(date)" >> "$BASE/progress.txt"

# Stages 1+2 once, in the first arm's dir; clone the artifacts to the rest.
A1=$BASE/e1.5_h10
"$GM" -n 128 -o "$A1" -s 1 --mcd 3000 --stop-after 2 > "$BASE/gen.log" 2>&1 \
  || { echo "GEN FAILED $(date)" >> "$BASE/progress.txt"; exit 1; }
for arm in e1.5_h30 e2_h10 e2_h30; do
  mkdir -p "$BASE/$arm"
  cp "$A1/gss.mpmct1" "$A1/gss.mpmct1.sandwich.mpmct1" \
     "$A1/gss.mpmct1.source_c.g57" "$BASE/$arm/"
done
echo "GADGET READY $(wc -l < "$A1/gss.mpmct1") lines $(date)" >> "$BASE/progress.txt"

run_arm() {
  local ARM=$1 E=$2 H=$3
  "$GM" -n 128 -o "$BASE/$ARM" -s 1 --mcd 3000 --expand "$E" --hold "$H" \
      > "$BASE/$ARM.driver.log" 2>&1
  echo "$ARM exit=$? $(date)" >> "$BASE/progress.txt"
}
run_arm e1.5_h10 1.5 10 &
run_arm e1.5_h30 1.5 30 &
run_arm e2_h10 2 10 &
run_arm e2_h30 2 30 &
wait
echo "GSSMIX4 COMPLETE $(date)" >> "$BASE/progress.txt"
