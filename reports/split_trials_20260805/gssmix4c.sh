#!/bin/bash
# gssmix4c (2026-08-06): the four short arms (|C|=|D|=3000) REDONE with
# INDEPENDENT CSPRNG seeds — each arm generates its own C/sandwich/gadget
# (no sharing), per the random-seed directive. Seeds are drawn by the driver
# itself and land in each arm's SEED file (mode 600), never in a log.
# NOTE: independent C's mean the arms are NOT a paired knob comparison —
# knob differences are confounded with C-to-C variation. These are
# deliverable-grade runs.
set -u
export FROZEN_DB_DIR=$HOME/frozen_m1_m11
export FROZEN_CURATED_DIR=$HOME/frozen_curated_m1_m11
export FROZEN_CURATED_VALUE_CONVENTION=legacy-swapped-controls
GM=$HOME/local_mixing_sd/scripts/gss_mix.sh
BASE=$HOME/tds/gssmix128_indep_20260806
mkdir -p "$BASE"
echo "GSSMIX4C START $(date)" >> "$BASE/progress.txt"

run_arm() {
  local ARM=$1 E=$2 H=$3
  "$GM" -n 128 -o "$BASE/$ARM" --mcd 3000 --expand "$E" --hold "$H" \
      > "$BASE/$ARM.driver.log" 2>&1
  echo "$ARM exit=$? $(date)" >> "$BASE/progress.txt"
}
run_arm e1.5_h10 1.5 10 &
run_arm e1.5_h30 1.5 30 &
run_arm e2_h10 2 10 &
run_arm e2_h30 2 30 &
wait
echo "GSSMIX4C COMPLETE $(date)" >> "$BASE/progress.txt"
