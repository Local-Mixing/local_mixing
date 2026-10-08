#!/bin/bash
# A/B stage 3 (db_mixing) for the v3 true-friend (ncdb) stores: identical
# input, seed, binary and flags; only FROZEN_CURATED_DIR differs per arm.
#   ./run_stage3.sh c|bp
# Mirrors runs/canon_ab_20260928/run_stage3.sh (mini profile 2,2,2,2,2,
# moves = round(2*2*G_IN*1.3), seed = mini_sandwich SEED + 2), with two
# changes: the curated store is a circuit-keyed ncdb (FROZEN_CURATED_KEY=
# circuit) and the binary carries the circuit-key lookup switch.
set -euo pipefail
ARM=$1
AB=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$AB/../.." && pwd)
OUT="$AB/$ARM"
mkdir -p "$OUT"
IN="$AB/stage2_tdp.mpmct1"
SEED=5336678513626647688            # runs/mini_sandwich_20260927/SEED
G_IN=$(head -1 "$IN" | awk '{print $3}')
A_MOVES=$(python3 -c "print(round(2*2*$G_IN*1.3))")
case "$ARM" in
  c)  CUR="$ROOT/frozen_db/frozen_tf_c_20260930" ;;
  bp) CUR="$ROOT/frozen_db/frozen_tf_bp_20260930" ;;
  *)  echo "arm must be c or bp" >&2; exit 2 ;;
esac
export CANON_RULE_L_BRANCH_CAP=512 CANON_MONOMIAL_CAP=200000 LOOKUP_CACHE_MB=2048
export FROZEN_DB_DIR=$ROOT/frozen_db/frozen_m1_m11
export FROZEN_CURATED_DIR=$CUR
export FROZEN_CURATED_KEY=circuit
export FROZEN_FILTER=1
export MIXER_STOP_FLAG=$OUT/stage3.stop MIXER_DUMP_FLAG=$OUT/stage3.dump
rm -f "$MIXER_STOP_FLAG"
{
  echo "arm=$ARM"
  echo "curated=$CUR key_mode=circuit"
  echo "binary=$AB/bin/circuit_mixer.ncdb"
  echo "started=$(date -u +%FT%TZ)"
} > "$OUT/run.info"
"$AB/bin/circuit_mixer.ncdb" --input "$IN" --tdp --db-mixing --profile 2,2,2,2,2 \
  --moves "$A_MOVES" --seed "$((SEED + 2))" \
  --p-db 1.0 --p-comp 1.0 --p-any 0.1 \
  --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048 \
  --no-local-verify --verify-every 2000000 --report-every 100000 \
  --state-out "$OUT/db_mixing.state" --output "$OUT/stage3_out.mpmct1" > "$OUT/stage3.log" 2>&1
echo "exit=$? finished=$(date -u +%FT%TZ)" >> "$OUT/run.info"
