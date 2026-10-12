#!/bin/bash
# delivery_parity.sh OUT [PACKING_DIR]: the Rust delivery loop (`tdp_unpacking run
# --refine`, docs/DELIVERY_STAGE.md §4) against the Python reference
# (`scripts/windowed_delivery/waist/deliver.py`, §3) on the n8_L8 fixture.
#
# Builds the packing of the committed n8_L8 inputs with `tdp_packing run` (unless
# PACKING_DIR, a directory with rep512.mpmct1, rep512.bounds, deliverable.bin and
# deliverable.tdp4, is given), runs both sides with the same seed, time limit and
# restarts, and cmp's bounds.final, deliverable.tdp4 and unpacked.mpmct1 (and the
# per-round bounds.<r> / deliverable.<r>.tdp4). The reference drives
# `tdp_unpacking run --windows` per round, `tdp_unpacking canon` for the final
# order and `hmap_affine --g-cuts` for the cut scan, so all three binaries are
# needed:
#   TDP_UNPACKING=.../release/tdp_unpacking TDP_PACKING=.../release/tdp_packing \
#   HMAP_AFFINE=.../release/hmap_affine bash tests/manual/delivery_parity.sh OUT
# Knobs: SEED (3), TIME_LIMIT (30: no window must hit the deadline, or the
# synthesized output is timing-dependent), RESTARTS (2: window 1 of the fixture
# then stalls and is split, so the refinement path is exercised), THREADS (4).
# Exit 0 when every file is identical and both sides report status=ok.
set -uo pipefail
: "${TDP_UNPACKING:?path to tdp_unpacking}"
: "${TDP_PACKING:?path to tdp_packing}"
: "${HMAP_AFFINE:?path to hmap_affine (built with --features security-tools)}"
SEED=${SEED:-3}; TIME_LIMIT=${TIME_LIMIT:-30}; RESTARTS=${RESTARTS:-2}; THREADS=${THREADS:-4}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
FX=$ROOT/tests/fixtures/stage4
OUT=${1:?usage: delivery_parity.sh OUT [PACKING_DIR]}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
PK=${2:-}
if [ -z "$PK" ]; then
  PK=$OUT/packing; rm -rf "$PK"
  "$TDP_PACKING" run --stage2 "$FX/cview_n8/g2.mpmct1" --stage3 "$FX/n8_L8/g3.mpmct1" \
      --source-c "$FX/n8_L8/c.g57" --n 8 --out-dir "$PK" --label n8_L8 --no-oracle-files \
      > "$OUT/packing.stdout" 2> "$OUT/packing.err" || { echo "tdp_packing run failed (see $OUT/packing.err)"; exit 1; }
fi
C=$FX/n8_L8/c.g57
fail=0
rm -rf "$OUT/rust" "$OUT/py"
"$TDP_UNPACKING" run --refine --deliverable "$PK/deliverable.tdp4" --packed "$PK/rep512.mpmct1" \
    --bounds "$PK/rep512.bounds" --source-c "$C" --n 8 --out-dir "$OUT/rust" --seed "$SEED" \
    --threads "$THREADS" --restarts "$RESTARTS" --time-limit "$TIME_LIMIT" --label n8 \
    > "$OUT/rust.stdout" 2> "$OUT/rust.err" || { echo "tdp_unpacking --refine failed (see $OUT/rust.err)"; fail=1; }
python3 -I "$ROOT/scripts/windowed_delivery/waist/deliver.py" --rep "$PK/rep512.mpmct1" --bounds "$PK/rep512.bounds" \
    --body "$PK/deliverable.bin" --source-c "$C" --n 8 --out "$OUT/py" --time-limit "$TIME_LIMIT" \
    --restarts "$RESTARTS" --threads "$THREADS" --seed "$SEED" --synth tdp_unpacking \
    --synth-bin "$TDP_UNPACKING" --canon-bin "$TDP_UNPACKING" --hmap "$HMAP_AFFINE" --label n8 \
    > "$OUT/py.stdout" 2> "$OUT/py.err" || { echo "deliver.py failed (see $OUT/py.err)"; fail=1; }
for f in bounds.final deliverable.tdp4 unpacked.mpmct1 $(cd "$OUT/py" 2>/dev/null && ls bounds.[0-9]* deliverable.[0-9]*.tdp4 2>/dev/null); do
  if cmp -s "$OUT/rust/$f" "$OUT/py/$f"; then echo "$f: identical"; else echo "$f: DIFFER"; fail=1; fi
done
rs=$(grep -E '^status=' "$OUT/rust/unpacking.report" 2>/dev/null); ps=$(grep -E '^status=' "$OUT/py/delivery.report" 2>/dev/null)
echo "rust: $rs $(grep -E '^(rounds|windows_final|splits|unclean_cuts|verbatim|deadline_hits)=' "$OUT/rust/unpacking.report" | tr '\n' ' ')"
echo "py:   $ps $(grep -E '^(rounds|windows_final|splits|unclean_cuts|verbatim)=' "$OUT/py/delivery.report" | tr '\n' ' ')"
[ "$rs" = "status=ok" ] && [ "$ps" = "status=ok" ] || fail=1
grep -q '^deadline_hits=0$' "$OUT/rust/unpacking.report" || { echo "WARNING: a window hit the deadline; raise TIME_LIMIT"; fail=1; }
[ $fail = 0 ] && echo "PARITY OK" || echo "PARITY FAILED"
exit $fail
