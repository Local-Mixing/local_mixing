#!/bin/bash
# reproduce_val_chain.sh N [L8|A] [MC] : the reproduction test (required
# acceptance, docs/STAGE4_RUST_DESIGN.md §10.7). One run of the MANAGED DRIVER
# (local_mixing_bin tdp_gen, config v4 with nothing but source.wires, the store,
# run.skip_stages = [4, 5] + delivery.anf_only (the oracle chain's four stages),
# and a calibration seed file; preset A only for CONFIG=A) and one run of the
# Python/bash reference chain scripts/windowed_delivery/waist/val_chain.sh in
# a path-rewritten sandbox, with the SAME fresh seeds, on the SAME stage
# binaries; every artifact is compared byte for byte.
#
#   N=16 MC=64 or N=8 MC=16: minutes; N=64 (MC 1000 from the table): ~40 min per side.
#
# Environment: BIN_DIR (release dir with gen_sandwich_gadget, circuit_mixer,
# tdp_packing, local_mixing_bin; default $CARGO_TARGET_DIR/release),
# HMAP_BASELINE (pinned baseline hmap_affine, tests/fixtures/stage4/README.md),
# ORACLE_PY (an interpreter passing pyset_selftest.py; default /usr/bin/python3),
# FROZEN_DB_DIR (the regular store), WORK (durable scratch; default
# port_runs/repro_<stamp>). Seeds are calibration values, kept in the private
# $WORK/SEEDS.private and never printed. Exit 0 = every comparison identical
# (or "faithful (failure path)", §10.7 step 5).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
N=${1:?usage: reproduce_val_chain.sh N [L8|A] [MC]}; CFG=${2:-L8}; MC=${3:-}
case $CFG in L8|A) ;; *) echo "CONFIG must be L8 or A" >&2; exit 2 ;; esac
case $N in 64) MC=${MC:-1000} ;; 128) MC=${MC:-3000} ;; *) : "${MC:?MC is required for n=$N}" ;; esac
BIN_DIR=${BIN_DIR:-${CARGO_TARGET_DIR:?set BIN_DIR or CARGO_TARGET_DIR}/release}
: "${HMAP_BASELINE:?pinned baseline hmap_affine}" "${FROZEN_DB_DIR:?the regular frozen store}"
ORACLE_PY=${ORACLE_PY:-/usr/bin/python3}
WORK=${WORK:-$REPO/../port_runs/repro_$(date +%Y%m%d_%H%M%S)_n${N}_$CFG}
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd); umask 077
for b in gen_sandwich_gadget circuit_mixer tdp_packing local_mixing_bin; do
  [ -x "$BIN_DIR/$b" ] || { echo "missing $BIN_DIR/$b" >&2; exit 2; }
done
S=$("$ORACLE_PY" -I -c "import math,sys; n=int(sys.argv[1]); print(max(n, round(n*math.log2(n))))" "$N")
SLICE=$((20 * N)); TAG=r${N}$CFG
echo "reproduce: n=$N CONFIG=$CFG |C|=|D|=$MC s=$S slice=$SLICE work=$WORK"

# ---- seeds (calibration; never printed) ----
SF=$WORK/SEEDS.private
"$ORACLE_PY" -I -c "import secrets; L=9223372036854775792; print(f'gen_seed={secrets.randbelow(L+1)}'); print(f'mix_seed={secrets.randbelow(L+1)}')" > "$SF"
chmod 600 "$SF"

# ---- reference side: sandboxed val_chain.sh on the same binaries ----
SB=$WORK/oracle
RD=$(HMAP_BASELINE=$HMAP_BASELINE ORACLE_PY=$ORACLE_PY bash "$REPO/tests/fixtures/stage4/oracle_sandbox.sh" "$SB") || exit 2
mkdir -p "$SB/target/release"
ln -s "$BIN_DIR/gen_sandwich_gadget" "$BIN_DIR/circuit_mixer" "$SB/target/release/"
ln -s "$HMAP_BASELINE" "$SB/target/release/hmap_affine"
VC=$RD/waist/val_chain.sh
cp "$REPO/scripts/windowed_delivery/waist/val_chain.sh" "$VC"
rewrite() { # FILE FROM TO COUNT
  local have; have=$(grep -oF -- "$2" "$1" | wc -l | tr -d ' ')
  [ "$have" = "$4" ] || { echo "val_chain rewrite: '$2' found $have times, expected $4" >&2; exit 2; }
  FROM="$2" TO="$3" perl -pi -e 's/\Q$ENV{FROM}\E/$ENV{TO}/g' "$1"
}
rewrite "$VC" "ROOT=/Users/rancanetti/code/local_mixing" "ROOT=$SB" 1
rewrite "$VC" "SEED=\$(python3 -c 'import secrets;print(secrets.randbelow(2**31))')" 'SEED=$VAL_GEN_SEED' 1
rewrite "$VC" "MSEED=\$(python3 -c 'import secrets;print(secrets.randbelow(2**40))')" 'MSEED=$VAL_MIX_SEED' 1
rewrite "$VC" 'FROZEN_DB_DIR=$HOME/code/local_mixing/frozen_db/frozen_m1_m11' "FROZEN_DB_DIR=$FROZEN_DB_DIR" 1
rewrite "$VC" 'python3 ' "$ORACLE_PY " 4
grep -qE 'ROOT=/Users/rancanetti/code/local_mixing|local_mixing/runs' "$VC" && { echo "val_chain sandbox: a path to the real run tree survived" >&2; exit 2; }
t0=$(date +%s)
( G=$(sed -n 's/^gen_seed=//p' "$SF"); M=$(sed -n 's/^mix_seed=//p' "$SF")
  env -i HOME="$HOME" PATH=/usr/bin:/bin VAL_GEN_SEED="$G" VAL_MIX_SEED="$M" \
    bash "$VC" "$TAG" "$N" "$MC" "$S" "$SLICE" "$CFG" > "$WORK/val_chain.out" 2>&1 )
t_ref=$(( $(date +%s) - t0 ))
REF=$RD/waist/$TAG

# ---- managed side: the driver with nothing but n, store and seeds ----
MT=$WORK/managed_target/$(rustc -vV 2>/dev/null | sed -n 's/^host: //p')/release
mkdir -p "$MT"
for b in gen_sandwich_gadget circuit_mixer tdp_packing; do ln -sf "$BIN_DIR/$b" "$MT/$b"; done
# the stage-2 span report's tool, as val_chain.sh runs it (report only)
ln -sf "$HMAP_BASELINE" "$MT/hmap_affine"
MAN=$WORK/managed; mkdir -p "$MAN"
size=""; case $N in 64|128) [ "$MC" = "$( [ "$N" = 64 ] && echo 1000 || echo 3000)" ] || size="gates = $MC" ;; *) size="gates = $MC" ;; esac
preset=""; [ "$CFG" = A ] && preset='preset = "A"'
cat > "$MAN/config.toml" <<CFG
config_version = 4
[run]
directory = "$MAN/run"
build_directory = "$WORK/managed_target"
build_binaries = false
skip_stages = [4, 5]
[source]
wires = $N
$size
[database]
regular_dir = "$FROZEN_DB_DIR"
[preprocessing]
$preset
[delivery]
anf_only = true
[calibration]
enabled = true
seed_file = "$SF"
CFG
# the driver finds a bash >= 5.1 itself (PATH, then the usual install
# locations); it needs rustc on PATH for its host query
CARGO_BIN=$(dirname "$(command -v cargo)"):$(dirname "$(command -v rustc)")
# decoys: the driver must refuse inherited output-affecting variables
for decoy in MIXER_DB_LEN_BAND=3 SLICE_GUARD_LEGACY=1 EMBEDDED_MASKING_RETIRE=4 CANON_MONOMIAL_CAP=10; do
  if env -i HOME="$HOME" PATH="$CARGO_BIN:/usr/bin:/bin" "$decoy" "$BIN_DIR/local_mixing_bin" tdp_gen \
       --config "$MAN/config.toml" > "$WORK/decoy.out" 2>&1; then
    echo "FAIL: the driver accepted inherited $decoy" >&2; exit 1
  fi
  grep -q "${decoy%%=*} is not honoured" "$WORK/decoy.out" || { cat "$WORK/decoy.out" >&2; exit 1; }
  [ ! -e "$MAN/run" ] || { echo "FAIL: a refused run created $MAN/run" >&2; exit 1; }
done
echo "decoys refused: MIXER_DB_LEN_BAND SLICE_GUARD_LEGACY EMBEDDED_MASKING_RETIRE CANON_MONOMIAL_CAP"
t0=$(date +%s)
env -i HOME="$HOME" PATH="$CARGO_BIN:/usr/bin:/bin" "$BIN_DIR/local_mixing_bin" tdp_gen \
  --config "$MAN/config.toml" > "$WORK/managed.out" 2>&1
mrc=$?
t_man=$(( $(date +%s) - t0 ))
RUN=$MAN/run; P=$RUN/packing

# ---- compare ----
fail=0; failure_path=0
row() { printf '  %-44s %s\n' "$1" "$2" | tee -a "$WORK/summary.txt"; }
same() { # LABEL A B
  if cmp -s "$2" "$3"; then row "$1" IDENTICAL; else row "$1" DIFFERENT; fail=1; fi
}
{ echo "reproduction n=$N CONFIG=$CFG |C|=|D|=$MC (seeds: calibration, private)"; } | tee "$WORK/summary.txt"
row "managed driver exit" "$mrc"
for f in tdp.mpmct1 tdp.mpmct1.source_c.g57 tdp.mpmct1.sandwich.mpmct1; do same "stage 2: $f" "$REF/$f" "$RUN/$f"; done
same "stage 3: s3.mpmct1 = db_mixing.mpmct1" "$REF/s3.mpmct1" "$RUN/db_mixing.mpmct1"
same "stage 2 report: mult_s2 (label normalized)" <(sed "s/^ $TAG:/ X:/" "$REF/mult_s2.txt") <(sed "s/^ tdp:/ X:/" "$RUN/stage2_mult.txt")
same "stage 2 report: span_s2.txt = stage2_span.txt" "$REF/span_s2.txt" "$RUN/stage2_span.txt"
same "stage 2 report: span_s2.tsv = stage2_span.tsv" "$REF/span_s2.tsv" "$RUN/stage2_span.tsv"
grep -q '^stage2_span_report=hmap_affine$' "$RUN/tdp_command.conf" && row "manifest: stage2_span_report" hmap_affine ||
  { row "manifest: stage2_span_report" "not recorded"; fail=1; }
grep -q '^frozen_filter=1$' "$RUN/stage3.settings" && row "stage3.settings: frozen_filter" "1 (= val_chain.sh)" ||
  { row "stage3.settings: frozen_filter" "not 1"; fail=1; }
grep -qE '^stop_reason=(moves budget spent|profile complete \(size schedule finished\))$' "$RUN/stage3.settings" &&
  row "stage3.settings: stop_reason" "$(sed -n 's/^stop_reason=//p' "$RUN/stage3.settings")" ||
  { row "stage3.settings: stop_reason" "not the schedule's end"; fail=1; }
# the validated store and sizes: the manifest names the chain (n=64/128 only)
case $N in 64|128)
  [ -z "$size" ] && want_pin=val_chain-$CFG-20261007 || want_pin=none ;;
  *) want_pin=none ;;
esac
grep -q "^pinned_recipe=$want_pin\$" "$RUN/tdp_command.conf" && row "manifest: pinned_recipe" "$want_pin" ||
  { row "manifest: pinned_recipe" "$(sed -n 's/^pinned_recipe=//p' "$RUN/tdp_command.conf") (want $want_pin)"; fail=1; }
for f in "$RUN/stage12.log" "$RUN/stage3.log" "$RUN/tdp_gen.log" "$RUN/tdp_command.conf" "$WORK/managed.out"; do
  if grep -qF -f <(sed -nE 's/^(gen|mix)_seed=//p' "$SF") "$f"; then row "seed values in $(basename "$f")" PRESENT; fail=1; fi
done
for f in packed_G512.mpmct1 packed_G512.mpmct1.bounds rep512.mpmct1 rep512.bounds \
         rep512.exposure.tsv rep512.allcuts.tsv rep512.wit.tsv; do
  same "stage 4: $f" "$REF/$f" "$P/$f"
done
norm_pack() { sed "s/$1-G512/X-G512/g" "$2"; }
same "stage 4: pack_G512.txt (label normalized)" <(norm_pack "$TAG" "$REF/pack_G512.txt") <(norm_pack tdp "$P/pack_G512.txt")
same "stage 4: anf_rep512.txt (label)" <(sed "s/^$TAG-rep512:/X:/" "$REF/anf_rep512.txt") <(sed "s/^tdp-rep512:/X:/" "$P/anf_rep512.txt")
same "stage 4: link_rep512.txt (label)" <(sed "s/^$TAG-rep512:/X:/" "$REF/link_rep512.txt") <(sed "s/^tdp-rep512:/X:/" "$P/link_rep512.txt")
if [ "$mrc" = 0 ]; then
  same "payload: deliverable.bin" "$REF/deliverable.bin" "$P/deliverable.bin"
  same "payload: deliverable.tdp4[48:]" "$REF/deliverable.bin" <(tail -c +49 "$P/deliverable.tdp4")
  "$ORACLE_PY" -I "$HERE/container_header.py" "$REF/deliverable.bin" "$WORK/want.tdp4" "$N" $((4 * N)) &&
    same "container: deliverable.tdp4 = container_header.py(oracle body)" "$WORK/want.tdp4" "$P/deliverable.tdp4"
  grep -q '^status=ok$' "$P/packing.report" && row "packing.report status" ok || { row "packing.report status" "not ok"; fail=1; }
  grep -q 'PIPELINE COMPLETE (stages 1-3,6)' "$RUN/tdp_gen.log" && row "driver log" "PIPELINE COMPLETE" || { row "driver log" "no PIPELINE COMPLETE"; fail=1; }
elif [ "$mrc" = 5 ] || [ "$mrc" = 6 ]; then
  # §10.7 step 5: the oracle shipped what the identity/budget test refuses.
  allow=$([ "$mrc" = 5 ] && echo --allow-floor-miss || echo --allow-over-budget)
  "$BIN_DIR/tdp_packing" check-final --rep "$REF/rep512.mpmct1" --bounds "$REF/rep512.bounds" \
      --source-c "$REF/tdp.mpmct1.source_c.g57" --n "$N" > "$WORK/check_final.out" 2>&1
  cf=$?
  [ "$cf" = "$mrc" ] && row "check-final on the ORACLE's rep512" "exit $cf (same violation)" || { row "check-final on the ORACLE's rep512" "exit $cf"; fail=1; }
  env -i PATH=/usr/bin:/bin "$BIN_DIR/tdp_packing" run --stage2 "$RUN/tdp.mpmct1" --stage3 "$RUN/db_mixing.mpmct1" \
      --source-c "$RUN/tdp.mpmct1.source_c.g57" --n "$N" --out-dir "$WORK/allowed" "$allow" > "$WORK/allowed.out" 2>&1
  same "payload under $allow: deliverable.bin" "$REF/deliverable.bin" "$WORK/allowed/deliverable.bin"
  failure_path=1
else
  row "managed driver" "FAILED (exit $mrc); see $WORK/managed.out"; fail=1
fi
row "stage 1-4 wall: val_chain.sh / managed (s)" "$t_ref / $t_man"
if [ $fail = 0 ]; then
  if [ $failure_path = 1 ]; then echo "RESULT: faithful (failure path)" | tee -a "$WORK/summary.txt"
  else echo "RESULT: ALL IDENTICAL" | tee -a "$WORK/summary.txt"; fi
else
  echo "RESULT: MISMATCH" | tee -a "$WORK/summary.txt"
fi
exit $fail
