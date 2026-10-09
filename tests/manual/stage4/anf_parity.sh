#!/bin/bash
# anf_parity.sh OUT TAG... : parity of the Rust ANF emission and size report
# with the Python oracle on the archived validation runs
# (docs/STAGE4_RUST_DESIGN.md §10.5 steps 3-5, ANF part).
#
# For each TAG of runs/ncdb_ab_20261001/waist (read-only), into OUT/TAG/:
#   serialize  tdp_packing serialize rep512.mpmct1 rep512.bounds (+ container,
#              exact window checks, DE's eval check, zero slice against C);
#              sha256 of the raw body must equal Appendix B's deliverable.bin
#              AND the archived file; container[48:] must equal the body.
#   anf        tdp_packing anf-report rep512 65536 TAG-rep512 500: stdout must
#              equal the archived anf_rep512.txt (all four lines).
#   decode     tdp_packing decode of the ARCHIVED deliverable.bin (validator).
# Wall time and peak RSS come from /usr/bin/time -l. No seed or hash of a
# private input (C, G2, G3) is printed.
#   TDP_PACKING=.../release/tdp_packing bash anf_parity.sh OUT vA64a vL128b ...
set -uo pipefail
OUT=$1; shift
W=${WAIST:-/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001/waist}
: "${TDP_PACKING:?path to tdp_packing}"
mkdir -p "$OUT"
SUM=$OUT/summary.txt
: > "$SUM"
fail=0
note() { echo "$*" | tee -a "$SUM"; }

# Appendix B: deliverable.bin (raw body) sha256 per tag.
appendix_b() {
  case $1 in
    vA64a) echo 430c6e3b249629f14508467ae33175ced9918efd7fc85886a1a3d2693c4ef93a ;;
    vA64b) echo 9b3c0a593ad36dab7bf111f2c0a510c5a28878fc831e2c249fe96e3bbf4f8a8a ;;
    vL64a) echo e32e0cf271373d6bd8046a527cbdce6ad1404d60dce75a6234b9d94adfd6cc7b ;;
    vL64b) echo a9e5ebc4df0e710f19eebac7b44fcd55185b0ef93dc7c67edebcd67acbacdf25 ;;
    vA128a) echo 7b1ad3801169ff65bb27fd308d035f07ce1132d18308ae6884e362b807ebf26d ;;
    vA128b) echo 73fb706e9d5558be291aa75d4d2307fef6ddb7c9545eaf7dd75cb6eab109ae9b ;;
    vL128a) echo aa2d8845db0be1c367c2e345334f1b9a2792695b726c73950990076ae437b11e ;;
    vL128b) echo 710baf59423c7f2a5ab2cb84d24d432f206c334c6ec39b7e4bebda2edc6bf68c ;;
    *) echo none ;;
  esac
}

timed() { # DIR CMD... : run with /usr/bin/time -l, stdout to DIR/stdout.txt
  local dir=$1; shift
  mkdir -p "$dir"
  env -i PATH=/usr/bin:/bin /usr/bin/time -l "$@" > "$dir/stdout.txt" 2> "$dir/time.txt"
  local st=$?
  local real rss
  real=$(grep -oE "[0-9.]+ real" "$dir/time.txt" | cut -d' ' -f1)
  rss=$(grep -E "maximum resident set size" "$dir/time.txt" | awk '{printf "%.0f", $1/1048576}')
  echo "exit=$st ${real}s ${rss}MB"
}

for TAG in "$@"; do
  R=$W/$TAG
  D=$OUT/$TAG
  mkdir -p "$D"
  N=$(( $(head -1 "$R/rep512.mpmct1" | awk '{print $2}') / 4 ))
  t=$(timed "$D/ser" "$TDP_PACKING" serialize "$R/rep512.mpmct1" "$R/rep512.bounds" \
        "$D/deliverable.bin" "$TAG" --container "$D/deliverable.tdp4" --source-wires "$N" \
        --zero-slice-c "$R/tdp.mpmct1.source_c.g57")
  got=$(shasum -a 256 "$D/deliverable.bin" 2>/dev/null | cut -d' ' -f1)
  arch=$(shasum -a 256 "$R/deliverable.bin" | cut -d' ' -f1)
  want=$(appendix_b "$TAG")
  if [ "$got" = "$want" ] && [ "$got" = "$arch" ] \
     && cmp -s <(tail -c +49 "$D/deliverable.tdp4") "$R/deliverable.bin"; then
    note "$TAG serialize: body == Appendix B == archive; container[48:] == body ($t)"
  else
    note "$TAG serialize: MISMATCH (got ${got:0:12} want ${want:0:12} archive ${arch:0:12}) ($t)"
    fail=1
  fi
  sed -n 2p "$D/ser/stdout.txt" | sed "s/^/  /" | tee -a "$SUM"
  t=$(timed "$D/anf" "$TDP_PACKING" anf-report "$R/rep512.mpmct1" "$R/rep512.bounds" 65536 "$TAG-rep512" 500)
  if cmp -s "$D/anf/stdout.txt" "$R/anf_rep512.txt"; then
    note "$TAG anf-report: identical to anf_rep512.txt ($t)"
  else
    note "$TAG anf-report: DIFFERS from anf_rep512.txt ($t)"
    diff "$D/anf/stdout.txt" "$R/anf_rep512.txt" | sed 's/^/  /' | tee -a "$SUM"
    fail=1
  fi
  t=$(timed "$D/dec" "$TDP_PACKING" decode "$R/deliverable.bin" --format raw --n "$N" --stats)
  note "$TAG decode(archived body): $(head -1 "$D/dec/stdout.txt" | cut -d' ' -f2-3) ($t)"
done
if [ $fail = 0 ]; then note "ALL TAGS IDENTICAL"; else note "SOME TAGS DIFFER"; exit 1; fi
