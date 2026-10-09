#!/bin/bash
# run_parity.sh OUT TAG... : `tdp_packing run` (the production stage 4) against
# the archived oracle chain (docs/STAGE4_RUST_DESIGN.md §10.5, P6 acceptance).
#
# For each TAG (read-only in $WAIST/TAG): one `tdp_packing run` on the archived
# tdp.mpmct1 (G2), s3.mpmct1 (G3) and source C, then cmp against the archive:
#   packed_G512.mpmct1(.bounds), pack_G512.txt, rep512.mpmct1, rep512.bounds,
#   rep512.{exposure,allcuts,wit}.tsv, anf_rep512.txt, link_rep512.txt,
#   deliverable.bin (= the payload); deliverable.tdp4 against the container
#   container_header.py builds from the ARCHIVED body; rep512.cview.log and
#   repair512.log after the directory prefix is normalized (the two
#   verify_equiv lines compared as a set: the oracle runs the first one in the
#   background); time and peak RSS (/usr/bin/time -l).
#   TDP_PACKING=.../release/tdp_packing bash run_parity.sh port_runs/p6_parity vA64a ...
set -uo pipefail
W=${WAIST:-/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001/waist}
: "${TDP_PACKING:?path to tdp_packing}"
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?usage: run_parity.sh OUT TAG...}; shift
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
norm() { sed "s|$1/|<DIR>/|g" "$2"; }
fail=0
for TAG in "$@"; do
  R=$W/$TAG; O=$OUT/$TAG; rm -rf "$O"
  case $TAG in *64*) N=64 ;; *) N=128 ;; esac
  /usr/bin/time -l "$TDP_PACKING" run --stage2 "$R/tdp.mpmct1" --stage3 "$R/s3.mpmct1" \
      --source-c "$R/tdp.mpmct1.source_c.g57" --n $N --out-dir "$O" --label "$TAG" \
      > "$OUT/$TAG.stdout" 2> "$OUT/$TAG.err"
  rc=$?
  st=ok
  [ $rc = 0 ] || st="exit $rc"
  for f in packed_G512.mpmct1 packed_G512.mpmct1.bounds pack_G512.txt rep512.mpmct1 rep512.bounds \
           rep512.exposure.tsv rep512.allcuts.tsv rep512.wit.tsv anf_rep512.txt link_rep512.txt deliverable.bin; do
    cmp -s "$O/$f" "$R/$f" || st="$st; $f != archive"
  done
  python3 -I "$HERE/container_header.py" "$R/deliverable.bin" "$OUT/$TAG.want.tdp4" $N $((4 * N)) \
    && cmp -s "$OUT/$TAG.want.tdp4" "$O/deliverable.tdp4" || st="$st; deliverable.tdp4 != container_header.py(archived body)"
  rm -f "$OUT/$TAG.want.tdp4"
  cmp -s <(norm "$O" "$O/rep512.cview.log") <(norm "$R" "$R/rep512.cview.log") || st="$st; cview.log differs"
  norm "$R" "$R/repair512.log" > "$O/want.log"; norm "$O" "$O/repair512.log" > "$O/got.log"
  cmp -s <(grep -v 'EQUIVALENT (1024 random states)' "$O/want.log") \
         <(grep -v 'EQUIVALENT (1024 random states)' "$O/got.log") || st="$st; repair log differs"
  cmp -s <(grep 'EQUIVALENT (1024 random states)' "$O/want.log") \
         <(grep 'EQUIVALENT (1024 random states)' "$O/got.log") || st="$st; verify lines differ"
  grep -q '^status=ok$' "$O/packing.report" || st="$st; packing.report status != ok"
  real=$(awk '/ real /{print $1}' "$OUT/$TAG.err")
  rss=$(awk '/maximum resident set size/{printf "%.0f", $1/1048576}' "$OUT/$TAG.err")
  [ "$st" = ok ] || fail=1
  echo "$TAG: $st; ${real}s, ${rss} MB; $(grep '^run:' "$OUT/$TAG.err")" | tee -a "$OUT/summary.txt"
  grep -E '^time\.' "$O/packing.report" | tr '\n' ' ' | sed "s/^/  $TAG /" >> "$OUT/summary.txt"; echo >> "$OUT/summary.txt"
  rm -f "$O/rep512.mpmct1" "$O/packed_G512.mpmct1" "$O/deliverable.bin" "$O/deliverable.tdp4"   # large; cmp recorded above
done
[ $fail = 0 ] && echo "ALL RUNS IDENTICAL" | tee -a "$OUT/summary.txt"
exit $fail
