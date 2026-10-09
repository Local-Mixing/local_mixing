#!/bin/bash
# cview_parity.sh OUT TAG... : P2 parity of the C-view exposure engine
# (docs/STAGE4_RUST_DESIGN.md §10.5 step 0 and the P2 row of §12).
#
# For each TAG of runs/ncdb_ab_20261001/waist (read-only), runs the PINNED
# baseline hmap_affine (built from 492564f0) and the refactored one (library
# engine) on the same inputs, each in its own directory under OUT/TAG/, and
# `cmp`s stdout and every output file:
#   rep     rep512 + rep512.bounds, ALLCUTS + WITNESS  (the repair.sh scan)
#   noall   rep512 + rep512.bounds, WITNESS only        (skip_settled path)
#   packed  packed_G512 + its pre-repair bounds, ALLCUTS + WITNESS
#   split   rep512 + archived split_cands_*.txt, ALLCUTS only (anf_split2)
#   span    SPAN_INTERSECT on rep512 + bounds           (first tag only)
#   heat    small affine heatmap, no exposure           (first tag only)
# Provenance (step 0): the baseline's rep/split outputs against the archived
# rep512.{exposure,allcuts,wit}.tsv, line 2 of rep512.cview.log and the
# archived split_cands_*.{exposure,allcuts}.tsv.
#   HMAP_BASELINE=... HMAP_NEW=... bash cview_parity.sh OUT vA64a vA128a
set -uo pipefail
OUT=$1; shift
W=${WAIST:-/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001/waist}
: "${HMAP_BASELINE:?baseline hmap_affine}" "${HMAP_NEW:?refactored hmap_affine}"
PIN=$(cd "$(dirname "$0")/../../fixtures/stage4/cview_n8" && cut -d' ' -f1 baseline_sha256.txt)
got=$(shasum -a 256 "$HMAP_BASELINE" | cut -d' ' -f1)
echo "$PIN" | grep -qx "$got" || { echo "refusing: baseline sha256 $got is not pinned" >&2; exit 1; }
mkdir -p "$OUT"
SUM=$OUT/summary.txt
fail=0
note() { echo "$*" | tee -a "$SUM"; }

# run SIDE DIR ENV... -- ARGS...
run() {
  local bin=$1 dir=$2; shift 2
  local envs=(PATH=/usr/bin:/bin)
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  mkdir -p "$dir"
  ( cd "$dir" && env -i "${envs[@]}" /usr/bin/time -l "$bin" "$@" > stdout.txt 2> time.txt )
  local st=$?
  local real rss
  real=$(grep -oE "[0-9.]+ real" "$dir/time.txt" | cut -d' ' -f1)
  rss=$(grep -oE "[0-9]+ +maximum resident" "$dir/time.txt" | grep -oE "^[0-9]+")
  echo "$st ${real}s $((rss / 1048576))MB"
}

variant() {  # variant TAG NAME ENV... -- ARGS...
  local tag=$1 name=$2; shift 2
  local d=$OUT/$tag/$name
  rm -rf "$d"
  local b n
  b=$(run "$HMAP_BASELINE" "$d/base" "$@")
  n=$(run "$HMAP_NEW" "$d/new" "$@")
  local verdict=IDENTICAL files=0
  for f in $(cd "$d/base" && ls | grep -v '^time.txt$'); do
    files=$((files + 1))
    cmp -s "$d/base/$f" "$d/new/$f" || { verdict="DIFF($f)"; }
  done
  for f in $(cd "$d/new" && ls | grep -v '^time.txt$'); do
    [ -e "$d/base/$f" ] || verdict="EXTRA($f)"
  done
  [ "$verdict" = IDENTICAL ] || fail=1
  note "$tag $name: $verdict ($files files) base[exit time rss]=[$b] new=[$n]"
}

prov() {  # prov TAG FILE_IN_RUN ARCHIVED
  if cmp -s "$2" "$3"; then note "$1 provenance $(basename "$3"): baseline == archive";
  else note "$1 provenance $(basename "$3"): baseline != archive"; fi
}

first=1
for tag in "$@"; do
  R=$W/$tag
  case $tag in *128*) N=128 ;; *) N=64 ;; esac
  C=(--c "$R/tdp.mpmct1.source_c.g57" --c-format g57 --n "$N" --exposure --c-step 1)
  variant "$tag" rep EXPOSURE_WITNESS=rep512.wit.tsv EXPOSURE_ALLCUTS=rep512.allcuts.tsv -- \
    "${C[@]}" --g "$R/rep512.mpmct1" --g-cuts "$R/rep512.bounds" --out rep512
  B=$OUT/$tag/rep/base
  for f in exposure.tsv allcuts.tsv wit.tsv; do prov "$tag" "$B/rep512.$f" "$R/rep512.$f"; done
  if [ "$(sed -n 2p "$B/stdout.txt")" = "$(sed -n 2p "$R/rep512.cview.log")" ]; then
    note "$tag provenance cview.log line 2: baseline == archive"
  else note "$tag provenance cview.log line 2: baseline != archive"; fi
  variant "$tag" noall EXPOSURE_WITNESS=rep512.wit.tsv -- \
    "${C[@]}" --g "$R/rep512.mpmct1" --g-cuts "$R/rep512.bounds" --out rep512
  variant "$tag" packed EXPOSURE_WITNESS=p.wit.tsv EXPOSURE_ALLCUTS=p.allcuts.tsv -- \
    "${C[@]}" --g "$R/packed_G512.mpmct1" --g-cuts "$R/packed_G512.mpmct1.bounds" --out p
  for sc in "$R"/split_cands_*.txt; do
    [ -e "$sc" ] || continue
    s=$(basename "$sc" .txt)
    variant "$tag" "split_$s" EXPOSURE_ALLCUTS="$s.allcuts.tsv" -- \
      "${C[@]}" --g "$R/rep512.mpmct1" --g-cuts "$sc" --out "$s"
    prov "$tag" "$OUT/$tag/split_$s/base/$s.allcuts.tsv" "$R/$s.allcuts.tsv"
    prov "$tag" "$OUT/$tag/split_$s/base/$s.exposure.tsv" "$R/$s.exposure.tsv"
  done
  if [ $first = 1 ]; then
    variant "$tag" span SPAN_INTERSECT=span.tsv -- \
      "${C[@]}" --g "$R/rep512.mpmct1" --g-cuts "$R/rep512.bounds" --out span
    variant "$tag" heat -- --c "$R/tdp.mpmct1.source_c.g57" --c-format g57 --n "$N" \
      --g "$R/rep512.mpmct1" --c-step 100 --g-step 200000 --out heat
    first=0
  fi
done
[ $fail = 0 ] && note "ALL VARIANTS IDENTICAL" || note "SOME VARIANTS DIFFER"
exit $fail
