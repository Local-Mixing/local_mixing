#!/bin/bash
# repair_parity.sh OUT TAG... : the Rust repair against the archived oracle
# repairs (docs/STAGE4_RUST_DESIGN.md §10.5 step 2, P4 acceptance).
#
# For each TAG (read-only in $WAIST/TAG): `tdp_packing repair` on the archived
# packed_G512.mpmct1 and its UNREPAIRED bounds (as repair.sh starts), then
#   * sha256 of rep512.mpmct1 and rep512.bounds == Appendix B, and cmp against
#     the archive;
#   * rep512.{exposure,allcuts,wit}.tsv cmp-equal to the archive (the §10.5
#     step-0 targets: the pinned baseline hmap_affine reproduces them);
#   * rep512.cview.log equal after the directory prefix is normalized;
#   * the repair log equal to the archived repair512.log up to the report
#     part ("== TAG rep512 final"), after normalizing the directory and
#     setting aside the two verify_equiv lines (the oracle runs the first one
#     in the background, so its position in the log is a race); both must be
#     present and equal;
#   * time and peak RSS (/usr/bin/time -l).
#   TDP_PACKING=.../release/tdp_packing bash repair_parity.sh port_runs/p4_parity vA64a ...
set -uo pipefail
W=${WAIST:-/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001/waist}
: "${TDP_PACKING:?path to tdp_packing}"
OUT=${1:?usage: repair_parity.sh OUT TAG...}; shift
mkdir -p "$OUT"
appendix_b() { # TAG -> "rep_sha bounds_sha"
  case $1 in
    vA64a) echo fcec8a2cc34f0d3250b24104e5b1d54f1cfb9ddc2047c10f7b6505564ab25b1a 489606afcb18fc989ed16a9e588229c6ec3b28f93106bfef20e403a2f190c800 ;;
    vA64b) echo dbe5220b1273fe8e6b9968e6d696f2b9a026f719493af5d47325d4b9bd2783f4 21e3ca2d9738fd361bf41ac80e094dad82949b614707e2d42d8fb3cdcfbde9c5 ;;
    vL64a) echo 274570154da5ec5152e8f520d3c552b9fdbee4ebe023c1915e30e82e1441210c 4e9ccd63faa03dc18021debe54d43a3cce510490cd80289cbcffcd90d225b90d ;;
    vL64b) echo fd9e4bb5d86cc6a22ccb6f156ce1a78dd902461ea1ec9503d25f388362f2e321 47acd95851c85cd9a58331903e1ac156301960cdbaa1c07bb77c9f8ede537dc8 ;;
    vA128a) echo d335e9868f25ebcc6ee738be4e791f433e202b9436c1446a975b4666f8f2aba5 55819935cedc23705a55223484bad33cae81b9caa5830abb62d2897981abafc6 ;;
    vA128b) echo 4ae19d27b7b66a23d4db7a5646338a77f46332b8cd6851d7f1b2cf99d58b13d0 a6384f2a4fccd8f56c6b68cb2274476403cfd9a8bd6d07b4dae120c1e6546003 ;;
    vL128a) echo e987ce0407b5160ff0359f08d4480ead39c63625b2c97b1235d6728afdb8131f 9ed0c259af91a4cd6c1a2bf882c86804409a6cc9b5a13d3cfc9c75a649895f99 ;;
    vL128b) echo 6ed654388b6e52951eb0165c536c5a119e137eb08746daf4b26cb51b4abf2b1a 3150c3a4d8fc3abbc6c4c86bc358793a4c15fda95c1c92acfe3ef1284697dfa8 ;;
    *) echo "- -" ;;
  esac
}
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
# norm DIR FILE : directory prefix -> <DIR>
norm() { sed "s|$1/|<DIR>/|g" "$2"; }
fail=0
for TAG in "$@"; do
  R=$W/$TAG; O=$OUT/$TAG; rm -rf "$O"; mkdir -p "$O"
  case $TAG in *64*) N=64 ;; *) N=128 ;; esac
  /usr/bin/time -l "$TDP_PACKING" repair --packed "$R/packed_G512.mpmct1" \
      --bounds "$R/packed_G512.mpmct1.bounds" --source-c "$R/tdp.mpmct1.source_c.g57" \
      --n $N --out-dir "$O" --stats > "$O/stdout.txt" 2> "$O/repair.err"
  rc=$?
  read -r want_r want_b <<< "$(appendix_b "$TAG")"
  st=ok
  [ $rc = 0 ] || st="exit $rc"
  [ "$(sha "$O/rep512.mpmct1" 2>/dev/null)" = "$want_r" ] || st="$st; rep sha256 != Appendix B"
  [ "$(sha "$O/rep512.bounds" 2>/dev/null)" = "$want_b" ] || st="$st; bounds sha256 != Appendix B"
  for f in rep512.mpmct1 rep512.bounds rep512.exposure.tsv rep512.allcuts.tsv rep512.wit.tsv; do
    cmp -s "$O/$f" "$R/$f" || st="$st; $f != archive"
  done
  cmp -s <(norm "$O" "$O/rep512.cview.log") <(norm "$R" "$R/rep512.cview.log") \
    || st="$st; cview.log differs"
  # repair log: up to the report part, verify lines set aside
  sed -n "/^== $TAG rep512 final/q;p" "$R/repair512.log" | sed "s|$R/|<DIR>/|g" > "$O/want.log"
  norm "$O" "$O/repair512.log" > "$O/got.log"
  cmp -s <(grep -v 'EQUIVALENT (1024 random states)' "$O/want.log") \
         <(grep -v 'EQUIVALENT (1024 random states)' "$O/got.log") || st="$st; repair log differs"
  cmp -s <(grep 'EQUIVALENT (1024 random states)' "$O/want.log") \
         <(grep 'EQUIVALENT (1024 random states)' "$O/got.log") || st="$st; verify lines differ"
  cmp -s "$O/repair512.log" "$O/stdout.txt" || st="$st; stdout != repair512.log"
  real=$(awk '/ real /{print $1}' "$O/repair.err")
  rss=$(awk '/maximum resident set size/{printf "%.0f", $1/1048576}' "$O/repair.err")
  summ=$(grep '^repair:' "$O/repair.err")
  [ "$st" = ok ] || fail=1
  echo "$TAG: $st; ${real}s, ${rss} MB; $summ" | tee -a "$OUT/summary.txt"
  grep '^stats: repair\.' "$O/repair.err" | sed "s/^/  $TAG /" >> "$OUT/summary.txt"
  rm -f "$O/rep512.mpmct1"   # 30-100 MB; the sha256 and cmp are recorded above
done
[ $fail = 0 ] && echo "ALL REPAIRS IDENTICAL" | tee -a "$OUT/summary.txt"
exit $fail
