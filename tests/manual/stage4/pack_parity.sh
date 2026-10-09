#!/bin/bash
# pack_parity.sh OUT TAG... : the Rust packer against the archived oracle
# packs (docs/STAGE4_RUST_DESIGN.md §10.5 step 1, P3 acceptance).
#
# For each TAG (read-only in $WAIST/TAG): `tdp_packing pack tdp.mpmct1
# s3.mpmct1 512 512 24576 3000 TAG-G512 OUT/TAG/packed_G512.mpmct1`, then
#   * sha256 of packed_G512.mpmct1 and .bounds == Appendix B;
#   * cmp against the archived files;
#   * stdout == the archived pack_G512.txt (the oracle's log), byte for byte;
#   * time and peak RSS (/usr/bin/time -l).
# Optional: THREADS1=1 also reruns with --threads 1 and compares the outputs.
#   TDP_PACKING=.../release/tdp_packing bash pack_parity.sh port_runs/p3_parity vA64a ...
set -uo pipefail
W=${WAIST:-/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001/waist}
: "${TDP_PACKING:?path to tdp_packing}"
OUT=${1:?usage: pack_parity.sh OUT TAG...}; shift
mkdir -p "$OUT"
appendix_b() { # TAG -> "packed_sha bounds_sha"
  case $1 in
    vA64a) echo bde5e0fc6f170b725be484433e50e3a1d1a245fdaec17da6e5918c36b23419cc 18138db7c3a4c1287ef98ac73ba56e126369174bec93a6740591d83514a9247e ;;
    vA64b) echo 699d3c6eed512ba1af47bee929022d0cfdf3ffbe782f66d8cfed7f0be2fceb61 2a8624a5b8084ceef8d7ab706bf3b799bae2399d20d19bb04c6b49adacc1e98c ;;
    vL64a) echo f2c6b53af825f84b6698bf95c01f653d087f7f1e30bf678279a194e3a720f69e a5cbe27daee2e9b638ee30ecd64d309f65740b89ec50aed53d5c749411658076 ;;
    vL64b) echo fe4d3d7db336a1ff33bc8b91ab9453cbced29e42dc7f8204ad4f3bebe3b30ab0 01f3ce503e98c7352890118c9b911d5d2aacdba88ef85317a3dbb82fc17dec23 ;;
    vA128a) echo 8ed20637c33b3dfe709af129976a8f5e10b44ae1d44fb69dd62d8073f2d051a1 09654ee2e91efca4c77a37cff34d81f8eb0c9b9dabe41356044338e92019796c ;;
    vA128b) echo deac6d86150738c304ee7a92dd9af22f282def29f24680487e6aec8bd7c3a28f 7ccc526dc4563a3d9781ad9f48265d6974d28f7e45a563f154be1ccbf633c19a ;;
    vL128a) echo 932fc7b45794ee04519a07c82fe9d70d10ef0a8cc4e90867f560ad0116aa528b bf19c8a1434ea4b99b2c642b72f2e9436341f137f7dde44fb293c3c90f208832 ;;
    vL128b) echo f9d22195aeeec4b629cfdc196e69155dffafac92824502770e328fd961ee59d9 8a19eb139ff6c18522917a1810306187e7d595efc9ccf8a2ed14b1c595d639f7 ;;
    *) echo "- -" ;;
  esac
}
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
fail=0
for TAG in "$@"; do
  R=$W/$TAG; O=$OUT/$TAG; mkdir -p "$O"
  /usr/bin/time -l "$TDP_PACKING" pack "$R/tdp.mpmct1" "$R/s3.mpmct1" 512 512 24576 3000 \
      "$TAG-G512" "$O/packed_G512.mpmct1" --stats > "$O/pack_G512.txt" 2> "$O/pack.err"
  rc=$?
  read -r want_p want_b <<< "$(appendix_b "$TAG")"
  got_p=$(sha "$O/packed_G512.mpmct1" 2>/dev/null); got_b=$(sha "$O/packed_G512.mpmct1.bounds" 2>/dev/null)
  st=ok
  [ $rc = 0 ] || st="exit $rc"
  [ "$got_p" = "$want_p" ] || st="$st; packed sha256 != Appendix B"
  [ "$got_b" = "$want_b" ] || st="$st; bounds sha256 != Appendix B"
  cmp -s "$O/packed_G512.mpmct1" "$R/packed_G512.mpmct1" || st="$st; packed != archive"
  cmp -s "$O/packed_G512.mpmct1.bounds" "$R/packed_G512.mpmct1.bounds" || st="$st; bounds != archive"
  cmp -s "$O/pack_G512.txt" "$R/pack_G512.txt" || st="$st; log != pack_G512.txt"
  if [ -n "${THREADS1:-}" ]; then
    "$TDP_PACKING" pack "$R/tdp.mpmct1" "$R/s3.mpmct1" 512 512 24576 3000 "$TAG-G512" \
        "$O/t1.mpmct1" --threads 1 > "$O/t1.txt" 2>/dev/null
    { cmp -s "$O/t1.mpmct1" "$O/packed_G512.mpmct1" && cmp -s "$O/t1.mpmct1.bounds" \
        "$O/packed_G512.mpmct1.bounds" && cmp -s "$O/t1.txt" "$O/pack_G512.txt"; } \
      || st="$st; --threads 1 differs"
    rm -f "$O/t1.mpmct1" "$O/t1.mpmct1.bounds"
  fi
  real=$(awk '/ real /{print $1}' "$O/pack.err")
  rss=$(awk '/maximum resident set size/{printf "%.0f", $1/1048576}' "$O/pack.err")
  stats=$(grep '^stats:' "$O/pack.err" | sed 's/; read .*//')
  [ "$st" = ok ] || fail=1
  echo "$TAG: $st; ${real}s, ${rss} MB; $stats" | tee -a "$OUT/summary.txt"
  rm -f "$O/packed_G512.mpmct1"   # 30-100 MB; the sha256 and cmp are recorded above
done
[ $fail = 0 ] && echo "ALL PACKS IDENTICAL" | tee -a "$OUT/summary.txt"
exit $fail
