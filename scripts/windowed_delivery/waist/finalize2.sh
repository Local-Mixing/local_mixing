#!/bin/bash
# finalize.sh TAG G N : iterate C-view merge-repair + ANF overflow split on
# packed_G<G> until the C-view is at the output floor and no window
# overflows; then link census + ANF size. Writes TAG/final_G<G>.bounds.
set -euo pipefail
cd /Users/rancanetti/code/local_mixing
TAG=$1; G=$2; N=$3
R=runs/ncdb_ab_20261001/waist/$TAG
P=/Users/rancanetti/code/local_mixing/$R/packed_G$G.mpmct1
cp $P.bounds $R/final_G$G.bounds
for it in 1 2 3 4; do
  EXPOSURE_ALLCUTS=$R/fin.allcuts.tsv target/release/hmap_affine --c $R/tdp.mpmct1.source_c.g57 --c-format g57 --g $P --n $N --exposure --c-step 1 --g-cuts $R/final_G$G.bounds --out $R/fin > $R/fin.log 2>&1
  vis=$(sed -n 2p $R/fin.log | grep -oE "[0-9]+ visible" | grep -oE "[0-9]+")
  echo "iter $it: C-view visible $vis (floor $N)"
  if [ "$vis" -gt "$N" ]; then
    python3 runs/ncdb_ab_20261001/waist/merge_bounds2.py $R/fin.exposure.tsv $R/fin.allcuts.tsv $R/final_G$G.bounds $R/final_G$G.bounds
  fi
  out=$(python3 runs/ncdb_ab_20261001/waist/anf_split.py $P $R/final_G$G.bounds 65536 $R/final_G$G.bounds)
  echo "  $out"
  if [ "$vis" -le "$N" ] && echo "$out" | grep -q ": 0 splits"; then break; fi
done
cd runs/ncdb_ab_20261001
python3 anf_compose.py $P waist/$TAG/final_G$G.bounds 65536 $TAG-final$G 500 > waist/$TAG/anf_final$G.txt 2>&1
python3 waist/link_census.py waist/$TAG/tdp.mpmct1 $P waist/$TAG/final_G$G.bounds $TAG-final$G > waist/$TAG/link_final$G.txt 2>&1
echo "== $TAG final G$G"; head -1 waist/$TAG/anf_final$G.txt; grep -oE "size: ANF [0-9.]+ MB" waist/$TAG/anf_final$G.txt; grep spot waist/$TAG/anf_final$G.txt; sed -n 2,3p waist/$TAG/link_final$G.txt
