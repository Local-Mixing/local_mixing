#!/bin/bash
# repair.sh TAG G N : commutation-aware repair of TAG/packed_G<G> (from its
# UNREPAIRED bounds), then merge/split fallback, then scoring. Outputs in
# TAG/rep<G>.* ; log on stdout. Never edit while running.
set -uo pipefail
ROOT=/Users/rancanetti/code/local_mixing
W=$ROOT/runs/ncdb_ab_20261001/waist
TAG=$1; G=$2; N=$3
R=$W/$TAG
P0=$R/packed_G$G.mpmct1
P=$R/rep$G.mpmct1
B=$R/rep$G.bounds
cd $ROOT
python3 $W/order_repair.py $P0 $P0.bounds $P $B cancel
python3 $W/verify_equiv.py $P0 $P &
VER=$!
cview() {
  EXPOSURE_WITNESS=$R/rep$G.wit.tsv EXPOSURE_ALLCUTS=$R/rep$G.allcuts.tsv target/release/hmap_affine \
    --c $R/tdp.mpmct1.source_c.g57 --c-format g57 --g $P --n $N --exposure --c-step 1 --g-cuts $B --out $R/rep$G > $R/rep$G.cview.log 2>&1
  sed -n 2p $R/rep$G.cview.log | grep -oE "[0-9]+ visible" | grep -oE "[0-9]+"
}
for it in 1 2 3 4; do
  vis=$(cview); echo "fix iter $it: C-view visible $vis (floor $N)"
  [ "$vis" -le "$N" ] && break
  python3 $W/order_repair.py $P $B $P $B fix $R/rep$G.wit.tsv $R/rep$G.exposure.tsv | tail -1
done
python3 $W/verify_equiv.py $P0 $P
for it in 1 2 3 4; do
  vis=$(cview); echo "fallback iter $it: C-view visible $vis"
  if [ "$vis" -gt "$N" ]; then python3 $W/merge_bounds2.py $R/rep$G.exposure.tsv $R/rep$G.allcuts.tsv $B $B; fi
  out=$(SPLIT_P=$P SPLIT_EXP=$R/rep$G.exposure.tsv python3 $W/anf_split2.py $TAG $G $N 65536 $B $B); echo "  $out"
  if [ "$vis" -le "$N" ] && echo "$out" | grep -q ": 0 splits"; then break; fi
done
wait $VER
cd $ROOT/runs/ncdb_ab_20261001
python3 anf_compose.py $P $B 65536 $TAG-rep$G 500 > $R/anf_rep$G.txt 2>&1
python3 $W/link_census.py $R/tdp.mpmct1 $P $B $TAG-rep$G > $R/link_rep$G.txt 2>&1
echo "== $TAG rep$G final"; head -1 $R/anf_rep$G.txt; grep -oE "size: ANF [0-9.]+ MB" $R/anf_rep$G.txt; grep spot $R/anf_rep$G.txt
sed -n 2,3p $R/link_rep$G.txt; sed -n 2p $R/rep$G.cview.log
