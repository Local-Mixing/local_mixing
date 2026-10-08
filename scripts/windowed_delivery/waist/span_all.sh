#!/bin/bash
# span_all.sh TAG N STEP : span-intersection (excess beyond inputs) of C values
# vs stage-2 wire values at every STEP-th stage-2 cut.
cd /Users/rancanetti/code/local_mixing
T=runs/ncdb_ab_20261001/waist/$1
SPAN_INTERSECT=$T/span_s2.tsv target/release/hmap_affine --c $T/tdp.mpmct1.source_c.g57 --c-format g57 --g $T/tdp.mpmct1 --n $2 --exposure --c-step 1 --g-step $3 --out $T/span_s2 2>&1 | grep "\[span\]" > $T/span_s2.txt
echo "$1 $(cat $T/span_s2.txt)"
