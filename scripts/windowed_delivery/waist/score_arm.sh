#!/bin/bash
# Linkable-exposure scoring of a finished arm at G=128 (chain packing) and
# repacked G=256 / G=512: ANF size + link census for each. New file; never
# edit while running (bash reads scripts by offset).
set -euo pipefail
cd /Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001
T=waist/$1
python3 waist/link_census.py $T/tdp.mpmct1 $T/packed.mpmct1 $T/packed.mpmct1.bounds $1-G128 > $T/link.txt 2>&1 &
for G in 256 512; do
  ( python3 window_pack3.py $T/tdp.mpmct1 $T/s3.mpmct1 $G $G $((48*G)) 3000 $1-G$G $T/packed_G$G.mpmct1 > $T/pack_G$G.txt 2>&1 \
    && python3 anf_compose.py $T/packed_G$G.mpmct1 $T/packed_G$G.mpmct1.bounds 65536 $1-G$G 500 > $T/anf_G$G.txt 2>&1 \
    && python3 waist/link_census.py $T/tdp.mpmct1 $T/packed_G$G.mpmct1 $T/packed_G$G.mpmct1.bounds $1-G$G > $T/link_G$G.txt 2>&1 ) &
done
wait
echo "$1 SCORED"
