#!/bin/bash
# Post-mixer steps of run_ret.sh (pack -> ANF -> censuses -> exposure scans),
# for resuming chains whose mixer already wrote s3.mpmct1. Never edit a bash
# script while instances are running: bash reads it incrementally by offset.
set -euo pipefail
cd /Users/rancanetti/code/local_mixing
TAG=$1; N=$2
RD=runs/ncdb_ab_20261001
R=$RD/waist/$TAG
G=$(head -1 "$R/tdp.mpmct1" | awk '{print $3}')
cd "$RD"
T=waist/$TAG
python3 window_pack3.py "$T/tdp.mpmct1" "$T/s3.mpmct1" 128 128 6144 3000 "$TAG" "$T/packed.mpmct1" > "$T/pack.txt" 2>&1
python3 anf_compose.py "$T/packed.mpmct1" "$T/packed.mpmct1.bounds" 16384 "$TAG" 500 > "$T/anf.txt" 2>&1
python3 waist/recurrence2.py "$T/tdp.mpmct1" "$T/packed.mpmct1" "$T/packed.mpmct1.bounds" "$TAG" > "$T/recur.txt" 2>&1
python3 waist/mult_census.py "$T/tdp.mpmct1" "$TAG" > "$T/mult.txt" 2>&1
python3 boundary_cluster.py "$T/tdp.mpmct1" "$T/packed.mpmct1" "$T/packed.mpmct1.bounds" "$TAG" > "$T/bc.txt" 2>&1
python3 wire_balance.py "$T/s3.mpmct1" "$TAG-S3" > "$T/wb.txt" 2>&1
cd /Users/rancanetti/code/local_mixing
CSTEP=$(python3 -c "print(max(1, $G//8000))")
GSTEP=$(python3 -c "print(max(1, $G//3000))")
EXPOSURE_WITNESS="$R/exp_C_vs_out.witness.tsv" target/release/hmap_affine --c "$R/tdp.mpmct1.source_c.g57" --c-format g57 --g "$R/packed.mpmct1" --n "$N" \
  --exposure --c-step 1 --g-cuts "$R/packed.mpmct1.bounds" --out "$R/exp_C_vs_out" > "$R/exp_C_vs_out.log" 2>&1
target/release/hmap_affine --c "$R/tdp.mpmct1.source_c.g57" --c-format g57 --g "$R/tdp.mpmct1" --n "$N" \
  --exposure --c-step 1 --g-step "$GSTEP" --out "$R/exp_C_vs_s2" > "$R/exp_C_vs_s2.log" 2>&1
target/release/hmap_affine --c "$R/tdp.mpmct1" --c-format mpmct1 --g "$R/packed.mpmct1" --n $((4*N)) \
  --exposure --c-step "$CSTEP" --g-cuts "$R/packed.mpmct1.bounds" --out "$R/exp_s2_vs_out" > "$R/exp_s2_vs_out.log" 2>&1
cd "$RD"
python3 waist/anf_dump_eval.py "$T/packed.mpmct1" "$T/packed.mpmct1.bounds" "$T/deliverable.bin" "$TAG" > "$T/compress_eval.txt" 2>&1
gzip -9 -kf "$T/deliverable.bin"
xz -9 -kf "$T/deliverable.bin" 2>/dev/null || true
ls -la "$T/deliverable.bin"* >> "$T/compress_eval.txt"
echo "$TAG CHAIN DONE"
