#!/bin/bash
# val_chain.sh TAG N MC S SLICE CONFIG : validation run of the 4-stage
# pipeline. CONFIG = A (no chains) or L8 (bounded chains CHAIN=2 LEN=8).
# Both: PREMASK + SEED_AVOID_C1, migration dose 8, no retirement.
# Stage 4 = pack G=512 + repair.sh; then real serialization + timing.
set -uo pipefail
ROOT=/Users/rancanetti/code/local_mixing
RD=$ROOT/runs/ncdb_ab_20261001; W=$RD/waist
TAG=$1; N=$2; MC=$3; S=$4; SLICE=$5; CFG=$6
R=$W/$TAG; mkdir -p $R; cd $ROOT
SEED=$(python3 -c 'import secrets;print(secrets.randbelow(2**31))')
MSEED=$(python3 -c 'import secrets;print(secrets.randbelow(2**40))')
CH=""; [ "$CFG" = L8 ] && CH="EMBEDDED_MASKING_CHAIN=2 EMBEDDED_MASKING_CHAIN_LEN=8"
{ echo "$TAG gen seed: $SEED"; echo "mix seed: $MSEED"; echo "config $CFG: PREMASK SEED_AVOID_C1 SHUFFLING=8 RETURN_HOME=0 $CH"; } > $R/seed.txt
env EMBEDDED_MASKING_SHUFFLING=8 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=0 EMBEDDED_MASKING_PREMASK=1 EMBEDDED_MASKING_SEED_AVOID_C1=1 $CH \
  target/release/gen_sandwich_gadget $R/tdp.mpmct1 $N $MC $MC $S 1 $SLICE $SEED $SEED $SEED > $R/gen.log 2>&1
G=$(head -1 $R/tdp.mpmct1 | awk '{print $3}')
MOVES=$(python3 -c "print(int($G*2*2*1.3))")
FROZEN_DB_DIR=$HOME/code/local_mixing/frozen_db/frozen_m1_m11 FROZEN_FILTER=1 \
  target/release/circuit_mixer --input $R/tdp.mpmct1 --output $R/s3.mpmct1 --tdp --db-mixing \
  --profile 2,2,2,2,2 --moves $MOVES --p-db 1.0 \
  --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048 \
  --no-local-verify --seed $MSEED > $R/s3.log 2>&1
python3 $W/mult_census.py $R/tdp.mpmct1 $TAG > $R/mult_s2.txt 2>&1 &
STEP=$([ $N = 64 ] && echo 64 || echo 256)
SPAN_OUTPUTS_TRIVIAL=1 SPAN_INTERSECT=$R/span_s2.tsv target/release/hmap_affine --c $R/tdp.mpmct1.source_c.g57 --c-format g57 \
  --g $R/tdp.mpmct1 --n $N --exposure --c-step 1 --g-step $STEP --out $R/span_s2 2>&1 | grep "\[span\]" > $R/span_s2.txt &
cd $RD
python3 window_pack3.py $R/tdp.mpmct1 $R/s3.mpmct1 512 512 24576 3000 $TAG-G512 $R/packed_G512.mpmct1 > $R/pack_G512.txt 2>&1
bash $W/repair.sh $TAG 512 $N > $R/repair512.log 2>&1
python3 $W/anf_dump_eval.py $R/rep512.mpmct1 $R/rep512.bounds $R/deliverable.bin $TAG > $R/deliver.txt 2>&1
gzip -9 -kf $R/deliverable.bin; xz -9 -kf $R/deliverable.bin
ls -la $R/deliverable.bin* | awk '{print $5, $9}' >> $R/deliver.txt
wait
echo "$TAG VALIDATION DONE"
