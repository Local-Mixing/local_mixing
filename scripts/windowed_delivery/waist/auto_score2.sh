#!/bin/bash
# bash-3.2-safe: poll until each arm's chain is done, then score it once
# (marker file <arm>/.scored). Exits when all are scored or after ~6 h.
cd /Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001/waist
for i in $(seq 1 360); do
  left=0
  for t in "$@"; do
    [ -f $t/.scored ] && continue
    if grep -q "born values" $t/exp_C_vs_out.log 2>/dev/null && [ -s $t/compress_eval.txt ]; then
      touch $t/.scored; bash score_arm.sh $t > ${t}_score.log 2>&1 &
    else
      left=$((left+1))
    fi
  done
  [ $left -eq 0 ] && break
  sleep 60
done
wait
echo "ALL SCORED: $*"
