#!/bin/bash
# Poll until each named arm's chain has finished (C-view scan written), then
# run score_arm.sh on it. Exits when all are scored or after ~5 h.
cd /Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001/waist
declare -A done_
for i in $(seq 1 300); do
  left=0
  for t in "$@"; do
    [ -n "${done_[$t]:-}" ] && continue
    if grep -q "born values" $t/exp_C_vs_out.log 2>/dev/null && [ -s $t/compress_eval.txt ]; then
      done_[$t]=1; bash score_arm.sh $t > ${t}_score.log 2>&1 &
    else
      left=$((left+1))
    fi
  done
  [ $left -eq 0 ] && break
  sleep 60
done
wait
echo "ALL SCORED: $*"
