#!/bin/bash
# Once an arm's ledger is written, drop the bulky observer artifacts (keep summaries, labels, protocols).
cd ~/cdesign/run
while true; do
  for lj in ledgers/*/*/ledger.json; do
    [ -f "$lj" ] || continue
    arm=$(basename $(dirname $(dirname $lj))); case=$(basename $(dirname $lj)); name=${case#full_}
    for d in observer/$arm/$name/r*; do
      [ -d "$d" ] || continue
      find "$d" -type f -size +1M ! -path "*/labels/*" ! -name summary.json ! -name protocol.json -delete 2>/dev/null
    done
  done
  sleep 300
done
