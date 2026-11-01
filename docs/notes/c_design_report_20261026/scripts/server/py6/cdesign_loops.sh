#!/bin/bash
# v2 arms: relaunch stage6 (variant v2, no round cap) until plateau; then start the resilience campaign. Idempotent.
cd ~/cdesign/py6
while true; do
  for a in bal_v2 burst67_v2; do
    fin=../run/stage6/$a/v2/final.json
    if [ -f "$fin" ] && python3 -c "import json,sys;sys.exit(0 if json.load(open('$fin')).get('plateau') else 1)"; then
      if ! pgrep -f "resilience.py $a v2\$" >/dev/null && [ ! -f ../run/resilience/${a}_v2/DONE.json ]; then
        nice -n 5 setsid nohup python3 resilience.py $a v2 >> ../run/resilience_${a}_v2.log 2>&1 < /dev/null &
        echo "$(date -u +%H:%MZ) launched resilience $a"; sleep 5
      fi
      continue
    fi
    if pgrep -f "stage6_run.py $a v2\$" >/dev/null; then continue; fi
    S6_PROCS=3 S6_THREADS=8 setsid nohup python3 stage6_run.py $a v2 >> ../run/stage6_${a}_v2.log 2>&1 < /dev/null &
    echo "$(date -u +%H:%MZ) (re)launched stage6 $a v2"; sleep 5
  done
  sleep 180
done
