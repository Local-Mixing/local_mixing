#!/bin/bash
# Wait for stage 5 of the four C-design arms, classify each source once, run stage 6 and the narrow portfolio.
cd ~/cdesign
ARMS="bal_base bal_v2 burst67_base burst67_v2"
for a in $ARMS; do until [ -f run/stages35/$a/DONE.json ]; do sleep 60; done; done
for c in bal burst67; do SRC=run/stages35/${c}_v2/source.mpmct
  if [ ! -f run/boundary/$(sha256sum $SRC | cut -c1-64).json ]; then ( cd py6 && ~/satvenv/bin/python boundary_run.py ../$SRC ) > run/boundary_$c.log 2>&1; fi
done
cd py6
for a in bal_base burst67_base; do S6_PROCS=2 S6_THREADS=8 S6_ROUNDS=2 setsid nohup python3 stage6_run.py $a asis > ../run/stage6_${a}_asis.log 2>&1 < /dev/null & done
setsid nohup ./cdesign_loops.sh > ../run/cdesign_loops.log 2>&1 < /dev/null &
cd ../py && nice -n 10 python3 attacks_port.py bal_base,bal_v2,burst67_base,burst67_v2 4 > ../run/attacks.log 2>&1
echo attacks done
