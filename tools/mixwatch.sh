#!/bin/bash
# mixwatch — one-glance status of the mixing runs on the experiment server.
#
#   tools/mixwatch.sh          one snapshot of every run with a log touched in the last 48h
#   tools/mixwatch.sh -w       refresh every 60s (Ctrl-C to stop)
#   tools/mixwatch.sh -f NAME  stream a run's log live (tail -f), e.g. -f gad_n128m900_mg80
#
# Runs stay fully detached on the server (an ssh disconnect must never kill a
# multi-hour job); this just reads their logs, which capture everything.

SERVER=ai-claude@129.114.108.89

snapshot() {
  ssh -o ConnectTimeout=15 "$SERVER" '
    for L in $(ls -t ~/tds/*.log 2>/dev/null); do
      # only mixing-run logs, active within 48h
      [ -z "$(find "$L" -mmin -2880)" ] && continue
      grep -aqm1 "\[ssg\]\|stage D" "$L" || continue
      nm=$(basename "$L" .log)
      if grep -aq "Final circuit written" "$L"; then st="DONE   "
      elif grep -aq "FAILURE at stage" "$L"; then st="BROKE  "
      elif [ "$(pgrep -cf "$nm")" -gt 0 ]; then st="RUNNING"
      else st="DEAD?  "; fi
      printf "%-9s %s\n" "$st" "$nm"
      stage=$(grep -a "stage-D\] stage [0-9]* |" "$L" | tail -1)
      prog=$(grep -a "progress:" "$L" | tail -1)
      pass=$(grep -a "stage-C pass" "$L" | tail -1)
      [ -n "$stage" ] && echo "    $stage"
      [ -n "$prog" ]  && echo "    $prog"
      [ -n "$pass" ]  && echo "    $pass"
    done
    echo
    echo "load: $(cut -d" " -f1-3 /proc/loadavg) (160 cores)"'
}

case "$1" in
  -w)
    while true; do clear; date; echo; snapshot; sleep 60; done ;;
  -f)
    [ -z "$2" ] && { echo "usage: mixwatch.sh -f <run-name>"; exit 1; }
    exec ssh -t "$SERVER" "tail -f ~/tds/$2.log" ;;
  *)
    snapshot ;;
esac
