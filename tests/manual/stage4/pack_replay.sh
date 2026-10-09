#!/bin/bash
# pack_replay.sh TAG... : replay the packer's incremental window ANF
# (WindowState on the CPython 3.9 set emulator, product cap 8*24576) over the
# archived packed_G512 windows of each TAG (read-only), under both cap
# semantics (docs/STAGE4_RUST_DESIGN.md §10.5 step 6, P1 acceptance).
#
# Reference (instrumented oracle, port_runs/spec_pack3, MULSTATS lines):
#   vA64a   mul calls 3267293, ambiguous 40,  aborts 2, order-dependent 2
#   vL128b  mul calls 9104992, ambiguous 242, aborts 6, order-dependent 6,
#           152 windows with an ambiguous product (replay_vL128b.log)
# Every abort must be a window's last gate (the packer closes the window
# right after a failed compose).
#   TDP_PACKING=.../release/tdp_packing bash pack_replay.sh vA64a vL128b
set -uo pipefail
W=${WAIST:-/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001/waist}
: "${TDP_PACKING:?path to tdp_packing}"
fail=0
# expect TAG : "key value ..." pairs known from the instrumented oracle
expect() {
  case $1 in
    vA64a) echo "aborts 2 order_dependent 2 mul_calls 3267293 ambiguous 40" ;;
    vL128b) echo "aborts 6 order_dependent 6 windows_with_ambiguous_mul 152 mul_calls 9104992 ambiguous 242" ;;
    *) echo "" ;;
  esac
}
# field LINE KEY : the number after KEY in LINE
field() { echo "$1" | awk -v k="$2" '{for (i = 1; i < NF; i++) if ($i == k) print $(i + 1)}'; }
# check_fields LINE KEY VALUE ... : every KEY's number in LINE equals VALUE
check_fields() {
  local line=$1 bad=0; shift
  while [ $# -ge 2 ]; do
    if [ "$(field "$line" "$1")" != "$2" ]; then echo "  MISMATCH $1: expected $2"; bad=1; fi
    shift 2
  done
  return $bad
}
for TAG in "$@"; do
  R=$W/$TAG
  for sem in cpython39 canonical; do
    line=$("$TDP_PACKING" pack-replay "$R/packed_G512.mpmct1" "$R/packed_G512.mpmct1.bounds" 24576 \
             --anf-cap-semantics $sem)
    echo "$TAG $line"
    if [ "$(echo "$line" | sed 's/.*not at window end: \([0-9]*\).*/\1/')" != 0 ]; then
      echo "  an abort is not at a window end"; fail=1
    fi
    [ $sem = cpython39 ] || continue
    # shellcheck disable=SC2046
    check_fields "$line" $(expect "$TAG") || fail=1
  done
done
exit $fail
