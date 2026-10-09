#!/bin/bash
# oracle_sandbox.sh SANDBOX_ROOT : copy the stage-4 Python/bash oracle
# (scripts/windowed_delivery) into a fresh sandbox laid out like the run tree
# (SANDBOX_ROOT/runs/ncdb_ab_20261001/{window_pack3.py,anf_compose.py,waist/}),
# rewriting only paths and the interpreter (docs/STAGE4_RUST_DESIGN.md §10.3):
#
#   /Users/rancanetti/code/local_mixing -> SANDBOX_ROOT
#   target/release/hmap_affine          -> $HMAP_BASELINE (pinned sha256)
#   python3 (repair.sh)                 -> $ORACLE_PY (set-order self-test)
#
# Every rewrite is asserted to apply exactly the expected number of times, so
# an oracle edit cannot silently escape the sandbox. The tracked oracle is
# never edited. Optional literal rewrites for small cases:
#   SPLIT_BUDGET (repair.sh literal 65536), FIX_ITERS, FALLBACK_ITERS.
#
# Environment: HMAP_BASELINE, ORACLE_PY (required).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
SB=${1:?usage: oracle_sandbox.sh SANDBOX_ROOT}
: "${HMAP_BASELINE:?set HMAP_BASELINE to the pinned baseline hmap_affine}"
: "${ORACLE_PY:?set ORACLE_PY to the oracle interpreter}"

got=$(shasum -a 256 "$HMAP_BASELINE" | cut -d' ' -f1)
if ! grep -q "^$got " "$HERE/cview_n8/baseline_sha256.txt"; then
  echo "refusing: $HMAP_BASELINE has sha256 $got, not a pinned baseline" >&2
  exit 1
fi
"$ORACLE_PY" -I "$HERE/pyset_selftest.py" "$HERE/pyset39_traces.txt" >/dev/null || {
  echo "refusing: $ORACLE_PY fails the CPython 3.9 set-order self-test" >&2
  exit 1
}

[ -e "$SB" ] && { echo "refusing: sandbox $SB exists" >&2; exit 1; }
RD=$SB/runs/ncdb_ab_20261001
mkdir -p "$RD/waist"
SRC=$REPO/scripts/windowed_delivery
# Only the oracle scripts the stage-4 chain runs (and their imports).
for f in window_pack3.py anf_compose.py carrier_refresh.py boundary_cluster.py; do
  cp "$SRC/$f" "$RD/"
done
for f in order_repair.py merge_bounds2.py anf_split2.py verify_equiv.py anf_dump_eval.py \
         link_census.py mult_census.py repair.sh; do
  cp "$SRC/waist/$f" "$RD/waist/"
done

OLD=/Users/rancanetti/code/local_mixing
# rewrite FILE FROM TO COUNT : replace every FROM by TO, asserting COUNT hits.
rewrite() {
  local f=$1 from=$2 to=$3 want=$4 have
  have=$(grep -oF -- "$from" "$f" | wc -l | tr -d ' ')
  if [ "$have" != "$want" ]; then
    echo "sandbox rewrite: $f has $have x '$from', expected $want" >&2
    exit 1
  fi
  FROM="$from" TO="$to" perl -pi -e 's/\Q$ENV{FROM}\E/$ENV{TO}/g' "$f"
}
W=$RD/waist
rewrite "$W/repair.sh" "$OLD" "$SB" 1
rewrite "$W/repair.sh" "target/release/hmap_affine" "$HMAP_BASELINE" 1
rewrite "$W/repair.sh" "python3 " "$ORACLE_PY " 8
rewrite "$W/anf_split2.py" "$OLD" "$SB" 2
rewrite "$W/anf_split2.py" "{ROOT}/target/release/hmap_affine" "$HMAP_BASELINE" 1
rewrite "$W/anf_dump_eval.py" "$OLD" "$SB" 1
rewrite "$W/verify_equiv.py" "$OLD" "$SB" 1
rewrite "$W/link_census.py" "$OLD" "$SB" 1
rewrite "$W/mult_census.py" "$OLD" "$SB" 1
if [ -n "${SPLIT_BUDGET:-}" ]; then
  rewrite "$W/repair.sh" " 65536 \$B \$B" " $SPLIT_BUDGET \$B \$B" 1
fi
if [ -n "${FIX_ITERS:-}" ] || [ -n "${FALLBACK_ITERS:-}" ]; then
  rewrite "$W/repair.sh" "for it in 1 2 3 4; do" "for it in @ITERS@; do" 2
  fix=$(seq -s ' ' 1 "${FIX_ITERS:-4}")
  fb=$(seq -s ' ' 1 "${FALLBACK_ITERS:-4}")
  perl -0pi -e "s/\@ITERS\@/$fix/; s/\@ITERS\@/$fb/" "$W/repair.sh"
fi
# Nothing may still point at the real run tree.
if grep -rlF "$OLD" "$RD" >/dev/null; then
  echo "sandbox: a path to the real run tree survived:" >&2
  grep -rnF "$OLD" "$RD" >&2
  exit 1
fi
echo "$RD"
