#!/bin/bash
# differential.sh SEED [K] : the CI differential test of the Python ORACLE
# against tdp_packing on fresh random inputs (docs/STAGE4_RUST_DESIGN.md §10.6).
#
# For k = 1..K (default 20), from synth.py with seeds derived from SEED:
#   pack   window_pack3.py vs `tdp_packing pack` on a random (A, B) pair at a
#          random argv: stdout, the packed circuit and its bounds byte for byte;
#   repair repair.sh (sandboxed; split budget and iteration counts rewritten
#          exactly once) vs `tdp_packing repair` on a random (C, P, bounds):
#          rep512.mpmct1, rep512.bounds, the three TSVs byte for byte,
#          cview.log and the repair log after the directory prefix is
#          normalized (verify lines as a set);
#   emit   anf_dump_eval.py vs `tdp_packing serialize` on the oracle's
#          repaired circuit: the body byte for byte.
# Plus the committed fixtures' expected files are re-derived by the oracle only
# where cheap (the integration tests compare Rust with them on every push).
#
# Environment: TDP_PACKING (release binary), HMAP_BASELINE (baseline
# hmap_affine built from 492564f0), ORACLE_PY (passes pyset_selftest.py),
# WORK (scratch). On a mismatch the failing inputs stay in WORK/fail_<k>_<kind>.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
SEED=${1:?usage: differential.sh SEED [K]}; K=${2:-20}
: "${TDP_PACKING:?}" "${HMAP_BASELINE:?}" "${ORACLE_PY:?}" "${WORK:?}"
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
FX=$REPO/tests/fixtures/stage4
"$ORACLE_PY" -I "$FX/pyset_selftest.py" "$FX/pyset39_traces.txt" > /dev/null ||
  { echo "ORACLE_PY fails the CPython 3.9 set-order self-test" >&2; exit 2; }
echo "differential: SEED=$SEED (calibration), K=$K, oracle $("$ORACLE_PY" -c 'import sys; print(sys.version.split()[0])')"
RD=$(bash "$FX/oracle_sandbox.sh" "$WORK/sb_pack") || exit 2
fail=0
note_fail() { echo "  MISMATCH k=$1 $2: $3"; fail=1; }
norm() { sed "s|$1/|<DIR>/|g" "$2"; }
split_verify() { grep -v 'EQUIVALENT (1024 random states)' "$1"; }
verify_set() { grep 'EQUIVALENT (1024 random states)' "$1" | sort; }
for k in $(seq 1 "$K"); do
  s=$("$ORACLE_PY" -I -c "import random,sys; r=random.Random(int(sys.argv[1])*1000003+int(sys.argv[2])); print(r.randrange(1<<40))" "$SEED" "$k")
  # ---- pack
  D=$WORK/k$k/pack; mkdir -p "$D"
  args=$("$ORACLE_PY" -I "$HERE/synth.py" "$D" pack "$s") || { echo "synth.py pack failed (k=$k)" >&2; exit 2; }
  read -r wc g bm sc <<< "$args"
  (cd "$RD" && "$ORACLE_PY" window_pack3.py "$D/a.mpmct1" "$D/b.mpmct1" "$wc" "$g" "$bm" "$sc" lbl "$D/o.mpmct1" > "$D/o.txt" 2>&1)
  "$TDP_PACKING" pack "$D/a.mpmct1" "$D/b.mpmct1" "$wc" "$g" "$bm" "$sc" lbl "$D/r.mpmct1" > "$D/r.txt" 2> "$D/r.err"
  st=ok
  cmp -s "$D/o.txt" "$D/r.txt" || st="$st; stdout"
  cmp -s "$D/o.mpmct1" "$D/r.mpmct1" || st="$st; packed"
  cmp -s "$D/o.mpmct1.bounds" "$D/r.mpmct1.bounds" || st="$st; bounds"
  if [ "$st" = ok ]; then echo "  k=$k pack ($wc $g $bm $sc; $(head -c 40 "$D/a.mpmct1" | head -1)): identical"
  else note_fail "$k" pack "$st"; cp -r "$D" "$WORK/fail_${k}_pack"; fi
  # ---- repair + emit
  D=$WORK/k$k/repair; mkdir -p "$D"
  args=$("$ORACLE_PY" -I "$HERE/synth.py" "$D/in" repair "$s") || { echo "synth.py repair failed (k=$k)" >&2; exit 2; }
  read -r n bud fix fb <<< "$args"
  SB=$WORK/k$k/sb_repair
  RDR=$(SPLIT_BUDGET=$bud FIX_ITERS=$fix FALLBACK_ITERS=$fb bash "$FX/oracle_sandbox.sh" "$SB") || exit 2
  R=$RDR/waist/x; mkdir -p "$R"
  cp "$D/in/p.mpmct1" "$R/packed_G512.mpmct1"; cp "$D/in/p.bounds" "$R/packed_G512.mpmct1.bounds"
  cp "$D/in/c.g57" "$R/tdp.mpmct1.source_c.g57"; cp "$D/in/p.mpmct1" "$R/tdp.mpmct1"
  bash "$RDR/waist/repair.sh" x 512 "$n" > "$R/repair512.log" 2>&1
  "$TDP_PACKING" repair --packed "$D/in/p.mpmct1" --bounds "$D/in/p.bounds" --source-c "$D/in/c.g57" --n "$n" \
     --out-dir "$D/rust" --split-budget "$bud" --fix-iterations "$fix" --fallback-iterations "$fb" \
     --allow-floor-miss --allow-over-budget > /dev/null 2> "$D/rust.err"
  st=ok
  for f in rep512.mpmct1 rep512.bounds rep512.exposure.tsv rep512.allcuts.tsv rep512.wit.tsv; do
    cmp -s "$R/$f" "$D/rust/$f" || st="$st; $f"
  done
  cmp -s <(norm "$R" "$R/rep512.cview.log") <(norm "$D/rust" "$D/rust/rep512.cview.log") || st="$st; cview.log"
  sed -n "/^== x rep512 final/q;p" "$R/repair512.log" | sed "s|$R/|<DIR>/|g" > "$D/want.log"
  norm "$D/rust" "$D/rust/repair512.log" > "$D/got.log"
  cmp -s <(split_verify "$D/want.log") <(split_verify "$D/got.log") || st="$st; repair log"
  cmp -s <(verify_set "$D/want.log") <(verify_set "$D/got.log") || st="$st; verify lines"
  # emit
  if [ -s "$R/rep512.mpmct1" ]; then
    "$ORACLE_PY" "$RDR/waist/anf_dump_eval.py" "$R/rep512.mpmct1" "$R/rep512.bounds" "$D/o.bin" x > /dev/null 2>&1
    "$TDP_PACKING" serialize "$R/rep512.mpmct1" "$R/rep512.bounds" "$D/r.bin" x > /dev/null 2> "$D/ser.err"
    cmp -s "$D/o.bin" "$D/r.bin" || st="$st; deliverable body"
  fi
  if [ "$st" = ok ]; then
    echo "  k=$k repair+emit (n=$n budget=$bud fix=$fix fallback=$fb; $(grep -c 'splits added' "$R/repair512.log") split rounds): identical"
  else note_fail "$k" repair "$st"; cp -r "$D" "$WORK/fail_${k}_repair"; cp -r "$R" "$WORK/fail_${k}_repair/oracle"; fi
  rm -rf "$SB"
done
[ $fail = 0 ] && echo "DIFFERENTIAL: ALL IDENTICAL ($K pack + $K repair/emit cases)"
exit $fail
