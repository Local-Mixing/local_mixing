#!/usr/bin/env bash
# Stage-4 resume safety (docs/STAGE4_RUST_DESIGN.md §11.1, §10.9 item 8):
# the keyed completion marker, forced reruns, allowed failures, and exits 5-7.
set -euo pipefail
# errexit ignores a `!`-negated command and every command of an `&&` list but
# the last: each assertion below is a plain command or an explicit `if`.
if [[ ${1-} != --isolated-environment ]]; then
  exec env -i PATH="$PATH" HOME="${HOME:-/tmp}" bash "$0" --isolated-environment
fi
shift
root=$(cd "$(dirname "$0")/../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export TDP_BIN_DIR="$tmp/bin" TDP_GEN_ALLOW_EMPTY_STORE=1
mkdir -p "$TDP_BIN_DIR"
for b in gen_sandwich_gadget circuit_mixer tdp_packing; do ln -s "$root/tests/shell/lib/tdp_mock.py" "$TDP_BIN_DIR/$b"; done
R=$tmp/run
no_complete() { # LOG: fails if the log claims PIPELINE COMPLETE
  if grep -q 'PIPELINE COMPLETE' "$1"; then echo "PIPELINE COMPLETE in $1" >&2; exit 1; fi
}
drive() { bash "$root/scripts/tdp_gen.sh" --source-wires 4 --source-gates 1 --run-directory "$R" "$@"; }
count() { python3 -I -c "import json,sys; print(sum(1 for l in open(sys.argv[1]) if json.loads(l)['name']==sys.argv[2] and (len(sys.argv)<4 or json.loads(l)['args'][0]==sys.argv[3])))" "$R.calls" "$@"; }
drive > "$tmp/1.log" 2>&1
grep -q 'PIPELINE COMPLETE (stages 1-4)' "$tmp/1.log"
[ "$(count tdp_packing run)" = 1 ]
# an unchanged resume skips stage 4 after re-checking the marker
drive > "$tmp/2.log" 2>&1
grep -q 'matches its completion marker, skipping' "$tmp/2.log"
[ "$(count tdp_packing run)" = 1 ]
# a tampered marker, a changed deliverable or a changed parameter forces stage 4
sed -i.bak 's/^mac.stage3=.*/mac.stage3=00/' "$R/packing.recipe"
drive > "$tmp/3.log" 2>&1
grep -q 'completion marker does not match' "$tmp/3.log"
[ "$(count tdp_packing run)" = 2 ]
printf 'x' >> "$R/packing/deliverable.tdp4"
drive > "$tmp/4.log" 2>&1
grep -q 'completion marker does not match' "$tmp/4.log"
[ "$(count tdp_packing run)" = 3 ]
drive --packing-split-budget 4096 > "$tmp/5.log" 2>&1
grep -q 'completion marker does not match' "$tmp/5.log"
[ "$(count tdp_packing run)" = 4 ]
grep -q -- '--split-budget' <(tail -1 "$R.calls")
grep -q '^params=--split-budget 4096$' "$R/packing.recipe"
# regenerating stage 3 forces stage 4 (and only stages 3-4)
drive --packing-split-budget 4096 --run-rerun-from-stage 3 > "$tmp/6.log" 2>&1
[ "$(count circuit_mixer)" = 2 ]
[ "$(count gen_sandwich_gadget)" = 1 ]
[ "$(count tdp_packing run)" = 5 ]
# regenerating stage 2 forces stages 3 and 4
drive --packing-split-budget 4096 --run-rerun-from-stage 2 > "$tmp/7.log" 2>&1
[ "$(count gen_sandwich_gadget)" = 2 ]
[ "$(count circuit_mixer)" = 3 ]
[ "$(count tdp_packing run)" = 6 ]
# exits 5, 6, 7 are reported with their meanings; no PIPELINE COMPLETE
for rc in 5 6 7; do
  printf 'packing_exit=%s\n' "$rc" > "$R.mock"
  set +e; drive --run-rerun-from-stage 4 > "$tmp/e$rc.log" 2>&1; got=$?; set -e
  [ "$got" = "$rc" ] || { cat "$tmp/e$rc.log" >&2; exit 1; }
  case $rc in 5) m='exposure floor missed' ;; 6) m='over budget' ;; 7) m='I/O error writing outputs' ;; esac
  grep -q "stage 4 FAILED: exit $rc ($m)" "$tmp/e$rc.log"
  no_complete "$tmp/e$rc.log"
  test ! -e "$R/packing.recipe"
done
# an allowed failure: marker status, NOT_RECOMMENDED, no PIPELINE COMPLETE, also on resume
printf 'packing_status=floor_miss\n' > "$R.mock"
drive --packing-allow-floor-miss > "$tmp/a1.log" 2>&1
grep -q 'ALLOWED FAILURE (status=floor_miss)' "$tmp/a1.log"
test -e "$R/packing/NOT_RECOMMENDED"
grep -q '^status=floor_miss$' "$R/packing.recipe"
no_complete "$tmp/a1.log"
drive --packing-allow-floor-miss > "$tmp/a2.log" 2>&1
grep -q 'matches its completion marker, skipping' "$tmp/a2.log"
grep -q 'ALLOWED FAILURE (status=floor_miss)' "$tmp/a2.log"
no_complete "$tmp/a2.log"
# dropping the allow flag re-runs stage 4 (a parameter change)
rm -f "$R.mock"
drive > "$tmp/a3.log" 2>&1
grep -q 'PIPELINE COMPLETE (stages 1-4)' "$tmp/a3.log"
test ! -e "$R/packing/NOT_RECOMMENDED"
echo 'TDP stage-4 resume shell tests passed'
