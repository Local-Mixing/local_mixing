#!/usr/bin/env bash
# Stage resume safety (docs/STAGE4_RUST_DESIGN.md §11.1, §10.9 item 8), for
# every stage after 3: the keyed completion markers, forced reruns, allowed
# failures, the stage-6 exits 5-7 and the stage-7 exits, and the chaining of
# the optional stages 4, 5 and 7.
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
for b in gen_sandwich_gadget circuit_mixer tdp_packing tdp_unpacking; do ln -s "$root/tests/shell/lib/tdp_mock.py" "$TDP_BIN_DIR/$b"; done
R=$tmp/run
no_complete() { # LOG: fails if the log claims PIPELINE COMPLETE
  if grep -q 'PIPELINE COMPLETE' "$1"; then echo "PIPELINE COMPLETE in $1" >&2; exit 1; fi
}
# The first part runs the four-stage chain (stages 4, 5, 7 skipped): stage 6's
# marker, reruns, exits and allowed failures, and stage 3's marker.
FOUR=(--run-skip-stages 4,5,7)
drive() { bash "$root/scripts/tdp_gen.sh" --source-wires 4 --source-gates 1 --run-directory "$R" ${FOUR[@]+"${FOUR[@]}"} "$@"; }
count() { python3 -I -c "import json,sys; print(sum(1 for l in open(sys.argv[1]) if json.loads(l)['name']==sys.argv[2] and (len(sys.argv)<4 or json.loads(l)['args'][0]==sys.argv[3])))" "$R.calls" "$@"; }
drive > "$tmp/1.log" 2>&1
grep -q 'PIPELINE COMPLETE (stages 1-3,6)' "$tmp/1.log"
[ "$(count tdp_packing run)" = 1 ]
# an unchanged resume skips stage 6 after re-checking the marker
drive > "$tmp/2.log" 2>&1
grep -q 'matches its completion marker, skipping' "$tmp/2.log"
[ "$(count tdp_packing run)" = 1 ]
# a tampered marker, a changed deliverable or a changed parameter forces stage 6
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
# regenerating stage 3 forces stage 6 (and only stages 3 and 6)
drive --packing-split-budget 4096 --run-rerun-from-stage 3 > "$tmp/6.log" 2>&1
[ "$(count circuit_mixer)" = 2 ]
[ "$(count gen_sandwich_gadget)" = 1 ]
[ "$(count tdp_packing run)" = 5 ]
# regenerating stage 2 forces stages 3 and 6
drive --packing-split-budget 4096 --run-rerun-from-stage 2 > "$tmp/7.log" 2>&1
[ "$(count gen_sandwich_gadget)" = 2 ]
[ "$(count circuit_mixer)" = 3 ]
[ "$(count tdp_packing run)" = 6 ]
# exits 5, 6, 7 are reported with their meanings; no PIPELINE COMPLETE
for rc in 5 6 7; do
  printf 'packing_exit=%s\n' "$rc" > "$R.mock"
  set +e; drive --run-rerun-from-stage 6 > "$tmp/e$rc.log" 2>&1; got=$?; set -e
  [ "$got" = "$rc" ] || { cat "$tmp/e$rc.log" >&2; exit 1; }
  case $rc in 5) m='exposure floor missed' ;; 6) m='over budget' ;; 7) m='I/O error writing outputs' ;; esac
  grep -q "stage 6 FAILED: exit $rc ($m)" "$tmp/e$rc.log"
  no_complete "$tmp/e$rc.log"
  test ! -e "$R/packing.recipe"
  # the marker the managed driver reads to attach a stage-6 meaning
  [ "$(cat "$R/packing.failed")" = "exit=$rc" ]
done
# an operator stop flag ends stage 3 below its dose: refused, never shipped,
# and a resume re-runs stage 3 (the output is not under the canonical name)
printf 'mixer_stop=stop flag\n' > "$R.mock"
set +e; drive --run-rerun-from-stage 3 > "$tmp/s1.log" 2>&1; got=$?; set -e
[ "$got" = 2 ] || { cat "$tmp/s1.log" >&2; exit 1; }
grep -q 'FATAL: stage 3 ended by stop flag, below the recipe' "$tmp/s1.log"
no_complete "$tmp/s1.log"
test ! -e "$R/db_mixing.mpmct1"
test -s "$R/db_mixing.stopped.mpmct1"
grep -q '^stop_reason=stop flag$' "$R/stage3.settings"
mixers=$(count circuit_mixer); packs=$(count tdp_packing run)
rm -f "$R.mock"
drive > "$tmp/s2.log" 2>&1
grep -q 'PIPELINE COMPLETE (stages 1-3,6)' "$tmp/s2.log"
[ "$(count circuit_mixer)" = $((mixers + 1)) ]
[ "$(count tdp_packing run)" = $((packs + 1)) ]
grep -q '^stop_reason=moves budget spent$' "$R/stage3.settings"
test ! -e "$R/packing.failed"
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
# dropping the allow flag re-runs stage 6 (a parameter change)
rm -f "$R.mock"
drive > "$tmp/a3.log" 2>&1
grep -q 'PIPELINE COMPLETE (stages 1-3,6)' "$tmp/a3.log"
test ! -e "$R/packing/NOT_RECOMMENDED"
# stage 3 is skipped only for an output its completion marker vouches for
grep -q '^tdp_stage3_recipe=1$' "$R/stage3.recipe"
grep -q '^mac.db_mixing=' "$R/stage3.recipe"
mixers=$(count circuit_mixer); packs=$(count tdp_packing run)
drive > "$tmp/m1.log" 2>&1
grep -q 'stage 3: .* matches its completion marker, skipping' "$tmp/m1.log"
[ "$(count circuit_mixer)" = "$mixers" ]
# a truncated (interrupted) output is re-run, and stage 6 with it
head -c 5 "$R/db_mixing.mpmct1" > "$R/partial" && mv "$R/partial" "$R/db_mixing.mpmct1"
drive > "$tmp/m2.log" 2>&1
grep -q 'no matching completion marker' "$tmp/m2.log"
grep -q 'PIPELINE COMPLETE (stages 1-3,6)' "$tmp/m2.log"
[ "$(count circuit_mixer)" = $((mixers + 1)) ]
[ "$(count tdp_packing run)" = $((packs + 1)) ]
# a changed schedule re-runs stage 3
drive --db-mixing-schedule custom > "$tmp/m3.log" 2>&1
grep -q 'no matching completion marker' "$tmp/m3.log"
[ "$(count circuit_mixer)" = $((mixers + 2)) ]
drive > "$tmp/m4.log" 2>&1
[ "$(count circuit_mixer)" = $((mixers + 3)) ]
# an older run (no stage-3 marker): adopted only when stage 6 consumed these bytes
rm -f "$R/stage3.recipe"
drive > "$tmp/m5.log" 2>&1
grep -q 'stage 3: adopted' "$tmp/m5.log"
grep -q '^adopted=packing.recipe$' "$R/stage3.recipe"
[ "$(count circuit_mixer)" = $((mixers + 3)) ]
rm -f "$R/stage3.recipe" "$R/packing.recipe"
drive > "$tmp/m6.log" 2>&1
grep -q 'no matching completion marker' "$tmp/m6.log"
[ "$(count circuit_mixer)" = $((mixers + 4)) ]

# ---- the optional stages 4, 5 and 7: markers, chaining and reruns ----
R=$tmp/seven
FOUR=()
drive > "$tmp/v1.log" 2>&1
grep -q 'PIPELINE COMPLETE (stages 1-7)' "$tmp/v1.log"
[ "$(count circuit_mixer)" = 3 ]
[ "$(count tdp_unpacking run)" = 1 ]
for f in stage4.recipe stage5.recipe unpacking.recipe split.mpmct1 split.state crossing.mpmct1 unpacking/unpacked.mpmct1; do test -s "$R/$f"; done
grep -q '^stop_reason=split stage complete (stopped at the stage boundary)$' "$R/stage4.recipe"
grep -q '^stage5_source=resume split.state$' "$R/stage5.recipe"
grep -q '^seed_mac.split=' "$R/stage4.recipe"
grep -q '^seed_mac.unpack=' "$R/unpacking.recipe"
# an unchanged resume skips every stage
drive > "$tmp/v2.log" 2>&1
[ "$(grep -c 'matches its completion marker, skipping' "$tmp/v2.log")" = 5 ]
[ "$(count circuit_mixer)" = 3 ]
# a changed stage-5 parameter re-runs stages 5, 6 and 7 only
drive --crossing-target-size-factor 3 > "$tmp/v3.log" 2>&1
grep -q 'stage 5: .* no matching completion marker' "$tmp/v3.log"
[ "$(count circuit_mixer)" = 4 ]
[ "$(count tdp_packing run)" = 2 ]
[ "$(count tdp_unpacking run)" = 2 ]
grep -q 'PIPELINE COMPLETE (stages 1-7)' "$tmp/v3.log"
# a changed stage-7 parameter re-runs stage 7 only
drive --crossing-target-size-factor 3 --unpacking-restarts 2 > "$tmp/v4.log" 2>&1
grep -q 'stage 7: completion marker does not match' "$tmp/v4.log"
[ "$(count tdp_packing run)" = 2 ]
[ "$(count tdp_unpacking run)" = 3 ]
grep -q '^params=--restarts 2$' "$R/unpacking.recipe"
# a tampered split output re-runs stages 4-7
printf 'x' >> "$R/split.mpmct1"
drive --crossing-target-size-factor 3 --unpacking-restarts 2 > "$tmp/v5.log" 2>&1
grep -q 'stage 4: .* no matching completion marker' "$tmp/v5.log"
[ "$(count circuit_mixer)" = 6 ]
[ "$(count tdp_packing run)" = 3 ]
# rerun from stage 4: stages 4-7 again, stage 3 kept
drive --crossing-target-size-factor 3 --unpacking-restarts 2 --run-rerun-from-stage 4 > "$tmp/v6.log" 2>&1
grep -q 'stage 3: .* matches its completion marker, skipping' "$tmp/v6.log"
[ "$(count circuit_mixer)" = 8 ]
# a split stage ended by a stop flag is refused like stage 3, and re-run on resume
printf 'split_stop=stop flag\n' > "$R.mock"
set +e; drive --run-rerun-from-stage 4 > "$tmp/v7.log" 2>&1; got=$?; set -e
[ "$got" = 2 ] || { cat "$tmp/v7.log" >&2; exit 1; }
grep -q 'FATAL: stage 4 ended by stop flag' "$tmp/v7.log"
test ! -e "$R/split.mpmct1"
test -s "$R/split.stopped.mpmct1"
no_complete "$tmp/v7.log"
rm -f "$R.mock"
drive > "$tmp/v8.log" 2>&1
grep -q 'PIPELINE COMPLETE (stages 1-7)' "$tmp/v8.log"
# stage-7 exits are reported with their meanings and the unpacking.failed marker
for rc in 3 4 7; do
  printf 'unpacking_exit=%s\n' "$rc" > "$R.mock"
  set +e; drive --run-rerun-from-stage 7 > "$tmp/u$rc.log" 2>&1; got=$?; set -e
  [ "$got" = "$rc" ] || { cat "$tmp/u$rc.log" >&2; exit 1; }
  case $rc in 3) m='input error' ;; 4) m='verification failure' ;; 7) m='I/O error writing outputs' ;; esac
  grep -q "stage 7 FAILED: exit $rc ($m)" "$tmp/u$rc.log"
  no_complete "$tmp/u$rc.log"
  test ! -e "$R/unpacking.recipe"
  [ "$(cat "$R/unpacking.failed")" = "exit=$rc" ]
done
rm -f "$R.mock"
drive > "$tmp/u0.log" 2>&1
grep -q 'PIPELINE COMPLETE (stages 1-7)' "$tmp/u0.log"
test ! -e "$R/unpacking.failed"
# a stage-7 status other than ok is noted
printf 'unpacking_status=fallback_used\n' > "$R.mock"
drive --run-rerun-from-stage 7 > "$tmp/u9.log" 2>&1
grep -q 'stage 7 status=fallback_used' "$tmp/u9.log"
grep -q 'PIPELINE COMPLETE (stages 1-7)' "$tmp/u9.log"
rm -f "$R.mock"
# stage 5 without stage 4 starts from the stage-3 output with the cross seed;
# stage 7 cannot run without stage 6
R=$tmp/s5
drive --run-skip-stages 4 > "$tmp/w1.log" 2>&1
grep -q 'stage 5: crossing walk (input db_mixing.mpmct1)' "$tmp/w1.log"
grep -q '^seed_mac.cross=' "$R/stage5.recipe"
grep -q 'PIPELINE COMPLETE (stages 1-3,5-7)' "$tmp/w1.log"
set +e; drive --run-skip-stages 6 > "$tmp/w2.log" 2>&1; got=$?; set -e
[ "$got" = 2 ]
grep -q 'stage 7 (unpacking) needs stage 6 (packing)' "$tmp/w2.log"
echo 'TDP stage resume shell tests passed'
