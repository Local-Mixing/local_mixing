#!/usr/bin/env bash
# Stage resume safety (docs/STAGE4_RUST_DESIGN.md §11.1, §10.9 item 8), for
# every stage after 3: the keyed completion markers, forced reruns, allowed
# failures, the delivery stage's packing exits 5-7 and unpacking exits, the
# chaining of the optional stages 4 and 5, and the two shapes of stage 6
# (packing + unpacking with refinement, or anf_only).
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
failed_marker() { # STEP RC: delivery.failed names the sub-step and its exit code
  [ "$(cat "$R/delivery.failed")" = "$(printf 'exit=%s\nstep=%s' "$2" "$1")" ]
}
# The first part runs the four-stage chain (stages 4 and 5 skipped, stage 6
# ending after the packer's deliverable): stage 6's marker, reruns, exits and
# allowed failures, and stage 3's marker.
FOUR=(--run-skip-stages 4,5 --delivery-anf-only)
drive() { bash "$root/scripts/tdp_gen.sh" --source-wires 4 --source-gates 1 --run-directory "$R" ${FOUR[@]+"${FOUR[@]}"} "$@"; }
count() { python3 -I -c "import json,sys; print(sum(1 for l in open(sys.argv[1]) if json.loads(l)['name']==sys.argv[2] and (len(sys.argv)<4 or json.loads(l)['args'][0]==sys.argv[3])))" "$R.calls" "$@"; }
drive > "$tmp/1.log" 2>&1
grep -q 'PIPELINE COMPLETE (stages 1-3,6): .*/packing/deliverable.tdp4' "$tmp/1.log"
[ "$(count tdp_packing run)" = 1 ]
[ "$(count tdp_unpacking run)" = 0 ]
grep -q '^anf_only=1$' "$R/delivery.recipe"
grep -q '^unpacking_status=skipped$' "$R/delivery.recipe"
if grep -q 'tdp_unpacking_sha256' "$R/delivery.recipe"; then echo "anf_only marker names tdp_unpacking" >&2; exit 1; fi
test ! -e "$R/unpacking"
# an unchanged resume skips stage 6 after re-checking the marker
drive > "$tmp/2.log" 2>&1
grep -q 'matches its completion marker, skipping' "$tmp/2.log"
[ "$(count tdp_packing run)" = 1 ]
# a tampered marker, a changed deliverable or a changed parameter forces stage 6
sed -i.bak 's/^mac.stage3=.*/mac.stage3=00/' "$R/delivery.recipe"
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
grep -q '^packing_params=--split-budget 4096$' "$R/delivery.recipe"
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
# packing exits 5, 6, 7 are reported with their meanings; no PIPELINE COMPLETE
for rc in 5 6 7; do
  printf 'packing_exit=%s\n' "$rc" > "$R.mock"
  set +e; drive --run-rerun-from-stage 6 > "$tmp/e$rc.log" 2>&1; got=$?; set -e
  [ "$got" = "$rc" ] || { cat "$tmp/e$rc.log" >&2; exit 1; }
  case $rc in 5) m='exposure floor missed' ;; 6) m='over budget' ;; 7) m='I/O error writing outputs' ;; esac
  grep -q "stage 6 FAILED: packing exit $rc ($m)" "$tmp/e$rc.log"
  no_complete "$tmp/e$rc.log"
  test ! -e "$R/delivery.recipe"
  # the marker the managed driver reads to attach a sub-step's exit meaning
  failed_marker packing "$rc"
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
test ! -e "$R/delivery.failed"
# an allowed failure: marker status, NOT_RECOMMENDED next to the final
# deliverable (packing/ under anf_only), no PIPELINE COMPLETE, also on resume
printf 'packing_status=floor_miss\n' > "$R.mock"
drive --packing-allow-floor-miss > "$tmp/a1.log" 2>&1
grep -q 'ALLOWED FAILURE (status=floor_miss)' "$tmp/a1.log"
test -e "$R/packing/NOT_RECOMMENDED"
grep -q '^status=floor_miss$' "$R/packing/NOT_RECOMMENDED"
grep -q '^status=floor_miss$' "$R/delivery.recipe"
grep -q '^packing_status=floor_miss$' "$R/delivery.recipe"
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
grep -q '^adopted=delivery.recipe$' "$R/stage3.recipe"
[ "$(count circuit_mixer)" = $((mixers + 3)) ]
# ... also through the seven-stage driver's packing.recipe (the same tag lines)
rm -f "$R/stage3.recipe"; mv "$R/delivery.recipe" "$R/packing.recipe"
drive > "$tmp/m5b.log" 2>&1
grep -q 'stage 3: adopted' "$tmp/m5b.log"
grep -q '^adopted=packing.recipe$' "$R/stage3.recipe"
[ "$(count circuit_mixer)" = $((mixers + 3)) ]
rm -f "$R/stage3.recipe" "$R/packing.recipe" "$R/delivery.recipe"
drive > "$tmp/m6.log" 2>&1
grep -q 'no matching completion marker' "$tmp/m6.log"
[ "$(count circuit_mixer)" = $((mixers + 4)) ]

# ---- the optional stages 4 and 5, and the full delivery stage: markers, chaining and reruns ----
R=$tmp/six
FOUR=()
drive > "$tmp/v1.log" 2>&1
grep -q 'PIPELINE COMPLETE (stages 1-6): .*/unpacking/unpacked.mpmct1' "$tmp/v1.log"
[ "$(count circuit_mixer)" = 3 ]
[ "$(count tdp_packing run)" = 1 ]
[ "$(count tdp_unpacking run)" = 1 ]
for f in stage4.recipe stage5.recipe delivery.recipe stage6.log split.mpmct1 split.state crossing.mpmct1 \
         packing/deliverable.tdp4 unpacking/deliverable.tdp4 unpacking/bounds.final unpacking/unpacked.mpmct1 \
         unpacking/rounds.tsv unpacking/unpacking.report; do test -s "$R/$f"; done
grep -q '^stop_reason=split stage complete (stopped at the stage boundary)$' "$R/stage4.recipe"
grep -q '^stage5_source=resume split.state$' "$R/stage5.recipe"
grep -q '^seed_mac.split=' "$R/stage4.recipe"
# the delivery marker: both binaries, both parameter sets, the delivery seed,
# the three inputs, the packer's deliverable + the refined one + the circuit
for line in '^delivery_recipe=1$' '^tdp_packing_sha256=' '^tdp_unpacking_sha256=' '^packing_params=default$' \
            '^anf_only=0$' '^delivery_params=--refine --max-rounds 16$' '^seed_mac.unpack=' '^mac.stage2=' '^mac.stage3=' \
            '^mac.source_c=' '^mac.deliverable_tdp4=' '^mac.refined_tdp4=' '^mac.unpacked=' '^status=ok$' \
            '^packing_status=ok$' '^unpacking_status=ok$'; do
  grep -q "$line" "$R/delivery.recipe" || { echo "delivery.recipe lacks $line" >&2; cat "$R/delivery.recipe" >&2; exit 1; }
done
grep -q -- '^---- unpacking: tdp_unpacking run --refine --max-rounds 16 ----$' "$R/stage6.log"
# an unchanged resume skips every stage (3, 4, 5, 6)
drive > "$tmp/v2.log" 2>&1
[ "$(grep -c 'matches its completion marker, skipping' "$tmp/v2.log")" = 4 ]
[ "$(count circuit_mixer)" = 3 ]
# a changed stage-5 parameter re-runs stages 5 and 6 only
drive --crossing-target-size-factor 3 > "$tmp/v3.log" 2>&1
grep -q 'stage 5: .* no matching completion marker' "$tmp/v3.log"
[ "$(count circuit_mixer)" = 4 ]
[ "$(count tdp_packing run)" = 2 ]
[ "$(count tdp_unpacking run)" = 2 ]
grep -q 'PIPELINE COMPLETE (stages 1-6)' "$tmp/v3.log"
# a changed delivery parameter re-runs stage 6 (one stage: both sub-steps)
drive --crossing-target-size-factor 3 --delivery-restarts 2 > "$tmp/v4.log" 2>&1
grep -q 'stage 6: completion marker does not match' "$tmp/v4.log"
[ "$(count tdp_packing run)" = 3 ]
[ "$(count tdp_unpacking run)" = 3 ]
grep -q '^delivery_params=--refine --max-rounds 16 --restarts 2$' "$R/delivery.recipe"
# a tampered split output re-runs stages 4-6
printf 'x' >> "$R/split.mpmct1"
drive --crossing-target-size-factor 3 --delivery-restarts 2 > "$tmp/v5.log" 2>&1
grep -q 'stage 4: .* no matching completion marker' "$tmp/v5.log"
[ "$(count circuit_mixer)" = 6 ]
[ "$(count tdp_packing run)" = 4 ]
[ "$(count tdp_unpacking run)" = 4 ]
# a tampered refined deliverable re-runs stage 6
printf 'x' >> "$R/unpacking/deliverable.tdp4"
drive --crossing-target-size-factor 3 --delivery-restarts 2 > "$tmp/v5b.log" 2>&1
grep -q 'stage 6: completion marker does not match' "$tmp/v5b.log"
[ "$(count tdp_packing run)" = 5 ]
[ "$(count tdp_unpacking run)" = 5 ]
# rerun from stage 4: stages 4-6 again, stage 3 kept
drive --crossing-target-size-factor 3 --delivery-restarts 2 --run-rerun-from-stage 4 > "$tmp/v6.log" 2>&1
grep -q 'stage 3: .* matches its completion marker, skipping' "$tmp/v6.log"
[ "$(count circuit_mixer)" = 8 ]
[ "$(count tdp_unpacking run)" = 6 ]
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
grep -q 'PIPELINE COMPLETE (stages 1-6)' "$tmp/v8.log"
packs=$(count tdp_packing run); unpacks=$(count tdp_unpacking run)
# unpacking exits are reported with their meanings and the delivery.failed
# marker (the packing sub-step ran again first: one stage)
for rc in 3 4 7; do
  printf 'unpacking_exit=%s\n' "$rc" > "$R.mock"
  set +e; drive --run-rerun-from-stage 6 > "$tmp/u$rc.log" 2>&1; got=$?; set -e
  [ "$got" = "$rc" ] || { cat "$tmp/u$rc.log" >&2; exit 1; }
  case $rc in 3) m='input error' ;; 4) m='verification or synthesis failure' ;; 7) m='I/O error writing outputs' ;; esac
  grep -q "stage 6 FAILED: unpacking exit $rc ($m)" "$tmp/u$rc.log"
  no_complete "$tmp/u$rc.log"
  test ! -e "$R/delivery.recipe"
  failed_marker unpacking "$rc"
  packs=$((packs + 1)); unpacks=$((unpacks + 1))
  [ "$(count tdp_packing run)" = "$packs" ]
  [ "$(count tdp_unpacking run)" = "$unpacks" ]
done
rm -f "$R.mock"
drive > "$tmp/u0.log" 2>&1
grep -q 'PIPELINE COMPLETE (stages 1-6)' "$tmp/u0.log"
test ! -e "$R/delivery.failed"
# a window emitted verbatim by the refinement (status=verbatim_used) is an
# allowed failure: reported, NOT_RECOMMENDED next to the final output, no
# PIPELINE COMPLETE, the same on resume; cleared by a clean re-run
printf 'unpacking_status=verbatim_used\n' > "$R.mock"
drive --run-rerun-from-stage 6 > "$tmp/u9.log" 2>&1
grep -q 'ALLOWED FAILURE (status=verbatim_used)' "$tmp/u9.log"
grep -q 'PIPELINE FINISHED WITH AN ALLOWED FAILURE (stages 1-6; status=verbatim_used)' "$tmp/u9.log"
no_complete "$tmp/u9.log"
test -e "$R/unpacking/NOT_RECOMMENDED"
test ! -e "$R/packing/NOT_RECOMMENDED"
grep -q '^status=verbatim_used$' "$R/delivery.recipe"
grep -q '^unpacking_status=verbatim_used$' "$R/unpacking/NOT_RECOMMENDED"
drive > "$tmp/u10.log" 2>&1
grep -q 'matches its completion marker, skipping' "$tmp/u10.log"
grep -q 'ALLOWED FAILURE (status=verbatim_used)' "$tmp/u10.log"
no_complete "$tmp/u10.log"
rm -f "$R.mock"
drive --run-rerun-from-stage 6 > "$tmp/u11.log" 2>&1
grep -q 'PIPELINE COMPLETE (stages 1-6)' "$tmp/u11.log"
test ! -e "$R/unpacking/NOT_RECOMMENDED"
# stage 5 without stage 4 starts from the stage-3 output with the cross seed
R=$tmp/s5
drive --run-skip-stages 4 > "$tmp/w1.log" 2>&1
grep -q 'stage 5: crossing walk (input db_mixing.mpmct1)' "$tmp/w1.log"
grep -q '^seed_mac.cross=' "$R/stage5.recipe"
grep -q 'PIPELINE COMPLETE (stages 1-3,5-6)' "$tmp/w1.log"
# anf_only and back: toggling it changes the stage's recipe, so stage 6 is
# re-run in full either way, and the stale unpacking/ directory goes with it
R=$tmp/anf
drive --run-skip-stages 4,5 --delivery-anf-only > "$tmp/x1.log" 2>&1
grep -q 'PIPELINE COMPLETE (stages 1-3,6): .*/packing/deliverable.tdp4' "$tmp/x1.log"
[ "$(count tdp_unpacking run)" = 0 ]
drive --run-skip-stages 4,5 > "$tmp/x2.log" 2>&1
grep -q 'stage 6: completion marker does not match' "$tmp/x2.log"
grep -q 'PIPELINE COMPLETE (stages 1-3,6): .*/unpacking/unpacked.mpmct1' "$tmp/x2.log"
[ "$(count tdp_packing run)" = 2 ]
[ "$(count tdp_unpacking run)" = 1 ]
grep -q '^anf_only=0$' "$R/delivery.recipe"
drive --run-skip-stages 4,5 --delivery-anf-only > "$tmp/x3.log" 2>&1
grep -q 'stage 6: completion marker does not match' "$tmp/x3.log"
[ "$(count tdp_packing run)" = 3 ]
[ "$(count tdp_unpacking run)" = 1 ]
test ! -e "$R/unpacking"
grep -q '^anf_only=1$' "$R/delivery.recipe"
echo 'TDP stage resume shell tests passed'
