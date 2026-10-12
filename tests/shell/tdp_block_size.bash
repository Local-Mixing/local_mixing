#!/usr/bin/env bash
# Fast plumbing smoke test of the seven-stage driver: stage-3 piece flags,
# stop/rerun points, the stage-3 schedule, the stage calls (the four-stage
# chain under --run-skip-stages 4,5,7, and the full seven-stage chain). Mock
# binaries (tests/shell/lib/tdp_mock.py) avoid mixing or release builds.
set -euo pipefail
if [[ ${1:-} != --isolated-environment ]]; then
  exec env -i PATH="$PATH" HOME="${HOME:-/tmp}" bash "$0" --isolated-environment
fi
shift
root=$(cd "$(dirname "$0")/../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export TDP_BIN_DIR="$tmp/bin" TDP_GEN_ALLOW_EMPTY_STORE=1
unset FROZEN_DB_DIR
mkdir -p "$TDP_BIN_DIR"
for binary in gen_sandwich_gadget circuit_mixer tdp_packing tdp_unpacking; do
  ln -s "$root/tests/shell/lib/tdp_mock.py" "$TDP_BIN_DIR/$binary"
done
reject() {
  if bash "$root/scripts/tdp_gen.sh" --source-wires 4 --source-gates 1 --run-directory "$tmp/rejected" "$@" >"$tmp/error" 2>&1; then
    echo "unexpected success: $*" >&2; exit 1
  fi
  test ! -e "$tmp/rejected"
  grep -q 'FATAL:' "$tmp/error"
  if grep -Eq 'unbound variable|missing.*cargo build' "$tmp/error"; then
    cat "$tmp/error" >&2; exit 1
  fi
}
for value in 0 1 -2 2.5 abc 1000000001 999999999999999999999999999999; do
  reject --parallel-target-piece-gates "$value"
done
for value in 0 65 999999999999999999999999999999; do
  reject --parallel-pieces "$value"
done
reject --parallel-pieces 2 --parallel-threads 1025
reject --parallel-pieces 2 --parallel-threads 999999999999999999999999999999
reject --parallel-target-piece-gates 2 --parallel-pieces 1
reject --parallel-pieces 4 --parallel-target-piece-gates 2
reject --parallel-threads 2
# stop and rerun points are 2..7
for stage in 0 1 8; do
  reject --run-stop-after-stage "$stage"
  reject --run-rerun-from-stage "$stage"
done
reject --parallel-target-piece-gates 2 --parallel-threads 0
# the validated schedule is authoritative
reject --db-mixing-hold-work-units 27
reject --db-mixing-target-size-factor 2
reject --db-mixing-schedule custom --db-mixing-target-size-factor 1
for flag in --parallel-target-piece-gates --parallel-pieces --parallel-threads; do
  reject "$flag"
  reject "$flag" --run-stop-after-stage 2
  reject "$flag" ''
done
run_case() {
  local mode=$1; shift
  bash "$root/scripts/tdp_gen.sh" --source-wires 4 --source-gates 1 --run-directory "$tmp/$mode" "$@" >"$tmp/$mode.log" 2>&1 ||
    { cat "$tmp/$mode.log" >&2; exit 1; }
}
# the four-stage chain (stages 4, 5 and 7 skipped) ...
FOUR=(--run-skip-stages 4,5,7)
run_case auto "${FOUR[@]}" --parallel-target-piece-gates 0002 --parallel-threads 3
run_case auto_default "${FOUR[@]}" --parallel-target-piece-gates 2
run_case fixed "${FOUR[@]}" --parallel-pieces 0004 --parallel-threads 0003
run_case serial "${FOUR[@]}"
run_case one "${FOUR[@]}" --parallel-pieces 1
run_case canonical "${FOUR[@]}" --parallel-target-piece-gates 2 --parallel-threads 3 --db-mixing-schedule custom --db-mixing-target-size-factor 2 --db-mixing-hold-work-units 27
run_case stop3 --run-stop-after-stage 3
# ... and the full seven-stage chain, with every stage-5/7 knob set
run_case seven --crossing-target-size-factor 2.5 --crossing-width-penalty-base 4 --crossing-width-penalty-threshold 2 \
  --crossing-size-tolerance-divisor 20 --crossing-move-attempts 1234 --unpacking-time-limit 12.5 --unpacking-restarts 3 \
  --unpacking-beam 2 --unpacking-no-fallback --threads 2
reject --parallel-pieces 4 --parallel-target-piece-gates 2
python3 -I - "$tmp" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
for mode in ('auto', 'auto_default', 'fixed', 'serial', 'one', 'canonical', 'stop3'):
    calls = [json.loads(l) for l in (root / f'{mode}.calls').read_text().splitlines()]
    names = [c['name'] for c in calls]
    stage4 = mode != 'stop3'
    assert names.count('gen_sandwich_gadget') == 1 and names.count('circuit_mixer') == 1, names
    runs = [c for c in calls if c['name'] == 'tdp_packing' and c['args'][0] == 'run']
    assert len(runs) == (1 if stage4 else 0), names
    mixer = next(c['args'] for c in calls if c['name'] == 'circuit_mixer')
    assert mixer[mixer.index('--p-db') + 1] == '0', mixer  # empty-store plumbing mode
    for gone in ('--p-comp', '--p-any', '--resume', '--verify-every', '--report-every', '--state-out'):
        assert gone not in mixer, mixer
    profile = mixer[mixer.index('--profile') + 1]
    moves = int(mixer[mixer.index('--moves') + 1])
    if mode == 'canonical':
        assert profile == '3,30,30,2,2' and moves == int(30.0 * 2.0 * 1 * 1.3), mixer
    else:
        assert profile == '2,2,2,2,2' and moves == int(1 * 2 * 2 * 1.3), mixer
    log = (root / mode / 'tdp_gen.log').read_text()
    if stage4:
        assert 'PIPELINE COMPLETE (stages 1-3,6)' in log, log
        assert (root / mode / 'packing' / 'deliverable.tdp4').is_file()
        assert (root / mode / 'packing.recipe').is_file()
        r = runs[0]['args']
        assert r[:2] == ['run', '--stage2'] and '--out-dir' in r, r
        assert not any(a.startswith(('--window', '--split', '--allow')) for a in r), r  # defaults: no PACKFLAGS
    else:
        assert 'PIPELINE STOPPED after stage 3' in log, log
    assert (root / mode / 'db_mixing.mpmct1').is_file()
    enabled = mode in ('auto', 'auto_default', 'fixed', 'canonical')
    assert ('--parallel-threads' in mixer) == (enabled and mode != 'auto_default'), mixer
    assert ('--parallel-target-piece-gates' in mixer) == (enabled and (mode.startswith('auto') or mode == 'canonical')), mixer
    assert ('--parallel-pieces' in mixer) == (enabled and mode == 'fixed'), mixer
    if enabled and mode != 'auto_default':
        assert mixer[mixer.index('--parallel-threads') + 1] == '3', mixer
    if mode.startswith('auto') or mode == 'canonical':
        assert 'automatic pieces target_piece_gates=2' in log
# the seven-stage chain: stage order, inputs chained, every knob forwarded
calls = [json.loads(l) for l in (root / 'seven.calls').read_text().splitlines()]
names = [c['name'] for c in calls if not (c['name'] == 'tdp_packing' and c['args'][0] != 'run')]
assert names == ['gen_sandwich_gadget', 'circuit_mixer', 'circuit_mixer', 'circuit_mixer', 'tdp_packing', 'tdp_unpacking'], names
base = lambda a: [x.rsplit('/', 1)[-1] for x in a]
s3, s4, s5 = (base(c['args']) for c in calls if c['name'] == 'circuit_mixer')
assert s3[:4] == ['--input', 'tdp.mpmct1', '--output', 'db_mixing.mpmct1'], s3
assert s4[:2] == ['--input', 'db_mixing.mpmct1'] and '--split' in s4 and '--split-stop' in s4, s4
assert s4[s4.index('--p-join') + 1] == '0.8' and s4[s4.index('--k-max') + 1] == '12', s4
assert s4[s4.index('--state-out') + 1] == 'split.state' and s4[s4.index('--output') + 1] == 'split.mpmct1', s4
assert s5[:2] == ['--resume', 'split.state'] and '--input' not in s5 and '--seed' not in s5, s5
assert s5[s5.index('--target-size') + 1] == '5', s5            # round(2 gates * 2.5)
assert s5[s5.index('--temp') + 1] == '64', s5                  # max(64, round(5 / 20))
assert s5[s5.index('--split-base') + 1] == '4' and s5[s5.index('--split-damp') + 1] == '2', s5
assert s5[s5.index('--moves') + 1] == str(7 + 1234), s5        # the state's counter + the budget
assert s5[s5.index('--output') + 1] == 'crossing.mpmct1', s5
pack = base(next(c['args'] for c in calls if c['name'] == 'tdp_packing' and c['args'][0] == 'run'))
assert pack[pack.index('--stage2') + 1] == 'tdp.mpmct1' and pack[pack.index('--stage3') + 1] == 'crossing.mpmct1', pack
assert pack[pack.index('--threads') + 1] == '2', pack
un = base(next(c['args'] for c in calls if c['name'] == 'tdp_unpacking'))
assert un[:2] == ['run', '--deliverable'] and un[un.index('--packed') + 1] == 'rep512.mpmct1', un
assert un[un.index('--time-limit') + 1] == '12.5' and un[un.index('--restarts') + 1] == '3', un
assert un[un.index('--beam') + 1] == '2' and '--no-fallback' in un and un[un.index('--threads') + 1] == '2', un
log = (root / 'seven' / 'tdp_gen.log').read_text()
assert 'PIPELINE COMPLETE (stages 1-7)' in log, log
for f in ('split.mpmct1', 'split.state', 'crossing.mpmct1', 'stage4.recipe', 'stage5.recipe', 'packing.recipe', 'unpacking.recipe', 'unpacking/unpacked.mpmct1'):
    assert (root / 'seven' / f).is_file(), f
PY
# Continuation must recognize every saved artifact and avoid another call.
rm "$tmp/canonical.calls"
run_case canonical "${FOUR[@]}" --parallel-target-piece-gates 2 --parallel-threads 3 --db-mixing-schedule custom --db-mixing-target-size-factor 2 --db-mixing-hold-work-units 27
test ! -e "$tmp/canonical.calls"
grep -q 'matches its completion marker, skipping' "$tmp/canonical.log"
# ... while dropping the skip list resumes the same run with stages 4-7 added
run_case canonical --parallel-target-piece-gates 2 --parallel-threads 3 --db-mixing-schedule custom --db-mixing-target-size-factor 2 --db-mixing-hold-work-units 27
grep -q 'stage 3: .* matches its completion marker, skipping' "$tmp/canonical.log"
grep -q 'PIPELINE COMPLETE (stages 1-7)' "$tmp/canonical.log"
python3 -I -c "import json,sys; names=[json.loads(l)['name'] for l in open(sys.argv[1])]; assert names == ['circuit_mixer', 'circuit_mixer', 'tdp_packing', 'tdp_unpacking'], names" "$tmp/canonical.calls"
echo 'TDP block-size shell smoke tests passed'
