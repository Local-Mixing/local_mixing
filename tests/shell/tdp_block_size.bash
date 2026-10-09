#!/usr/bin/env bash
# Fast plumbing smoke test of the four-stage driver: stage-3 piece flags,
# stop/rerun points, the stage-3 schedule, and the stage-4 call. Mock binaries
# (tests/shell/lib/tdp_mock.py) avoid mixing or release builds.
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
for binary in gen_sandwich_gadget circuit_mixer tdp_packing; do
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
# The old stages 5-6 (crossing walk, fcompress) are historic: neither stop nor
# rerun points, and the crossing walk's flags are refused. Stage 4 is packing.
for stage in 1 5 6; do
  reject --run-stop-after-stage "$stage"
  reject --run-rerun-from-stage "$stage"
done
grep -q 'historic' "$tmp/error"
reject --crossing-target-size-factor 2
reject --xmoves 100
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
run_case auto --parallel-target-piece-gates 0002 --parallel-threads 3
run_case auto_default --parallel-target-piece-gates 2
run_case fixed --parallel-pieces 0004 --parallel-threads 0003
run_case serial
run_case one --parallel-pieces 1
run_case canonical --parallel-target-piece-gates 2 --parallel-threads 3 --db-mixing-schedule custom --db-mixing-target-size-factor 2 --db-mixing-hold-work-units 27
run_case stop3 --run-stop-after-stage 3
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
        assert 'PIPELINE COMPLETE (stages 1-4)' in log, log
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
PY
# Continuation must recognize every saved artifact and avoid another call.
rm "$tmp/canonical.calls"
run_case canonical --parallel-target-piece-gates 2 --parallel-threads 3 --db-mixing-schedule custom --db-mixing-target-size-factor 2 --db-mixing-hold-work-units 27
test ! -e "$tmp/canonical.calls"
grep -q 'matches its completion marker, skipping' "$tmp/canonical.log"
echo 'TDP block-size shell smoke tests passed'
