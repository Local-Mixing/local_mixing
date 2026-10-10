#!/usr/bin/env bash
# Exercise the public driver's stage-2 surface without release builds or real
# mixing: presets, custom knobs, the inherited-environment policy, and the
# stage-2 marker. Mock binaries record argv + environment (tests/shell/lib/tdp_mock.py).
set -euo pipefail
# Run with a clean environment so inherited driver controls cannot affect cases.
if [[ ${1-} != --isolated-environment ]]; then
  exec env -i PATH="$PATH" HOME="${HOME:-/tmp}" bash "$0" --isolated-environment
fi
shift
root=$(cd "$(dirname "$0")/../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export TDP_BIN_DIR="$tmp/bin"
mkdir -p "$TDP_BIN_DIR"
for binary in gen_sandwich_gadget circuit_mixer tdp_packing; do
  ln -s "$root/tests/shell/lib/tdp_mock.py" "$TDP_BIN_DIR/$binary"
done
reject() { # reject PATTERN ARGS... : fails, with a FATAL line matching PATTERN, creating nothing
  local pattern=$1; shift
  if bash "$root/scripts/tdp_gen.sh" --source-wires 4 --source-gates 1 --run-directory "$tmp/rejected" --run-stop-after-stage 2 "$@" >"$tmp/error" 2>&1; then
    echo "unexpected success: $*" >&2; exit 1
  fi
  test ! -e "$tmp/rejected"
  grep -q 'FATAL:' "$tmp/error" || { cat "$tmp/error" >&2; exit 1; }
  grep -Eq -- "$pattern" "$tmp/error" || { echo "expected /$pattern/ for: $*" >&2; cat "$tmp/error" >&2; exit 1; }
  if grep -Eq 'unbound variable|missing.*cargo build|integer expression expected' "$tmp/error"; then
    cat "$tmp/error" >&2; exit 1
  fi
}
for mode in unknown-mode embedded_masking EMBEDDED-MASKING; do
  reject 'preprocessing-mode' --preprocessing-mode "$mode"
done
knobs=(mask-pair-wires max-open-masks min-open-masks balanced-masks shuffling-segments
       shuffling-return-home premask seed-avoid-c1 chain chain-len)
for k in "${knobs[@]}"; do
  flag=--preprocessing-$k
  reject 'requires an argument' "$flag"
  reject 'requires an argument' "$flag" ''
  # presets are authoritative: any knob next to L8 (default) or A is fatal
  reject "is set by preset L8" "$flag" 1
  reject "is set by preset A" --preprocessing-preset A "$flag" 1
  reject 'only valid with --preprocessing-mode embedded-masking' --preprocessing-mode nonlinear291 "$flag" 2
  reject 'must be' --preprocessing-preset custom "$flag" 999999999999999999999999999
done
C=(--preprocessing-preset custom)
for value in 0 1 65 2.5 abc; do reject 'mask-pair-wires' "${C[@]}" --preprocessing-mask-pair-wires "$value"; done
for value in 0 1 65 abc; do reject 'max-open-masks' "${C[@]}" --preprocessing-max-open-masks "$value"; done
for value in 0 64 abc; do reject 'min-open-masks' "${C[@]}" --preprocessing-min-open-masks "$value"; done
for value in 2 abc; do reject 'balanced-masks' "${C[@]}" --preprocessing-balanced-masks "$value"; done
for value in 1 7 abc 8.5 -8 9223372036854775808 9999999999999999999; do
  reject 'shuffling-segments' "${C[@]}" --preprocessing-shuffling-segments "$value"
done
for value in 2 abc; do reject 'return-home' "${C[@]}" --preprocessing-shuffling-return-home "$value"; done
for value in 2 abc; do reject 'premask' "${C[@]}" --preprocessing-premask "$value"; done
for value in 2 abc; do reject 'seed-avoid-c1' "${C[@]}" --preprocessing-seed-avoid-c1 "$value"; done
for value in 65 abc; do reject 'chain' "${C[@]}" --preprocessing-chain "$value"; done
reject 'max-open-masks' "${C[@]}" --preprocessing-max-open-masks 2
reject 'less than' "${C[@]}" --preprocessing-min-open-masks 4 --preprocessing-max-open-masks 3
reject '2 \* N > K' "${C[@]}" --preprocessing-mask-pair-wires 8
reject 'preset' --preprocessing-preset L9
reject 'applies only to embedded-masking' --preprocessing-mode nonlinear291 --preprocessing-preset L8
reject 'source-wires' --source-wires 2
reject 'source-wires' --source-wires 4096
# packing budgets are resource bounds (MAX_MONOMIAL_BUDGET = 2^18)
reject 'packing-monomial-budget' --packing-monomial-budget 262145
reject 'packing-split-budget' --packing-split-budget 262145
# nonlinear291 wider than stage 4's 1024 state wires is refused before stage 1
# (reject() stops after stage 2, so this case runs with the default stop)
if bash "$root/scripts/tdp_gen.sh" --source-wires 16 --source-gates 64 --preprocessing-mode nonlinear291 \
     --run-directory "$tmp/nl-wide" >"$tmp/error" 2>&1; then
  echo "unexpected success: nonlinear291 n=16 with stage 4" >&2; exit 1
fi
test ! -e "$tmp/nl-wide"
grep -q 'FATAL: nonlinear291 makes a 3677-wire stage-2 circuit; stage 4 supports at most 1024' "$tmp/error" ||
  { cat "$tmp/error" >&2; exit 1; }
# inherited stage-2 variables are refused, even with explicit flags (§11.5)
EMBEDDED_MASKING_K=1 reject 'EMBEDDED_MASKING_K is not honoured'
EMBEDDED_MASKING_SHUFFLING=8 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=false reject \
  'EMBEDDED_MASKING_SHUFFLING, EMBEDDED_MASKING_SHUFFLING_RETURN_HOME are not honoured'
EMBEDDED_MASKING_CHAIN=2 reject 'not honoured' "${C[@]}" --preprocessing-chain 2
EMBEDDED_MASKING_SHUFFLING=0 reject 'not honoured' --preprocessing-mode nonlinear291
EMBEDDED_MASKING_RETIRE=4 reject 'EMBEDDED_MASKING_RETIRE is not honoured'
BV5_K=2 reject 'BV5_K is not honoured'
PROD_K=2 reject 'PROD_K is not honoured'
for flag in --unknown-option --preprocessing-unknown; do
  reject 'unknown argument' "$flag" 2
done
reject 'duplicate' --preprocessing-mode embedded-masking --preprocessing-mode embedded-masking
reject 'duplicate' "${C[@]}" --preprocessing-chain 1 --preprocessing-chain 1
reject 'seeds on argv' -s 42
reject 'seeds on argv' --calibration-seed 42

run_case() {
  local label=$1; shift
  if ! bash "$root/scripts/tdp_gen.sh" --source-wires 4 --run-directory "$tmp/$label" --source-gates 1 --run-stop-after-stage 2 "$@" >"$tmp/$label.log" 2>&1; then
    cat "$tmp/$label.log" >&2; exit 1
  fi
}
run_case defaults
run_case preset-L8 --preprocessing-preset L8
run_case preset-A --preprocessing-preset A
run_case custom-base "${C[@]}"
run_case custom-masks "${C[@]}" --preprocessing-mask-pair-wires 0004 --preprocessing-max-open-masks 0004 --preprocessing-min-open-masks 0001 --preprocessing-balanced-masks 00
run_case custom-narrow "${C[@]}" --preprocessing-max-open-masks 2 --preprocessing-min-open-masks 1
run_case custom-l8 "${C[@]}" --preprocessing-shuffling-segments 8 --preprocessing-shuffling-return-home 0 --preprocessing-premask 1 --preprocessing-seed-avoid-c1 1 --preprocessing-chain 2 --preprocessing-chain-len 8
run_case custom-home "${C[@]}" --preprocessing-shuffling-segments 0008 --preprocessing-shuffling-return-home true
run_case nonlinear291 --preprocessing-mode nonlinear291
python3 -I - "$tmp" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
L8 = {'EMBEDDED_MASKING_SHUFFLING': '8', 'EMBEDDED_MASKING_SHUFFLING_RETURN_HOME': '0',
      'EMBEDDED_MASKING_PREMASK': '1', 'EMBEDDED_MASKING_SEED_AVOID_C1': '1',
      'EMBEDDED_MASKING_CHAIN': '2', 'EMBEDDED_MASKING_CHAIN_LEN': '8'}
A = {k: v for k, v in L8.items() if 'CHAIN' not in k}
expected = {
    'defaults': (L8, 'L8'),
    'preset-L8': (L8, 'L8'),
    'preset-A': (A, 'A'),
    'custom-base': ({}, 'custom'),
    'custom-masks': ({'EMBEDDED_MASKING_K': '4', 'EMBEDDED_MASKING_MAX_OPEN': '4',
                      'EMBEDDED_MASKING_MIN_OPEN': '1', 'EMBEDDED_MASKING_BALANCED': '0'}, 'custom'),
    'custom-narrow': ({'EMBEDDED_MASKING_MAX_OPEN': '2', 'EMBEDDED_MASKING_MIN_OPEN': '1'}, 'custom'),
    'custom-l8': (L8, 'custom'),
    'custom-home': ({'EMBEDDED_MASKING_SHUFFLING': '8'}, 'custom'),
    'nonlinear291': ({}, 'none'),
}
for name, (env, preset) in expected.items():
    calls = [json.loads(l) for l in (root / f'{name}.calls').read_text().splitlines()]
    gen = [c for c in calls if c['name'] == 'gen_sandwich_gadget']
    assert len(gen) == 1, (name, calls)
    gen = gen[0]
    got = {k: v for k, v in gen['env'].items() if k.startswith('EMBEDDED_MASKING_')}
    assert got == env, (name, got)
    # no other recipe variable reaches gen
    assert not any(k.startswith(('SANDWICH_', 'TDP_SOURCE', 'SLICE_GUARD', 'PROD_')) for k in gen['env']), gen['env']
    mode = 'nonlinear291' if name.startswith('nonlinear') else 'embedded-masking'
    args = gen['args']
    assert args[1:] == ['4', '1', '1', '8', '1', '80'] + [args[7]] * 3 + (['nonlinear291'] if mode == 'nonlinear291' else []), args
    # the seeds reach gen through descriptors, never on its argv
    assert gen['argv'][7:10] == ['@/dev/fd/3', '@/dev/fd/4', '@/dev/fd/5'], gen['argv']
    recipe = (root / name / 'stage12.recipe').read_text()
    assert recipe.startswith('tdp_stage12_recipe=7\n'), recipe
    assert f'preprocessing_mode={mode}\n' in recipe, recipe
    assert f'preprocessing_preset={preset}\n' in recipe, recipe
    assert 'seed=' not in recipe.replace('seeds_mac=', ''), recipe
PY
# A same-driver restart must skip the saved stage-2 artifact and retain its marker.
cp "$tmp/defaults/stage12.recipe" "$tmp/original.recipe"
rm "$tmp/defaults.calls"
run_case defaults
if [ -e "$tmp/defaults.calls" ] && grep -q gen_sandwich_gadget "$tmp/defaults.calls"; then
  echo 'restart regenerated stage 2' >&2; exit 1
fi
cmp "$tmp/original.recipe" "$tmp/defaults/stage12.recipe"
# A changed stage-2 recipe must fail before regenerating or replacing its marker.
# (each case replaces the base --source-gates 1, so a changed |C| reaches the
# recipe comparison instead of the duplicate-flag check)
for changed in "--source-gates 1 --preprocessing-mode nonlinear291" "--source-gates 1 --preprocessing-preset A" \
               "--source-gates 1 --preprocessing-preset custom" "--source-gates 2"; do
  # shellcheck disable=SC2086
  if bash "$root/scripts/tdp_gen.sh" --source-wires 4 --run-directory "$tmp/defaults" --run-stop-after-stage 2 $changed >"$tmp/changed.log" 2>&1; then
    echo "changed preprocessing recipe unexpectedly resumed: $changed" >&2; exit 1
  fi
  grep -q 'stage-2 recipe.*does not match' "$tmp/changed.log" || { echo "expected a recipe mismatch for: $changed" >&2; cat "$tmp/changed.log" >&2; exit 1; }
  if [ -e "$tmp/defaults.calls" ] && grep -q gen_sandwich_gadget "$tmp/defaults.calls"; then
    echo 'changed recipe regenerated stage 2' >&2; exit 1
  fi
  cmp "$tmp/original.recipe" "$tmp/defaults/stage12.recipe"
done
echo 'TDP preprocessing flag shell regression tests passed'
