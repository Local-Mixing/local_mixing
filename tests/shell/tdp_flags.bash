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
for binary in gen_sandwich_gadget circuit_mixer tdp_packing tdp_adaptive; do
  ln -s "$root/tests/shell/lib/tdp_mock.py" "$TDP_BIN_DIR/$binary"
done
printf '0 1 2\n' > "$tmp/src.g57"
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
E=(--preprocessing-mode embedded-masking)
for k in "${knobs[@]}"; do
  flag=--preprocessing-$k
  reject 'requires an argument' "$flag"
  reject 'requires an argument' "$flag" ''
  # presets are authoritative: any knob next to L8 (embedded masking's default) or A is fatal
  reject "is set by preset L8" "${E[@]}" "$flag" 1
  reject "is set by preset A" "${E[@]}" --preprocessing-preset A "$flag" 1
  reject 'only valid with --preprocessing-mode embedded-masking' --preprocessing-mode nonlinear291 "$flag" 2
  # ... and under the default mode (three-mask) too
  reject 'only valid with --preprocessing-mode embedded-masking' "$flag" 2
  reject 'must be' "${E[@]}" --preprocessing-preset custom "$flag" 999999999999999999999999999
done
C=(--preprocessing-mode embedded-masking --preprocessing-preset custom)
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
reject 'preset' "${E[@]}" --preprocessing-preset L9
reject 'applies only to embedded-masking' --preprocessing-mode nonlinear291 --preprocessing-preset L8
reject 'applies only to embedded-masking' --preprocessing-preset L8
# the three-mask stage 2: its knobs only under its mode, no preset, no embedded-masking knob
reject 'applies only to embedded-masking' --preprocessing-mode three-mask --preprocessing-preset L8
reject 'only valid with --preprocessing-mode embedded-masking' --preprocessing-mode three-mask --preprocessing-chain 2
reject 'only valid with --preprocessing-mode three-mask' "${E[@]}" --three-mask-banks 8
reject 'only valid with --preprocessing-mode three-mask' --preprocessing-mode nonlinear291 --three-mask-growth 8
reject 'unknown argument' --preprocessing-mode three-mask --three-mask-invented 1
reject 'requires an argument' --preprocessing-mode three-mask --three-mask-banks
for value in 48 72 abc; do reject 'band-wires' --preprocessing-mode three-mask --three-mask-band-wires "$value"; done
for value in 2 7; do reject 'banks' --preprocessing-mode three-mask --three-mask-banks "$value"; done
reject 'max-retiring' --preprocessing-mode three-mask --three-mask-banks 8 --three-mask-max-retiring 4
for value in 0 4; do reject 'mask-count' --preprocessing-mode three-mask --three-mask-mask-count "$value"; done
reject 'random-helpers' --preprocessing-mode three-mask --three-mask-random-helpers 2
# the source construction
reject 'source-construction' --source-construction tidy
reject 'exclusive' --source-construction balanced --source-path "$tmp/src.g57"
# the stage-6 mode decides which parameter family applies
reject 'delivery-mode' --delivery-mode invented
reject 'applies only to --delivery-mode windowed' --delivery-anf-only
reject 'applies only to --delivery-mode windowed' --packing-window-gates 256
reject 'applies only to --delivery-mode windowed' --delivery-restarts 2
reject 'applies only to --delivery-mode adaptive' --delivery-mode windowed --adaptive-max-rounds 3
reject 'applies only to --delivery-mode adaptive' --delivery-mode windowed --adaptive-no-canonical-order
reject 'adaptive-max-rounds' --adaptive-max-rounds 1001
reject 'adaptive-stop-fraction' --adaptive-stop-fraction 1
reject 'adaptive-stop-fraction' --adaptive-stop-fraction -0.1
reject 'adaptive-seeds-per-block' --adaptive-seeds-per-block 0
reject 'adaptive-seeds-per-block' --adaptive-seeds-per-block 17
reject 'adaptive-time-limit' --adaptive-time-limit 0
reject 'adaptive-restarts' --adaptive-restarts 0
reject 'adaptive-beam' --adaptive-beam abc
reject 'adaptive-term-cap' --adaptive-term-cap 63
reject 'adaptive-max-block-gates' --adaptive-max-block-gates 1
reject 'adaptive-segments' --adaptive-segments 0
reject 'adaptive-refinement-levels' --adaptive-refinement-levels 17
reject 'adaptive-min-refine-gates' --adaptive-min-refine-gates 0
reject 'adaptive-census-batches' --adaptive-census-batches 0
reject 'adaptive-accept' --adaptive-accept largest
reject 'requires an argument' --adaptive-max-rounds
reject 'source-wires' --source-wires 2
reject 'source-wires' --source-wires 4096
# packing budgets are resource bounds (MAX_MONOMIAL_BUDGET = 2^18)
reject 'packing-monomial-budget' --delivery-mode windowed --packing-monomial-budget 262145
reject 'packing-split-budget' --delivery-mode windowed --packing-split-budget 262145
# nonlinear291 wider than stage 6's 1024 state wires is refused before stage 1
# (reject() stops after stage 2, so this case runs with the default stop);
# skipping stage 6 (delivery) lifts the limit
if bash "$root/scripts/tdp_gen.sh" --source-wires 16 --source-gates 64 --preprocessing-mode nonlinear291 \
     --run-directory "$tmp/nl-wide" >"$tmp/error" 2>&1; then
  echo "unexpected success: nonlinear291 n=16 with stage 6" >&2; exit 1
fi
test ! -e "$tmp/nl-wide"
grep -q 'FATAL: nonlinear291 makes a 3677-wire stage-2 circuit; stage 6 (delivery) supports at most 1024' "$tmp/error" ||
  { cat "$tmp/error" >&2; exit 1; }
if ! bash "$root/scripts/tdp_gen.sh" --source-wires 16 --source-gates 64 --preprocessing-mode nonlinear291 \
     --run-directory "$tmp/nl-wide" --run-skip-stages 6 --run-stop-after-stage 2 >"$tmp/error" 2>&1; then
  cat "$tmp/error" >&2; exit 1
fi
# the optional-stage surface: stages 2..6 (the former stage 7 is part of 6)
reject 'run-skip-stages' --run-skip-stages 1
reject 'run-skip-stages' --run-skip-stages 7
reject 'run-skip-stages' --run-skip-stages 8
reject 'run-skip-stages' --run-skip-stages 4,,5
reject 'run-skip-stages' --run-skip-stages 4,5,
reject 'run-skip-stages' --run-skip-stages ,4
reject 'run-skip-stages' --run-skip-stages ''
reject 'run-stop-after-stage' --run-stop-after-stage 7
reject 'run-rerun-from-stage' --run-rerun-from-stage 7
reject 'crossing-target-size-factor' --crossing-target-size-factor 0.5
reject 'crossing-target-size-factor' --crossing-target-size-factor 17
reject 'crossing-width-penalty-base' --crossing-width-penalty-base 0
reject 'crossing-size-tolerance-divisor' --crossing-size-tolerance-divisor 0
reject 'crossing-move-attempts' --crossing-move-attempts 0
reject 'crossing-move-budget-factor' --crossing-move-budget-factor 0
reject 'crossing-move-budget-factor' --crossing-move-budget-factor 1001
reject 'exclusive' --crossing-move-budget-factor 2 --crossing-move-attempts 5
W=(--delivery-mode windowed)
reject 'delivery-time-limit' "${W[@]}" --delivery-time-limit 0
reject 'delivery-restarts' "${W[@]}" --delivery-restarts 0
reject 'delivery-beam' "${W[@]}" --delivery-beam abc
reject 'delivery-max-rounds' "${W[@]}" --delivery-max-rounds 0
reject 'delivery-max-rounds' "${W[@]}" --delivery-max-rounds 1001
reject 'requires an argument' "${W[@]}" --delivery-max-rounds
# the seven-stage driver's unpacking flags point at their delivery names
reject 'is now --delivery-time-limit' --unpacking-time-limit 5
reject 'is now --delivery-restarts' --unpacking-restarts 2
reject 'is gone' --unpacking-no-fallback
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
run_case embedded "${E[@]}"
run_case preset-L8 "${E[@]}" --preprocessing-preset L8
run_case preset-A "${E[@]}" --preprocessing-preset A
run_case custom-base "${C[@]}"
run_case custom-masks "${C[@]}" --preprocessing-mask-pair-wires 0004 --preprocessing-max-open-masks 0004 --preprocessing-min-open-masks 0001 --preprocessing-balanced-masks 00
run_case custom-narrow "${C[@]}" --preprocessing-max-open-masks 2 --preprocessing-min-open-masks 1
run_case custom-l8 "${C[@]}" --preprocessing-shuffling-segments 8 --preprocessing-shuffling-return-home 0 --preprocessing-premask 1 --preprocessing-seed-avoid-c1 1 --preprocessing-chain 2 --preprocessing-chain-len 8
run_case custom-home "${C[@]}" --preprocessing-shuffling-segments 0008 --preprocessing-shuffling-return-home true
run_case nonlinear291 --preprocessing-mode nonlinear291
run_case three-mask --preprocessing-mode three-mask
run_case three-mask-knobs --preprocessing-mode three-mask --three-mask-band-wires 0256 --three-mask-unique-cover 0 --three-mask-banks 8 --three-mask-max-retiring 3
run_case balanced --source-construction balanced
python3 -I - "$tmp" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
L8 = {'EMBEDDED_MASKING_SHUFFLING': '8', 'EMBEDDED_MASKING_SHUFFLING_RETURN_HOME': '0',
      'EMBEDDED_MASKING_PREMASK': '1', 'EMBEDDED_MASKING_SEED_AVOID_C1': '1',
      'EMBEDDED_MASKING_CHAIN': '2', 'EMBEDDED_MASKING_CHAIN_LEN': '8'}
A = {k: v for k, v in L8.items() if 'CHAIN' not in k}
expected = {
    'defaults': ({}, 'none'),
    'embedded': (L8, 'L8'),
    'preset-L8': (L8, 'L8'),
    'preset-A': (A, 'A'),
    'custom-base': ({}, 'custom'),
    'custom-masks': ({'EMBEDDED_MASKING_K': '4', 'EMBEDDED_MASKING_MAX_OPEN': '4',
                      'EMBEDDED_MASKING_MIN_OPEN': '1', 'EMBEDDED_MASKING_BALANCED': '0'}, 'custom'),
    'custom-narrow': ({'EMBEDDED_MASKING_MAX_OPEN': '2', 'EMBEDDED_MASKING_MIN_OPEN': '1'}, 'custom'),
    'custom-l8': (L8, 'custom'),
    'custom-home': ({'EMBEDDED_MASKING_SHUFFLING': '8'}, 'custom'),
    'nonlinear291': ({}, 'none'),
    'three-mask': ({}, 'none'),
    'three-mask-knobs': ({'THREE_MASK_BAND_WIRES': '256', 'THREE_MASK_UNIQUE_COVER': '0', 'THREE_MASK_BANKS': '8', 'THREE_MASK_MAX_RETIRING': '3'}, 'none'),
    'balanced': ({}, 'none'),
}
for name, (env, preset) in expected.items():
    calls = [json.loads(l) for l in (root / f'{name}.calls').read_text().splitlines()]
    gen = [c for c in calls if c['name'] == 'gen_sandwich_gadget']
    assert len(gen) == 1, (name, calls)
    gen = gen[0]
    got = {k: v for k, v in gen['env'].items() if k.startswith(('EMBEDDED_MASKING_', 'THREE_MASK_'))}
    assert got == env, (name, got)
    # no other recipe variable reaches gen (the balanced construction travels as TDP_SOURCE_CONSTRUCTION)
    assert not any(k.startswith(('SANDWICH_', 'SLICE_GUARD', 'PROD_')) or k == 'TDP_SOURCE_C' for k in gen['env']), gen['env']
    assert gen['env'].get('TDP_SOURCE_CONSTRUCTION') == ('balanced' if name == 'balanced' else None), (name, gen['env'])
    mode = 'nonlinear291' if name.startswith('nonlinear') else 'embedded-masking' if name in ('embedded', 'preset-L8', 'preset-A') or name.startswith('custom') else 'three-mask'
    args = gen['args']
    assert args[1:] == ['4', '1', '1', '8', '1', '80'] + [args[7]] * 3 + ([mode] if mode != 'embedded-masking' else []), args
    # the seeds reach gen through descriptors, never on its argv
    assert gen['argv'][7:10] == ['@/dev/fd/3', '@/dev/fd/4', '@/dev/fd/5'], gen['argv']
    recipe = (root / name / 'stage12.recipe').read_text()
    assert recipe.startswith('tdp_stage12_recipe=8\n'), recipe
    assert f'preprocessing_mode={mode}\n' in recipe, recipe
    assert f'preprocessing_preset={preset}\n' in recipe, recipe
    assert f"source_construction={'balanced' if name == 'balanced' else 'random'}\n" in recipe, recipe
    if mode == 'three-mask':
        assert f"three_mask_banks={'8' if 'knobs' in name else '0'}\n" in recipe and 'three_mask_outer_blocks=L8\n' in recipe, recipe
        assert f"three_mask_band_wires={'256' if 'knobs' in name else '0'}\n" in recipe, recipe
        assert 'preprocessing_mask_pair_wires' not in recipe, recipe
    else:
        assert 'three_mask_' not in recipe, recipe
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
for changed in "--source-gates 1 --preprocessing-mode nonlinear291" "--source-gates 1 --preprocessing-mode embedded-masking" \
               "--source-gates 1 --preprocessing-mode embedded-masking --preprocessing-preset A" "--source-gates 2" \
               "--source-gates 1 --three-mask-banks 8" "--source-gates 1 --source-construction balanced"; do
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
