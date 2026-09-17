#!/usr/bin/env bash
# Exercise the public driver without release builds or real mixing.
set -euo pipefail
# Run with a clean environment so inherited driver controls cannot affect cases.
if [[ ${1-} != --isolated-environment ]]; then
  exec env -i PATH="$PATH" bash "$0" --isolated-environment
fi
shift
root=$(cd "$(dirname "$0")/../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export TDP_BIN_DIR="$tmp/bin"
mkdir -p "$TDP_BIN_DIR"
reject() {
  if bash "$root/scripts/tdp_gen.sh" --source-wires 4 --run-directory "$tmp/rejected" --run-stop-after-stage 2 "$@" >"$tmp/error" 2>&1; then
    echo "unexpected success: $*" >&2; exit 1
  fi
  test ! -e "$tmp/rejected"
  grep -q 'FATAL:' "$tmp/error"
  if grep -Eq 'unbound variable|missing.*cargo build|integer expression expected' "$tmp/error"; then
    cat "$tmp/error" >&2; exit 1
  fi
}
for mode in unknown-mode embedded_masking EMBEDDED-MASKING; do
  reject --preprocessing-mode "$mode"
done
for flag in --preprocessing-mask-pair-wires --preprocessing-max-open-masks --preprocessing-min-open-masks --preprocessing-balanced-masks --preprocessing-shuffling-segments --preprocessing-shuffling-return-home; do
  reject "$flag"
  reject "$flag" --run-stop-after-stage 2
  reject "$flag" ''
  reject "$flag" 999999999999999999999999999
  reject --preprocessing-mode nonlinear291 "$flag" 2
done
for value in 0 1 65 2.5 abc; do reject --preprocessing-mask-pair-wires "$value"; done
for value in 0 1 65 abc; do reject --preprocessing-max-open-masks "$value"; done
for value in 0 64 abc; do reject --preprocessing-min-open-masks "$value"; done
for value in 2 abc; do reject --preprocessing-balanced-masks "$value"; done
for value in 1 7 abc 8.5 -8 9223372036854775808 9999999999999999999; do reject --preprocessing-shuffling-segments "$value"; done
for value in 2 abc; do reject --preprocessing-shuffling-return-home "$value"; done
reject --preprocessing-shuffling-segments 8 --preprocessing-shuffling-return-home 0
EMBEDDED_MASKING_SHUFFLING=8 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=false reject
EMBEDDED_MASKING_SHUFFLING=0 reject --preprocessing-mode nonlinear291
reject --preprocessing-max-open-masks 2
reject --preprocessing-min-open-masks 4 --preprocessing-max-open-masks 3
reject --preprocessing-mask-pair-wires 8
reject --source-wires 2
reject --source-wires 4096
EMBEDDED_MASKING_K=1 reject
EMBEDDED_MASKING_MAX_OPEN=1 reject
EMBEDDED_MASKING_MIN_OPEN=4 reject
EMBEDDED_MASKING_BALANCED=2 reject
for flag in --unknown-option --preprocessing-unknown; do
  reject "$flag" 2
done
reject --preprocessing-mode embedded-masking --preprocessing-mode embedded-masking
for suffix in mask-pair-wires max-open-masks min-open-masks balanced-masks shuffling-segments shuffling-return-home; do
  reject "--preprocessing-$suffix" 1 "--preprocessing-$suffix" 1
done
cat > "$TDP_BIN_DIR/mock" <<'MOCK'
#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
assert Path(sys.argv[0]).name == 'gen_sandwich_gadget', sys.argv
keys = ('EMBEDDED_MASKING_K', 'EMBEDDED_MASKING_MAX_OPEN', 'EMBEDDED_MASKING_MIN_OPEN', 'EMBEDDED_MASKING_BALANCED', 'EMBEDDED_MASKING_SHUFFLING', 'EMBEDDED_MASKING_SHUFFLING_RETURN_HOME')
Path(os.environ['MOCK_CALLS']).write_text(json.dumps({
    'args': sys.argv[1:], 'env': {key: os.environ.get(key) for key in keys},
}))
Path(sys.argv[1]).write_text('4\nx\nx\n')
Path(sys.argv[1] + '.sandwich.mpmct1').write_text('4\nx\nx\n')
MOCK
chmod +x "$TDP_BIN_DIR/mock"
for binary in gen_sandwich_gadget circuit_mixer fcompress; do
  ln -s mock "$TDP_BIN_DIR/$binary"
done
run_case() {
  local label=$1; shift
  export MOCK_CALLS="$tmp/$label.calls"
  if ! bash "$root/scripts/tdp_gen.sh" --source-wires 4 --run-directory "$tmp/$label" --calibration-seed 42 --source-gates 1 --run-stop-after-stage 2 "$@" >"$tmp/$label.log" 2>&1; then
    cat "$tmp/$label.log" >&2; exit 1
  fi
}
run_case defaults
run_case embedded-masking --preprocessing-mode embedded-masking --preprocessing-mask-pair-wires 0004 --preprocessing-max-open-masks 0004 --preprocessing-min-open-masks 0001 --preprocessing-balanced-masks 00
run_case narrow --preprocessing-max-open-masks 2 --preprocessing-min-open-masks 1
run_case canonical --preprocessing-mode embedded-masking --preprocessing-mask-pair-wires 4 --preprocessing-max-open-masks 4 --preprocessing-min-open-masks 1 --preprocessing-balanced-masks 0
EMBEDDED_MASKING_K=4 EMBEDDED_MASKING_MAX_OPEN=4 EMBEDDED_MASKING_MIN_OPEN=1 EMBEDDED_MASKING_BALANCED=0 run_case environment
EMBEDDED_MASKING_K=1 EMBEDDED_MASKING_MAX_OPEN=1 EMBEDDED_MASKING_MIN_OPEN=9 EMBEDDED_MASKING_BALANCED=2 run_case precedence --preprocessing-mask-pair-wires 2 --preprocessing-max-open-masks 3 --preprocessing-min-open-masks 2 --preprocessing-balanced-masks 1
run_case nonlinear291 --preprocessing-mode nonlinear291
run_case shuffled --preprocessing-shuffling-segments 0008
EMBEDDED_MASKING_SHUFFLING=16 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=true run_case shuffled-environment
EMBEDDED_MASKING_SHUFFLING=7 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=0 run_case shuffled-precedence --preprocessing-shuffling-segments 8 --preprocessing-shuffling-return-home 1
python3 -I - "$tmp" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
expected = {
    'defaults': ['2', '3', '2', '1', '0', '1'],
    'embedded-masking': ['4', '4', '1', '0', '0', '1'],
    'narrow': ['2', '2', '1', '1', '0', '1'],
    'canonical': ['4', '4', '1', '0', '0', '1'],
    'environment': ['4', '4', '1', '0', '0', '1'],
    'precedence': ['2', '3', '2', '1', '0', '1'],
    'nonlinear291': [None] * 6,
    'shuffled': ['2', '3', '2', '1', '8', '1'],
    'shuffled-environment': ['2', '3', '2', '1', '16', '1'],
    'shuffled-precedence': ['2', '3', '2', '1', '8', '1'],
}
for name, values in expected.items():
    call = json.loads((root / f'{name}.calls').read_text())
    mode = name if name.startswith('nonlinear') else 'embedded-masking'
    assert call['args'][-1] == mode, call
    assert list(call['env'].values()) == values, call
    recipe = (root / name / 'stage12.recipe').read_text()
    assert recipe.startswith('tdp_stage12_recipe=6\n'), recipe
    assert f'preprocessing_mode={mode}\n' in recipe, recipe
    if mode == 'embedded-masking':
        assert f'shuffling_segments={values[4]}\n' in recipe, recipe
        assert f'shuffling_return_home={values[5]}\n' in recipe, recipe
PY
# A same-driver restart must skip the saved stage-2 artifact and retain its marker.
cp "$tmp/defaults/stage12.recipe" "$tmp/original.recipe"
rm "$tmp/defaults.calls"
run_case defaults
test ! -e "$tmp/defaults.calls"
cmp "$tmp/original.recipe" "$tmp/defaults/stage12.recipe"
# A changed stage-2 recipe must fail before regenerating or replacing its marker.
for changed_setting in mode mask shuffling; do
  changed_flags=(--preprocessing-mode nonlinear291)
  if [ "$changed_setting" = mask ]; then
    changed_flags=(--preprocessing-mask-pair-wires 4)
  fi
  if [ "$changed_setting" = shuffling ]; then
    changed_flags=(--preprocessing-shuffling-segments 8)
  fi
  if bash "$root/scripts/tdp_gen.sh" --source-wires 4 --run-directory "$tmp/defaults" --calibration-seed 42 --source-gates 1 --run-stop-after-stage 2 "${changed_flags[@]}" >"$tmp/changed.log" 2>&1; then
    echo 'changed preprocessing recipe unexpectedly resumed' >&2; exit 1
  fi
  test ! -e "$tmp/defaults.calls"
  cmp "$tmp/original.recipe" "$tmp/defaults/stage12.recipe"
done
echo 'TDP preprocessing flag shell regression tests passed'
