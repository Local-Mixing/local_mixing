#!/usr/bin/env bash
# Exercise the public driver without release builds or real mixing.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export GSS_BIN_DIR="$tmp/bin"
# Tests use only their declared gadget controls, irrespective of the caller.
while IFS= read -r key; do unset "$key"; done < <(compgen -A variable BV5_ || true)
while IFS= read -r key; do unset "$key"; done < <(compgen -A variable PROD_ || true)
unset GSS_SOURCE_C SANDWICH_VARIANT
mkdir -p "$GSS_BIN_DIR"
reject() {
  if bash "$root/scripts/gss_mix.sh" -n 4 -o "$tmp/rejected" --stop-after 2 "$@" >"$tmp/error" 2>&1; then
    echo "unexpected success: $*" >&2; exit 1
  fi
  test ! -e "$tmp/rejected"
  grep -q 'FATAL:' "$tmp/error"
  if grep -Eq 'unbound variable|missing.*cargo build|integer expression expected' "$tmp/error"; then
    cat "$tmp/error" >&2; exit 1
  fi
}
for mode in product-2223 2223 nonlinear193 unknown; do
  reject --gadgetization-mode "$mode"
done
for flag in --bv5-k --bv5-max-open --bv5-min-open --bv5-balanced --gadget-mask-pair-wires --gadget-max-open-masks --gadget-min-open-masks --gadget-balanced-masks --preprocessing-mask-pair-wires --preprocessing-max-open-masks --preprocessing-min-open-masks --preprocessing-balanced-masks; do
  reject "$flag"
  reject "$flag" --stop-after 2
  reject "$flag" ''
  reject "$flag" 999999999999999999999999999
  reject --gadgetization-mode nonlinear291 "$flag" 2
done
for value in 0 1 65 2.5 abc; do reject --bv5-k "$value"; done
for value in 0 1 65 abc; do reject --bv5-max-open "$value"; done
for value in 0 64 abc; do reject --bv5-min-open "$value"; done
for value in 2 abc; do reject --bv5-balanced "$value"; done
reject --bv5-max-open 2
reject --bv5-min-open 4 --bv5-max-open 3
reject --bv5-k 8
reject -n 2
reject -n 4096
BV5_K=1 reject
BV5_MAX_OPEN=1 reject
BV5_MIN_OPEN=4 reject
BV5_BALANCED=2 reject
PROD_PRESET=production reject
reject --preprocessing-mode quadratic-masking --gadget-mode ran-balanced
reject --gadgetization-mode nonlinear291 --preprocessing-mode nonlinear291
for suffix in mask-pair-wires max-open-masks min-open-masks balanced-masks; do
  reject "--preprocessing-$suffix" 1 "--gadget-$suffix" 1
done
reject --preprocessing-mask-pair-wires 2 --bv5-k 2
cat > "$GSS_BIN_DIR/mock" <<'MOCK'
#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
assert Path(sys.argv[0]).name == 'gen_sandwich_gadget', sys.argv
keys = ('BV5_K', 'BV5_MAX_OPEN', 'BV5_MIN_OPEN', 'BV5_BALANCED')
Path(os.environ['MOCK_CALLS']).write_text(json.dumps({
    'args': sys.argv[1:], 'env': {key: os.environ.get(key) for key in keys},
}))
Path(sys.argv[1]).write_text('4\nx\nx\n')
Path(sys.argv[1] + '.sandwich.mpmct1').write_text('4\nx\nx\n')
MOCK
chmod +x "$GSS_BIN_DIR/mock"
for binary in gen_sandwich_gadget fmix fcompress; do
  ln -s mock "$GSS_BIN_DIR/$binary"
done
run_case() {
  local label=$1; shift
  export MOCK_CALLS="$tmp/$label.calls"
  if ! bash "$root/scripts/gss_mix.sh" -n 4 -o "$tmp/$label" -s 42 --mcd 1 --stop-after 2 "$@" >"$tmp/$label.log" 2>&1; then
    cat "$tmp/$label.log" >&2; exit 1
  fi
}
run_case defaults
for mode in quadratic-masking ran-balanced blinded-v5 blinded_v5; do
  run_case "$mode" --gadgetization-mode "$mode" --bv5-k 0004 --bv5-max-open 0004 --bv5-min-open 0001 --bv5-balanced 00
done
run_case narrow --bv5-max-open 2 --bv5-min-open 1
run_case canonical --preprocessing-mode quadratic-masking --preprocessing-mask-pair-wires 4 --preprocessing-max-open-masks 4 --preprocessing-min-open-masks 1 --preprocessing-balanced-masks 0
BV5_K=4 BV5_MAX_OPEN=4 BV5_MIN_OPEN=1 BV5_BALANCED=0 run_case environment
BV5_K=1 BV5_MAX_OPEN=1 BV5_MIN_OPEN=9 BV5_BALANCED=2 run_case precedence --bv5-k 2 --bv5-max-open 3 --bv5-min-open 2 --bv5-balanced 1
run_case nonlinear291 --gadgetization-mode nonlinear291
python3 -I - "$tmp" <<'PY'
from pathlib import Path
import json, sys
root = Path(sys.argv[1])
expected = {
    'defaults': ['2', '3', '2', '1'],
    'quadratic-masking': ['4', '4', '1', '0'],
    'ran-balanced': ['4', '4', '1', '0'],
    'blinded-v5': ['4', '4', '1', '0'],
    'blinded_v5': ['4', '4', '1', '0'],
    'narrow': ['2', '2', '1', '1'],
    'canonical': ['4', '4', '1', '0'],
    'environment': ['4', '4', '1', '0'],
    'precedence': ['2', '3', '2', '1'],
    'nonlinear291': [None] * 4,
}
for name, values in expected.items():
    call = json.loads((root / f'{name}.calls').read_text())
    mode = name if name.startswith('nonlinear') else 'quadratic-masking'
    assert call['args'][-1] == mode, call
    assert list(call['env'].values()) == values, call
    recipe = (root / name / 'stage12.recipe').read_text()
    assert f'preprocessing_mode={mode}\n' in recipe, recipe
PY
# A same-driver restart must skip the saved stage-2 artifact and retain its marker.
cp "$tmp/defaults/stage12.recipe" "$tmp/original.recipe"
rm "$tmp/defaults.calls"
run_case defaults
test ! -e "$tmp/defaults.calls"
cmp "$tmp/original.recipe" "$tmp/defaults/stage12.recipe"
# A changed stage-2 recipe must fail before regenerating or replacing its marker.
for changed_setting in mode mask; do
  changed_flags=(--preprocessing-mode nonlinear291)
  if [ "$changed_setting" = mask ]; then
    changed_flags=(--preprocessing-mask-pair-wires 4)
  fi
  if bash "$root/scripts/gss_mix.sh" -n 4 -o "$tmp/defaults" -s 42 --mcd 1 --stop-after 2 "${changed_flags[@]}" >"$tmp/changed.log" 2>&1; then
    echo 'changed preprocessing recipe unexpectedly resumed' >&2; exit 1
  fi
  test ! -e "$tmp/defaults.calls"
  cmp "$tmp/original.recipe" "$tmp/defaults/stage12.recipe"
done
echo 'GSS preprocessing flag shell regression tests passed'
