#!/usr/bin/env bash
# Fast plumbing smoke test; mock binaries avoid mixing or release builds.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export GSS_BIN_DIR="$tmp/bin" GSS_MIX_ALLOW_EMPTY_STORE=1
unset FROZEN_DB_DIR FROZEN_CURATED_DIR
mkdir -p "$GSS_BIN_DIR"
reject() {
  if bash "$root/scripts/gss_mix.sh" -n 4 -o "$tmp/rejected" "$@" >"$tmp/error" 2>&1; then
    echo "unexpected success: $*" >&2; exit 1
  fi
  test ! -e "$tmp/rejected"
  grep -q 'FATAL:' "$tmp/error"
  if grep -Eq 'unbound variable|missing.*cargo build' "$tmp/error"; then
    cat "$tmp/error" >&2; exit 1
  fi
}
for value in 0 1 -2 2.5 abc 1000000001 999999999999999999999999999999; do
  reject --min-block-size "$value"
done
for value in 0 65 999999999999999999999999999999; do
  reject --pieces "$value"
done
reject --pieces 2 --piece-threads 1025
reject --pieces 2 --piece-threads 999999999999999999999999999999
reject --min-block-size 2 --pieces 1
reject --pieces 4 --min-block-size 2
reject --piece-threads 2
reject --min-block-size 2 --piece-threads 0
for flag in --min-block-size --pieces --piece-threads; do
  reject "$flag"
  reject "$flag" --stop-after 2
  reject "$flag" ''
done
cat > "$GSS_BIN_DIR/mock" <<'MOCK'
#!/usr/bin/env python3
import os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['MOCK_CALLS'], 'a') as calls:
    calls.write(name + ' ' + ' '.join(args) + '\n')
if name == 'gen_sandwich_gadget':
    out = args[0]
    Path(out + '.sandwich.mpmct1').write_text('4\nx\nx\n')
else:
    out = args[args.index('--output') + 1]
    if '--state-out' in args:
        Path(args[args.index('--state-out') + 1]).write_text('moves 0\n')
Path(out).write_text('4\nx\nx\n')
MOCK
chmod +x "$GSS_BIN_DIR/mock"
for binary in gen_sandwich_gadget fmix fcompress; do
  ln -s mock "$GSS_BIN_DIR/$binary"
done
run_case() {
  local mode=$1; shift
  export MOCK_CALLS="$tmp/$mode.calls"
  bash "$root/scripts/gss_mix.sh" -n 4 -o "$tmp/$mode" -s 42 "$@" >"$tmp/$mode.log" 2>&1
}
run_case auto --min-block-size 0002 --piece-threads 3
run_case auto_default --min-block-size 2
run_case fixed --pieces 0004 --piece-threads 0003
run_case serial
run_case one --pieces 1
run_case canonical --parallel-target-piece-gates 2 --parallel-threads 3 --db-mixing-target-size-factor 2 --db-mixing-hold-work-units 27
reject --parallel-pieces 4 --min-block-size 2
reject --pieces 4 --parallel-target-piece-gates 2
python3 -I - "$tmp" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
for mode in ('auto', 'auto_default', 'fixed', 'serial', 'one', 'canonical'):
    lines = (root / f'{mode}.calls').read_text().splitlines()
    assert len(lines) == 5, lines
    assert '--db-mixing' in lines[1], lines
    assert (root / mode / 'db_mixing.mpmct1').is_file()
    assert (root / mode / 'db_mixing.state').is_file()
    assert not (root / mode / 'phaseA.mpmct1').exists()
    for i, line in enumerate(lines):
        args = line.split()
        enabled = i in (1, 2) and mode in ('auto', 'auto_default', 'fixed', 'canonical')
        assert ('--parallel-threads' in args) == (enabled and mode != 'auto_default'), line
        assert ('--parallel-target-piece-gates' in args) == (enabled and (mode.startswith('auto') or mode == 'canonical')), line
        assert ('--parallel-pieces' in args) == (enabled and mode == 'fixed'), line
        if enabled:
            if mode != 'auto_default':
                assert args[args.index('--parallel-threads') + 1] == '3', line
            flag, value = ('--parallel-target-piece-gates', '2') if (mode.startswith('auto') or mode == 'canonical') else ('--parallel-pieces', '4')
            assert args[args.index(flag) + 1] == value, line
    if (mode.startswith('auto') or mode == 'canonical'):
        assert 'automatic pieces target_piece_gates=2' in (root / mode / 'gss_mix.log').read_text()
PY
# Stage-3 continuation must recognize the new artifact and avoid another mix.
rm "$tmp/canonical.calls"
run_case canonical --parallel-target-piece-gates 2 --parallel-threads 3 --db-mixing-target-size-factor 2 --db-mixing-hold-work-units 27
test ! -e "$tmp/canonical.calls"
echo 'GSS block-size shell smoke tests passed' 
