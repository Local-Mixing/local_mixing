#!/usr/bin/env bash
# Seeds (docs/STAGE4_RUST_DESIGN.md §8, §10.9 item 7): the private SEEDS file,
# strict parsing, legacy inputs rejected with migration messages, the gen
# banner check, and no seed value in the narrative log, stdout or argv
# renderings. Mock binaries (tests/shell/lib/tdp_mock.py).
set -euo pipefail
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
drive() { # RUNDIR ARGS...
  local d=$1; shift
  bash "$root/scripts/tdp_gen.sh" --source-wires 4 --source-gates 1 --run-directory "$d" "$@"
}
expect_fail() { # PATTERN RUNDIR ARGS...
  local pattern=$1; shift
  if drive "$@" > "$tmp/out" 2>&1; then echo "unexpected success: $*" >&2; exit 1; fi
  grep -Eq -- "$pattern" "$tmp/out" || { echo "expected /$pattern/:" >&2; cat "$tmp/out" >&2; exit 1; }
}
# 1. a fresh random run: SEEDS (format 2, 0600), values never printed
drive "$tmp/r1" > "$tmp/r1.stdout" 2>&1
S=$tmp/r1/SEEDS
[ "$(stat -f %Lp "$S" 2>/dev/null || stat -c %a "$S")" = 600 ]
mapfile -t L < "$S"
[ "${#L[@]}" = 5 ] && [ "${L[0]}" = seeds_format=2 ] && [ "${L[4]}" = seed_source=random ]
gen=${L[1]#gen_seed=}; mix=${L[2]#mix_seed=}; key=${L[3]#run_key=}
[[ $gen =~ ^[0-9]+$ ]] && [[ $mix =~ ^[0-9]+$ ]] && [[ $key =~ ^[0-9a-f]{64}$ ]]
for v in "$gen" "$mix" "$key"; do
  for f in "$tmp/r1.stdout" "$tmp/r1/tdp_gen.log" "$tmp/r1/stage12.recipe" "$tmp/r1/packing.recipe"; do
    if grep -qF -- "$v" "$f"; then echo "a seed value reached $f" >&2; exit 1; fi
  done
done
grep -q 'gen seeds: verified' "$tmp/r1/tdp_gen.log"
# the gen and mix seeds reach exactly their binaries' argv (as in val_chain.sh)
python3 -I - "$tmp/r1.calls" "$gen" "$mix" <<'PY'
import json, sys
calls = [json.loads(l) for l in open(sys.argv[1])]
gen, mix = sys.argv[2], sys.argv[3]
for c in calls:
    a = c["args"]
    if c["name"] == "gen_sandwich_gadget":
        assert a[7:10] == [gen] * 3, a
        assert mix not in a
    elif c["name"] == "circuit_mixer":
        assert a[a.index("--seed") + 1] == mix and gen not in a, a
    else:
        assert gen not in a and mix not in a, c
PY
# two fresh runs draw different seeds
drive "$tmp/r2" > /dev/null 2>&1
cmp -s <(sed -n 2,3p "$tmp/r1/SEEDS") <(sed -n 2,3p "$tmp/r2/SEEDS") && { echo "two runs drew the same seeds" >&2; exit 1; }
# 2. explicit seed file (calibration) and its validation
sf=$tmp/seeds; printf 'gen_seed=11\nmix_seed=1013\n' > "$sf"; chmod 600 "$sf"
drive "$tmp/e1" --seed-file "$sf" > "$tmp/e1.out" 2>&1
grep -q '^seed_source=explicit$' "$tmp/e1/SEEDS"
grep -q 'calibration only, NOT a deliverable' "$tmp/e1/tdp_gen.log"
grep -q '"11", "11", "11"' "$tmp/e1.calls"
# resume with the same file is fine; a different one is fatal
drive "$tmp/e1" --seed-file "$sf" > /dev/null 2>&1
sf2=$tmp/seeds2; printf 'gen_seed=12\nmix_seed=1013\n' > "$sf2"; chmod 600 "$sf2"
expect_fail 'conflicts with the existing' "$tmp/e1" --seed-file "$sf2"
# legacy and malformed inputs
printf '42\n' > "$tmp/v1"; chmod 600 "$tmp/v1"
expect_fail 'seed file format 1 holds one seed; recipe v10 needs gen_seed= and mix_seed=' "$tmp/x1" --seed-file "$tmp/v1"
expect_fail 'seeds on argv are no longer accepted' "$tmp/x2" -s 42
expect_fail 'seeds on argv are no longer accepted' "$tmp/x3" --calibration-seed 42
chmod 644 "$sf"; expect_fail 'chmod 600' "$tmp/x4" --seed-file "$sf"; chmod 600 "$sf"
for bad in 'gen_seed=1' 'gen_seed=1\nmix_seed=x' 'gen_seed=1\nmix_seed=9223372036854775793' \
           'gen_seed=1\nmix_seed=2\nother=3' 'gen_seed=-1\nmix_seed=2'; do
  printf "$bad\n" > "$tmp/bad"; chmod 600 "$tmp/bad"
  expect_fail 'seed-file' "$tmp/x5" --seed-file "$tmp/bad"
done
test ! -e "$tmp/x1" && test ! -e "$tmp/x5"
# 3. malformed SEEDS in a run directory are fatal (exit 2)
mk() { mkdir -p "$tmp/$1"; printf "$2" > "$tmp/$1/SEEDS"; chmod 600 "$tmp/$1/SEEDS"; }
K=$(printf 'a%.0s' $(seq 64))
mk m1 "seeds_format=2\ngen_seed=1\nmix_seed=2\nrun_key=$K\n"                       # missing key
mk m2 "seeds_format=2\ngen_seed=x\nmix_seed=2\nrun_key=$K\nseed_source=random\n"    # non-decimal
mk m3 "seeds_format=2\ngen_seed=9223372036854775793\nmix_seed=2\nrun_key=$K\nseed_source=random\n"
mk m4 "seeds_format=2\ngen_seed=1\nmix_seed=2\nrun_key=abc\nseed_source=random\n"  # short run_key
for d in m1 m2 m3 m4; do
  set +e; drive "$tmp/$d" > "$tmp/out" 2>&1; rc=$?; set -e
  [ "$rc" = 2 ] || { echo "$d: exit $rc" >&2; cat "$tmp/out" >&2; exit 1; }
  grep -q 'malformed\|out of range' "$tmp/out"
done
# a v9 run directory (one-line SEED) is refused
mkdir -p "$tmp/v9"; echo 42 > "$tmp/v9/SEED"
expect_fail 'recipe v9' "$tmp/v9"
# 4. the gen banner check: a gen that used another seed is fatal
printf 'gen_seed_override=7\n' > "$tmp/g1.mock"
expect_fail 'gen used a different seed than SEEDS' "$tmp/g1"
test ! -e "$tmp/g1/tdp.mpmct1"
echo 'TDP seed shell tests passed'
