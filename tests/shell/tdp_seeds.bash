#!/usr/bin/env bash
# Seeds (docs/STAGE4_RUST_DESIGN.md §8, §10.9 item 7): the private SEEDS file,
# strict parsing, legacy inputs rejected with migration messages, the gen
# banner check, and no seed value in the narrative log, stdout or argv
# renderings, and never on a stage binary's argv (ps): gen and the mixer read
# them from inherited descriptors. Mock binaries (tests/shell/lib/tdp_mock.py).
# (`[ a ] && [ b ]` is not checked by errexit except for its last command, so
# every assertion below is its own command.)
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
for b in gen_sandwich_gadget circuit_mixer tdp_packing tdp_unpacking; do ln -s "$root/tests/shell/lib/tdp_mock.py" "$TDP_BIN_DIR/$b"; done
drive() { # RUNDIR ARGS...
  local d=$1; shift
  bash "$root/scripts/tdp_gen.sh" --source-wires 4 --source-gates 1 --run-directory "$d" "$@"
}
expect_fail() { # PATTERN RUNDIR ARGS...
  local pattern=$1; shift
  if drive "$@" > "$tmp/out" 2>&1; then echo "unexpected success: $*" >&2; exit 1; fi
  grep -Eq -- "$pattern" "$tmp/out" || { echo "expected /$pattern/:" >&2; cat "$tmp/out" >&2; exit 1; }
}
# 1. a fresh random run: SEEDS (format 3, 0600), values never printed
drive "$tmp/r1" > "$tmp/r1.stdout" 2>&1
S=$tmp/r1/SEEDS
[ "$(stat -f %Lp "$S" 2>/dev/null || stat -c %a "$S")" = 600 ]
mapfile -t L < "$S"
[ "${#L[@]}" = 8 ]
[ "${L[0]}" = seeds_format=3 ]
[[ ${L[1]} =~ ^gen_seed=[0-9]+$ ]]
[[ ${L[2]} =~ ^mix_seed=[0-9]+$ ]]
[[ ${L[3]} =~ ^split_seed=[0-9]+$ ]]
[[ ${L[4]} =~ ^cross_seed=[0-9]+$ ]]
[[ ${L[5]} =~ ^unpack_seed=[0-9]+$ ]]
[[ ${L[6]} =~ ^run_key=[0-9a-f]{64}$ ]]
[ "${L[7]}" = seed_source=random ]
gen=${L[1]#gen_seed=}; mix=${L[2]#mix_seed=}; split=${L[3]#split_seed=}; cross=${L[4]#cross_seed=}
unpack=${L[5]#unpack_seed=}; key=${L[6]#run_key=}
for v in "$gen" "$mix" "$split" "$cross" "$unpack" "$key"; do
  for f in "$tmp/r1.stdout" "$tmp/r1/tdp_gen.log" "$tmp/r1/stage12.recipe" "$tmp/r1/stage3.recipe" \
           "$tmp/r1/stage4.recipe" "$tmp/r1/stage5.recipe" "$tmp/r1/packing.recipe" "$tmp/r1/unpacking.recipe" \
           "$tmp/r1/stage12.log" "$tmp/r1/stage3.log" "$tmp/r1/stage4.log" "$tmp/r1/stage5.log" "$tmp/r1/stage7.log" \
           "$tmp/r1/stage3.settings"; do
    if grep -qF -- "$v" "$f"; then echo "a seed value reached $f" >&2; exit 1; fi
  done
done
grep -q 'gen seeds: verified' "$tmp/r1/tdp_gen.log"
# each seed reaches exactly its binary (gen and mix as in val_chain.sh), but
# through descriptors: no seed value on any stage binary's raw argv. The cross
# seed is unused here: stage 5 resumes the split state (its RNG continues).
python3 -I - "$tmp/r1.calls" "$gen" "$mix" "$split" "$cross" "$unpack" <<'PY'
import json, sys
calls = [json.loads(l) for l in open(sys.argv[1])]
gen, mix, split, cross, unpack = sys.argv[2:7]
seeds = {gen, mix, split, cross, unpack}
names = []
for c in calls:
    a, raw = c["args"], c["argv"]
    names.append(c["name"])
    assert not seeds & set(raw), ("seed on argv", c["name"], raw)
    if c["name"] == "gen_sandwich_gadget":
        assert a[7:10] == [gen] * 3, a
        assert raw[7:10] == ["@/dev/fd/3", "@/dev/fd/4", "@/dev/fd/5"], raw
        assert not (seeds - {gen}) & set(a)
    elif c["name"] == "circuit_mixer":
        want = mix if "--db-mixing" in a else split if "--split" in a else None
        if want is None:
            assert "--resume" in a and "--seed" not in a and not seeds & set(a), a
        else:
            assert a[a.index("--seed") + 1] == want and not (seeds - {want}) & set(a), a
            assert "--seed" not in raw and raw[raw.index("--seed-file") + 1] == "/dev/fd/3", raw
    elif c["name"] == "tdp_unpacking":
        assert a[a.index("--seed") + 1] == unpack and not (seeds - {unpack}) & set(a), a
        assert "--seed" not in raw and raw[raw.index("--seed-file") + 1] == "/dev/fd/3", raw
    else:
        assert not seeds & set(a), c
assert names.count("circuit_mixer") == 3 and "tdp_unpacking" in names, names
PY
# the cross seed is used when stage 5 starts from a circuit file (stage 4 skipped)
drive "$tmp/r1x" --run-skip-stages 4 > /dev/null 2>&1
cross_x=$(sed -n 's/^cross_seed=//p' "$tmp/r1x/SEEDS")
python3 -I - "$tmp/r1x.calls" "$cross_x" <<'PY'
import json, sys
calls = [json.loads(l) for l in open(sys.argv[1])]
cross = sys.argv[2]
c = next(c for c in calls if c["name"] == "circuit_mixer" and "--target-size" in c["args"])
assert c["args"][c["args"].index("--seed") + 1] == cross and "--input" in c["args"], c["args"]
assert cross not in c["argv"] and c["argv"][c["argv"].index("--seed-file") + 1] == "/dev/fd/3", c["argv"]
PY
# two fresh runs draw different seeds
drive "$tmp/r2" > /dev/null 2>&1
cmp -s <(sed -n 2,6p "$tmp/r1/SEEDS") <(sed -n 2,6p "$tmp/r2/SEEDS") && { echo "two runs drew the same seeds" >&2; exit 1; }
# 2. explicit seed file (calibration) and its validation: gen and mix are
# required, the stage-4/5/7 seeds optional (drawn fresh when absent)
sf=$tmp/seeds; printf 'gen_seed=11\nmix_seed=1013\n' > "$sf"; chmod 600 "$sf"
drive "$tmp/e1" --seed-file "$sf" > "$tmp/e1.out" 2>&1
grep -q '^seed_source=explicit$' "$tmp/e1/SEEDS"
grep -q '^split_seed=[0-9]' "$tmp/e1/SEEDS"
grep -q 'calibration only, NOT a deliverable' "$tmp/e1/tdp_gen.log"
grep -q '"11", "11", "11"' "$tmp/e1.calls"
# resume with the same file is fine; a different one is fatal (also for an
# optional seed that the run drew differently)
drive "$tmp/e1" --seed-file "$sf" > /dev/null 2>&1
sf2=$tmp/seeds2; printf 'gen_seed=12\nmix_seed=1013\n' > "$sf2"; chmod 600 "$sf2"
expect_fail 'conflicts with the existing' "$tmp/e1" --seed-file "$sf2"
sf3=$tmp/seeds3; printf 'gen_seed=11\nmix_seed=1013\nunpack_seed=7\n' > "$sf3"; chmod 600 "$sf3"
expect_fail 'conflicts with the existing' "$tmp/e1" --seed-file "$sf3"
# all five explicit: each reaches its binary
sf5=$tmp/seeds5; printf 'gen_seed=21\nmix_seed=22\nsplit_seed=2323\ncross_seed=2424\nunpack_seed=2525\n' > "$sf5"; chmod 600 "$sf5"
drive "$tmp/e5" --seed-file "$sf5" --run-skip-stages 4 > /dev/null 2>&1
grep -q '"--seed", "2424"' "$tmp/e5.calls"
grep -q '"--seed", "2525"' "$tmp/e5.calls"
grep -q '^split_seed=2323$' "$tmp/e5/SEEDS"
# legacy and malformed inputs
printf '42\n' > "$tmp/v1"; chmod 600 "$tmp/v1"
expect_fail 'seed file format 1 holds one seed; recipe v11 needs gen_seed= and mix_seed=' "$tmp/x1" --seed-file "$tmp/v1"
expect_fail 'seeds on argv are no longer accepted' "$tmp/x2" -s 42
expect_fail 'seeds on argv are no longer accepted' "$tmp/x3" --calibration-seed 42
chmod 644 "$sf"; expect_fail 'chmod 600' "$tmp/x4" --seed-file "$sf"; chmod 600 "$sf"
for bad in 'gen_seed=1' 'gen_seed=1\nmix_seed=x' 'gen_seed=1\nmix_seed=9223372036854775793' \
           'gen_seed=1\nmix_seed=2\nother=3' 'gen_seed=-1\nmix_seed=2' 'gen_seed=1\nmix_seed=2\nsplit_seed=x' \
           'gen_seed=1\nmix_seed=2\ncross_seed=9223372036854775793'; do
  printf "$bad\n" > "$tmp/bad"; chmod 600 "$tmp/bad"
  expect_fail 'seed-file' "$tmp/x5" --seed-file "$tmp/bad"
done
test ! -e "$tmp/x1"
test ! -e "$tmp/x5"
# 3. malformed SEEDS in a run directory are fatal (exit 2)
mk() { mkdir -p "$tmp/$1"; printf "$2" > "$tmp/$1/SEEDS"; chmod 600 "$tmp/$1/SEEDS"; }
K=$(printf 'a%.0s' $(seq 64))
S3="split_seed=3\ncross_seed=4\nunpack_seed=5"
mk m1 "seeds_format=3\ngen_seed=1\nmix_seed=2\n$S3\nrun_key=$K\n"                       # missing source
mk m2 "seeds_format=3\ngen_seed=x\nmix_seed=2\n$S3\nrun_key=$K\nseed_source=random\n"    # non-decimal
mk m3 "seeds_format=3\ngen_seed=9223372036854775793\nmix_seed=2\n$S3\nrun_key=$K\nseed_source=random\n"
mk m4 "seeds_format=3\ngen_seed=1\nmix_seed=2\n$S3\nrun_key=abc\nseed_source=random\n"  # short run_key
mk m5 "seeds_format=3\ngen_seed=1\nmix_seed=2\nsplit_seed=3\nrun_key=$K\nseed_source=random\n"  # format-3 header, format-2 body
mk m6 "seeds_format=2\ngen_seed=1\nmix_seed=x\nrun_key=$K\nseed_source=random\n"         # format 2, non-decimal
for d in m1 m2 m3 m4 m5 m6; do
  set +e; drive "$tmp/$d" > "$tmp/out" 2>&1; rc=$?; set -e
  [ "$rc" = 2 ] || { echo "$d: exit $rc" >&2; cat "$tmp/out" >&2; exit 1; }
  grep -q 'malformed\|out of range' "$tmp/out"
done
# a format-2 SEEDS (the four-stage driver) is upgraded in place on resume:
# gen and mix kept, the three new seeds drawn, the file rewritten as format 3
mk u1 "seeds_format=2\ngen_seed=11\nmix_seed=1013\nrun_key=$K\nseed_source=explicit\n"
drive "$tmp/u1" > "$tmp/u1.out" 2>&1
grep -q 'upgraded from format 2' "$tmp/u1/tdp_gen.log"
mapfile -t U < "$tmp/u1/SEEDS"
[ "${#U[@]}" = 8 ]
[ "${U[0]}" = seeds_format=3 ]
[ "${U[1]}" = gen_seed=11 ]
[ "${U[2]}" = mix_seed=1013 ]
[[ ${U[3]} =~ ^split_seed=[0-9]+$ ]]
[ "${U[6]}" = "run_key=$K" ]
[ "${U[7]}" = seed_source=explicit ]
grep -q '"11", "11", "11"' "$tmp/u1.calls"
# a v9 run directory (one-line SEED) is refused
mkdir -p "$tmp/v9"; echo 42 > "$tmp/v9/SEED"
expect_fail 'recipe v9' "$tmp/v9"
# 4. the gen banner check: a gen that used another seed is fatal
printf 'gen_seed_override=7\n' > "$tmp/g1.mock"
expect_fail 'gen used a different seed than SEEDS' "$tmp/g1"
test ! -e "$tmp/g1/tdp.mpmct1"
echo 'TDP seed shell tests passed'
