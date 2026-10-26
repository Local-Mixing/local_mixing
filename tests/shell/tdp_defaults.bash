#!/usr/bin/env bash
# The defaults requirement, CI-sized (docs/STAGE4_RUST_DESIGN.md §10.9 item 1):
# a transcript diff of the managed driver against the ORACLE SCRIPT ITSELF.
# A path-rewritten sandbox copy of scripts/windowed_delivery/waist/val_chain.sh
# runs on recording mocks (gen_sandwich_gadget, circuit_mixer, hmap_affine;
# stage-4 Python scripts replaced by recording stubs); scripts/tdp_gen.sh (and,
# when a local_mixing_bin is available, the Rust driver with a v2 TOML) runs
# with the same seeds on the same mocks. Asserted per case:
#   (a) gen argv identical (output directory normalized);
#   (b) mixer argv identical as an option set, positional order kept;
#   (c) the effective value (unset = the registry default) of every recipe and
#       driver variable identical, and no forbidden variable set;
#   (d) the managed binaries' environment is exactly allow-list + recipe + driver;
#   (e) the oracle's stage-4 calls are window_pack3.py ... 512 512 24576 3000,
#       repair.sh TAG 512 N and anf_dump_eval.py; the managed stage 6 is
#       `tdp_packing run` with every parameter at its default and no
#       unpacking sub-step (the managed driver runs with --run-skip-stages 4,5
#       --delivery-mode windowed --delivery-anf-only: the oracle chain's stages).
# Plus: G with 4*G*1.3 having a fractional part >= .5 (int, not round), the
# n=64/128 default sizes, n=32 without size (fatal), and one case per
# forbidden variable exported in the caller (fatal, naming it).
set -euo pipefail
if [[ ${1-} != --isolated-environment ]]; then
  exec env -i PATH="$PATH" HOME="${HOME:-/tmp}" LOCAL_MIXING_BIN="${LOCAL_MIXING_BIN:-}" \
    bash "$0" --isolated-environment
fi
shift
root=$(cd "$(dirname "$0")/../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
MOCK=$root/tests/shell/lib/tdp_mock.py
STUB=$root/tests/shell/lib/oracle_stub.py
# a fake store: only its path is used (the mocks never open it)
STORE=$tmp/store; mkdir -p "$STORE"
# ---- the sandboxed oracle driver ----
VR=$tmp/vc_root; RD=$VR/runs/ncdb_ab_20261001; W=$RD/waist
mkdir -p "$VR/target/release" "$W"
for b in gen_sandwich_gadget circuit_mixer hmap_affine; do ln -s "$MOCK" "$VR/target/release/$b"; done
ln -s "$STUB" "$RD/window_pack3.py"
for s in repair.sh anf_dump_eval.py mult_census.py; do ln -s "$STUB" "$W/$s"; done
VC=$W/val_chain.sh
cp "$root/scripts/windowed_delivery/waist/val_chain.sh" "$VC"
rewrite() { # FILE FROM TO COUNT (exactly COUNT hits, or fail)
  local have; have=$(grep -oF -- "$2" "$1" | wc -l | tr -d ' ')
  [ "$have" = "$4" ] || { echo "rewrite: '$2' found $have times in $1, expected $4" >&2; exit 1; }
  FROM="$2" TO="$3" perl -pi -e 's/\Q$ENV{FROM}\E/$ENV{TO}/g' "$1"
}
rewrite "$VC" "ROOT=/Users/rancanetti/code/local_mixing" "ROOT=$VR" 1
rewrite "$VC" "SEED=\$(python3 -c 'import secrets;print(secrets.randbelow(2**31))')" 'SEED=$VAL_GEN_SEED' 1
rewrite "$VC" "MSEED=\$(python3 -c 'import secrets;print(secrets.randbelow(2**40))')" 'MSEED=$VAL_MIX_SEED' 1
rewrite "$VC" 'FROZEN_DB_DIR=$HOME/code/local_mixing/frozen_db/frozen_m1_m11' "FROZEN_DB_DIR=$STORE" 1
rewrite "$VC" "python3 \$W/mult_census.py" "\$W/mult_census.py" 1
rewrite "$VC" "python3 window_pack3.py" "./window_pack3.py" 1
rewrite "$VC" "bash \$W/repair.sh" "\$W/repair.sh" 1
rewrite "$VC" "python3 \$W/anf_dump_eval.py" "\$W/anf_dump_eval.py" 1
# ---- the managed side's mocks ----
BIN=$tmp/bin; mkdir -p "$BIN"
for b in gen_sandwich_gadget circuit_mixer tdp_packing tdp_unpacking tdp_adaptive; do ln -s "$MOCK" "$BIN/$b"; done

seedfile() { # FILE GEN MIX
  printf 'gen_seed=%s\nmix_seed=%s\n' "$2" "$3" > "$1"; chmod 600 "$1"
}
run_pair() { # LABEL N MC CFG GATES GEN MIX [managed size flags...]
  local label=$1 n=$2 mc=$3 cfg=$4 gates=$5 gs=$6 ms=$7; shift 7
  local s; s=$(python3 -I -c "import math,sys; n=int(sys.argv[1]); print(max(n, round(n*math.log2(n))))" "$n")
  # reference
  printf 'gen_gates=%s\n' "$gates" > "$W/$label.mock"
  env -i HOME="$HOME" PATH="$PATH" VAL_GEN_SEED="$gs" VAL_MIX_SEED="$ms" \
    bash "$VC" "$label" "$n" "$mc" "$s" $((20 * n)) "$cfg" > "$tmp/$label.vc.log" 2>&1 ||
    { cat "$tmp/$label.vc.log" >&2; exit 1; }
  # managed (bash driver)
  local sf=$tmp/$label.seeds; seedfile "$sf" "$gs" "$ms"
  mkdir -p "$tmp/m"; printf 'gen_gates=%s\n' "$gates" > "$tmp/m/$label.mock"
  local preset=(); [ "$cfg" = A ] && preset=(--preprocessing-preset A)
  # --run-skip-stages 4,5 --delivery-mode windowed --delivery-anf-only: the oracle chain has four stages
  env -i HOME="$HOME" PATH="$PATH" TDP_BIN_DIR="$BIN" FROZEN_DB_DIR="$STORE" \
    bash "$root/scripts/tdp_gen.sh" --source-wires "$n" --run-directory "$tmp/m/$label" \
      --seed-file "$sf" --run-skip-stages 4,5 --delivery-mode windowed --delivery-anf-only ${preset[@]+"${preset[@]}"} "$@" > "$tmp/$label.m.log" 2>&1 ||
    { cat "$tmp/$label.m.log" >&2; exit 1; }
}
python_compare() { # LABEL N MANAGED_CALLS
  python3 -I - "$1" "$2" "$3" "$W" "$root/scripts/tdp_env_registry.tsv" "$STORE" <<'PY'
import json, os, sys
label, n, mcalls, W, registry, store = sys.argv[1:7]
ref = [json.loads(l) for l in open(f"{W}/{label}.calls")]
man = [json.loads(l) for l in open(mcalls)]
def one(calls, name, first=None):
    c = [x for x in calls if x["name"] == name and (first is None or x["args"][0] == first)]
    assert len(c) == 1, (name, [x["name"] for x in calls])
    return c[0]
reg = []
for line in open(registry):
    if line.startswith("#"):
        continue
    nm, kind, cls, dflt = line.rstrip("\n").split("\t")
    reg.append((nm, kind, cls, dflt))
def classify(k):
    ex = [r for r in reg if r[1] == "exact" and r[0] == k]
    if ex:
        return ex[0]
    pre = sorted([r for r in reg if r[1] == "prefix" and k.startswith(r[0])], key=lambda r: -len(r[0]))
    return pre[0] if pre else None
PLUMBING = {"MIXER_STOP_FLAG", "MIXER_DUMP_FLAG", "FROZEN_DB_DIR"}  # paths; set/unset compared below
SHIM = {"CPATH", "LIBRARY_PATH", "MANPATH", "SDKROOT", "__CF_USER_TEXT_ENCODING", "__PYVENV_LAUNCHER__",
        "PWD", "SHLVL", "_", "OLDPWD"}
def effective(env):
    out = {}
    for nm, kind, cls, dflt in reg:
        if cls not in ("recipe", "driver") or kind != "exact" or nm in PLUMBING:
            continue
        out[nm] = env.get(nm, dflt)
    return out
def norm_paths(args):
    return [os.path.basename(a) if "/" in a else a for a in args]
# (a) gen argv (the managed seeds resolved from their descriptors; raw, they
# are never on argv)
rg, mg = one(ref, "gen_sandwich_gadget"), one(man, "gen_sandwich_gadget")
assert norm_paths(rg["args"]) == norm_paths(mg["args"]), (rg["args"], mg["args"])
assert mg["argv"][7:10] == ["@/dev/fd/3", "@/dev/fd/4", "@/dev/fd/5"], mg["argv"]
for c in man:
    for secret in (rg["args"][7], one(ref, "circuit_mixer")["args"][-1]):
        if len(secret) >= 4:  # (short test seeds such as 1 collide with argv values)
            assert secret not in c["argv"], (c["name"], "seed on argv")
# (b) mixer argv as an option set (positional order kept)
def optset(args):
    pos, opts, i = [], {}, 0
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                opts[a] = os.path.basename(args[i + 1]); i += 2
            else:
                opts[a] = None; i += 1
        else:
            pos.append(a); i += 1
    return pos, opts
rm_, mm = one(ref, "circuit_mixer"), one(man, "circuit_mixer")
rp, ro = optset(rm_["args"]); mp, mo = optset(mm["args"])
assert "--seed" not in mm["argv"] and ro["--seed"] not in mm["argv"], mm["argv"]
assert mm["argv"][mm["argv"].index("--seed-file") + 1] == "/dev/fd/3", mm["argv"]
ro["--output"] = "db_mixing.mpmct1"  # val_chain's s3.mpmct1 is the driver's db_mixing.mpmct1
assert (rp, ro) == (mp, mo), (ro, mo)
# (c) effective recipe/driver values; no forbidden variable
for name, r, m in (("gen", rg, mg), ("mixer", rm_, mm)):
    er, em = effective(r["env"]), effective(m["env"])
    assert er == em, (name, {k: (er[k], em[k]) for k in er if er[k] != em[k]})
    for env in (r["env"], m["env"]):
        for k in env:
            c = classify(k)
            assert not (c and c[2] in ("forbidden", "diagnostic")), (name, k)
assert os.path.realpath(rm_["env"]["FROZEN_DB_DIR"]) == os.path.realpath(mm["env"]["FROZEN_DB_DIR"]) == os.path.realpath(store)
assert rm_["env"].get("FROZEN_FILTER") == mm["env"].get("FROZEN_FILTER") == "1"
# (d) managed environment = allow-list + recipe + driver (+ neutral cache sizes)
ALLOW = {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL"}
for c in man:
    for k in c["env"]:
        if k in ALLOW or k in SHIM:
            continue
        cls = classify(k)
        ok = cls and cls[2] in ("recipe", "driver", "neutral", "input")
        if c["name"] == "hmap_affine":
            ok = ok or k.startswith("SPAN_")
        assert ok, (c["name"], k)
    if c["name"] != "circuit_mixer":
        assert not any(classify(k) and classify(k)[2] == "neutral" for k in c["env"]), c["name"]
# (e) stage 4
wp, rep, de = one(ref, "window_pack3.py"), one(ref, "repair.sh"), one(ref, "anf_dump_eval.py")
assert wp["args"][2:6] == ["512", "512", "24576", "3000"], wp
assert rep["args"] == [label, "512", n], rep
run = one(man, "tdp_packing", "run")
a = run["args"]
assert [x for x in a if x.startswith("--")] == ["--stage2", "--stage3", "--source-c", "--n", "--out-dir"], a
assert a[a.index("--n") + 1] == n
assert not any(c["name"] == "tdp_unpacking" for c in man), "anf_only: no unpacking sub-step"
print(f"{label}: (a) gen argv, (b) mixer argv, (c) effective recipe env, (d) allow-list, (e) stage 4: identical")
PY
}
G=4611686018427387903
run_pair d64L8 64 1000 L8 3 "$G" 1099511627775
python_compare d64L8 64 "$tmp/m/d64L8.calls"
run_pair d64A 64 1000 A 7 12345 67890
python_compare d64A 64 "$tmp/m/d64A.calls"
run_pair d128 128 3000 L8 3 1 2
python_compare d128 128 "$tmp/m/d128.calls"
run_pair d16 16 64 L8 3 9223372036854775792 0 --source-gates 64
python_compare d16 16 "$tmp/m/d16.calls"
# int, not round: G=3 -> 4*3*1.3 = 15.6 -> 15 moves
grep -q -- '"--moves", "15"' "$tmp/m/d64L8.calls"
# a literal pin of the managed stage-3 argv (secondary to the oracle transcript)
python3 -I - "$tmp/m/d64A.calls" <<'PY'
import json, os, sys
c = [json.loads(l) for l in open(sys.argv[1]) if '"circuit_mixer"' in l][0]["args"]
c = [os.path.basename(x) if "/" in x else x for x in c]
want = ["--input", "tdp.mpmct1", "--output", "db_mixing.mpmct1", "--tdp", "--db-mixing", "--profile",
        "2,2,2,2,2", "--moves", str(int(7 * 2 * 2 * 1.3)), "--p-db", "1.0", "--db-max-degree", "9",
        "--db-max-span", "30", "--db-wire-terms", "1024", "--db-total-terms", "2048", "--no-local-verify",
        "--seed", "67890"]
assert c == want, c
PY
# the Rust driver (when built): same transcript from a v5 TOML holding only
# n, the store, the seeds, run.skip_stages = [4, 5], delivery.mode = "windowed" and delivery.anf_only
# (preset A only for d64A), for every documented default path: L8 and A at
# n=64, the n=128 validated size, against the same oracle transcripts as the
# bash driver
if [ -n "${LOCAL_MIXING_BIN:-}" ] && [ -x "$LOCAL_MIXING_BIN" ]; then
  host=$(rustc -vV | sed -n 's/^host: //p')
  RT=$tmp/rtarget/$host/release; mkdir -p "$RT"
  for b in gen_sandwich_gadget circuit_mixer tdp_packing tdp_unpacking tdp_adaptive; do ln -s "$MOCK" "$RT/$b"; done
  # the fake store has no tables: validate_frozen_store would refuse it, so give it the files
  : > "$STORE/tables.bin"; echo x > "$STORE/tables.bin"
  for i in $(seq 0 255); do printf 'x' > "$STORE/shard_$(printf %02x "$i").frz"; done
  rust_case() { # LABEL N CFG GATES GEN MIX
    local label=$1 n=$2 cfg=$3 gates=$4 gs=$5 ms=$6 run=$tmp/rust_$1
    local sf=$run.seeds; printf 'gen_seed=%s\nmix_seed=%s\n' "$gs" "$ms" > "$sf"; chmod 600 "$sf"
    printf 'gen_gates=%s\n' "$gates" > "$run.mock"
    local preset=""; [ "$cfg" = A ] && preset='preset = "A"'
    cat > "$run.toml" <<TOML
config_version = 5
[run]
directory = "$run"
build_directory = "$tmp/rtarget"
build_binaries = false
skip_stages = [4, 5]
[source]
wires = $n
[database]
regular_dir = "$STORE"
[preprocessing]
$preset
[delivery]
mode = "windowed"
anf_only = true
[calibration]
enabled = true
seed_file = "$sf"
TOML
    env -i HOME="$HOME" PATH="$PATH" "$LOCAL_MIXING_BIN" tdp_gen --config "$run.toml" > "$run.log" 2>&1 ||
      { cat "$run.log" >&2; exit 1; }
    python_compare "$label" "$n" "$run.calls"
    if [ "${#gs}" -ge 4 ] && grep -q "$gs" "$run.log" "$run/tdp_gen.log" "$run/tdp_command.conf"; then
      echo "a seed value reached the Rust driver's output or logs ($label)" >&2; exit 1
    fi
    # a test store is not the validated one: the manifest must not claim the recipe
    grep -q '^pinned_recipe=none$' "$run/tdp_command.conf"
    grep -q "^preprocessing_preset=$cfg\$" "$run/tdp_command.conf"
  }
  rust_case d64L8 64 L8 3 "$G" 1099511627775
  rust_case d64A 64 A 7 12345 67890
  rust_case d128 128 L8 3 1 2
  echo "Rust driver: d64L8, d64A, d128 identical to the oracle transcripts"
else
  echo "(Rust driver case skipped: set LOCAL_MIXING_BIN to target/release/local_mixing_bin)"
fi
# n=32 has no validated size: fatal unless sized
if env -i HOME="$HOME" PATH="$PATH" TDP_BIN_DIR="$BIN" FROZEN_DB_DIR="$STORE" \
     bash "$root/scripts/tdp_gen.sh" --source-wires 32 --run-directory "$tmp/m/n32" > "$tmp/n32.log" 2>&1; then
  echo "n=32 without a size was accepted" >&2; exit 1
fi
grep -q 'no validated source size for n=32' "$tmp/n32.log"
test ! -e "$tmp/m/n32"
# every forbidden variable exported in the caller is fatal and named
for v in MIXER_DB_LEN_BAND=3 SLICE_GUARD_LEGACY=1 MIXER_STOP_AT_PHASE=2 MIXER_STOP_AT_SPLICES=9 \
         TWIST_G57_NO_RETRY=1 TWIST_G57_NO_SLIDE=1 MIN_DIR_LOOKUP=1 CANON_RULE_L_BRANCH_CAP=512 \
         CANON_MONOMIAL_CAP=200000 EMBEDDED_MASKING_RETIRE=4 EMBEDDED_MASKING_QUAD_FIRE=0 \
         FROZEN_FILTER=1 SANDWICH_VARIANT=balanced TDP_SOURCE_C=/x FROZEN_CURATED_DIR=/x DB_QC=1; do
  if env -i HOME="$HOME" PATH="$PATH" TDP_BIN_DIR="$BIN" FROZEN_DB_DIR="$STORE" "$v" \
       bash "$root/scripts/tdp_gen.sh" --source-wires 64 --run-directory "$tmp/m/forbidden" > "$tmp/f.log" 2>&1; then
    echo "inherited $v was accepted" >&2; exit 1
  fi
  grep -q "FATAL: ${v%%=*} is not honoured by the managed recipe" "$tmp/f.log" || { cat "$tmp/f.log" >&2; exit 1; }
  test ! -e "$tmp/m/forbidden"
done
# neutral cache sizes inherited are ignored with a note (the driver's values win)
env -i HOME="$HOME" PATH="$PATH" TDP_BIN_DIR="$BIN" FROZEN_DB_DIR="$STORE" LOOKUP_CACHE_MB=1 \
  bash "$root/scripts/tdp_gen.sh" --source-wires 64 --run-directory "$tmp/m/neutral" --run-stop-after-stage 3 > "$tmp/neutral.log" 2>&1
grep -q 'inherited LOOKUP_CACHE_MB ignored' "$tmp/neutral.log"
grep -q '"LOOKUP_CACHE_MB": "2048"' "$tmp/m/neutral.calls"
echo 'TDP defaults transcript tests passed'
