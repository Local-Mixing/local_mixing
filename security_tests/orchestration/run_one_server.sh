#!/usr/bin/env bash
# Run one isolated, deliverable-grade n=128 GSS-MIX circuit and verify the
# end-to-end zero-slice wire-shift contract against its original source C.
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: run_one_server.sh TOOLROOT RUNROOT SERVER_ALIAS SOURCE_COMMIT" >&2
  exit 2
fi

toolroot=$1
runroot=$2
server_alias=$3
source_commit=$4
status=$runroot/STATUS

umask 077
mkdir -p "$runroot"
if [[ -e "$status" && "${RED_TEAM_ALLOW_RESUME:-0}" != 1 ]]; then
  echo "refusing to reuse $runroot without RED_TEAM_ALLOW_RESUME=1" >&2
  exit 3
fi

on_exit() {
  rc=$?
  trap - EXIT
  if [[ $rc -eq 0 ]]; then
    printf 'COMPLETE\n' > "$status"
  else
    printf 'FAILED rc=%s\n' "$rc" > "$status"
  fi
  date -u +finished_utc=%FT%TZ >> "$runroot/run.env"
  df -h "$runroot" > "$runroot/disk_after.txt" 2>&1 || true
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

gen=$toolroot/target/release/gen_sandwich_gadget
fmix=$toolroot/target/release/fmix
fcompress=$toolroot/target/release/fcompress
verify=$toolroot/target/release/verify_zero_slice
driver=$toolroot/scripts/gss_mix.sh

for file in "$gen" "$fmix" "$fcompress" "$verify" "$driver"; do
  [[ -x "$file" ]] || { echo "missing executable: $file" >&2; exit 4; }
done

check_hash() {
  expected=$1
  file=$2
  actual=$(sha256sum "$file" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    echo "hash mismatch for $file: expected=$expected actual=$actual" >&2
    exit 5
  }
}
check_hash 141c08b5c78e42982be9eddeaa9e9e7ebd8cb62637aea661ab9111ae05dcd5aa "$gen"
check_hash 343ac77a4ff4f3edf3a2f889e5f33951ad0a481c7958ea465668934c97e04971 "$fmix"
check_hash 2e801b5dabe173beb468ed175199c2a916d58217d8d0bcc30f4c5197be2742bb "$fcompress"
check_hash bdadc70be97444d2357cb34f8357f8ba06e85b3495121e4dc80343730a0e89ac "$verify"
check_hash 357e2fc5e63f51ab2808860ba35e0a953ca9c253e86e182c0604535aa226de30 "$driver"

regular_store=/home/cc/frozen_m1_m11
curated_store=/home/cc/frozen_curated_m1_m11
regular_shards=("$regular_store"/shard_*.frz)
curated_shards=("$curated_store"/shard_*.frz)
[[ ${#regular_shards[@]} -eq 256 ]] || { echo "regular store is incomplete" >&2; exit 6; }
[[ ${#curated_shards[@]} -eq 256 ]] || { echo "curated store is incomplete" >&2; exit 6; }
[[ -r "$regular_store/tables.bin" && -r "$regular_store/filters.bin" ]] || exit 6
[[ -r "$curated_store/tables.bin" && -r "$curated_store/filters.bin" ]] || exit 6

available_kib=$(df -Pk "$runroot" | awk 'NR == 2 {print $4}')
memory_kib=$(awk '$1 == "MemAvailable:" {print $2}' /proc/meminfo)
(( available_kib >= 20 * 1024 * 1024 )) || { echo "less than 20 GiB free" >&2; exit 7; }
(( memory_kib >= 100 * 1024 * 1024 )) || { echo "less than 100 GiB RAM available" >&2; exit 7; }

printf 'RUNNING pid=%s\n' "$$" > "$status"
{
  date -u +started_utc=%FT%TZ
  printf 'server_alias=%s\n' "$server_alias"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'source_commit=%s\n' "$source_commit"
  printf 'n=128\n'
  printf 'source_gate_rule=round(n*(log2(n))^2)\n'
  printf 'phase_a_expand=2\nphase_a_hold_effs=30\n'
  printf 'cross_xr=2\ncross_xb=3\ncross_xc=1\ncross_xtdiv=25\n'
  printf 'cross_xmoves=default_6x_target_stop_at_arrival\n'
  printf 'seed=private_pipeline_SEED_file_CSPRNG\n'
  printf 'regular_store=%s\ncurated_store=%s\n' "$regular_store" "$curated_store"
  printf 'regular_value_convention=native\n'
  printf 'curated_value_convention=legacy-swapped-controls\n'
  printf 'frozen_filter=1\n'
} > "$runroot/run.env"

sha256sum "$gen" "$fmix" "$fcompress" "$verify" "$driver" \
  > "$runroot/toolchain.sha256"
{
  uname -a
  lscpu
  free -h
  df -h "$runroot"
} > "$runroot/host_before.txt"

# Ensure the production preset is actually the compiled default, independent
# of any ambient experiment shell.
unset PROD_K PROD_DEG PROD_K_HI PROD_DEG_HI PROD_BAND PROD_RSRC
unset PROD_MAX_WIDTH PROD_FILL_NL PROD_ROLL PROD_SRC_DIST PROD_SRC_HORIZON
unset PROD_SRC_LO PROD_SRC_HI PROD_FILL_PIVOTS PROD_G57_NARROW PROD_LADDER_CAP
unset PROD_CG_JITTER PROD_RUNG_MENU PROD_EPOCH PROD_REFILL_DATA PROD_SINGLE
unset PROD_GRAY_FOLD GSS_MIX_ALLOW_EMPTY_STORE

export FROZEN_DB_DIR=$regular_store
export FROZEN_CURATED_DIR=$curated_store
export FROZEN_REGULAR_VALUE_CONVENTION=native
export FROZEN_CURATED_VALUE_CONVENTION=legacy-swapped-controls
export FROZEN_FILTER=1

pipeline=$runroot/pipeline
"$driver" -n 128 -o "$pipeline" \
  --expand 2 --hold 30 --xr 2 --xb 3 --xc 1 --xtdiv 25 \
  > "$runroot/driver.log" 2>&1

source_c=$pipeline/gss.mpmct1.source_c.g57
final=$pipeline/final.mpmct1
[[ -s "$source_c" && -s "$final" ]] || { echo "pipeline outputs missing" >&2; exit 8; }

# A fresh verifier seed does not regenerate C and is safe to record. The
# verifier directly evaluates the original n-wire C and the final 4n-wire
# circuit on 512 independently sampled zero-slice inputs.
verify_seed=$(python3 -c 'import secrets; print(secrets.randbits(64))')
"$verify" "$source_c" "$final" 128 512 "$verify_seed" \
  > "$runroot/verify_zero_slice.log" 2>&1
grep -q '^\[verify\] PASS:' "$runroot/verify_zero_slice.log"
printf 'verify_samples=512\nverify_seed=%s\n' "$verify_seed" >> "$runroot/run.env"

head -n 1 "$pipeline/gss.mpmct1" > "$runroot/initial_gss.header"
head -n 1 "$final" > "$runroot/final.header"
sha256sum \
  "$source_c" \
  "$pipeline/gss.mpmct1.sandwich.mpmct1" \
  "$pipeline/gss.mpmct1" \
  "$final" \
  > "$runroot/circuits.sha256"

