#!/usr/bin/env bash
# Inspect one frozen store without modifying it or contacting another machine.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash db_gen/maintenance/check_filters.sh FROZEN_DIR [EXPECTED_SHA256]

Require tables.bin, all 256 shard files, and filters.bin; print the filter's
SHA-256 and optionally compare it with a known digest. This checks file layout
and identity, not filter membership. frozen_filters_build validates membership
against every retained key when constructing the filter.
USAGE
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if (($# < 1 || $# > 2)); then usage >&2; exit 2; fi
store=$1
expected=${2:-}
if [[ -n "$expected" && ! "$expected" =~ ^[[:xdigit:]]{64}$ ]]; then
  printf 'Expected SHA-256 must contain exactly 64 hexadecimal digits.\n' >&2
  exit 2
fi

for name in tables.bin filters.bin; do
  if [[ ! -f "$store/$name" ]]; then
    printf 'Missing file: %s/%s\n' "$store" "$name" >&2
    exit 1
  fi
done
for ((shard=0; shard<256; shard++)); do
  printf -v name 'shard_%02x.frz' "$shard"
  if [[ ! -f "$store/$name" ]]; then
    printf 'Missing shard: %s/%s\n' "$store" "$name" >&2
    exit 1
  fi
done

digest_line=$(sha256sum -- "$store/filters.bin")
# GNU sha256sum prefixes escaped filenames with a backslash. Compare only its
# hexadecimal digest, while retaining the complete output for the caller.
digest=${digest_line#\\}
digest=${digest%% *}
printf '%s\n' "$digest_line"
if [[ -n "$expected" && "${digest,,}" != "${expected,,}" ]]; then
  printf 'Filter SHA-256 does not match the expected digest.\n' >&2
  exit 1
fi
printf 'Frozen-store file layout complete: %s\n' "$store"
