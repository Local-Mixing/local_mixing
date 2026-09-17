#!/usr/bin/env bash
# Build and validate a missing filter directly from one explicit frozen store.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash db_gen/maintenance/build_filters.sh [--skip-build] FROZEN_DIR

Build filters.bin from the store's encoded keys and validate every retained
key before publishing it. Existing filters.bin files are never overwritten.
No filters are copied or deployed to another store.

--skip-build uses an existing release/frozen_filters_build executable.
DB_GEN_TARGET_DIR (or CARGO_TARGET_DIR) selects its build directory;
DB_GEN_CARGO selects Cargo, which otherwise comes from PATH.
USAGE
}

skip_build=false
while (($#)); do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --skip-build) skip_build=true; shift ;;
    --) shift; break ;;
    -*) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    *) break ;;
  esac
done
if (($# != 1)); then usage >&2; exit 2; fi

store=$1
if [[ ! -f "$store/tables.bin" ]]; then
  printf 'Missing frozen-store tables.bin: %s\n' "$store" >&2
  exit 1
fi
if [[ -e "$store/filters.bin" || -L "$store/filters.bin" ]]; then
  printf 'Refusing to overwrite existing filters.bin: %s\n' "$store" >&2
  exit 1
fi
# Resolve before sourcing build paths; paths supplied by the caller are relative
# to the caller's working directory, not to the source checkout.
store=$(cd -- "$store" && pwd -P)
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"
if [[ "$skip_build" == false ]]; then
  db_gen_build_release frozen_filters_build
fi
builder="$DB_GEN_TARGET_DIR/release/frozen_filters_build"
if [[ ! -x "$builder" ]]; then
  printf 'Missing executable: %s\n' "$builder" >&2
  exit 1
fi
"$builder" from-frozen "$store"
sha256sum -- "$store/filters.bin"
