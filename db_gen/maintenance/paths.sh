#!/bin/bash
# Sourced by repository-based DB tools. This only resolves source/build/data
# paths; it never creates, scans, or modifies a store.
DB_GEN_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) || return 1
DB_GEN_REPO_ROOT=$(cd -- "${DB_GEN_REPO_ROOT:-$DB_GEN_SCRIPT_DIR/../..}" && pwd -P) || return 1
if [ ! -f "$DB_GEN_REPO_ROOT/Cargo.toml" ]; then
  printf 'DB_GEN_REPO_ROOT has no Cargo.toml: %s\n' "$DB_GEN_REPO_ROOT" >&2
  return 1
fi

# Keep the historical run root independent of the source checkout.
DB_GEN_RUN_ROOT=${DB_GEN_RUN_ROOT:-"$HOME/curated_diversity_20260815"}
DB_GEN_REPORT_ROOT=${DB_GEN_REPORT_ROOT:-"$DB_GEN_REPO_ROOT/reports/curated_diversity_20260815"}
case "$DB_GEN_RUN_ROOT" in /*) ;; *) DB_GEN_RUN_ROOT="$PWD/$DB_GEN_RUN_ROOT" ;; esac
case "$DB_GEN_REPORT_ROOT" in /*) ;; *) DB_GEN_REPORT_ROOT="$PWD/$DB_GEN_REPORT_ROOT" ;; esac

# Match build and executable lookup even when invoked outside the checkout.
DB_GEN_TARGET_DIR=${DB_GEN_TARGET_DIR:-${CARGO_TARGET_DIR:-"$DB_GEN_REPO_ROOT/target"}}
case "$DB_GEN_TARGET_DIR" in /*) ;; *) DB_GEN_TARGET_DIR="$DB_GEN_REPO_ROOT/$DB_GEN_TARGET_DIR" ;; esac
DB_GEN_CARGO=${DB_GEN_CARGO:-"$HOME/.cargo/bin/cargo"}
export DB_GEN_REPO_ROOT DB_GEN_RUN_ROOT DB_GEN_REPORT_ROOT DB_GEN_TARGET_DIR DB_GEN_CARGO

db_gen_build_release() (
  cd -- "$DB_GEN_REPO_ROOT" || exit 1
  "$DB_GEN_CARGO" build --release --features db-tools \
    --manifest-path "$DB_GEN_REPO_ROOT/Cargo.toml" \
    --target-dir "$DB_GEN_TARGET_DIR" --bin "$1"
)
