#!/bin/bash
# On n64tests: build + run the regular->curated coverage census, detached.
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh" || exit 1
[ -d "$DB_GEN_RUN_ROOT" ] || { printf 'Missing run root: %s\n' "$DB_GEN_RUN_ROOT" >&2; exit 1; }
setsid nohup bash -c '
  source "$1" || exit 1
  db_gen_build_release curated_coverage_census > "$DB_GEN_RUN_ROOT/coverage_build.log" 2>&1 &&
    "$DB_GEN_TARGET_DIR/release/curated_coverage_census" "$HOME/frozen_m1_m11" "$DB_GEN_RUN_ROOT/composite_v2_sieved" > "$DB_GEN_RUN_ROOT/coverage.log" 2>&1
  rc=$?
  echo done >> "$DB_GEN_RUN_ROOT/coverage.log"
  exit "$rc"
' _ "$DB_GEN_SCRIPT_DIR/paths.sh" < /dev/null > /dev/null 2>&1 &
sleep 1
echo "coverage census launched"
