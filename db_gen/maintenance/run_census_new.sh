#!/bin/bash
# On n64tests: min-gates census of the new sieved store.
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh" || exit 1
db_gen_build_release curated_size_census > /dev/null 2>&1 || exit 1
"$DB_GEN_TARGET_DIR/release/curated_size_census" "$DB_GEN_RUN_ROOT/composite_v2_sieved" --top 10 2>&1 | grep -vE '^\[census\] keys='
