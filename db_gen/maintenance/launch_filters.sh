#!/bin/bash
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh" || exit 1
[ -d "$DB_GEN_RUN_ROOT" ] || { printf 'Missing run root: %s\n' "$DB_GEN_RUN_ROOT" >&2; exit 1; }
setsid nohup bash "$DB_GEN_SCRIPT_DIR/build_filters.sh" < /dev/null > "$DB_GEN_RUN_ROOT/filters_build.log" 2>&1 &
sleep 1
echo "filters build launched"
