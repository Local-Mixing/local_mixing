#!/bin/bash
# On n64tests: build filters.bin for the v2 store and install into the
# deployed dir (hardlink; same filesystem).
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh" || exit 1
log() { printf '%s %s\n' "$(date -Is)" "$*"; }
log "building binary"
db_gen_build_release frozen_filters_build || exit 1
log "building filters.bin"
"$DB_GEN_TARGET_DIR/release/frozen_filters_build" from-frozen "$DB_GEN_RUN_ROOT/frozen_curated_v2" || exit 2
ln -f "$DB_GEN_RUN_ROOT/frozen_curated_v2/filters.bin" "$HOME/frozen_curated_m1_m11_native/filters.bin" || exit 3
sha256sum "$DB_GEN_RUN_ROOT/frozen_curated_v2/filters.bin"
log "FILTERS DONE"
