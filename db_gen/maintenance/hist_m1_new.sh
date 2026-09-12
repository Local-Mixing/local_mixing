#!/bin/bash
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh" || exit 1
"$DB_GEN_TARGET_DIR/release/curated_key_histogram" "$DB_GEN_RUN_ROOT/composite_v2_sieved" 66ca88066ac7c7a878442082635feaeb 2>/dev/null
