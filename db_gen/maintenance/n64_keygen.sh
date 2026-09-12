#!/bin/bash
# On n64tests: transfer key + manifest of the new frozen store.
set -e
KEY=~/.ssh/fleet_xfer_ed25519
ROOT=${DB_GEN_RUN_ROOT:-"$HOME/curated_diversity_20260815"}
case "$ROOT" in /*) ;; *) ROOT="$PWD/$ROOT" ;; esac
if [ ! -f "$KEY" ]; then
  ssh-keygen -t ed25519 -N "" -f "$KEY" -C "n64tests-xfer-20260816" >/dev/null
fi
cd -- "$ROOT/frozen_curated_v2"
sha256sum * > "$ROOT/frozen_curated_v2.sha256"
wc -l "$ROOT/frozen_curated_v2.sha256"
cat "$KEY.pub"
