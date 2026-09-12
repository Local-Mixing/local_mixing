#!/bin/bash
# On n64tests: ship filters.bin to the five deployed boxes, then full-deploy
# the store (with filters) to ran-gpu.
ROOT=${DB_GEN_RUN_ROOT:-"$HOME/curated_diversity_20260815"}
case "$ROOT" in /*) ;; *) ROOT="$PWD/$ROOT" ;; esac
SRC=$ROOT/frozen_curated_v2
SSHCMD="ssh -i $HOME/.ssh/fleet_xfer_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
log() { printf '%s %s\n' "$(date -Is)" "$*"; }
fail=0

if [ ! -f "$SRC/filters.bin" ]; then log "STOP: no filters.bin"; exit 90; fi
( cd "$SRC" && sha256sum * > "$ROOT/frozen_curated_v2.sha256" )
want=$(sha256sum "$SRC/filters.bin" | cut -d' ' -f1)

for ip in 129.114.108.242 129.114.109.41 129.114.108.159 129.114.109.6 129.114.108.89; do
  rsync -a -e "$SSHCMD" "$SRC/filters.bin" "cc@$ip:frozen_curated_m1_m11_native/filters.bin"
  got=$($SSHCMD -n "cc@$ip" "sha256sum ~/frozen_curated_m1_m11_native/filters.bin | cut -d' ' -f1")
  if [ "$got" = "$want" ]; then log "[$ip] FILTERS OK"; else log "[$ip] FILTERS MISMATCH"; fail=1; fi
done

log "[ran-gpu] full store deploy"
rsync -a --delete -e "$SSHCMD" "$SRC/" "cc@129.114.109.22:frozen_curated_m1_m11_native.incoming/" \
  && rsync -a -e "$SSHCMD" "$ROOT/frozen_curated_v2.sha256" "cc@129.114.109.22:frozen_curated_m1_m11_native.incoming.sha256"
if [ $? -ne 0 ]; then log "[ran-gpu] RSYNC FAILED"; fail=1; else
  $SSHCMD -n cc@129.114.109.22 '
    cd ~/frozen_curated_m1_m11_native.incoming || exit 1
    sha256sum --quiet -c ~/frozen_curated_m1_m11_native.incoming.sha256 || exit 2
    cd ~
    [ -d frozen_curated_m1_m11_native ] && mv frozen_curated_m1_m11_native "frozen_curated_m1_m11_native_pre_$(date +%s)"
    mv frozen_curated_m1_m11_native.incoming frozen_curated_m1_m11_native
    rm -f frozen_curated_m1_m11_native.incoming.sha256
    du -sh frozen_curated_m1_m11_native
  '
  if [ $? -ne 0 ]; then log "[ran-gpu] VERIFY/SWAP FAILED"; fail=1; else log "[ran-gpu] DEPLOYED"; fi
fi

if [ $fail -eq 0 ]; then log "FILTER DISTRIBUTION DONE"; else log "FILTER DISTRIBUTION FAILED"; fi
