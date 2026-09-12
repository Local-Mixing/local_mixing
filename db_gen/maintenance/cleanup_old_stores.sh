#!/bin/bash
# On n64tests: remove the superseded curated stores fleet-wide, leaving only
# the deployed v2 under ~/frozen_curated_m1_m11_native. Run ONLY after local
# archives of the old store are verified.
SSHCMD="ssh -i $HOME/.ssh/fleet_xfer_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
log() { printf '%s %s\n' "$(date -Is)" "$*"; }
for ip in 129.114.108.242 129.114.109.41 129.114.108.159 129.114.109.6 129.114.108.89; do
  $SSHCMD -n "cc@$ip" '
    rm -rf ~/frozen_curated_m1_m11 ~/frozen_curated_m1_m11_native_v1
    ls -d ~/frozen_curated_m1_m11* 2>/dev/null
  '
  log "[$ip] cleaned"
done
rm -rf ~/frozen_curated_m1_m11 ~/frozen_curated_m1_m11_native_v1
ls -d ~/frozen_curated_m1_m11* 2>/dev/null
log "n64tests cleaned"
log "CLEANUP DONE"
