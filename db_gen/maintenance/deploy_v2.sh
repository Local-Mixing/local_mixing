#!/bin/bash
# On n64tests: deploy frozen_curated_v2 to the fleet as frozen_curated_m1_m11_native.
# Per target: rsync -> .incoming, sha256 verify, swap (old kept as _v1).
ROOT=${DB_GEN_RUN_ROOT:-"$HOME/curated_diversity_20260815"}
case "$ROOT" in /*) ;; *) ROOT="$PWD/$ROOT" ;; esac
SRC=$ROOT/frozen_curated_v2
MANIFEST=$ROOT/frozen_curated_v2.sha256
SSHCMD="ssh -i $HOME/.ssh/fleet_xfer_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
TARGETS="129.114.108.242 129.114.109.41 129.114.108.159 129.114.109.6 129.114.108.89"
log() { printf '%s %s\n' "$(date -Is)" "$*"; }

fail=0
for ip in $TARGETS; do
  log "[$ip] rsync"
  rsync -a --delete -e "$SSHCMD" "$SRC/" "cc@$ip:frozen_curated_m1_m11_native.incoming/" \
    && rsync -a -e "$SSHCMD" "$MANIFEST" "cc@$ip:frozen_curated_m1_m11_native.incoming.sha256"
  if [ $? -ne 0 ]; then log "[$ip] RSYNC FAILED"; fail=1; continue; fi
  log "[$ip] verify + swap"
  $SSHCMD -n "cc@$ip" '
    cd ~/frozen_curated_m1_m11_native.incoming || exit 1
    sha256sum --quiet -c ~/frozen_curated_m1_m11_native.incoming.sha256 || exit 2
    cd ~
    if [ -d frozen_curated_m1_m11_native ] && [ ! -d frozen_curated_m1_m11_native_v1 ]; then
      mv frozen_curated_m1_m11_native frozen_curated_m1_m11_native_v1
    elif [ -d frozen_curated_m1_m11_native ]; then
      mv frozen_curated_m1_m11_native "frozen_curated_m1_m11_native_pre_$(date +%s)"
    fi
    mv frozen_curated_m1_m11_native.incoming frozen_curated_m1_m11_native
    rm -f frozen_curated_m1_m11_native.incoming.sha256
    du -sh frozen_curated_m1_m11_native
  '
  rc=$?
  if [ $rc -ne 0 ]; then log "[$ip] VERIFY/SWAP FAILED rc=$rc"; fail=1; else log "[$ip] DEPLOYED"; fi
done

log "local n64tests swap (hardlink copy)"
cd ~
if [ -d frozen_curated_m1_m11_native ] && [ ! -d frozen_curated_m1_m11_native_v1 ]; then
  mv frozen_curated_m1_m11_native frozen_curated_m1_m11_native_v1
fi
rm -rf frozen_curated_m1_m11_native.tmp
cp -al "$SRC" frozen_curated_m1_m11_native.tmp \
  && (cd frozen_curated_m1_m11_native.tmp && sha256sum --quiet -c "$MANIFEST") \
  && mv frozen_curated_m1_m11_native.tmp frozen_curated_m1_m11_native \
  && du -sh frozen_curated_m1_m11_native \
  || { log "LOCAL SWAP FAILED"; fail=1; }

if [ $fail -eq 0 ]; then log "DEPLOY ALL DONE"; else log "DEPLOY FINISHED WITH FAILURES"; fi
