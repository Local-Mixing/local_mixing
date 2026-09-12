#!/bin/bash
# On n64tests: copy the 323G regular frozen store to test-1 and ran-gpu.
# Staged into .incoming, verified (count + sizes + spot hashes), then swapped.
SRC=~/frozen_m1_m11
SSHCMD="ssh -i $HOME/.ssh/fleet_xfer_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
log() { printf '%s %s\n' "$(date -Is)" "$*"; }
fail=0

# Spot-hash manifest: tables.bin, filters.bin, and 8 fixed shards.
SPOT="tables.bin filters.bin shard_00.frz shard_1f.frz shard_40.frz shard_7e.frz shard_9c.frz shard_b3.frz shard_dd.frz shard_ff.frz"
( cd $SRC && sha256sum $SPOT > /tmp/regular_spot.sha256 )
src_count=$(ls $SRC | wc -l)
src_bytes=$(du -sb $SRC | cut -f1)
log "source: $src_count files, $src_bytes bytes"

for ip in 129.114.108.89 129.114.109.22; do
  avail=$($SSHCMD -n "cc@$ip" "df --output=avail -BG ~ | tail -1 | tr -dc 0-9")
  log "[$ip] ${avail}G available"
  if [ "$avail" -lt 330 ]; then log "[$ip] SKIP: not enough room"; fail=1; continue; fi
  log "[$ip] rsync starting"
  rsync -a --whole-file -e "$SSHCMD" "$SRC/" "cc@$ip:frozen_m1_m11.incoming/" \
    && rsync -a -e "$SSHCMD" /tmp/regular_spot.sha256 "cc@$ip:frozen_m1_m11.incoming.spot"
  if [ $? -ne 0 ]; then log "[$ip] RSYNC FAILED"; fail=1; continue; fi
  $SSHCMD -n "cc@$ip" "
    cd ~/frozen_m1_m11.incoming || exit 1
    [ \$(ls | wc -l) -eq $src_count ] || exit 2
    [ \$(du -sb ~/frozen_m1_m11.incoming | cut -f1) -eq $src_bytes ] || exit 3
    sha256sum --quiet -c ~/frozen_m1_m11.incoming.spot || exit 4
    cd ~
    mv frozen_m1_m11.incoming frozen_m1_m11
    rm -f frozen_m1_m11.incoming.spot
    du -sh frozen_m1_m11
  "
  rc=$?
  if [ $rc -ne 0 ]; then log "[$ip] VERIFY/SWAP FAILED rc=$rc"; fail=1; else log "[$ip] REGULAR DEPLOYED"; fi
done

if [ $fail -eq 0 ]; then log "REGULAR DEPLOY DONE"; else log "REGULAR DEPLOY FAILED"; fi
