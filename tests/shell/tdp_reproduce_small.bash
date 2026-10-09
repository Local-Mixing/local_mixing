#!/usr/bin/env bash
# The reproduction test at CI size (docs/STAGE4_RUST_DESIGN.md §10.9 item 4):
# tests/manual/stage4/reproduce_val_chain.sh at N=8, |C|=|D|=16, CONFIG L8,
# for real (managed driver vs sandboxed val_chain.sh, same seeds and
# binaries, every artifact byte-compared). It needs the regular frozen store
# (FROZEN_DB_DIR), the pinned baseline hmap_affine (HMAP_BASELINE) and release
# binaries including local_mixing_bin (BIN_DIR); it skips with a message
# otherwise (as in CI, where the stage4-oracle differential covers stage 4).
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
BIN_DIR=${BIN_DIR:-${CARGO_TARGET_DIR:-$root/target}/release}
missing=()
[ -n "${FROZEN_DB_DIR:-}" ] && [ -s "${FROZEN_DB_DIR:-/nonexistent}/tables.bin" ] || missing+=("FROZEN_DB_DIR (a regular frozen store)")
[ -n "${HMAP_BASELINE:-}" ] && [ -x "${HMAP_BASELINE:-/nonexistent}" ] || missing+=("HMAP_BASELINE (the pinned baseline hmap_affine)")
for b in gen_sandwich_gadget circuit_mixer tdp_packing local_mixing_bin; do
  [ -x "$BIN_DIR/$b" ] || missing+=("$BIN_DIR/$b")
done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "SKIP tdp_reproduce_small: missing ${missing[*]}"
  exit 0
fi
work=$(mktemp -d "${TMPDIR:-/tmp}/tdp_repro_small.XXXXXX")
trap 'rm -rf "$work"' EXIT
BIN_DIR=$BIN_DIR WORK=$work/run bash "$root/tests/manual/stage4/reproduce_val_chain.sh" 8 L8 16
