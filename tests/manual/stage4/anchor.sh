#!/bin/bash
# anchor.sh gen|mix|managed TAG... : historic anchor (docs/STAGE4_RUST_DESIGN.md §10.8).
#   gen, mix  step A: the baseline binaries (492564f0) with the archived seeds.
#   managed   step B: the MANAGED DRIVER (local_mixing_bin tdp_gen, config v2,
#             nothing set but source.wires, the store and the archived seeds in a
#             private calibration seed file) on this checkout; every artifact is
#             cmp'd with the archive (db_mixing.mpmct1 vs s3.mpmct1, packing/*).
# Re-runs gen (all 8 tags) and the mixer (named tags) with the ARCHIVED seeds on the baseline
# binaries built from 492564f0, under env -i plus the tag's CONFIG environment, and cmp's the
# outputs with the archive.  Seeds are read from seed.txt inside this script and never printed.
# Reads the archive read-only; writes to $OUT (default port_runs/anchor_A).
set -uo pipefail
ARCH=${ARCH:-/Users/rancanetti/code/local_mixing/runs/ncdb_ab_20261001/waist}
BT=${BASELINE_BIN:-/Users/rancanetti/code/lm_worktrees/port_runs/baseline_target/release}
OUT=${OUT:-/Users/rancanetti/code/lm_worktrees/port_runs/anchor_A}
STORE=${FROZEN_DB_DIR:-$HOME/code/local_mixing/frozen_db/frozen_m1_m11}
step=${1:?usage: anchor.sh gen|mix TAG...}; shift
mkdir -p "$OUT"; umask 077
gen_seed() { sed -n 1p "$ARCH/$1/seed.txt" | sed -E 's/.*gen seed: ([0-9]+)$/\1/'; }
mix_seed() { sed -n 2p "$ARCH/$1/seed.txt" | sed -E 's/^mix seed: ([0-9]+)$/\1/'; }
cfg_env() {
  local e=(EMBEDDED_MASKING_SHUFFLING=8 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=0 EMBEDDED_MASKING_PREMASK=1 EMBEDDED_MASKING_SEED_AVOID_C1=1)
  case "$1" in vL*) e+=(EMBEDDED_MASKING_CHAIN=2 EMBEDDED_MASKING_CHAIN_LEN=8);; esac
  printf '%s\n' "${e[@]}"
}
case "$step" in
gen)
  tags=("$@"); [ ${#tags[@]} -eq 0 ] && tags=(vA64a vA64b vL64a vL64b vA128a vA128b vL128a vL128b)
  for t in "${tags[@]}"; do
    a=$ARCH/$t; o=$OUT/$t; mkdir -p "$o"
    hdr=$(head -1 "$a/tdp.mpmct1")
    n=$(awk '{print $2/4}' <<<"$hdr")
    case $n in 64) mc=1000; s=384; slice=1280;; 128) mc=3000; s=896; slice=2560;; *) echo "$t: unexpected n=$n"; exit 2;; esac
    sd=$(gen_seed "$t"); ce=($(cfg_env "$t"))
    env -i PATH=/usr/bin:/bin HOME="$HOME" "${ce[@]}" "$BT/gen_sandwich_gadget" "$o/tdp.mpmct1" "$n" "$mc" "$mc" "$s" 1 "$slice" "$sd" "$sd" "$sd" > "$o/gen.log" 2>&1
    rc=$?; unset sd
    res="$t gen exit=$rc"
    for f in tdp.mpmct1 tdp.mpmct1.source_c.g57 tdp.mpmct1.sandwich.mpmct1; do
      if cmp -s "$a/$f" "$o/$f"; then res="$res $f=IDENTICAL"; else res="$res $f=DIFFERENT"; fi
    done
    if cmp -s <(sed "1d; s|$a|<DIR>|g" "$a/gen.log") <(sed "1d; s|$o|<DIR>|g" "$o/gen.log"); then res="$res gen.log[2:](paths normalized)=IDENTICAL"; else res="$res gen.log[2:](paths normalized)=DIFFERENT"; fi
    echo "$res" | tee -a "$OUT/summary.txt"
  done ;;
mix)
  for t in "$@"; do
    a=$ARCH/$t; o=$OUT/$t; mkdir -p "$o"
    g=$(head -1 "$a/tdp.mpmct1" | awk '{print $3}')
    moves=$(python3 -c "print(int($g*2*2*1.3))")
    ms=$(mix_seed "$t")
    /usr/bin/time -l env -i PATH=/usr/bin:/bin HOME="$HOME" FROZEN_DB_DIR="$STORE" FROZEN_FILTER=1 \
      "$BT/circuit_mixer" --input "$a/tdp.mpmct1" --output "$o/s3.mpmct1" --tdp --db-mixing \
      --profile 2,2,2,2,2 --moves "$moves" --p-db 1.0 --db-max-degree 9 --db-max-span 30 \
      --db-wire-terms 1024 --db-total-terms 2048 --no-local-verify --seed "$ms" > "$o/s3.log" 2> "$o/s3.time"
    rc=$?; unset ms
    if cmp -s "$a/s3.mpmct1" "$o/s3.mpmct1"; then r=IDENTICAL; else r=DIFFERENT; fi
    echo "$t mix exit=$rc s3.mpmct1=$r" | tee -a "$OUT/summary.txt"
  done ;;
managed)
  REPO=$(cd "$(dirname "$0")/../../.." && pwd)
  LMB=${LOCAL_MIXING_BIN:?path to the local_mixing_bin built from this checkout}
  TGT=${MANAGED_TARGET_DIR:?cargo target directory for the managed build}
  for t in "$@"; do
    a=$ARCH/$t; o=$OUT/managed_$t; rm -rf "$o"; mkdir -p "$o"
    n=$(awk '{print $2/4}' <<<"$(head -1 "$a/tdp.mpmct1")")
    sf=$o/seeds.private
    ( set +x; g=$(gen_seed "$t"); m=$(mix_seed "$t")
      printf 'gen_seed=%s\nmix_seed=%s\n' "$g" "$m" > "$sf" ); chmod 600 "$sf"
    case $t in vA*) preset='preprocessing.preset = "A"' ;; *) preset="" ;; esac
    cat > "$o/config.toml" <<CFG
config_version = 2
[run]
directory = "$o/run"
build_directory = "$TGT"
[source]
wires = $n
[database]
regular_dir = "$STORE"
[preprocessing]
$preset
[calibration]
enabled = true
seed_file = "$sf"
CFG
    /usr/bin/time -l env -i PATH="$(dirname "$(command -v cargo)"):/opt/homebrew/bin:/usr/bin:/bin" HOME="$HOME" "$LMB" tdp_gen --config "$o/config.toml" \
      > "$o/driver.out" 2> "$o/driver.err"
    rc=$?
    r=$o/run; p=$r/packing
    res="$t managed exit=$rc"
    chk() { if cmp -s "$1" "$2"; then res="$res $3=IDENTICAL"; else res="$res $3=DIFFERENT"; fi; }
    chk "$a/tdp.mpmct1" "$r/tdp.mpmct1" tdp.mpmct1
    chk "$a/tdp.mpmct1.source_c.g57" "$r/tdp.mpmct1.source_c.g57" source_c
    chk "$a/tdp.mpmct1.sandwich.mpmct1" "$r/tdp.mpmct1.sandwich.mpmct1" sandwich
    chk "$a/s3.mpmct1" "$r/db_mixing.mpmct1" s3=db_mixing
    for f in packed_G512.mpmct1 packed_G512.mpmct1.bounds rep512.mpmct1 rep512.bounds \
             rep512.exposure.tsv rep512.allcuts.tsv rep512.wit.tsv deliverable.bin; do
      chk "$a/$f" "$p/$f" "$f"
    done
    chk "$a/deliverable.bin" <(tail -c +49 "$p/deliverable.tdp4") "deliverable.tdp4[48:]"
    chk <(sed "s/^$t-rep512:/X:/" "$a/anf_rep512.txt") <(sed "s/^tdp-rep512:/X:/" "$p/anf_rep512.txt") "anf_rep512(label)"
    chk <(sed -n 2,99p "$a/link_rep512.txt") <(sed -n 2,99p "$p/link_rep512.txt") "link_rep512[2:]"
    chk <(sed "s/^ $t:/ X:/" "$a/mult_s2.txt") <(sed "s/^ tdp:/ X:/" "$r/stage2_mult.txt") "mult_s2(label)"
    grep -q '^status=ok$' "$p/packing.report" && res="$res status=ok" || res="$res status!=ok"
    echo "$res" | tee -a "$OUT/summary.txt"
  done ;;
*) echo "usage: anchor.sh gen|mix|managed TAG..."; exit 2;;
esac
