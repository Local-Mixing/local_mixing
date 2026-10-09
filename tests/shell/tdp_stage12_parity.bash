#!/usr/bin/env bash
# Stage 1+2 parity on REAL binaries (docs/STAGE4_RUST_DESIGN.md §10.9 item 2):
# at n=16 with |C|=|D|=32, gen_sandwich_gadget run exactly as val_chain.sh runs
# it (argv and environment, CONFIG L8 and A) and scripts/tdp_gen.sh with the
# same seeds produce byte-identical tdp.mpmct1, .source_c.g57 and
# .sandwich.mpmct1. Binaries: $TDP_STAGE_BIN (default target/debug, then
# target/release); skipped with a message when none is built.
set -euo pipefail
if [[ ${1-} != --isolated-environment ]]; then
  exec env -i PATH="$PATH" HOME="${HOME:-/tmp}" TDP_STAGE_BIN="${TDP_STAGE_BIN:-}" \
    bash "$0" --isolated-environment
fi
shift
root=$(cd "$(dirname "$0")/../.." && pwd)
BIN=${TDP_STAGE_BIN:-}
if [ -z "$BIN" ]; then
  for d in "$root/target/debug" "$root/target/release"; do
    [ -x "$d/gen_sandwich_gadget" ] && [ -x "$d/circuit_mixer" ] && { BIN=$d; break; }
  done
fi
if [ -z "$BIN" ] || [ ! -x "$BIN/gen_sandwich_gadget" ]; then
  echo "SKIP tdp_stage12_parity: no built gen_sandwich_gadget (cargo build first, or set TDP_STAGE_BIN)"
  exit 0
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
read -r GS MS < <(python3 -I -c 'import secrets; print(secrets.randbelow(2**63 - 15), secrets.randbelow(2**63 - 15))')
sf=$tmp/seeds; printf 'gen_seed=%s\nmix_seed=%s\n' "$GS" "$MS" > "$sf"; chmod 600 "$sf"
for cfg in L8 A; do
  ref=$tmp/ref_$cfg; mkdir -p "$ref"
  env_cfg=(EMBEDDED_MASKING_SHUFFLING=8 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=0 EMBEDDED_MASKING_PREMASK=1 EMBEDDED_MASKING_SEED_AVOID_C1=1)
  [ "$cfg" = L8 ] && env_cfg+=(EMBEDDED_MASKING_CHAIN=2 EMBEDDED_MASKING_CHAIN_LEN=8)
  # val_chain.sh line 15-16: N MC MC S 1 SLICE SEED SEED SEED (s = max(n, round(n log2 n)) = 64, slice = 20n)
  env -i HOME="$HOME" PATH="$PATH" "${env_cfg[@]}" \
    "$BIN/gen_sandwich_gadget" "$ref/tdp.mpmct1" 16 32 32 64 1 320 "$GS" "$GS" "$GS" > "$ref/gen.log" 2>&1
  env -i HOME="$HOME" PATH="$PATH" TDP_BIN_DIR="$BIN" \
    bash "$root/scripts/tdp_gen.sh" --source-wires 16 --source-gates 32 --run-directory "$tmp/man_$cfg" \
      --seed-file "$sf" --preprocessing-preset "$cfg" --run-stop-after-stage 2 > "$tmp/man_$cfg.log" 2>&1 ||
    { cat "$tmp/man_$cfg.log" >&2; exit 1; }
  for f in tdp.mpmct1 tdp.mpmct1.source_c.g57 tdp.mpmct1.sandwich.mpmct1; do
    cmp "$ref/$f" "$tmp/man_$cfg/$f" || { echo "stage-2 parity FAILED ($cfg): $f" >&2; exit 1; }
  done
  # gen's stdout after its banner (paths normalized) is identical too
  cmp <(sed "1d; s|$ref/||g" "$ref/gen.log") <(sed "1d; s|$tmp/man_$cfg/||g" "$tmp/man_$cfg/stage12.log")
  echo "stage-2 parity $cfg: tdp.mpmct1, source C, sandwich identical ($(head -1 "$ref/tdp.mpmct1" | awk '{print $3}') gates)"
done
echo 'TDP stage 1+2 parity tests passed'
