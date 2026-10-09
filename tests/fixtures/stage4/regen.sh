#!/bin/bash
# regen.sh : regenerate the stage-4 oracle fixtures (docs/STAGE4_RUST_DESIGN.md
# §10.3). Every expected output comes from the Python/bash ORACLE run in a
# path-rewritten sandbox (oracle_sandbox.sh) under an interpreter that passes
# the CPython 3.9 set-order self-test, with the PINNED baseline hmap_affine.
#
#   HMAP_BASELINE=.../baseline_target/release/hmap_affine ORACLE_PY=/usr/bin/python3 \
#   CARGO_TARGET_DIR=... FROZEN_DB_DIR=... WORK=<durable scratch dir> \
#   bash regen.sh [n8_L8] [anf] [container] [pack]
#
# Parts:
#   n8_L8  real chain at n=8 (|C|=|D|=16, preset L8): gen_sandwich_gadget and
#          circuit_mixer from this worktree (release), then the oracle stage 4
#          exactly as val_chain.sh runs it (pack 512 512 24576 3000, repair.sh,
#          anf_dump_eval.py) plus anf_compose.py reports.
#   anf    crafted (circuit, bounds) cases from anf_cases.py: wide states
#          (W = 1, 2, 4, 8, 16 limbs), k=0 / complemented / negative gates,
#          duplicate and out-of-range bounds, empty windows; oracle
#          anf_dump_eval.py bodies, anf_compose.py reports at several budgets,
#          and per-window window_anf decisions (anf_windows.py).
#   container  48-byte container headers of every fixture body, from the
#          standalone writer tests/manual/stage4/container_header.py.
#   pack   the packer window_pack3.py on the cases of pack/cases.txt: the real
#          n=8 (G2, G3) pair at production and crafted argv, and synthetic
#          pairs from pack_cases.py (wide states, dense ANF, ancilla wires).
#          Expected: stdout, bounds, and blake2b-128 digests of the packed
#          circuit and of the emitted order (WP's PACK_DUMP_ORDER). Coverage
#          is asserted from the oracle logs and tdp_packing --stats.
#
# Seeds here are CALIBRATION values, never deliverable seeds.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
: "${HMAP_BASELINE:?}" "${ORACLE_PY:?}" "${WORK:?set WORK to a durable scratch directory}"
PARTS=("$@"); [ ${#PARTS[@]} -eq 0 ] && PARTS=(n8_L8 anf container pack)
want() { local p; for p in "${PARTS[@]}"; do [ "$p" = "$1" ] && return 0; done; return 1; }

if want n8_L8 || want anf || want pack; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  SBR=$WORK/fixture_$STAMP
  RD=$(bash "$HERE/oracle_sandbox.sh" "$SBR")
  W=$RD/waist
  echo "sandbox: $SBR"
fi

if want n8_L8; then
  : "${CARGO_TARGET_DIR:?}" "${FROZEN_DB_DIR:?}"
  BIN=$CARGO_TARGET_DIR/release
  (cd "$REPO" && cargo build --locked --release --bin gen_sandwich_gadget --bin circuit_mixer)
  TAG=n8_L8; N=8; MC=16; S=24; SLICE=160
  GEN_SEED=11; MIX_SEED=1013   # CALIBRATION
  R=$W/$TAG; mkdir -p "$R"
  env -i HOME="$HOME" PATH=/usr/bin:/bin \
    EMBEDDED_MASKING_SHUFFLING=8 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=0 \
    EMBEDDED_MASKING_PREMASK=1 EMBEDDED_MASKING_SEED_AVOID_C1=1 \
    EMBEDDED_MASKING_CHAIN=2 EMBEDDED_MASKING_CHAIN_LEN=8 \
    "$BIN/gen_sandwich_gadget" "$R/tdp.mpmct1" $N $MC $MC $S 1 $SLICE $GEN_SEED $GEN_SEED $GEN_SEED \
    > "$R/gen.log" 2>&1
  G=$(head -1 "$R/tdp.mpmct1" | awk '{print $3}')
  MOVES=$("$ORACLE_PY" -I -c "print(int($G*2*2*1.3))")
  env -i HOME="$HOME" PATH=/usr/bin:/bin FROZEN_DB_DIR="$FROZEN_DB_DIR" FROZEN_FILTER=1 \
    "$BIN/circuit_mixer" --input "$R/tdp.mpmct1" --output "$R/s3.mpmct1" --tdp --db-mixing \
    --profile 2,2,2,2,2 --moves "$MOVES" --p-db 1.0 \
    --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048 \
    --no-local-verify --seed $MIX_SEED > "$R/s3.log" 2>&1
  (cd "$RD" && "$ORACLE_PY" window_pack3.py "$R/tdp.mpmct1" "$R/s3.mpmct1" 512 512 24576 3000 \
     $TAG-G512 "$R/packed_G512.mpmct1" > "$R/pack_G512.txt" 2>&1)
  bash "$W/repair.sh" $TAG 512 $N > "$R/repair512.log" 2>&1
  "$ORACLE_PY" "$W/anf_dump_eval.py" "$R/rep512.mpmct1" "$R/rep512.bounds" "$R/deliverable.bin" $TAG \
    > "$R/deliver.txt" 2>&1
  for b in 64 256; do
    (cd "$RD" && "$ORACLE_PY" anf_compose.py "$R/rep512.mpmct1" "$R/rep512.bounds" $b $TAG-b$b 3 \
       > "$R/anf_b$b.txt" 2>&1)
  done
  out=$HERE/n8_L8; rm -rf "$out"; mkdir -p "$out"
  cp "$R/tdp.mpmct1.source_c.g57" "$out/c.g57"
  # G2 is committed once, as cview_n8/g2.mpmct1 (same calibration run)
  cmp "$R/tdp.mpmct1" "$HERE/cview_n8/g2.mpmct1" || { echo "n8 G2 != cview_n8/g2.mpmct1" >&2; exit 1; }
  cp "$R/s3.mpmct1" "$out/g3.mpmct1"
  cp "$R/rep512.mpmct1" "$out/rep.mpmct1"
  cp "$R/rep512.bounds" "$out/rep.bounds"
  cp "$R/deliverable.bin" "$out/deliverable.bin"
  cp "$R/anf_rep512.txt" "$out/anf_rep.txt"
  cp "$R/anf_b64.txt" "$R/anf_b256.txt" "$out/"
  head -1 "$R/deliver.txt" | sed 's/ in [0-9]*s,/ in <T>s,/' > "$out/deliver_line1.txt"
  grep -o 'match=[A-Za-z]*' "$R/deliver.txt" > "$out/deliver_match.txt"
  { echo "gen: $(grep -c . "$R/tdp.mpmct1") lines; s3: $(head -1 "$R/s3.mpmct1")"
    grep -E 'fix iter|fallback iter|splits added|EQUIVALENT|MISMATCH|WARNING' "$R/repair512.log"
    sed -n 1,3p "$R/anf_rep512.txt"; } | sed "s|$SBR|<SANDBOX>|g" > "$out/provenance.txt"
fi

if want anf; then
  out=$HERE/anf; rm -rf "$out"; mkdir -p "$out"
  "$ORACLE_PY" -I "$HERE/anf_cases.py" "$out"
  while read -r name budgets; do
    c=$out/$name.mpmct1; b=$out/$name.bounds
    "$ORACLE_PY" "$W/anf_dump_eval.py" "$c" "$b" "$out/$name.bin" "$name" > "$out/$name.deliver.txt" 2>&1
    sed -i '' -e 's/ in [0-9]*s,/ in <T>s,/' "$out/$name.deliver.txt"
    grep -o 'match=[A-Za-z]*' "$out/$name.deliver.txt" > "$out/$name.match"
    head -1 "$out/$name.deliver.txt" > "$out/$name.line1"; mv "$out/$name.line1" "$out/$name.deliver.txt"
    for bud in $budgets; do
      (cd "$RD" && "$ORACLE_PY" anf_compose.py "$c" "$b" "$bud" "$name-b$bud" 2 > "$out/$name.anf_b$bud.txt" 2>&1)
      "$ORACLE_PY" "$HERE/anf_windows.py" "$RD" "$c" "$b" "$bud" > "$out/$name.win_b$bud.txt"
    done
  done < "$out/cases.txt"
fi
if want container; then
  CH=$REPO/tests/manual/stage4/container_header.py
  hdr() { # BODY N STATE OUT
    "$ORACLE_PY" -I "$CH" "$1" "$4.full" "$2" "$3"
    head -c 48 "$4.full" > "$4"; rm -f "$4.full"
  }
  hdr "$HERE/n8_L8/deliverable.bin" 8 32 "$HERE/n8_L8/container_header.bin"
  while read -r name budgets; do
    state=$(head -1 "$HERE/anf/$name.mpmct1" | awk '{print $2}')
    hdr "$HERE/anf/$name.bin" 1 "$state" "$HERE/anf/$name.tdp4hdr"
  done < "$HERE/anf/cases.txt"
fi
if want pack; then
  : "${CARGO_TARGET_DIR:?}"
  (cd "$REPO" && cargo build --locked --release --bin tdp_packing)
  TP=$CARGO_TARGET_DIR/release/tdp_packing
  out=$HERE/pack; mkdir -p "$out"; rm -rf "$out/expected"; mkdir -p "$out/expected"
  "$ORACLE_PY" -I "$HERE/pack_cases.py" "$out"
  PW=$SBR/pack_work; mkdir -p "$PW"
  digest() { "$ORACLE_PY" -I -c 'import hashlib,sys; print(hashlib.blake2b(open(sys.argv[1],"rb").read(), digest_size=16).hexdigest())' "$1"; }
  : > "$PW/coverage.txt"
  while read -r name a b rest; do
    [ -z "$name" ] && continue
    args=(); rargs=(); has_out=0
    for t in $rest; do
      if [ "$t" = OUT ]; then
        args+=("$PW/$name.mpmct1"); rargs+=("$PW/rust_$name.mpmct1"); has_out=1
      else
        args+=("$t"); rargs+=("$t")
      fi
    done
    (cd "$RD" && PACK_DUMP_ORDER="$PW/$name.order" "$ORACLE_PY" window_pack3.py \
       "$HERE/$a" "$HERE/$b" ${args[@]+"${args[@]}"} > "$out/expected/$name.txt" 2>&1)
    { [ $has_out = 1 ] && echo "packed $(digest "$PW/$name.mpmct1")"
      echo "order $(digest "$PW/$name.order")"; } > "$out/expected/$name.digest"
    [ $has_out = 1 ] && cp "$PW/$name.mpmct1.bounds" "$out/expected/$name.bounds"
    st=$("$TP" pack "$HERE/$a" "$HERE/$b" ${rargs[@]+"${rargs[@]}"} --order-out "$PW/rust_$name.order" \
           --stats 2>&1 >/dev/null | grep '^stats:' | sed 's/; read .*//')
    echo "$name $st" >> "$PW/coverage.txt"
  done < "$out/cases.txt"
  # Coverage (docs/STAGE4_RUST_DESIGN.md §10.3 step 6, packer part).
  cov() { if ! eval "$2"; then echo "regen pack: coverage '$1' not reached" >&2; exit 1; fi; echo "  covered: $1"; }
  E=$out/expected
  cov "compose abort" "grep -q 'compose_aborts=[1-9]' '$PW/coverage.txt'"
  cov "order-dependent compose abort" "grep -q 'order_dependent=[1-9]' '$PW/coverage.txt'"
  cov "duplicate boundary" "grep -q 'duplicate_bounds=[1-9]' '$PW/coverage.txt'"
  cov "next-next-writer rescue" "grep -qE '; [1-9][0-9]* spans rescued' $E/*.txt"
  cov "long-span disqualification" "grep -qE 'disq: [1-9][0-9]* long-span' $E/*.txt"
  cov "big-interval disqualification" "grep -qE ', [1-9][0-9]* big-interval' $E/*.txt"
  cov "mid and band classes" "grep -q '  mid: ' $E/*.txt && grep -q '  band: ' $E/*.txt"
  cov "drain, burst and sibling deaths" "grep -q drain_close $E/*.txt && grep -q burst_close $E/*.txt && grep -q sibling $E/*.txt"
  cov "no deaths" "grep -q 'death causes: {}' $E/*.txt"
  cov "B wider than A" "grep -q 'input function check 128/128' $E/wide128_prod.txt"
  cov "W = 16 state" "grep -q 'w1024' '$out/cases.txt'"
  cp "$PW/coverage.txt" "$out/coverage.txt"
  sed -i '' -e "s|$HERE/||g" "$out/coverage.txt"
fi
du -sh "$HERE"
