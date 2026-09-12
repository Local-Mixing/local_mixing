#!/bin/bash
# GSS-MIX â€” the end-to-end mixing pipeline for GSS (gadgetized sliced
# sandwich) circuits. Manual: docs/GSS_MIX.md.
#
# Stages:
#   1+2  generate the sliced sandwich S and gadgetize it with the selected
#        representation family
#        (gen_sandwich_gadget; |C| = |D| = round(nÂ·(log2 n)^2), the library
#        convention; the production preset â€” mask plan [2,2,2,3] + Gray fold â€”
#        is the tool default)
#   3    fmix phase A: --gss --phase-a --profile 3,(3+HOLD),(3+HOLD),R1,R1
#        (no compression leg since 2026-08-17: it only simplified the
#        circuit â€” it made sense before phase B existed, not anymore)
#   4    fmix phase B part 1: the split stage, current defaults, to exhaustion
#        (stages 3 and 4 run piecewise-parallel under --pieces P > 1:
#        docs/FMIX_PIECEWISE.md; P = 1 is the serial run, unchanged)
#   5    fmix phase B part 2: the crossing walk (resumes the split state;
#        numerical defaults reflect the X-panel, but deliverable-promotion
#        status remains unresolved â€” see the manual)
#   6    fcompress
#
# Every stage writes its artifact + log into the run dir and is skipped when
# the artifact already exists (rerun with --force-from K). bash only â€” do not
# port to zsh (word-splitting changes silently).
set -euo pipefail

usage() {
  cat <<'EOF'
usage: gss_mix.sh -n N -o RUNDIR [options]
  -h, --help    show this help and exit
  -n N           wires of the source computation C (required)
  -o DIR         run directory (created; all artifacts + logs land here)
  -s SEED        master seed. DEFAULT: a fresh CSPRNG draw â€” the seed
                 regenerates the secret C, so a predictable seed (1, 2, a
                 counter) makes the output reconstructible and worthless as
                 a deliverable. Pass -s only for CALIBRATION arms that must
                 share an input; label such outputs calibration-only.
  --mcd M        override |C| = |D| gate count            [0 = round(n(log2 n)^2)]
  --gadgetization-mode MODE
                  stage-2 representation: product-2223, nonlinear193,
                  nonlinear291, or blinded-v5 (LGI compute) [product-2223;
                  legacy alias: 2223]
  --bv5-k K      blinded-v5 only: band wires per LGI (LGI length lever).
                 Exported as BV5_K to gen_sandwich_gadget [gen default 2]
  --bv5-max-open N
                 blinded-v5 only: open-mask cap per wire, exported as
                 BV5_MAX_OPEN [3; 2 = size-neutral balanced variant, weaker
                 against two-feature correlation scans]
  --bv5-balanced 0|1
                 blinded-v5 only: balanced masks (CNOT from a fresh band wire
                 per LGI), exported as BV5_BALANCED [1; 0 = plain g57 masks]
  --bv5-min-open N
                 blinded-v5 only: minimum open masks per data wire at every
                 instant, exported as BV5_MIN_OPEN [2; must be < max_open]
  --expand R     phase-A max expansion factor R1          [2]
  --hold E       phase-A hold duration in effs            [27 -> profile 3,30,30,2,2]
  --xr R         stage-5 crossing target factor           [2; 2.5 = max-spread point]
  --xb B         stage-5 width-damper base                [3   â€” X-panel calibrated]
  --xc C         stage-5 width-damper threshold           [1   â€” X-panel calibrated]
  --xtdiv D      stage-5 temperature = target/D           [25]
  --xmoves M     stage-5 move budget (default 6 x target: STOP AT ARRIVAL â€”
                 median spread peaks there and the hold erodes it)
  --stop-after K stop after stage K (2, 3, 4, 5 or 6)
  --force-from K rebuild from stage K even if artifacts exist
  --pieces P     stages 3-4 piecewise-parallel mixing: cut the circuit into
                 P pieces, mix them in parallel on one shared store,
                 concatenate, shift the cuts by half a slice, repeat
                 [1 = serial, unchanged; docs/FMIX_PIECEWISE.md]
  --min-block-size B  nominal divisor for automatic stages 3-4 sizing (2..1000000000);
                 each round P=max(1,current gates/B), then fixed-P cutting
                 (shifted ends may be smaller);
                 mutually exclusive with explicit --pieces, including 1
  --piece-threads T  threads for the piece pool (pieces > 1 or automatic mode)
                 [P+1 for fixed pieces; available CPU count in automatic mode]
env:
  FROZEN_DB_DIR       required for stage 3 (the frozen replacement store)
  FROZEN_CURATED_DIR  recommended for stage 3 (curated-first cascade).
                      Standard store: ~/frozen_curated_m1_m11_native (the FULL
                      untruncated curated DB, NATIVE convention; band stores
                      are opt-in by explicit path only)
  FROZEN_CURATED_VALUE_CONVENTION=legacy-swapped-controls
                      only for a historical pre-2ed0222a curated store;
                      NEVER with the standard native store above
  PROD_PRESET         product-2223-only preset: production (default), no-gray-phase-a,
                      micro-gray, sentinel-gray, no-gray-post-exact,
                      no-gray-post-native, five-carrier,
                      strong-five-carrier, six-carrier, strong-six-carrier,
                      or seven-carrier
  PROD_POST_FRAGMENT  product-2223-only optional post-layout wide-gate pass:
                      off, exact, or native-deep (overrides the named preset)
  GSS_MIX_ALLOW_EMPTY_STORE=1  testing only: run stage 3 with no store
  DB_QC=1             optional gate-wise audit and DB repair at the end of stage 3
  DB_QC_REFERENCE     original reference in matching input/wire coordinates
                      (mpmct1; default: stage-3 input); see docs/DB_QUALITY_CONTROL.md
  DB_QC_SEED          independent QC seed [20803]; existing stage-3 artifacts
                      are still skipped unless --force-from 3 is supplied
  GSS_BIN_DIR         directory containing gen_sandwich_gadget, fmix and
                      fcompress [default: repository target/release]
EOF
  exit "${1:-1}"
}

N=""; RUN=""; SEED=""; EXPAND=2; HOLD=27; MCD=0
GADGETIZATION_MODE=product-2223
BV5_K_ARG=""; BV5_MAX_OPEN_ARG=""; BV5_BALANCED_ARG=""; BV5_MIN_OPEN_ARG=""
XR=2; XB=3; XC=1; XTDIV=25; XMOVES=""
STOP_AFTER=6; FORCE_FROM=99
PIECES=1; PIECES_SET=0; MIN_BLOCK_SIZE=""; PIECE_THREADS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --pieces|--min-block-size|--piece-threads)
      if [ "$#" -lt 2 ] || [ -z "$2" ] || [[ $2 == -* ]]; then
        echo "FATAL: $1 requires an argument" >&2
        exit 2
      fi ;;
  esac
  case "$1" in
    -h|--help) usage 0 ;;
    -n) N=$2; shift 2 ;;
    -o) RUN=$2; shift 2 ;;
    -s) SEED=$2; shift 2 ;;
    --mcd) MCD=$2; shift 2 ;;
    --gadgetization-mode) GADGETIZATION_MODE=$2; shift 2 ;;
    --bv5-k) BV5_K_ARG=$2; shift 2 ;;
    --bv5-max-open) BV5_MAX_OPEN_ARG=$2; shift 2 ;;
    --bv5-balanced) BV5_BALANCED_ARG=$2; shift 2 ;;
    --bv5-min-open) BV5_MIN_OPEN_ARG=$2; shift 2 ;;
    --expand) EXPAND=$2; shift 2 ;;
    --hold) HOLD=$2; shift 2 ;;
    --xr) XR=$2; shift 2 ;;
    --xb) XB=$2; shift 2 ;;
    --xc) XC=$2; shift 2 ;;
    --xtdiv) XTDIV=$2; shift 2 ;;
    --xmoves) XMOVES=$2; shift 2 ;;
    --stop-after) STOP_AFTER=$2; shift 2 ;;
    --force-from) FORCE_FROM=$2; shift 2 ;;
    --pieces) PIECES=$2; PIECES_SET=1; shift 2 ;;
    --min-block-size) MIN_BLOCK_SIZE=$2; shift 2 ;;
    --piece-threads) PIECE_THREADS=$2; shift 2 ;;
    *) echo "unknown arg $1"; usage ;;
  esac
done
[ -n "$N" ] && [ -n "$RUN" ] || usage
case "$GADGETIZATION_MODE" in
  product-2223) ;;
  2223) GADGETIZATION_MODE=product-2223 ;;
  nonlinear193|nonlinear291) ;;
  blinded-v5|blinded_v5) GADGETIZATION_MODE=blinded-v5 ;;
  *)
    echo "FATAL: --gadgetization-mode must be product-2223, nonlinear193, nonlinear291, or blinded-v5 (2223 is a compatibility alias)" >&2
    exit 2
    ;;
esac
# Piecewise-parallel stages 3-4 (docs/FMIX_PIECEWISE.md). The serial command
# lines stay byte-identical: flags are passed only in piece mode. The
# same rules live in `local_mixing_bin gss`, so both entry points agree.
[[ $PIECES =~ ^[0-9]+$ ]] && [ "$PIECES" -ge 1 ] || {
  echo "FATAL: --pieces must be a positive integer (1 = serial)" >&2
  exit 2
}
if [ -n "$MIN_BLOCK_SIZE" ]; then
  [ "$PIECES_SET" -eq 0 ] || {
    echo "FATAL: --min-block-size is mutually exclusive with --pieces (including 1)" >&2
    exit 2
  }
  # Strip leading zeros before bounded arithmetic, avoiding octal and overflow.
  [[ $MIN_BLOCK_SIZE =~ ^[0-9]+$ ]] || {
    echo "FATAL: --min-block-size must be an integer in 2..1000000000" >&2
    exit 2
  }
  MIN_BLOCK_SIZE="${MIN_BLOCK_SIZE#"${MIN_BLOCK_SIZE%%[!0]*}"}"
  if [ "${#MIN_BLOCK_SIZE}" -gt 10 ] || [ -z "$MIN_BLOCK_SIZE" ] ||
     [ "$MIN_BLOCK_SIZE" -lt 2 ] || [ "$MIN_BLOCK_SIZE" -gt 1000000000 ]; then
    echo "FATAL: --min-block-size must be an integer in 2..1000000000" >&2
    exit 2
  fi
fi
if [ -n "$PIECE_THREADS" ]; then
  [[ $PIECE_THREADS =~ ^[0-9]+$ ]] && [ "$PIECE_THREADS" -ge 1 ] || {
    echo "FATAL: --piece-threads must be a positive integer" >&2
    exit 2
  }
  [ "$PIECES" -gt 1 ] || [ -n "$MIN_BLOCK_SIZE" ] || {
    echo "FATAL: --piece-threads applies only with --pieces > 1 or --min-block-size" >&2
    exit 2
  }
fi
PIECEFLAGS=()
PIECE_DESCRIPTION="pieces $PIECES"
if [ -n "$MIN_BLOCK_SIZE" ]; then
  PIECEFLAGS=(--min-block-size "$MIN_BLOCK_SIZE")
  PIECE_DESCRIPTION="automatic pieces min_block_size=$MIN_BLOCK_SIZE"
elif [ "$PIECES" -gt 1 ]; then
  PIECEFLAGS=(--pieces "$PIECES")
fi
[ -n "$PIECE_THREADS" ] && PIECEFLAGS+=(--piece-threads "$PIECE_THREADS")
[ -n "$BV5_K_ARG$BV5_MAX_OPEN_ARG$BV5_BALANCED_ARG$BV5_MIN_OPEN_ARG" ] && [ "$GADGETIZATION_MODE" != blinded-v5 ] && {
  echo "FATAL: --bv5-* flags are only valid with --gadgetization-mode blinded-v5" >&2; exit 2; }
case "${BV5_BALANCED_ARG:-1}" in 0|1) ;; *) echo "FATAL: --bv5-balanced must be 0 or 1" >&2; exit 2 ;; esac
case "${BV5_MAX_OPEN_ARG:-1}" in ''|*[!0-9]*|0) echo "FATAL: --bv5-max-open must be a positive integer" >&2; exit 2 ;; esac
case "${BV5_K_ARG:-2}" in ''|*[!0-9]*|0|1) echo "FATAL: --bv5-k must be an integer >= 2" >&2; exit 2 ;; esac
case "${BV5_MIN_OPEN_ARG:-2}" in ''|*[!0-9]*|0) echo "FATAL: --bv5-min-open must be a positive integer" >&2; exit 2 ;; esac
if [ "$GADGETIZATION_MODE" != product-2223 ]; then
  product_override_names=()
  while IFS= read -r _pvar; do
    [ -n "$_pvar" ] && product_override_names+=("$_pvar")
  done < <(compgen -A variable PROD_ || true)
  [ "${#product_override_names[@]}" -eq 0 ] || {
    echo "FATAL: all PROD_* controls must be unset outside --gadgetization-mode product-2223: ${product_override_names[*]}" >&2
    exit 2
  }
fi

BIN=${GSS_BIN_DIR:-$(cd "$(dirname "$0")/.." && pwd)/target/release}
for b in gen_sandwich_gadget fmix fcompress; do
  [ -x "$BIN/$b" ] || { echo "FATAL: $BIN/$b missing â€” cargo build --release first"; exit 1; }
done
mkdir -p "$RUN"; RUN=$(cd "$RUN" && pwd)
GADGET=$RUN/gss.mpmct1
STAGE12_RECIPE_FILE=$RUN/stage12.recipe
# A missing stage-2 artifact invalidates every downstream artifact, even when
# the caller did not explicitly request --force-from 2.
[ -s "$GADGET" ] || FORCE_FROM=2
LOGALL=$RUN/gss_mix.log
note() { echo "[gss-mix] $*" | tee -a "$LOGALL"; }

# The seed regenerates the secret C: default to the OS CSPRNG, never a
# constant or a counter (docs/GSS_MIX.md, "seeds"). An explicit -s is for
# calibration arms only.
SEED_SRC="RANDOM (CSPRNG)"
if [ "$FORCE_FROM" -gt 2 ] && [ -s "$GADGET" ] && [ ! -s "$RUN/SEED" ]; then
  echo "FATAL: existing stage-2 artifact has no SEED; use a fresh run directory or --force-from 2" >&2
  exit 2
fi
if [ -n "$SEED" ]; then
  if [ -s "$RUN/SEED" ]; then
    EXISTING_SEED=$(cat "$RUN/SEED")
    [ "$SEED" = "$EXISTING_SEED" ] || {
      echo "FATAL: explicit seed conflicts with the existing $RUN/SEED; use the original seed or a fresh run directory" >&2
      exit 1
    }
  fi
  SEED_SRC="EXPLICIT â€” calibration only, NOT a deliverable"
elif [ -s "$RUN/SEED" ]; then
  # A rerun of an existing run dir MUST keep the seed that built the
  # artifacts on disk, or <run>/SEED would stop describing them.
  SEED=$(cat "$RUN/SEED")
  SEED_SRC="RESUMED from $RUN/SEED"
else
  # 63-bit draw: the stage seeds are SEED+k, and a full 64-bit value
  # overflows bash's signed arithmetic into a negative number that fmix
  # parses as a flag ("unexpected argument '-8...'").
  SEED=$(python3 -I -c "import secrets; print(secrets.randbelow(2**63 - 16))")
fi

# Derived sizes (the library conventions, computed here so they are pinned in
# the log): |C| = |D| = round(n (log2 n)^2), s = round(n log2 n),
# slice_gates = 10 * 2n, rg_freq = 1.
read -r M_CD S_SL SLICE_G <<< "$(python3 -I - "$N" "$MCD" <<'EOF'
import math, sys
n, mcd = int(sys.argv[1]), int(sys.argv[2])
l = math.log2(n)
print(mcd if mcd > 0 else round(n * l * l), max(n, round(n * l)), 10 * 2 * n)
EOF
)"
STAGE12_RECIPE=(
  "gss_stage12_recipe=1"
  "gadgetization_mode=$GADGETIZATION_MODE"
  "n=$N"
  "m_cd=$M_CD"
  "s=$S_SL"
  "slice_gates=$SLICE_G"
  "rg_freq=1"
)
# blinded-v5 knobs are part of the stage-2 identity (a different K/max_open/balanced
# is a different gadget); appended only for that mode so other modes' markers keep their shape
if [ "$GADGETIZATION_MODE" = blinded-v5 ]; then
  STAGE12_RECIPE+=("bv5_k=${BV5_K_ARG:-${BV5_K:-2}}" "bv5_max_open=${BV5_MAX_OPEN_ARG:-${BV5_MAX_OPEN:-3}}" "bv5_balanced=${BV5_BALANCED_ARG:-${BV5_BALANCED:-1}}" "bv5_min_open=${BV5_MIN_OPEN_ARG:-${BV5_MIN_OPEN:-2}}")
fi
if [ "$FORCE_FROM" -gt 2 ] && [ -s "$GADGET" ]; then
  [ -s "$STAGE12_RECIPE_FILE" ] || {
    echo "FATAL: existing stage-2 artifact has no recipe marker; rerun with --force-from 2" >&2
    exit 2
  }
  mapfile -t stored_stage12_recipe < "$STAGE12_RECIPE_FILE"
  [ "${#stored_stage12_recipe[@]}" -eq "${#STAGE12_RECIPE[@]}" ] || {
    echo "FATAL: existing stage-2 recipe marker is malformed; rerun with --force-from 2" >&2
    exit 2
  }
  for recipe_index in "${!STAGE12_RECIPE[@]}"; do
    [ "${stored_stage12_recipe[$recipe_index]}" = "${STAGE12_RECIPE[$recipe_index]}" ] || {
      echo "FATAL: existing stage-2 recipe does not match the requested mode or dimensions; use a fresh run directory or --force-from 2" >&2
      exit 2
    }
  done
fi
gates_of() { python3 -I -c "import sys; print(sum(1 for _ in open(sys.argv[1])) - 1)" "$1"; }
state_moves() { awk '$1=="moves"{print $2; exit}' "$1"; }

note "run=$RUN n=$N gadgetization_mode=$GADGETIZATION_MODE |C|=|D|=$M_CD s=$S_SL slice_gates=$SLICE_G expand=$EXPAND hold=${HOLD}effs x=(r=$XR b=$XB c=$XC tdiv=$XTDIV) $PIECE_DESCRIPTION${PIECE_THREADS:+ piece_threads=$PIECE_THREADS}"
# The seed goes to the run dir, NOT to the shared narrative log: it is the
# secret that regenerates C.
note "seed source: $SEED_SRC (value in $RUN/SEED)"
umask 077; printf '%s\n' "$SEED" > "$RUN/SEED"; chmod 600 "$RUN/SEED"

PHASEA=$RUN/phaseA.mpmct1
SPLITB=$RUN/splitB.mpmct1
CROSSB=$RUN/crossB.mpmct1
FINAL=$RUN/final.esop1

# ---- stages 1+2: sandwich + selected gadgetization ----
if [ "$FORCE_FROM" -le 2 ] || [ ! -s "$GADGET" ]; then
  if [ "$GADGETIZATION_MODE" = product-2223 ]; then
    note "stage 1+2: gen_sandwich_gadget (mode=$GADGETIZATION_MODE, PROD_PRESET=${PROD_PRESET:-production}, PROD_POST_FRAGMENT=${PROD_POST_FRAGMENT:-preset/off})"
  elif [ "$GADGETIZATION_MODE" = blinded-v5 ]; then
    # blinded-v5 (LGI compute) reads its knobs from env: K (LGI mask width),
    # max_open (open-mask cap, 3), balanced (1). Rerand stays at the auto preset
    # (m/4K straddle burst slots x F=8K gates, no repair).
    [ -n "$BV5_K_ARG" ] && export BV5_K="$BV5_K_ARG"
    [ -n "$BV5_MAX_OPEN_ARG" ] && export BV5_MAX_OPEN="$BV5_MAX_OPEN_ARG"
    [ -n "$BV5_BALANCED_ARG" ] && export BV5_BALANCED="$BV5_BALANCED_ARG"
    [ -n "$BV5_MIN_OPEN_ARG" ] && export BV5_MIN_OPEN="$BV5_MIN_OPEN_ARG"
    note "stage 1+2: gen_sandwich_gadget (mode=$GADGETIZATION_MODE, BV5_K=${BV5_K:-2}, max_open=${BV5_MAX_OPEN:-3}, min_open=${BV5_MIN_OPEN:-2}, balanced=${BV5_BALANCED:-1}, quad-fire, rerand=auto burst slots m/4K x F=8K, min_mask=auto)"
  else
    note "stage 1+2: gen_sandwich_gadget (mode=$GADGETIZATION_MODE; experimental/capacity-limited)"
  fi
  "$BIN/gen_sandwich_gadget" "$GADGET" "$N" "$M_CD" "$M_CD" "$S_SL" 1 "$SLICE_G" \
      "$SEED" "$((SEED + 1))" "$SEED" "$GADGETIZATION_MODE" > "$RUN/stage12.log" 2>&1
  printf '%s\n' "${STAGE12_RECIPE[@]}" > "$STAGE12_RECIPE_FILE.tmp"
  mv "$STAGE12_RECIPE_FILE.tmp" "$STAGE12_RECIPE_FILE"
  note "stage 1+2 done: GSS $(gates_of "$GADGET") gates (S: $(gates_of "$GADGET.sandwich.mpmct1"))"
else
  note "stage 1+2: $GADGET exists, skipping"
fi
[ "$STOP_AFTER" -le 2 ] && { note "stopped after stage 2"; exit 0; }

# ---- stage 3: fmix phase A (--gss --phase-a --profile) ----
if [ "$FORCE_FROM" -le 3 ] || [ ! -s "$PHASEA" ]; then
  if [ -z "${FROZEN_DB_DIR:-}" ] && [ "${GSS_MIX_ALLOW_EMPTY_STORE:-0}" != "1" ]; then
    echo "FATAL: stage 3 needs FROZEN_DB_DIR (or GSS_MIX_ALLOW_EMPTY_STORE=1 for a plumbing test)"; exit 1
  fi
  [ -z "${FROZEN_CURATED_DIR:-}" ] && note "WARNING: FROZEN_CURATED_DIR unset â€” curated-first cascade OFF for phase A"
  G_IN=$(gates_of "$GADGET")
  read -r PROFILE A_MOVES <<< "$(python3 -I - "$EXPAND" "$HOLD" "$G_IN" <<'EOF'
import sys
r1, hold, g = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
# No compression leg (2026-08-17): it only simplified the circuit. It made
# sense when phase A was the whole pipeline; with phase B following, the run
# ends at the held size (N2 = N1, R2 = R1 â€” a zero-length leg is valid,
# prof_target never enters the interpolation branch).
n0 = 3.0
n1 = n0 + hold; n2 = n1
r2 = r1
# move ceiling: effs x peak size x margin
print(f"{n0:g},{n1:g},{n2:g},{r1:g},{r2:g}", round(n2 * r1 * g * 1.3))
EOF
)"
  note "stage 3: fmix phase A --gss --phase-a --profile $PROFILE (moves ceiling $A_MOVES, $PIECE_DESCRIPTION)"
  export CANON_RULE_L_BRANCH_CAP=${CANON_RULE_L_BRANCH_CAP:-512}
  export CANON_MONOMIAL_CAP=${CANON_MONOMIAL_CAP:-200000}
  # Lookup-cache headroom for long runs (512MB default never overflows at 200k
  # moves but 1M+ move runs would epoch-reset; 2GB is <1% of a server's RAM).
  export LOOKUP_CACHE_MB=${LOOKUP_CACHE_MB:-2048}
  # FROZEN_FILTER (measured on .32, 2026-08-09): on PRODUCTION runs (fresh
  # seed, filters.bin page-cached) the in-RAM miss filter cuts phase-A wall
  # ~33% at 200k moves and the win grows ~0.30s/1k moves, against a fixed
  # ~13s cached load. Same-seed reruns (+19%) and fully-cold caches (+41%)
  # lose â€” hence the RAM gate and the background prewarm below.
  if [ -z "${FROZEN_FILTER:-}" ] && [ -n "${FROZEN_DB_DIR:-}" ] \
     && [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 60 ]; then
    export FROZEN_FILTER=1
    # Warm the filter files while gen/stage-2 artifacts are checked; a cached
    # load is ~13s vs ~107s cold.
    { cat "$FROZEN_DB_DIR/filters.bin" "${FROZEN_CURATED_DIR:+$FROZEN_CURATED_DIR/filters.bin}" \
        > /dev/null 2>&1 & } 2>/dev/null
    note "FROZEN_FILTER=1 (auto: >=60GB available; prewarming filters.bin in background)"
  fi
  export FMIX_STOP_FLAG=$RUN/stage3.stop FMIX_DUMP_FLAG=$RUN/stage3.dump
  rm -f "$FMIX_STOP_FLAG"
  # Plumbing-test mode: fmix hard-requires the store whenever any DB channel
  # is armed, so the bypass must zero all three coins (a true null plant â€”
  # the profile still walks its eff schedule, no re-encoding happens).
  DBFLAGS=(--p-db 1.0 --p-comp 1.0 --p-any 0.1)
  if [ -z "${FROZEN_DB_DIR:-}" ]; then
    DBFLAGS=(--p-db 0 --p-comp 0 --p-any 0)
    note "WARNING: empty-store plumbing mode â€” phase A performs NO re-encoding"
  fi
  QCFLAGS=()
  if [ "${DB_QC:-0}" = "1" ]; then
    QCFLAGS=(--qc --qc-seed "${DB_QC_SEED:-20803}" --qc-report "$RUN/phaseA.qc.txt")
    if [ -n "${DB_QC_REFERENCE:-}" ]; then
      QCFLAGS+=(--qc-reference "$DB_QC_REFERENCE")
    fi
    note "stage 3: gate-wise QC repair enabled; report $RUN/phaseA.qc.txt"
  fi
  "$BIN/fmix" --input "$GADGET" --gss --phase-a --profile "$PROFILE" \
      --moves "$A_MOVES" --seed "$((SEED + 2))" \
      "${DBFLAGS[@]}" \
      "${QCFLAGS[@]}" \
      ${PIECEFLAGS[@]+"${PIECEFLAGS[@]}"} \
      --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048 \
      --no-local-verify --verify-every 2000000 --report-every 100000 \
      --state-out "$RUN/phaseA.state" --output "$PHASEA" > "$RUN/stage3.log" 2>&1
  note "stage 3 done: $(gates_of "$PHASEA") gates ($(grep -c '^\[fmix\] mv=' "$RUN/stage3.log" 2>/dev/null || true) report points)"
else
  note "stage 3: $PHASEA exists, skipping"
fi
[ "$STOP_AFTER" -le 3 ] && { note "stopped after stage 3"; exit 0; }

# ---- stage 4: the split stage (phase B part 1), current defaults ----
if [ "$FORCE_FROM" -le 4 ] || [ ! -s "$SPLITB" ]; then
  G_A=$(gates_of "$PHASEA")
  B_MOVES=$((G_A + 1000000))   # one move per split twist; comp count < G_A
  note "stage 4: fmix --split (current defaults), moves ceiling $B_MOVES, $PIECE_DESCRIPTION"
  export FMIX_STOP_FLAG=$RUN/stage4.stop FMIX_DUMP_FLAG=$RUN/stage4.dump
  rm -f "$FMIX_STOP_FLAG"
  "$BIN/fmix" --input "$PHASEA" --split --split-stop \
      --p-join 0.8 --split-reach-k 2 --split-fail-limit 100 --split-canaries 256 \
      --p-db 0 --p-comp 0 --p-any 0 --p-twist 0 --k-max 12 \
      ${PIECEFLAGS[@]+"${PIECEFLAGS[@]}"} \
      --moves "$B_MOVES" --seed "$((SEED + 3))" --report-every 1000000 \
      --state-out "$RUN/splitB.state" --output "$SPLITB" > "$RUN/stage4.log" 2>&1
  grep -E "split stage ENDED|split spans|canary deciles" "$RUN/stage4.log" | tee -a "$LOGALL" || true
  note "stage 4 done: $(gates_of "$SPLITB") gates"
else
  note "stage 4: $SPLITB exists, skipping"
fi
[ "$STOP_AFTER" -le 4 ] && { note "stopped after stage 4"; exit 0; }

# ---- stage 5: crossing walk (X-panel defaults; promotion status unresolved) ----
if [ "$FORCE_FROM" -le 5 ] || [ ! -s "$CROSSB" ]; then
  G_S=$(gates_of "$SPLITB")
  read -r X_TGT X_TEMP X_MOVES_ABS <<< "$(python3 -I - "$G_S" "$XR" "$XTDIV" "${XMOVES:-0}" "$(state_moves "$RUN/splitB.state")" <<'EOF'
import sys
g, xr, tdiv, xmoves, done = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
tgt = round(g * xr)
temp = max(64, round(tgt / tdiv))
# STOP AT ARRIVAL (X-panel 2026-08-05): median descendants AND median span
# peak when size reaches its damped equilibrium (~2-5 moves/target gate)
# and the hold then ERODES them; 6x covers arrival across r with only mild
# post-peak decay.
budget = xmoves if xmoves > 0 else 6 * tgt
print(tgt, temp, done + budget)   # --moves is ABSOLUTE on a resume
EOF
)"
  note "stage 5: crossing walk (resume) target=$X_TGT temp=$X_TEMP b=$XB c=$XC moves(abs)=$X_MOVES_ABS â€” calibrated defaults; promotion status unresolved, see manual"
  export FMIX_STOP_FLAG=$RUN/stage5.stop FMIX_DUMP_FLAG=$RUN/stage5.dump
  rm -f "$FMIX_STOP_FLAG"
  "$BIN/fmix" --resume "$RUN/splitB.state" \
      --target-size "$X_TGT" --temp "$X_TEMP" --split-base "$XB" --split-damp "$XC" \
      --p-twist 0 --p-db 0 --p-comp 0 --p-any 0 --k-max 12 \
      --moves "$X_MOVES_ABS" --report-every 500000 \
      --state-out "$RUN/crossB.state" --output "$CROSSB" > "$RUN/stage5.log" 2>&1
  note "stage 5 done: $(gates_of "$CROSSB") gates"
else
  note "stage 5: $CROSSB exists, skipping"
fi
[ "$STOP_AFTER" -le 5 ] && { note "stopped after stage 5"; exit 0; }

# ---- stage 6: fcompress ----
if [ "$FORCE_FROM" -le 6 ] || [ ! -s "$FINAL" ]; then
  note "stage 6: fcompress"
  # Compress (transport / separated reads / reversed gather, POSTMIX_MANUAL Â§3)
  # and PACK: the deliverable is the esop1 file, one generalized gate per
  # maximal same-target run, its activation function spelled as the ANF
  # compacted by the deterministic reducer (a function of the ANF alone, so
  # one spelling per function). The cube count (fcompress's "gates A -> B"
  # line) is the honest effective size; the packed count is the number of
  # generalized gates. Every mpmct1 reader loads esop1 transparently.
  "$BIN/fcompress" --input "$CROSSB" --output "$FINAL" --seed "$((SEED + 5))" \
      > "$RUN/stage6.log" 2>&1
  G_X=$(gates_of "$CROSSB"); G_P=$(gates_of "$FINAL")
  G_C=$(sed -n 's/.*done in .*gates [0-9]* -> \([0-9]*\) (.*/\1/p' "$RUN/stage6.log" | tail -1)
  G_C=${G_C:-$G_P}
  note "stage 6 done: $G_X -> $G_C cubes (residual $(python3 -I -c "print(f'{100*$G_C/$G_X:.1f}%')")) -> $G_P packed gates"
else
  note "stage 6: $FINAL exists, skipping"
fi

note "PIPELINE COMPLETE: $FINAL"
