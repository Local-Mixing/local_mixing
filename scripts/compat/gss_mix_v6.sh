#!/bin/bash
# GSS-MIX — the end-to-end mixing pipeline for GSS (gadgetized sliced
# sandwich) circuits. Manual: docs/GSS_MIX.md.
#
# Stages:
#   1+2  generate the sliced sandwich S and gadgetize it with the selected
#        representation family
#        (gen_sandwich_gadget; |C| = |D| = round(n·(log2 n)^2), the library
#        convention; Ran balanced with K=2, max_open=3, min_open=2
#        is the default)
#   3    fmix db_mixing: --gss --db-mixing --profile 3,(3+HOLD),(3+HOLD),R1,R1
#        (no compression leg since 2026-08-17: it only simplified the
#        circuit — it made sense before splitting and crossing existed, not anymore)
#   4    fmix the split stage, current defaults, to exhaustion
#        (stages 3 and 4 run piecewise-parallel under --parallel-pieces P > 1:
#        docs/FMIX_PIECEWISE.md; P = 1 is the serial run, unchanged)
#   5    fmix the crossing walk (resumes the split state;
#        numerical defaults reflect the X-panel, but deliverable-promotion
#        status remains unresolved — see the manual)
#   6    fcompress
#
# Every stage writes its artifact + log into the run dir and is skipped when
# the artifact already exists (rerun with --run-rerun-from-stage K). bash only — do not
# port to zsh (word-splitting changes silently).
set -euo pipefail

usage() {
  cat <<'EOF'
usage: gss_mix.sh --source-wires N --run-directory RUNDIR [options]
  -h, --help    show this help and exit
  --source-wires N
                 wires of the source computation C (required)
  --run-directory DIR
                 run directory (created; all artifacts + logs land here)
  --calibration-seed SEED
                 master seed. DEFAULT: a fresh CSPRNG draw — the seed
                 regenerates the secret C, so a predictable seed (1, 2, a
                 counter) makes the output reconstructible and worthless as
                 a deliverable. Pass --calibration-seed only for CALIBRATION arms that must
                 share an input; label such outputs calibration-only.
  --source-gates M
                 override |C| = |D| gate count            [0 = round(n(log2 n)^2)]
  --gadget-mode MODE
                  stage-2 representation: ran-balanced (default; aliases
                  blinded-v5, blinded_v5), nonlinear193, or nonlinear291
  --gadget-mask-pair-wires K
                 Ran balanced only: wires in each mask’s paired component.
                 Exported as BV5_K to gen_sandwich_gadget [2; range 2..64]
  --gadget-max-open-masks N
                 Ran balanced only: open-mask cap per wire, exported as
                 BV5_MAX_OPEN [3; range 2..64; 2 = size-neutral balanced variant, weaker
                 against two-feature correlation scans]
  --gadget-balanced-masks 0|1
                 Ran balanced only: balanced masks (CNOT from a fresh band wire
                 per LGI), exported as BV5_BALANCED [1; 0 = plain g57 masks]
  --gadget-min-open-masks N
                 Ran balanced only: minimum open masks per data wire at every
                 covered interval, exported as BV5_MIN_OPEN [2; range 1..63; must be < max_open]
  --db-mixing-target-size-factor R
                 db_mixing target gate-count multiplier          [2]
  --db-mixing-hold-work-units E
                 db_mixing hold duration in effective work units            [27 -> profile 3,30,30,2,2]
  --crossing-target-size-factor R
                 stage-5 crossing target factor           [2; 2.5 = max-spread point]
  --crossing-width-penalty-base B
                 stage-5 width-damper base                [3   — X-panel calibrated]
  --crossing-width-penalty-threshold C
                 stage-5 width-damper threshold           [1   — X-panel calibrated]
  --crossing-size-tolerance-divisor D
                 stage-5 temperature = target/D           [25]
  --crossing-move-attempts M
                 stage-5 move budget (default 6 x target: STOP AT ARRIVAL —
                 median spread peaks there and the hold erodes it)
  --run-stop-after-stage K stop after stage K (2, 3, 4, 5 or 6)      [6]
  --run-rerun-from-stage K rebuild from stage K even if artifacts exist [unset]
  --parallel-pieces P
                 stages 3-4 piecewise-parallel mixing: cut the circuit into
                 P pieces, mix them in parallel on one shared store,
                 concatenate, shift the cuts by half a slice, repeat
                 [1 = serial; range 1..64; docs/FMIX_PIECEWISE.md]
  --parallel-target-piece-gates B
                 nominal divisor for automatic stages 3-4 sizing (2..1000000000);
                 each round P=max(1,current gates/B), then fixed-P cutting
                 (shifted ends may be smaller);
                 mutually exclusive with explicit --parallel-pieces, including 1
  --parallel-threads T
                 threads for the piece pool (1..1024; requires pieces > 1 or auto)
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
  GSS_MIX_ALLOW_EMPTY_STORE=1  testing only: run stage 3 with no store
  DB_QC=1             optional gate-wise audit and DB repair at the end of stage 3
  DB_QC_REFERENCE     original reference in matching input/wire coordinates
                      (mpmct1; default: stage-3 input); see docs/DB_QUALITY_CONTROL.md
  DB_QC_SEED          independent QC seed [20803]; existing stage-3 artifacts
                      are still skipped unless --run-rerun-from-stage 3 is supplied
  GSS_BIN_DIR         directory containing gen_sandwich_gadget, fmix and
                      fcompress [default: repository target/release]
EOF
  exit "${1:-1}"
}

N=""; RUN=""; SEED=""; EXPAND=2; HOLD=27; MCD=0
GADGETIZATION_MODE=ran-balanced
BV5_K_ARG=""; BV5_MAX_OPEN_ARG=""; BV5_BALANCED_ARG=""; BV5_MIN_OPEN_ARG=""
XR=2; XB=3; XC=1; XTDIV=25; XMOVES=""
STOP_AFTER=6; FORCE_FROM=99
PIECES=1; PIECES_SET=0; MIN_BLOCK_SIZE=""; PIECE_THREADS=""
while [ $# -gt 0 ]; do
  # Historical spellings remain accepted; use canonical names below.
  case "$1" in
    -n) set -- --source-wires "${@:2}" ;;
    -o) set -- --run-directory "${@:2}" ;;
    -s) set -- --calibration-seed "${@:2}" ;;
    --mcd) set -- --source-gates "${@:2}" ;;
    --gadgetization-mode) set -- --gadget-mode "${@:2}" ;;
    --bv5-k) set -- --gadget-mask-pair-wires "${@:2}" ;;
    --bv5-max-open) set -- --gadget-max-open-masks "${@:2}" ;;
    --bv5-min-open) set -- --gadget-min-open-masks "${@:2}" ;;
    --bv5-balanced) set -- --gadget-balanced-masks "${@:2}" ;;
    --expand) set -- --db-mixing-target-size-factor "${@:2}" ;;
    --hold) set -- --db-mixing-hold-work-units "${@:2}" ;;
    --xr) set -- --crossing-target-size-factor "${@:2}" ;;
    --xb) set -- --crossing-width-penalty-base "${@:2}" ;;
    --xc) set -- --crossing-width-penalty-threshold "${@:2}" ;;
    --xtdiv) set -- --crossing-size-tolerance-divisor "${@:2}" ;;
    --xmoves) set -- --crossing-move-attempts "${@:2}" ;;
    --stop-after) set -- --run-stop-after-stage "${@:2}" ;;
    --force-from) set -- --run-rerun-from-stage "${@:2}" ;;
    --pieces) set -- --parallel-pieces "${@:2}" ;;
    --min-block-size) set -- --parallel-target-piece-gates "${@:2}" ;;
    --piece-threads) set -- --parallel-threads "${@:2}" ;;
  esac
  case "$1" in
    --source-wires|--run-directory|--calibration-seed|--source-gates|--gadget-mode|--gadget-mask-pair-wires|--gadget-max-open-masks|--gadget-balanced-masks|--gadget-min-open-masks|--db-mixing-target-size-factor|--db-mixing-hold-work-units|--crossing-target-size-factor|--crossing-width-penalty-base|--crossing-width-penalty-threshold|--crossing-size-tolerance-divisor|--crossing-move-attempts|--run-stop-after-stage|--run-rerun-from-stage|--parallel-pieces|--parallel-target-piece-gates|--parallel-threads)
      if [ "$#" -lt 2 ] || [ -z "$2" ] || [[ $2 == -* ]]; then
        echo "FATAL: $1 requires an argument" >&2
        exit 2
      fi ;;
  esac
  case "$1" in
    -h|--help) usage 0 ;;
    --source-wires) N=$2; shift 2 ;;
    --run-directory) RUN=$2; shift 2 ;;
    --calibration-seed) SEED=$2; shift 2 ;;
    --source-gates) MCD=$2; shift 2 ;;
    --gadget-mode) GADGETIZATION_MODE=$2; shift 2 ;;
    --gadget-mask-pair-wires) BV5_K_ARG=$2; shift 2 ;;
    --gadget-max-open-masks) BV5_MAX_OPEN_ARG=$2; shift 2 ;;
    --gadget-balanced-masks) BV5_BALANCED_ARG=$2; shift 2 ;;
    --gadget-min-open-masks) BV5_MIN_OPEN_ARG=$2; shift 2 ;;
    --db-mixing-target-size-factor) EXPAND=$2; shift 2 ;;
    --db-mixing-hold-work-units) HOLD=$2; shift 2 ;;
    --crossing-target-size-factor) XR=$2; shift 2 ;;
    --crossing-width-penalty-base) XB=$2; shift 2 ;;
    --crossing-width-penalty-threshold) XC=$2; shift 2 ;;
    --crossing-size-tolerance-divisor) XTDIV=$2; shift 2 ;;
    --crossing-move-attempts) XMOVES=$2; shift 2 ;;
    --run-stop-after-stage) STOP_AFTER=$2; shift 2 ;;
    --run-rerun-from-stage) FORCE_FROM=$2; shift 2 ;;
    --parallel-pieces) PIECES=$2; PIECES_SET=1; shift 2 ;;
    --parallel-target-piece-gates) MIN_BLOCK_SIZE=$2; shift 2 ;;
    --parallel-threads) PIECE_THREADS=$2; shift 2 ;;
    *) echo "unknown arg $1"; usage ;;
  esac
done
[ -n "$N" ] && [ -n "$RUN" ] || usage
case "$GADGETIZATION_MODE" in
  product-2223|2223)
    echo "FATAL: product-2223 is retired from new GSS runs; resume saved runs through gss with their original configuration" >&2
    exit 2
    ;;
  nonlinear193|nonlinear291) ;;
  ran-balanced|blinded-v5|blinded_v5) GADGETIZATION_MODE=ran-balanced ;;
  *)
    echo "FATAL: --gadget-mode must be ran-balanced, nonlinear193, or nonlinear291" >&2
    exit 2
    ;;
esac
# Piecewise-parallel stages 3-4 (docs/FMIX_PIECEWISE.md). The serial command
# lines stay byte-identical: flags are passed only in piece mode. The
# same rules live in `local_mixing_bin gss`, so both entry points agree.
# Normalize decimal input before arithmetic (leading zeros are not octal).
# Check digit length first so very large inputs cannot overflow shell integers.
bounded_uint() {
  local label=$1 value=$2 minimum=$3 maximum=$4
  if [[ $value =~ ^[0-9]+$ ]]; then
    value="${value#"${value%%[!0]*}"}"
    value=${value:-0}
    if [ "${#value}" -le "${#maximum}" ] &&
       [ "$value" -ge "$minimum" ] && [ "$value" -le "$maximum" ]; then
      printf '%s\n' "$value"
      return 0
    fi
  fi
  echo "FATAL: $label must be an integer in $minimum..$maximum" >&2
  return 2
}
N=$(bounded_uint --source-wires "$N" 3 4095)
PIECES=$(bounded_uint --parallel-pieces "$PIECES" 1 64)
if [ -n "$MIN_BLOCK_SIZE" ]; then
  [ "$PIECES_SET" -eq 0 ] || {
    echo "FATAL: --parallel-target-piece-gates is mutually exclusive with --parallel-pieces (including 1)" >&2
    exit 2
  }
  MIN_BLOCK_SIZE=$(bounded_uint --parallel-target-piece-gates "$MIN_BLOCK_SIZE" 2 1000000000)
fi
if [ -n "$PIECE_THREADS" ]; then
  PIECE_THREADS=$(bounded_uint --parallel-threads "$PIECE_THREADS" 1 1024)
  [ "$PIECES" -gt 1 ] || [ -n "$MIN_BLOCK_SIZE" ] || {
    echo "FATAL: --parallel-threads applies only with --parallel-pieces > 1 or --parallel-target-piece-gates" >&2
    exit 2
  }
fi
PIECEFLAGS=()
PIECE_DESCRIPTION="pieces $PIECES"
if [ -n "$MIN_BLOCK_SIZE" ]; then
  PIECEFLAGS=(--parallel-target-piece-gates "$MIN_BLOCK_SIZE")
  PIECE_DESCRIPTION="automatic pieces target_piece_gates=$MIN_BLOCK_SIZE"
elif [ "$PIECES" -gt 1 ]; then
  PIECEFLAGS=(--parallel-pieces "$PIECES")
fi
[ -n "$PIECE_THREADS" ] && PIECEFLAGS+=(--parallel-threads "$PIECE_THREADS")
if [ -n "$BV5_K_ARG$BV5_MAX_OPEN_ARG$BV5_BALANCED_ARG$BV5_MIN_OPEN_ARG" ] && [ "$GADGETIZATION_MODE" != ran-balanced ]; then
  echo "FATAL: gadget mask flags are only valid with --gadget-mode ran-balanced" >&2
  exit 2
fi
if [ "$GADGETIZATION_MODE" = ran-balanced ]; then
  BV5_K=$(bounded_uint --gadget-mask-pair-wires "${BV5_K_ARG:-${BV5_K:-2}}" 2 64)
  BV5_MAX_OPEN=$(bounded_uint --gadget-max-open-masks "${BV5_MAX_OPEN_ARG:-${BV5_MAX_OPEN:-3}}" 2 64)
  BV5_MIN_OPEN=$(bounded_uint --gadget-min-open-masks "${BV5_MIN_OPEN_ARG:-${BV5_MIN_OPEN:-2}}" 1 63)
  BV5_BALANCED=$(bounded_uint --gadget-balanced-masks "${BV5_BALANCED_ARG:-${BV5_BALANCED:-1}}" 0 1)
  [ "$BV5_MIN_OPEN" -lt "$BV5_MAX_OPEN" ] || {
    echo "FATAL: --gadget-min-open-masks must be less than --gadget-max-open-masks" >&2
    exit 2
  }
  [ "$((2 * N))" -gt "$((BV5_K - BV5_K % 2 + BV5_BALANCED))" ] || {
    echo "FATAL: Ran balanced requires 2 * N > K rounded down to an even number + balanced; reduce --gadget-mask-pair-wires or increase --source-wires" >&2
    exit 2
  }
  export BV5_K BV5_MAX_OPEN BV5_MIN_OPEN BV5_BALANCED
fi
product_override_names=()
while IFS= read -r _pvar; do
  [ -n "$_pvar" ] && product_override_names+=("$_pvar")
done < <(compgen -A variable PROD_ || true)
[ "${#product_override_names[@]}" -eq 0 ] || {
  echo "FATAL: PROD_* controls are retired from new GSS runs; unset: ${product_override_names[*]}" >&2
  exit 2
}

BIN=${GSS_BIN_DIR:-$(cd "$(dirname "$0")/.." && pwd)/target/release}
for b in gen_sandwich_gadget fmix fcompress; do
  [ -x "$BIN/$b" ] || { echo "FATAL: $BIN/$b missing — cargo build --release first"; exit 1; }
done
mkdir -p "$RUN"; RUN=$(cd "$RUN" && pwd)
GADGET=$RUN/gss.mpmct1
STAGE12_RECIPE_FILE=$RUN/stage12.recipe
# A missing stage-2 artifact invalidates every downstream artifact, even when
# the caller did not explicitly request --run-rerun-from-stage 2.
[ -s "$GADGET" ] || FORCE_FROM=2
LOGALL=$RUN/gss_mix.log
note() { echo "[gss-mix] $*" | tee -a "$LOGALL"; }

# The seed regenerates the secret C: default to the OS CSPRNG, never a
# constant or a counter (docs/GSS_MIX.md, "seeds"). An explicit -s is for
# calibration arms only.
SEED_SRC="RANDOM (CSPRNG)"
if [ "$FORCE_FROM" -gt 2 ] && [ -s "$GADGET" ] && [ ! -s "$RUN/SEED" ]; then
  echo "FATAL: existing stage-2 artifact has no SEED; use a fresh run directory or --run-rerun-from-stage 2" >&2
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
  SEED_SRC="EXPLICIT — calibration only, NOT a deliverable"
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
# Record the complete stage-2 implementation and selected environment. This
# prevents silently resuming a different binary, source, or comparison override.
STAGE12_FINGERPRINT=$(python3 -I - "$BIN/gen_sandwich_gadget" <<'PYHASH'
import hashlib, json, os, sys
h = hashlib.sha256()
with open(sys.argv[1], 'rb') as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b''):
        h.update(chunk)
controls = {key: value for key, value in os.environ.items() if key.startswith(('BV5_', 'PROD_')) or key == 'SANDWICH_VARIANT'}
h.update(json.dumps(controls, sort_keys=True).encode())
source_path = os.environ.get('GSS_SOURCE_C')
if source_path:
    with open(source_path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
print(h.hexdigest())
PYHASH
)
STAGE12_RECIPE=(
  "gss_stage12_recipe=2"
  "implementation_sha256=$STAGE12_FINGERPRINT"
  "sandwich_variant=${SANDWICH_VARIANT:-classic}"
  "gadgetization_mode=$GADGETIZATION_MODE"
  "n=$N"
  "m_cd=$M_CD"
  "s=$S_SL"
  "slice_gates=$SLICE_G"
  "rg_freq=1"
)
# Ran balanced knobs are part of the stage-2 identity (a different K/max_open/balanced
# is a different gadget); appended only for that mode so other modes' markers keep their shape
if [ "$GADGETIZATION_MODE" = ran-balanced ]; then
  STAGE12_RECIPE+=("bv5_k=$BV5_K" "bv5_max_open=$BV5_MAX_OPEN" "bv5_balanced=$BV5_BALANCED" "bv5_min_open=$BV5_MIN_OPEN")
fi
if [ "$FORCE_FROM" -gt 2 ] && [ -s "$GADGET" ]; then
  [ -s "$STAGE12_RECIPE_FILE" ] || {
    echo "FATAL: existing stage-2 artifact has no recipe marker; rerun with --run-rerun-from-stage 2" >&2
    exit 2
  }
  mapfile -t stored_stage12_recipe < "$STAGE12_RECIPE_FILE"
  [ "${#stored_stage12_recipe[@]}" -eq "${#STAGE12_RECIPE[@]}" ] || {
    echo "FATAL: existing stage-2 recipe marker is malformed; rerun with --run-rerun-from-stage 2" >&2
    exit 2
  }
  for recipe_index in "${!STAGE12_RECIPE[@]}"; do
    [ "${stored_stage12_recipe[$recipe_index]}" = "${STAGE12_RECIPE[$recipe_index]}" ] || {
      echo "FATAL: existing stage-2 recipe does not match the requested mode or dimensions; use a fresh run directory or --run-rerun-from-stage 2" >&2
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

DB_MIXING=$RUN/db_mixing.mpmct1
SPLIT=$RUN/split.mpmct1
CROSSING=$RUN/crossing.mpmct1
FINAL=$RUN/final.esop1

# ---- stages 1+2: sandwich + selected gadgetization ----
if [ "$FORCE_FROM" -le 2 ] || [ ! -s "$GADGET" ]; then
  if [ "$GADGETIZATION_MODE" = ran-balanced ]; then
    # blinded-v5 (LGI compute) reads its knobs from env: K (LGI mask width),
    # max_open (open-mask cap, 3), balanced (1). Rerand stays at the auto preset
    # (m/4K straddle burst slots x F=8K gates, no repair).
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

# ---- stage 3: fmix db_mixing (--gss --db-mixing --profile) ----
if [ "$FORCE_FROM" -le 3 ] || [ ! -s "$DB_MIXING" ]; then
  if [ -z "${FROZEN_DB_DIR:-}" ] && [ "${GSS_MIX_ALLOW_EMPTY_STORE:-0}" != "1" ]; then
    echo "FATAL: stage 3 needs FROZEN_DB_DIR (or GSS_MIX_ALLOW_EMPTY_STORE=1 for a plumbing test)"; exit 1
  fi
  [ -z "${FROZEN_CURATED_DIR:-}" ] && note "WARNING: FROZEN_CURATED_DIR unset — curated-first cascade OFF for db_mixing"
  G_IN=$(gates_of "$GADGET")
  read -r PROFILE A_MOVES <<< "$(python3 -I - "$EXPAND" "$HOLD" "$G_IN" <<'EOF'
import sys
r1, hold, g = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
# No compression leg (2026-08-17): it only simplified the circuit. It made
# sense when db_mixing was the whole pipeline; with splitting and crossing following, the run
# ends at the held size (N2 = N1, R2 = R1 — a zero-length leg is valid,
# prof_target never enters the interpolation branch).
n0 = 3.0
n1 = n0 + hold; n2 = n1
r2 = r1
# move ceiling: effs x peak size x margin
print(f"{n0:g},{n1:g},{n2:g},{r1:g},{r2:g}", round(n2 * r1 * g * 1.3))
EOF
)"
  note "stage 3: fmix db_mixing --gss --db-mixing --profile $PROFILE (moves ceiling $A_MOVES, $PIECE_DESCRIPTION)"
  export CANON_RULE_L_BRANCH_CAP=${CANON_RULE_L_BRANCH_CAP:-512}
  export CANON_MONOMIAL_CAP=${CANON_MONOMIAL_CAP:-200000}
  # Lookup-cache headroom for long runs (512MB default never overflows at 200k
  # moves but 1M+ move runs would epoch-reset; 2GB is <1% of a server's RAM).
  export LOOKUP_CACHE_MB=${LOOKUP_CACHE_MB:-2048}
  # FROZEN_FILTER (measured on .32, 2026-08-09): on PRODUCTION runs (fresh
  # seed, filters.bin page-cached) the in-RAM miss filter cuts db_mixing wall
  # ~33% at 200k moves and the win grows ~0.30s/1k moves, against a fixed
  # ~13s cached load. Same-seed reruns (+19%) and fully-cold caches (+41%)
  # lose — hence the RAM gate and the background prewarm below.
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
  # is armed, so the bypass must zero all three coins (a true null plant —
  # the profile still walks its eff schedule, no re-encoding happens).
  DBFLAGS=(--p-db 1.0 --p-comp 1.0 --p-any 0.1)
  if [ -z "${FROZEN_DB_DIR:-}" ]; then
    DBFLAGS=(--p-db 0 --p-comp 0 --p-any 0)
    note "WARNING: empty-store plumbing mode — db_mixing performs NO re-encoding"
  fi
  QCFLAGS=()
  if [ "${DB_QC:-0}" = "1" ]; then
    QCFLAGS=(--leakage-repair --leakage-repair-seed "${DB_QC_SEED:-20803}" --leakage-repair-report "$RUN/db_mixing.leakage_repair.txt")
    if [ -n "${DB_QC_REFERENCE:-}" ]; then
      QCFLAGS+=(--leakage-repair-reference "$DB_QC_REFERENCE")
    fi
    note "stage 3: leakage repair enabled; report $RUN/db_mixing.leakage_repair.txt"
  fi
  "$BIN/fmix" --input "$GADGET" --gss --db-mixing --profile "$PROFILE" \
      --moves "$A_MOVES" --seed "$((SEED + 2))" \
      "${DBFLAGS[@]}" \
      "${QCFLAGS[@]}" \
      ${PIECEFLAGS[@]+"${PIECEFLAGS[@]}"} \
      --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048 \
      --no-local-verify --verify-every 2000000 --report-every 100000 \
      --state-out "$RUN/db_mixing.state" --output "$DB_MIXING" > "$RUN/stage3.log" 2>&1
  note "stage 3 done: $(gates_of "$DB_MIXING") gates ($(grep -c '^\[fmix\] mv=' "$RUN/stage3.log" 2>/dev/null || true) report points)"
else
  note "stage 3: $DB_MIXING exists, skipping"
fi
[ "$STOP_AFTER" -le 3 ] && { note "stopped after stage 3"; exit 0; }

# ---- stage 4: splitting, current defaults ----
if [ "$FORCE_FROM" -le 4 ] || [ ! -s "$SPLIT" ]; then
  G_A=$(gates_of "$DB_MIXING")
  B_MOVES=$((G_A + 1000000))   # one move per split twist; comp count < G_A
  note "stage 4: fmix --split (current defaults), moves ceiling $B_MOVES, $PIECE_DESCRIPTION"
  export FMIX_STOP_FLAG=$RUN/stage4.stop FMIX_DUMP_FLAG=$RUN/stage4.dump
  rm -f "$FMIX_STOP_FLAG"
  "$BIN/fmix" --input "$DB_MIXING" --split --split-stop \
      --p-join 0.8 --split-reach-k 2 --split-fail-limit 100 --split-canaries 256 \
      --p-db 0 --p-comp 0 --p-any 0 --p-twist 0 --k-max 12 \
      ${PIECEFLAGS[@]+"${PIECEFLAGS[@]}"} \
      --moves "$B_MOVES" --seed "$((SEED + 3))" --report-every 1000000 \
      --state-out "$RUN/split.state" --output "$SPLIT" > "$RUN/stage4.log" 2>&1
  grep -E "split stage ENDED|split spans|canary deciles" "$RUN/stage4.log" | tee -a "$LOGALL" || true
  note "stage 4 done: $(gates_of "$SPLIT") gates"
else
  note "stage 4: $SPLIT exists, skipping"
fi
[ "$STOP_AFTER" -le 4 ] && { note "stopped after stage 4"; exit 0; }

# ---- stage 5: crossing walk (X-panel defaults; promotion status unresolved) ----
if [ "$FORCE_FROM" -le 5 ] || [ ! -s "$CROSSING" ]; then
  G_S=$(gates_of "$SPLIT")
  read -r X_TGT X_TEMP X_MOVES_ABS <<< "$(python3 -I - "$G_S" "$XR" "$XTDIV" "${XMOVES:-0}" "$(state_moves "$RUN/split.state")" <<'EOF'
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
  note "stage 5: crossing walk (resume) target=$X_TGT temp=$X_TEMP b=$XB c=$XC moves(abs)=$X_MOVES_ABS — calibrated defaults; promotion status unresolved, see manual"
  export FMIX_STOP_FLAG=$RUN/stage5.stop FMIX_DUMP_FLAG=$RUN/stage5.dump
  rm -f "$FMIX_STOP_FLAG"
  "$BIN/fmix" --resume "$RUN/split.state" \
      --target-size "$X_TGT" --temp "$X_TEMP" --split-base "$XB" --split-damp "$XC" \
      --p-twist 0 --p-db 0 --p-comp 0 --p-any 0 --k-max 12 \
      --moves "$X_MOVES_ABS" --report-every 500000 \
      --state-out "$RUN/crossing.state" --output "$CROSSING" > "$RUN/stage5.log" 2>&1
  note "stage 5 done: $(gates_of "$CROSSING") gates"
else
  note "stage 5: $CROSSING exists, skipping"
fi
[ "$STOP_AFTER" -le 5 ] && { note "stopped after stage 5"; exit 0; }

# ---- stage 6: fcompress ----
if [ "$FORCE_FROM" -le 6 ] || [ ! -s "$FINAL" ]; then
  note "stage 6: fcompress"
  # Compress (transport / separated reads / reversed gather, POSTMIX_MANUAL §3)
  # and PACK: the deliverable is the esop1 file, one generalized gate per
  # maximal same-target run, its activation function spelled as the ANF
  # compacted by the deterministic reducer (a function of the ANF alone, so
  # one spelling per function). The cube count (fcompress's "gates A -> B"
  # line) is the honest effective size; the packed count is the number of
  # generalized gates. Every mpmct1 reader loads esop1 transparently.
  "$BIN/fcompress" --input "$CROSSING" --output "$FINAL" --seed "$((SEED + 5))" \
      > "$RUN/stage6.log" 2>&1
  G_X=$(gates_of "$CROSSING"); G_P=$(gates_of "$FINAL")
  G_C=$(sed -n 's/.*done in .*gates [0-9]* -> \([0-9]*\) (.*/\1/p' "$RUN/stage6.log" | tail -1)
  G_C=${G_C:-$G_P}
  note "stage 6 done: $G_X -> $G_C cubes (residual $(python3 -I -c "print(f'{100*$G_C/$G_X:.1f}%')")) -> $G_P packed gates"
else
  note "stage 6: $FINAL exists, skipping"
fi

note "PIPELINE COMPLETE: $FINAL"
