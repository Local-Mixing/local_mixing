#!/bin/bash
# TDP generation: the end-to-end four-stage pipeline. Manual: docs/tdp_pipeline.md;
# design and the reference recipe: docs/STAGE4_RUST_DESIGN.md.
#
# Stages:
#   1+2  gen_sandwich_gadget: sliced sandwich S of a source C, then embedded
#        masking (preset L8 by default; A = no bounded chains) or nonlinear291
#   3    circuit_mixer db_mixing: --tdp --db-mixing, schedule `validated`
#        (--profile 2,2,2,2,2, --moves int(G*2*2*1.3)), regular frozen store
#   4    tdp_packing run: windowed ANF delivery (pack G=512, repair, ANF
#        serialization with budget 65536) -> packing/deliverable.tdp4
#
# DEFAULTS reproduce scripts/windowed_delivery/waist/val_chain.sh CONFIG=L8 as
# run for the chosen validation runs vL64a/b, vL128a/b: same gen argv and
# stage-2 environment, same mixer argv and environment (serial), the same
# stage 4 (bit-exact port of the Python oracle). Seeds are fresh CSPRNG draws
# kept in the private $RUN/SEEDS; explicit seeds (--seed-file) are for
# calibration and reproduction only. Unlike val_chain.sh, the seeds reach gen
# and the mixer on inherited descriptors (@/dev/fd/N, --seed-file /dev/fd/3),
# never on argv where ps shows them; their banners print "(file)".
#
# Environment discipline: every stage binary starts from an EMPTY environment
# plus PATH/HOME/TMPDIR/LANG/LC_ALL and the variables the recipe sets
# (scripts/tdp_env_registry.tsv). An inherited variable that could change
# output is a fatal error, never silently honoured or dropped.
#
# Every stage writes its artifact + log into the run dir and is skipped when
# the artifact already exists (rerun with --run-rerun-from-stage K). bash only:
# do not port to zsh (word-splitting changes silently).
#
# bash >= 5.1 (checked first, in syntax bash 3.2 parses): associative arrays
# and mapfile need 4.0, and 5.1 is the first bash that feeds a small
# here-string through a pipe instead of a temporary file under TMPDIR, which
# keeps the seed hand-over below off disk. macOS's /bin/bash is 3.2; the
# managed driver (local_mixing_bin tdp_gen) finds a current bash itself.
if [ -z "${BASH_VERSINFO:-}" ] || [ "${BASH_VERSINFO[0]}" -lt 5 ] ||
   { [ "${BASH_VERSINFO[0]}" -eq 5 ] && [ "${BASH_VERSINFO[1]}" -lt 1 ]; }; then
  echo "FATAL: scripts/tdp_gen.sh needs bash >= 5.1, this is ${BASH_VERSION:-not bash} ($BASH); macOS: brew install bash, then run it with that bash" >&2
  exit 2
fi
set -euo pipefail

usage() {
  cat <<'EOF'
usage: tdp_gen.sh --source-wires N --run-directory RUNDIR [options]
  -h, --help    show this help and exit
  --source-wires N         wires of the source computation C (required; 3..4095)
  --run-directory DIR      run directory (created; artifacts + logs land here)
  --seed-file PATH         CALIBRATION/REPRODUCTION ONLY: private file (chmod 600,
                           <= 256 bytes) with lines gen_seed=<u63> and mix_seed=<u63>.
                           Default: fresh CSPRNG seeds, kept in the private $RUN/SEEDS
  --source-gates M (-m)    |C| = |D| gate count [default: the validated table,
                           1000 at n=64 and 3000 at n=128]
  --source-gates-rule R    validated (default; other n need --source-gates) or
                           formula (round(n·log2(n)^2); prints UNVALIDATED SIZE)
  --source-path PATH       load C (G57) instead of generating it; needs --source-gates
  --preprocessing-mode M   embedded-masking (default) or nonlinear291
  --preprocessing-preset P L8 (default) | A | custom (embedded-masking only).
                           L8 = SHUFFLING=8 RETURN_HOME=0 PREMASK=1 SEED_AVOID_C1=1
                           CHAIN=2 CHAIN_LEN=8; A = L8 without the chains
    custom only (defaults = gen's base parameters):
    --preprocessing-mask-pair-wires K      [2; 2..64]
    --preprocessing-max-open-masks N       [3; 2..64]
    --preprocessing-min-open-masks N       [2; 1..63, < max_open]
    --preprocessing-balanced-masks 0|1     [1]
    --preprocessing-shuffling-segments N   [0; 0 = off, else >= 8]
    --preprocessing-shuffling-return-home 0|1  [1; 0 = persistent migration]
    --preprocessing-premask 0|1            [0]
    --preprocessing-seed-avoid-c1 0|1      [0]
    --preprocessing-chain K                [0; 0 = off, 0..64]
    --preprocessing-chain-len L            [0; 0..1000000]
  --db-mixing-schedule S   validated (default: profile 2,2,2,2,2, moves int(4·G·1.3))
                           or custom:
    --db-mixing-growth-work-units G        [3]  profile n0
    --db-mixing-hold-work-units E          [27] profile n1 = n2 = n0 + E
    --db-mixing-target-size-factor R       [2]  profile r1 = r2
  --database-lookup-miss-filter F   auto (default) | on | off (FROZEN_FILTER)
  --leakage-repair                  sampled gate-wise audit + DB repair after stage 3
  --leakage-repair-seed S           [20803]
  --leakage-repair-reference PATH   [stage-3 input]
  --packing-window-gates G          [512]
  --packing-interval-cap W          [512]
  --packing-monomial-budget B       [24576]
  --packing-span-cap S              [3000]
  --packing-split-budget B          [65536]
  --packing-fix-iterations K        [4]
  --packing-fallback-iterations K   [4]
  --packing-anf-cap-semantics C     cpython39 (default, the oracle) | canonical
  --packing-allow-floor-miss        ship despite a visible non-output row (NOT recommended)
  --packing-allow-over-budget       ship despite an over-budget window (NOT recommended)
  --threads T              pin RAYON_NUM_THREADS and tdp_packing --threads
  --run-stop-after-stage K   stop after stage K (2, 3 or 4)                [4]
  --run-rerun-from-stage K   rebuild from stage K (2, 3 or 4)          [unset]
  --parallel-pieces P      stage-3 piecewise-parallel mixing [1 = serial; 1..64]
                           (changes the stage-3 bytes; not the validated recipe)
  --parallel-target-piece-gates B   automatic piece sizing (2..1000000000)
  --parallel-threads T     piece pool threads (1..1024; needs pieces > 1 or auto)
env (inputs only; every other output-affecting variable is refused):
  FROZEN_DB_DIR       the regular frozen store (required for stage 3)
  TDP_GEN_ALLOW_EMPTY_STORE=1  testing only: run stage 3 with no store
  TDP_BIN_DIR         directory with gen_sandwich_gadget, circuit_mixer, tdp_packing
                      [default: repository target/release]
EOF
  exit "${1:-1}"
}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REGISTRY=$SCRIPT_DIR/tdp_env_registry.tsv
LIB=$SCRIPT_DIR/tdp_reports
fatal() { echo "FATAL: $*" >&2; exit 2; }

# ---- inherited environment against the registry (§11.5) ----
[ -s "$REGISTRY" ] || fatal "$REGISTRY missing"
REG_NAMES=(); REG_KINDS=(); REG_CLASSES=()
while IFS=$'\t' read -r entry kind class _; do
  case "$entry" in '#'*|'') continue ;; esac
  REG_NAMES+=("$entry"); REG_KINDS+=("$kind"); REG_CLASSES+=("$class")
done < "$REGISTRY"
classify_env() { # NAME -> class or empty
  local name=$1 i best="" bestlen=-1
  for i in "${!REG_NAMES[@]}"; do
    if [ "${REG_KINDS[$i]}" = exact ] && [ "${REG_NAMES[$i]}" = "$name" ]; then
      echo "${REG_CLASSES[$i]}"; return
    fi
    if [ "${REG_KINDS[$i]}" = prefix ] && [[ $name == "${REG_NAMES[$i]}"* ]] &&
       [ "${#REG_NAMES[$i]}" -gt "$bestlen" ]; then
      best=${REG_CLASSES[$i]}; bestlen=${#REG_NAMES[$i]}
    fi
  done
  echo "$best"
}
refused=(); ignored=()
while IFS= read -r var; do
  case "$(classify_env "$var")" in
    recipe|driver|forbidden|diagnostic) refused+=("$var") ;;
    neutral) ignored+=("$var") ;;
  esac
done < <(compgen -e)
if [ "${#refused[@]}" -gt 0 ]; then
  list=$(printf '%s, ' "${refused[@]}"); list=${list%, }
  if [ "${#refused[@]}" -eq 1 ]; then
    fatal "$list is not honoured by the managed recipe; unset it (use flags/config)"
  fi
  fatal "$list are not honoured by the managed recipe; unset them (use flags/config)"
fi

N=""; RUN=""; SEED_FILE=""; MCD=""; MCD_RULE=validated; SOURCE_PATH=""
PREPROCESSING_MODE=embedded-masking; PRESET=""
declare -A KNOB=()
SCHEDULE=validated; GROWTH=""; HOLD=""; EXPAND=""
STOP_AFTER=4; FORCE_FROM=99
PIECES=1; PIECES_SET=0; MIN_BLOCK_SIZE=""; PIECE_THREADS=""
FILTER=auto; QC=0; QC_SEED=20803; QC_REF=""; THREADS=""
PACK_WINDOW_GATES=512; PACK_INTERVAL_CAP=512; PACK_MONOMIAL_BUDGET=24576; PACK_SPAN_CAP=3000
PACK_SPLIT_BUDGET=65536; PACK_FIX_ITERATIONS=4; PACK_FALLBACK_ITERATIONS=4; PACK_CAP_SEM=cpython39
PACK_ALLOW_FLOOR=0; PACK_ALLOW_BUDGET=0
declare -A SEEN=()
KNOB_FLAGS=" --preprocessing-mask-pair-wires --preprocessing-max-open-masks --preprocessing-min-open-masks --preprocessing-balanced-masks --preprocessing-shuffling-segments --preprocessing-shuffling-return-home --preprocessing-premask --preprocessing-seed-avoid-c1 --preprocessing-chain --preprocessing-chain-len "
while [ $# -gt 0 ]; do
  case "$1" in
    -n) set -- --source-wires "${@:2}" ;;
    -o) set -- --run-directory "${@:2}" ;;
    -m|--mcd) set -- --source-gates "${@:2}" ;;
    --expand) set -- --db-mixing-target-size-factor "${@:2}" ;;
    --hold) set -- --db-mixing-hold-work-units "${@:2}" ;;
    --stop-after) set -- --run-stop-after-stage "${@:2}" ;;
    --force-from) set -- --run-rerun-from-stage "${@:2}" ;;
    --pieces) set -- --parallel-pieces "${@:2}" ;;
    --min-block-size) set -- --parallel-target-piece-gates "${@:2}" ;;
    --piece-threads) set -- --parallel-threads "${@:2}" ;;
  esac
  case "$1" in
    -s|--calibration-seed)
      fatal "seeds on argv are no longer accepted (they appear in ps); use --seed-file (lines gen_seed= and mix_seed=)" ;;
    --crossing-*|--xr|--xb|--xc|--xtdiv|--xmoves)
      fatal "$1 configured the crossing walk (old stage 5), which is historic; see branch ran/archive/curated-and-postmix" ;;
  esac
  case "$1" in
    --*)
      if [ "${SEEN[$1]+present}" ]; then fatal "duplicate setting $1"; fi
      SEEN[$1]=1 ;;
  esac
  case "$1" in
    -h|--help|--leakage-repair|--packing-allow-floor-miss|--packing-allow-over-budget) ;;
    --*)
      if [ "$#" -lt 2 ] || [ -z "$2" ] || [[ $2 == -* ]]; then fatal "$1 requires an argument"; fi ;;
  esac
  case "$1" in
    -h|--help) usage 0 ;;
    --source-wires) N=$2; shift 2 ;;
    --run-directory) RUN=$2; shift 2 ;;
    --seed-file) SEED_FILE=$2; shift 2 ;;
    --source-gates) MCD=$2; shift 2 ;;
    --source-gates-rule) MCD_RULE=$2; shift 2 ;;
    --source-path) SOURCE_PATH=$2; shift 2 ;;
    --preprocessing-mode) PREPROCESSING_MODE=$2; shift 2 ;;
    --preprocessing-preset) PRESET=$2; shift 2 ;;
    --preprocessing-*)
      [[ $KNOB_FLAGS == *" $1 "* ]] || fatal "unknown argument $1"
      KNOB[${1#--preprocessing-}]=$2; shift 2 ;;
    --db-mixing-schedule) SCHEDULE=$2; shift 2 ;;
    --db-mixing-growth-work-units) GROWTH=$2; shift 2 ;;
    --db-mixing-hold-work-units) HOLD=$2; shift 2 ;;
    --db-mixing-target-size-factor) EXPAND=$2; shift 2 ;;
    --database-lookup-miss-filter) FILTER=$2; shift 2 ;;
    --leakage-repair) QC=1; shift ;;
    --leakage-repair-seed) QC_SEED=$2; shift 2 ;;
    --leakage-repair-reference) QC_REF=$2; shift 2 ;;
    --packing-window-gates) PACK_WINDOW_GATES=$2; shift 2 ;;
    --packing-interval-cap) PACK_INTERVAL_CAP=$2; shift 2 ;;
    --packing-monomial-budget) PACK_MONOMIAL_BUDGET=$2; shift 2 ;;
    --packing-span-cap) PACK_SPAN_CAP=$2; shift 2 ;;
    --packing-split-budget) PACK_SPLIT_BUDGET=$2; shift 2 ;;
    --packing-fix-iterations) PACK_FIX_ITERATIONS=$2; shift 2 ;;
    --packing-fallback-iterations) PACK_FALLBACK_ITERATIONS=$2; shift 2 ;;
    --packing-anf-cap-semantics) PACK_CAP_SEM=$2; shift 2 ;;
    --packing-allow-floor-miss) PACK_ALLOW_FLOOR=1; shift ;;
    --packing-allow-over-budget) PACK_ALLOW_BUDGET=1; shift ;;
    --threads) THREADS=$2; shift 2 ;;
    --run-stop-after-stage) STOP_AFTER=$2; shift 2 ;;
    --run-rerun-from-stage) FORCE_FROM=$2; shift 2 ;;
    --parallel-pieces) PIECES=$2; PIECES_SET=1; shift 2 ;;
    --parallel-target-piece-gates) MIN_BLOCK_SIZE=$2; shift 2 ;;
    --parallel-threads) PIECE_THREADS=$2; shift 2 ;;
    *) echo "FATAL: unknown argument $1" >&2; usage 2 ;;
  esac
done
[ -n "$N" ] && [ -n "$RUN" ] || usage

# Normalize decimal input before arithmetic (leading zeros are not octal).
# Check digit length first so very large inputs cannot overflow shell integers.
bounded_uint() {
  local label=$1 value=$2 minimum=$3 maximum=$4
  if [[ $value =~ ^[0-9]+$ ]]; then
    value="${value#"${value%%[!0]*}"}"
    value=${value:-0}
    if { [ "${#value}" -lt "${#maximum}" ] ||
         { [ "${#value}" -eq "${#maximum}" ] &&
           { [[ $value < $maximum ]] || [ "$value" = "$maximum" ]; }; }; } &&
       [ "$value" -ge "$minimum" ]; then
      printf '%s\n' "$value"
      return 0
    fi
  fi
  echo "FATAL: $label must be an integer in $minimum..$maximum" >&2
  return 2
}
bounded_number() { # LABEL VALUE LO HI LO_OPEN(0|1) -> normalized number
  python3 -I - "$@" <<'PY'
import math, sys
label, value, lo, hi, lo_open = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5] == "1"
try:
    v = float(value)
except ValueError:
    v = math.nan
if not math.isfinite(v) or (v <= lo if lo_open else v < lo) or v > hi:
    sys.stderr.write(f"FATAL: {label} must be a number in {'(' if lo_open else '['}{lo:g}, {hi:g}]\n")
    sys.exit(2)
print(f"{v:g}")
PY
}
N=$(bounded_uint --source-wires "$N" 3 4095)
case "$PREPROCESSING_MODE" in
  product-2223|2223|nonlinear193)
    fatal "$PREPROCESSING_MODE is retired from new TDP runs; resume saved runs with the checkout that created them" ;;
  nonlinear291|embedded-masking) ;;
  *) fatal "--preprocessing-mode must be embedded-masking or nonlinear291" ;;
esac
for stage_arg in "$STOP_AFTER" "$FORCE_FROM"; do
  case "$stage_arg" in
    5|6) fatal "stage $stage_arg (old crossing/fcompress) is historic; use 2, 3 or 4 (see branch ran/archive/curated-and-postmix)" ;;
  esac
done
STOP_AFTER=$(bounded_uint --run-stop-after-stage "$STOP_AFTER" 2 4)
[ "$FORCE_FROM" = 99 ] || FORCE_FROM=$(bounded_uint --run-rerun-from-stage "$FORCE_FROM" 2 4)
PIECES=$(bounded_uint --parallel-pieces "$PIECES" 1 64)
if [ -n "$MIN_BLOCK_SIZE" ]; then
  [ "$PIECES_SET" -eq 0 ] || fatal "--parallel-target-piece-gates is mutually exclusive with --parallel-pieces (including 1)"
  MIN_BLOCK_SIZE=$(bounded_uint --parallel-target-piece-gates "$MIN_BLOCK_SIZE" 2 1000000000)
fi
if [ -n "$PIECE_THREADS" ]; then
  PIECE_THREADS=$(bounded_uint --parallel-threads "$PIECE_THREADS" 1 1024)
  [ "$PIECES" -gt 1 ] || [ -n "$MIN_BLOCK_SIZE" ] ||
    fatal "--parallel-threads applies only with --parallel-pieces > 1 or --parallel-target-piece-gates"
fi
PIECEFLAGS=()
PIECE_DESCRIPTION="serial"
if [ -n "$MIN_BLOCK_SIZE" ]; then
  PIECEFLAGS=(--parallel-target-piece-gates "$MIN_BLOCK_SIZE")
  PIECE_DESCRIPTION="automatic pieces target_piece_gates=$MIN_BLOCK_SIZE"
elif [ "$PIECES" -gt 1 ]; then
  PIECEFLAGS=(--parallel-pieces "$PIECES")
  PIECE_DESCRIPTION="pieces $PIECES"
fi
[ -n "$PIECE_THREADS" ] && PIECEFLAGS+=(--parallel-threads "$PIECE_THREADS")
[ -z "$THREADS" ] || THREADS=$(bounded_uint --threads "$THREADS" 1 1024)
case "$FILTER" in auto|on|off) ;; *) fatal "--database-lookup-miss-filter must be auto, on or off" ;; esac
if [ "$QC" != 1 ] && { [ -n "$QC_REF" ] || [ "${SEEN[--leakage-repair-seed]+x}" = x ]; }; then
  fatal "--leakage-repair-seed/--leakage-repair-reference require --leakage-repair"
fi
QC_SEED=$(bounded_uint --leakage-repair-seed "$QC_SEED" 0 9223372036854775807)

# ---- stage-2 preset (§11.1; docs/STAGE2_LINKABILITY.md §6) ----
# gen's base parameters (EmbeddedMaskingParams::managed_tdp): the effective
# value of each knob when its variable is unset.
declare -A BASE=([mask-pair-wires]=2 [max-open-masks]=3 [min-open-masks]=2 [balanced-masks]=1
  [shuffling-segments]=0 [shuffling-return-home]=1 [premask]=0 [seed-avoid-c1]=0 [chain]=0 [chain-len]=0)
declare -A VAR=([mask-pair-wires]=EMBEDDED_MASKING_K [max-open-masks]=EMBEDDED_MASKING_MAX_OPEN
  [min-open-masks]=EMBEDDED_MASKING_MIN_OPEN [balanced-masks]=EMBEDDED_MASKING_BALANCED
  [shuffling-segments]=EMBEDDED_MASKING_SHUFFLING [shuffling-return-home]=EMBEDDED_MASKING_SHUFFLING_RETURN_HOME
  [premask]=EMBEDDED_MASKING_PREMASK [seed-avoid-c1]=EMBEDDED_MASKING_SEED_AVOID_C1
  [chain]=EMBEDDED_MASKING_CHAIN [chain-len]=EMBEDDED_MASKING_CHAIN_LEN)
KNOB_ORDER=(mask-pair-wires max-open-masks min-open-masks balanced-masks shuffling-segments
  shuffling-return-home premask seed-avoid-c1 chain chain-len)
declare -A EFF=()
GEN_ENV=()
if [ "$PREPROCESSING_MODE" = nonlinear291 ]; then
  [ -z "$PRESET" ] || fatal "--preprocessing-preset applies only to embedded-masking"
  [ "${#KNOB[@]}" -eq 0 ] ||
    fatal "preprocessing mask and shuffling controls are only valid with --preprocessing-mode embedded-masking"
  PRESET=none
else
  PRESET=${PRESET:-L8}
  case "$PRESET" in
    L8|A)
      for k in "${!KNOB[@]}"; do
        fatal "--preprocessing-$k is set by preset $PRESET; remove it or use --preprocessing-preset custom"
      done
      for k in "${KNOB_ORDER[@]}"; do EFF[$k]=${BASE[$k]}; done
      EFF[shuffling-segments]=8; EFF[shuffling-return-home]=0; EFF[premask]=1; EFF[seed-avoid-c1]=1
      GEN_ENV=(EMBEDDED_MASKING_SHUFFLING=8 EMBEDDED_MASKING_SHUFFLING_RETURN_HOME=0
               EMBEDDED_MASKING_PREMASK=1 EMBEDDED_MASKING_SEED_AVOID_C1=1)
      if [ "$PRESET" = L8 ]; then
        EFF[chain]=2; EFF[chain-len]=8
        GEN_ENV+=(EMBEDDED_MASKING_CHAIN=2 EMBEDDED_MASKING_CHAIN_LEN=8)
      fi ;;
    custom)
      v() { echo "${KNOB[$1]:-${BASE[$1]}}"; }
      EFF[mask-pair-wires]=$(bounded_uint --preprocessing-mask-pair-wires "$(v mask-pair-wires)" 2 64)
      EFF[max-open-masks]=$(bounded_uint --preprocessing-max-open-masks "$(v max-open-masks)" 2 64)
      EFF[min-open-masks]=$(bounded_uint --preprocessing-min-open-masks "$(v min-open-masks)" 1 63)
      EFF[balanced-masks]=$(bounded_uint --preprocessing-balanced-masks "$(v balanced-masks)" 0 1)
      EFF[shuffling-segments]=$(bounded_uint --preprocessing-shuffling-segments "$(v shuffling-segments)" 0 9223372036854775807)
      home=$(v shuffling-return-home); case "$home" in true) home=1 ;; false) home=0 ;; esac
      EFF[shuffling-return-home]=$(bounded_uint --preprocessing-shuffling-return-home "$home" 0 1)
      EFF[premask]=$(bounded_uint --preprocessing-premask "$(v premask)" 0 1)
      EFF[seed-avoid-c1]=$(bounded_uint --preprocessing-seed-avoid-c1 "$(v seed-avoid-c1)" 0 1)
      EFF[chain]=$(bounded_uint --preprocessing-chain "$(v chain)" 0 64)
      EFF[chain-len]=$(bounded_uint --preprocessing-chain-len "$(v chain-len)" 0 1000000)
      [ "${EFF[shuffling-segments]}" -eq 0 ] || [ "${EFF[shuffling-segments]}" -ge 8 ] ||
        fatal "--preprocessing-shuffling-segments must be 0 (off) or at least 8"
      [ "${EFF[min-open-masks]}" -lt "${EFF[max-open-masks]}" ] ||
        fatal "--preprocessing-min-open-masks must be less than --preprocessing-max-open-masks"
      for k in "${KNOB_ORDER[@]}"; do
        [ "${EFF[$k]}" = "${BASE[$k]}" ] || GEN_ENV+=("${VAR[$k]}=${EFF[$k]}")
      done ;;
    *) fatal "--preprocessing-preset must be L8, A or custom" ;;
  esac
  K=${EFF[mask-pair-wires]}
  [ "$((2 * N))" -gt "$((K - K % 2 + EFF[balanced-masks]))" ] ||
    fatal "Embedded masking requires 2 * N > K rounded down to an even number + balanced; reduce --preprocessing-mask-pair-wires or increase --source-wires"
fi

# ---- stage-3 schedule ----
case "$SCHEDULE" in
  validated)
    for f in growth-work-units hold-work-units target-size-factor; do
      [ "${SEEN[--db-mixing-$f]+x}" != x ] ||
        fatal "--db-mixing-$f is set by schedule validated; remove it or use --db-mixing-schedule custom"
    done
    PROFILE=2,2,2,2,2 ;;
  custom)
    GROWTH=$(bounded_number --db-mixing-growth-work-units "${GROWTH:-3}" 0 10000 1) || exit 2
    HOLD=$(bounded_number --db-mixing-hold-work-units "${HOLD:-27}" 0 10000 0) || exit 2
    EXPAND=$(bounded_number --db-mixing-target-size-factor "${EXPAND:-2}" 1 16 1) || exit 2
    PROFILE=$(python3 -I -c "import sys; n0,h,r=map(float,sys.argv[1:]); n1=n0+h; print(f'{n0:g},{n1:g},{n1:g},{r:g},{r:g}')" "$GROWTH" "$HOLD" "$EXPAND") ;;
  *) fatal "--db-mixing-schedule must be validated or custom" ;;
esac

# ---- stage-4 parameters ----
PACK_WINDOW_GATES=$(bounded_uint --packing-window-gates "$PACK_WINDOW_GATES" 1 1000000)
PACK_INTERVAL_CAP=$(bounded_uint --packing-interval-cap "$PACK_INTERVAL_CAP" 1 1000000)
PACK_MONOMIAL_BUDGET=$(bounded_uint --packing-monomial-budget "$PACK_MONOMIAL_BUDGET" 1 1000000000)
PACK_SPAN_CAP=$(bounded_uint --packing-span-cap "$PACK_SPAN_CAP" 1 1000000000)
PACK_SPLIT_BUDGET=$(bounded_uint --packing-split-budget "$PACK_SPLIT_BUDGET" 1 1000000000)
PACK_FIX_ITERATIONS=$(bounded_uint --packing-fix-iterations "$PACK_FIX_ITERATIONS" 0 1000)
PACK_FALLBACK_ITERATIONS=$(bounded_uint --packing-fallback-iterations "$PACK_FALLBACK_ITERATIONS" 0 1000)
case "$PACK_CAP_SEM" in cpython39|canonical) ;; *) fatal "--packing-anf-cap-semantics must be cpython39 or canonical" ;; esac
[ "$STOP_AFTER" -lt 4 ] || [ "$N" -le 256 ] || fatal "stage 4 supports source wires up to 256"
PACKFLAGS=()
pf() { [ "$2" = "$3" ] || PACKFLAGS+=("$1" "$2"); }
pf --window-gates "$PACK_WINDOW_GATES" 512
pf --interval-cap "$PACK_INTERVAL_CAP" 512
pf --monomial-budget "$PACK_MONOMIAL_BUDGET" 24576
pf --span-cap "$PACK_SPAN_CAP" 3000
pf --split-budget "$PACK_SPLIT_BUDGET" 65536
pf --fix-iterations "$PACK_FIX_ITERATIONS" 4
pf --fallback-iterations "$PACK_FALLBACK_ITERATIONS" 4
pf --anf-cap-semantics "$PACK_CAP_SEM" cpython39
[ "$PACK_ALLOW_FLOOR" = 0 ] || PACKFLAGS+=(--allow-floor-miss)
[ "$PACK_ALLOW_BUDGET" = 0 ] || PACKFLAGS+=(--allow-over-budget)

# ---- source size (§2.4) ----
if [ -n "$SOURCE_PATH" ]; then
  [ -s "$SOURCE_PATH" ] || fatal "--source-path $SOURCE_PATH is not a readable file"
  [ -n "$MCD" ] || fatal "--source-path needs --source-gates (its gate count)"
  SOURCE_PATH=$(cd "$(dirname "$SOURCE_PATH")" && pwd)/$(basename "$SOURCE_PATH")
fi
case "$MCD_RULE" in validated|formula) ;; *) fatal "--source-gates-rule must be validated or formula" ;; esac
validated_gates() { case "$1" in 64) echo 1000 ;; 128) echo 3000 ;; *) echo "" ;; esac; }
if [ -n "$MCD" ]; then
  [ "${SEEN[--source-gates-rule]+x}" != x ] || fatal "--source-gates and --source-gates-rule are exclusive"
  MCD=$(bounded_uint --source-gates "$MCD" 1 1000000000)
  if [ "$(validated_gates "$N")" = "$MCD" ]; then MCD_RULE=validated; else MCD_RULE=explicit; fi
elif [ "$MCD_RULE" = validated ]; then
  MCD=$(validated_gates "$N")
  [ -n "$MCD" ] || fatal "no validated source size for n=$N; set --source-gates or --source-gates-rule formula"
fi
read -r M_CD S_SL SLICE_G <<< "$(python3 -I - "$N" "${MCD:-0}" <<'EOF'
import math, sys
n, mcd = int(sys.argv[1]), int(sys.argv[2])
l = math.log2(n)
print(mcd if mcd > 0 else round(n * l * l), max(n, round(n * l)), 10 * 2 * n)
EOF
)"

# Renamed stage artifacts must not make an existing legacy run look fresh.
for legacy_artifact in gss_command.conf gss.mpmct1; do
  [ ! -e "$RUN/$legacy_artifact" ] ||
    fatal "$RUN contains legacy GSS artifact $legacy_artifact; use a fresh run directory or the checkout that created it"
done
[ ! -e "$RUN/SEED" ] ||
  fatal "$RUN was created by recipe v9 (a one-line SEED file); resume it with the checkout that created it"

BIN=${TDP_BIN_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)/target/release}
need=(gen_sandwich_gadget circuit_mixer)
[ "$STOP_AFTER" -lt 4 ] || need+=(tdp_packing)
for b in "${need[@]}"; do
  [ -x "$BIN/$b" ] || { echo "FATAL: $BIN/$b missing — cargo build --release first" >&2; exit 1; }
done

# ---- seeds (§8): $RUN/SEEDS, format 2, private ----
U63=9223372036854775792
parse_seeds() { # FILE -> sets GEN_SEED MIX_SEED SEED_SOURCE; never prints values
  local f=$1 lines
  { [ -f "$f" ] && [ "$(wc -c < "$f")" -le 512 ]; } || fatal "$f is not a valid SEEDS file"
  mapfile -t lines < "$f"
  { [ "${#lines[@]}" -eq 5 ] && [ "${lines[0]}" = seeds_format=2 ] &&
    [[ ${lines[1]} =~ ^gen_seed=([0-9]+)$ ]] && GEN_SEED=${BASH_REMATCH[1]} &&
    [[ ${lines[2]} =~ ^mix_seed=([0-9]+)$ ]] && MIX_SEED=${BASH_REMATCH[1]} &&
    [[ ${lines[3]} =~ ^run_key=[0-9a-f]{64}$ ]] &&
    [[ ${lines[4]} =~ ^seed_source=(random|explicit)$ ]] && SEED_SOURCE=${BASH_REMATCH[1]}; } ||
    fatal "$f is malformed (expected seeds_format=2, gen_seed=, mix_seed=, run_key=<64 hex>, seed_source=)"
  GEN_SEED=$(bounded_uint gen_seed "$GEN_SEED" 0 "$U63" 2>/dev/null) || fatal "$f: gen_seed out of range"
  MIX_SEED=$(bounded_uint mix_seed "$MIX_SEED" 0 "$U63" 2>/dev/null) || fatal "$f: mix_seed out of range"
}
read_seed_file() { # PATH -> sets EXPL_GEN EXPL_MIX
  local f=$1 lines l mode
  [ -f "$f" ] || fatal "--seed-file $f is not a regular file"
  [ "$(wc -c < "$f")" -le 256 ] || fatal "--seed-file $f is too large (maximum 256 bytes)"
  mode=$(python3 -I -c 'import os,sys; print(oct(os.stat(sys.argv[1]).st_mode & 0o777)[2:])' "$f")
  [ $((8#$mode & 8#077)) -eq 0 ] || fatal "--seed-file $f must not be accessible by group/other (chmod 600)"
  mapfile -t lines < "$f"
  if [ "${#lines[@]}" -eq 1 ] && [[ ${lines[0]} =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]]; then
    fatal "seed file format 1 holds one seed; recipe v10 needs gen_seed= and mix_seed= lines (the old seed was the gen seed; the old mix seed was gen+2)"
  fi
  EXPL_GEN=""; EXPL_MIX=""
  for l in "${lines[@]}"; do
    if [[ $l =~ ^gen_seed=([0-9]+)$ ]] && [ -z "$EXPL_GEN" ]; then EXPL_GEN=${BASH_REMATCH[1]}
    elif [[ $l =~ ^mix_seed=([0-9]+)$ ]] && [ -z "$EXPL_MIX" ]; then EXPL_MIX=${BASH_REMATCH[1]}
    elif [ -n "$l" ]; then fatal "--seed-file $f: unexpected line (only gen_seed= and mix_seed=)"
    fi
  done
  { [ -n "$EXPL_GEN" ] && [ -n "$EXPL_MIX" ]; } || fatal "--seed-file $f needs both gen_seed= and mix_seed="
  EXPL_GEN=$(bounded_uint gen_seed "$EXPL_GEN" 0 "$U63" 2>/dev/null) || fatal "--seed-file: gen_seed out of range 0..$U63"
  EXPL_MIX=$(bounded_uint mix_seed "$EXPL_MIX" 0 "$U63" 2>/dev/null) || fatal "--seed-file: mix_seed out of range 0..$U63"
}
[ -z "$SEED_FILE" ] || read_seed_file "$SEED_FILE"

mkdir -p "$RUN"; RUN=$(cd "$RUN" && pwd)
GADGET=$RUN/tdp.mpmct1
DB_MIXING=$RUN/db_mixing.mpmct1
SEEDS=$RUN/SEEDS
STAGE12_RECIPE_FILE=$RUN/stage12.recipe
LOGALL=$RUN/tdp_gen.log
note() { echo "[tdp_gen] $*" | tee -a "$LOGALL"; }
[ -s "$GADGET" ] || FORCE_FROM=2
if [ -e "$SEEDS" ]; then
  parse_seeds "$SEEDS"
  if [ -n "$SEED_FILE" ] && { [ "$EXPL_GEN" != "$GEN_SEED" ] || [ "$EXPL_MIX" != "$MIX_SEED" ]; }; then
    fatal "--seed-file conflicts with the existing $SEEDS; use the original seeds or a fresh run directory"
  fi
  SEED_SRC="RESUMED from $SEEDS ($SEED_SOURCE)"
else
  [ ! -s "$GADGET" ] || fatal "existing stage-2 artifact has no SEEDS; use a fresh run directory"
  KEY=$(python3 -I -c 'import secrets; print(secrets.token_hex(32))')
  if [ -n "$SEED_FILE" ]; then
    GEN_SEED=$EXPL_GEN; MIX_SEED=$EXPL_MIX; SEED_SOURCE=explicit
  else
    read -r GEN_SEED MIX_SEED < <(python3 -I -c "import secrets; print(secrets.randbelow($U63 + 1), secrets.randbelow($U63 + 1))")
    SEED_SOURCE=random
  fi
  ( umask 077; set -o noclobber
    printf 'seeds_format=2\ngen_seed=%s\nmix_seed=%s\nrun_key=%s\nseed_source=%s\n' \
      "$GEN_SEED" "$MIX_SEED" "$KEY" "$SEED_SOURCE" > "$SEEDS" ) || fatal "cannot create $SEEDS"
  unset KEY
  SEED_SRC="NEW ($SEED_SOURCE) in $SEEDS"
fi
[ "$SEED_SOURCE" = random ] || SEED_SRC="$SEED_SRC — EXPLICIT: calibration only, NOT a deliverable"
umask 077

# ---- stage launcher: empty environment + allow-list + recipe variables ----
ALLOW_ENV=(PATH="$PATH" HOME="${HOME:-/}" LANG=C LC_ALL=C)
[ -z "${TMPDIR:-}" ] || ALLOW_ENV+=(TMPDIR="$TMPDIR")
[ -z "$THREADS" ] || ALLOW_ENV+=(RAYON_NUM_THREADS="$THREADS")
RECIPE_ENV=()
stage_exec() { env -i "${ALLOW_ENV[@]}" ${RECIPE_ENV[@]+"${RECIPE_ENV[@]}"} "$@"; }
gates_of() { head -1 "$1" | awk '{print $3}'; }
mac() { python3 -I "$LIB/run_mac.py" "$SEEDS" "$@"; }
sha_of() { python3 -I -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"; }

# ---- stage-2 identity marker (tdp_stage12_recipe=7) ----
STAGE12_RECIPE=(
  "tdp_stage12_recipe=7"
  "gen_binary_sha256=$(sha_of "$BIN/gen_sandwich_gadget")"
  "sandwich_variant=classic"
  "preprocessing_mode=$PREPROCESSING_MODE"
  "preprocessing_preset=$PRESET"
)
if [ "$PREPROCESSING_MODE" = embedded-masking ]; then
  for k in "${KNOB_ORDER[@]}"; do STAGE12_RECIPE+=("preprocessing_${k//-/_}=${EFF[$k]}"); done
fi
gen_env_text=$(printf '%s,' ${GEN_ENV[@]+"${GEN_ENV[@]}"}); gen_env_text=${gen_env_text%,}
STAGE12_RECIPE+=(
  "stage2_env=${gen_env_text:-none}"
  "gadget_seed_rule=gen"
  "n=$N" "m_cd=$M_CD" "source_gates_rule=$MCD_RULE" "s=$S_SL" "slice_gates=$SLICE_G" "rg_freq=1"
  "source=$([ -n "$SOURCE_PATH" ] && echo path || echo generated)"
  "$(mac --seeds)"
)
[ -z "$SOURCE_PATH" ] || STAGE12_RECIPE+=("$(mac source="$SOURCE_PATH")")
if [ "$FORCE_FROM" -gt 2 ] && [ -s "$GADGET" ]; then
  [ -s "$STAGE12_RECIPE_FILE" ] ||
    fatal "existing stage-2 artifact has no recipe marker; rerun with --run-rerun-from-stage 2"
  mapfile -t stored < "$STAGE12_RECIPE_FILE"
  [ "${#stored[@]}" -eq "${#STAGE12_RECIPE[@]}" ] ||
    fatal "existing stage-2 recipe marker does not match (recipe, seeds or binary); use a fresh run directory or --run-rerun-from-stage 2"
  for i in "${!STAGE12_RECIPE[@]}"; do
    [ "${stored[$i]}" = "${STAGE12_RECIPE[$i]}" ] ||
      fatal "existing stage-2 recipe does not match the requested mode, preset, dimensions, seeds or binary; use a fresh run directory or --run-rerun-from-stage 2"
  done
fi

size_note=""; [ "$MCD_RULE" != formula ] || size_note=" UNVALIDATED SIZE"
note "run=$RUN n=$N |C|=|D|=$M_CD (rule $MCD_RULE$size_note) s=$S_SL slice_gates=$SLICE_G mode=$PREPROCESSING_MODE preset=$PRESET schedule=$SCHEDULE profile=$PROFILE stage3=$PIECE_DESCRIPTION"
note "seeds: $SEED_SRC (values never printed)"
for v in ${ignored[@]+"${ignored[@]}"}; do note "inherited $v ignored: cache sizes are set by the driver (output-neutral)"; done

# ---- stages 1+2: sandwich + preprocessing ----
if [ "$FORCE_FROM" -le 2 ] || [ ! -s "$GADGET" ]; then
  RECIPE_ENV=(${GEN_ENV[@]+"${GEN_ENV[@]}"})
  [ -z "$SOURCE_PATH" ] || RECIPE_ENV+=(TDP_SOURCE_C="$SOURCE_PATH")
  # val_chain.sh's argv, except that the three seeds (all = the gen seed) arrive
  # as @/dev/fd/N here-strings: a seed on argv is visible in ps to every local
  # user, and the gen seed regenerates C. gen prints them as "(file)".
  GEN_ARGS=("$GADGET" "$N" "$M_CD" "$M_CD" "$S_SL" 1 "$SLICE_G" @/dev/fd/3 @/dev/fd/4 @/dev/fd/5)
  [ "$PREPROCESSING_MODE" = embedded-masking ] || GEN_ARGS+=("$PREPROCESSING_MODE")
  note "stage 1+2: gen_sandwich_gadget (mode=$PREPROCESSING_MODE preset=$PRESET env: ${gen_env_text:-none})"
  rm -f "$STAGE12_RECIPE_FILE" "$RUN/stage2_mult.txt" "$RUN/stage2_span.txt" "$RUN/stage2_span.tsv"
  stage_exec "$BIN/gen_sandwich_gadget" "${GEN_ARGS[@]}" 3<<< "$GEN_SEED" 4<<< "$GEN_SEED" 5<<< "$GEN_SEED" \
      > "$RUN/stage12.log" 2>&1 ||
    { note "stage 1+2 FAILED; see $RUN/stage12.log"; exit 1; }
  banner=$(head -1 "$RUN/stage12.log")
  if [[ $banner == *" seed=(file) gadget_seed=(file) sandwich_seed=(file) "* ]]; then
    note "gen seeds: verified (all three read from the driver's descriptors)"
  else
    rm -f "$GADGET"; fatal "gen used a different seed than SEEDS (its banner does not show the three driver-supplied seeds); stage-2 artifact removed"
  fi
  printf '%s\n' "${STAGE12_RECIPE[@]}" > "$STAGE12_RECIPE_FILE.tmp"
  mv "$STAGE12_RECIPE_FILE.tmp" "$STAGE12_RECIPE_FILE"
  [ "$FORCE_FROM" -le 3 ] || FORCE_FROM=3
  note "stage 1+2 done: TDP $(gates_of "$GADGET") gates (S: $(gates_of "$GADGET.sandwich.mpmct1"))"
else
  note "stage 1+2: $GADGET exists, skipping"
fi

# ---- stage-2 side reports (report-only, val_chain.sh lines 24-27) ----
SIDE_PIDS=()
if [ ! -s "$RUN/stage2_mult.txt" ] && [ -x "$BIN/tdp_packing" ]; then
  ( RECIPE_ENV=(); stage_exec "$BIN/tdp_packing" mult-census "$GADGET" tdp > "$RUN/stage2_mult.txt" 2>&1 ||
      echo "[tdp_gen] note: stage2_mult failed (report only)" >> "$LOGALL" ) &
  SIDE_PIDS+=($!)
fi
if [ ! -s "$RUN/stage2_span.txt" ]; then
  if [ -x "$BIN/hmap_affine" ]; then
    STEP=$([ "$N" = 64 ] && echo 64 || echo 256)
    ( RECIPE_ENV=(SPAN_OUTPUTS_TRIVIAL=1 SPAN_INTERSECT="$RUN/stage2_span.tsv")
      stage_exec "$BIN/hmap_affine" --c "$GADGET.source_c.g57" --c-format g57 --g "$GADGET" --n "$N" \
        --exposure --c-step 1 --g-step "$STEP" --out "$RUN/stage2_span" 2>&1 | grep "\[span\]" > "$RUN/stage2_span.txt" ||
        echo "[tdp_gen] note: stage2_span failed (report only)" >> "$LOGALL" ) &
    SIDE_PIDS+=($!)
  else
    note "stage2_span skipped: no $BIN/hmap_affine (the managed driver builds it; standalone: cargo build --release --features security-tools --bin hmap_affine, then place it in $BIN)"
  fi
fi
wait_side() { local p; for p in ${SIDE_PIDS[@]+"${SIDE_PIDS[@]}"}; do wait "$p" || true; done; SIDE_PIDS=(); }
if [ "$STOP_AFTER" -le 2 ]; then wait_side; note "stopped after stage 2"; exit 0; fi

# ---- stage 3: circuit_mixer db_mixing ----
if [ "$FORCE_FROM" -le 3 ] || [ ! -s "$DB_MIXING" ]; then
  if [ -z "${FROZEN_DB_DIR:-}" ] && [ "${TDP_GEN_ALLOW_EMPTY_STORE:-0}" != "1" ]; then
    echo "FATAL: stage 3 needs FROZEN_DB_DIR (or TDP_GEN_ALLOW_EMPTY_STORE=1 for a plumbing test)" >&2; exit 1
  fi
  G_IN=$(gates_of "$GADGET")
  if [ "$SCHEDULE" = validated ]; then
    A_MOVES=$(python3 -I -c "import sys; G=int(sys.argv[1]); print(int(G*2*2*1.3))" "$G_IN")
  else
    A_MOVES=$(python3 -I -c "import sys; n0,h,r=map(float,sys.argv[1:4]); G=int(sys.argv[4]); print(int((n0+h)*r*G*1.3))" "$GROWTH" "$HOLD" "$EXPAND" "$G_IN")
  fi
  rm -f "$RUN/stage3.settings"; FF=""
  RECIPE_ENV=(FROZEN_REGULAR_VALUE_CONVENTION=native LOOKUP_CACHE_MB=2048 CANON_CACHE_MB=256
              XPOLY_CANON_CACHE_MB=1024 MIXER_STOP_FLAG="$RUN/stage3.stop" MIXER_DUMP_FLAG="$RUN/stage3.dump")
  rm -f "$RUN/stage3.stop"
  DBFLAGS=(--p-db 1.0)
  FILTER_NOTE=""
  if [ -n "${FROZEN_DB_DIR:-}" ]; then
    RECIPE_ENV+=(FROZEN_DB_DIR="$FROZEN_DB_DIR")
    # D11: FILTER_NEUTRALITY_RECORDED flips to 1 once a filtered/unfiltered
    # db_mixing comparison is recorded (docs/STAGE4_RUST_DESIGN.md §10.9 item 6).
    # The managed manifest predicts this resolution (src/tdp/manifest.rs
    # resolved_frozen_filter, frozen_filter=auto(<value>)); keep them in step.
    # The value set is recorded in $RUN/stage3.settings.
    FILTER_NEUTRALITY_RECORDED=0
    case "$FILTER" in
      on) [ -s "$FROZEN_DB_DIR/filters.bin" ] || fatal "--database-lookup-miss-filter on needs $FROZEN_DB_DIR/filters.bin"
          FF=1 ;;
      off) FF=0 ;;
      auto)
        if [ ! -s "$FROZEN_DB_DIR/filters.bin" ]; then FF=1   # the reference situation: the mixer runs unfiltered
        elif [ "$FILTER_NEUTRALITY_RECORDED" = 1 ]; then
          avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo 2>/dev/null ||
                  vm_stat | awk '/page size of/{ps=$8} /Pages free|Pages inactive|Pages speculative/{gsub("\\.","",$NF); s+=$NF} END{print int(s*ps/1073741824)}')
          if [ "${avail:-0}" -ge 60 ]; then FF=1; else FF=0; fi
        else FF=0
        fi ;;
    esac
    RECIPE_ENV+=(FROZEN_FILTER=$FF)
    FILTER_NOTE=" FROZEN_FILTER=$FF ($FILTER)"
  else
    DBFLAGS=(--p-db 0)
    note "WARNING: empty-store plumbing mode — db_mixing performs NO re-encoding"
  fi
  QCFLAGS=()
  if [ "$QC" = 1 ]; then
    QCFLAGS=(--leakage-repair --leakage-repair-seed "$QC_SEED" --leakage-repair-report "$RUN/db_mixing.leakage_repair.txt")
    [ -z "$QC_REF" ] || QCFLAGS+=(--leakage-repair-reference "$QC_REF")
    note "stage 3: leakage repair enabled; report $RUN/db_mixing.leakage_repair.txt"
  fi
  note "stage 3: circuit_mixer --tdp --db-mixing --profile $PROFILE --moves $A_MOVES (schedule $SCHEDULE, $PIECE_DESCRIPTION)$FILTER_NOTE"
  # val_chain.sh's argv, in its order (the seed as --seed-file /dev/fd/3, off argv); optional flags only when enabled.
  stage_exec "$BIN/circuit_mixer" --input "$GADGET" --output "$DB_MIXING" --tdp --db-mixing \
      --profile "$PROFILE" --moves "$A_MOVES" "${DBFLAGS[@]}" \
      --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048 \
      --no-local-verify --seed-file /dev/fd/3 \
      ${QCFLAGS[@]+"${QCFLAGS[@]}"} ${PIECEFLAGS[@]+"${PIECEFLAGS[@]}"} 3<<< "$MIX_SEED" > "$RUN/stage3.log" 2>&1 ||
    { note "stage 3 FAILED; see $RUN/stage3.log"; exit 1; }
  # Why the chain ended (circuit_mixer's "chain done in Ts: <reason> (...)").
  # An operator stop flag ($RUN/stage3.stop) ends it "cleanly" with exit 0 but
  # below the recipe's dose: that output is never handed to stage 4 under the
  # canonical name, so neither this run nor a resume ships it.
  STOP_REASON=$(sed -n 's/^\[circuit_mixer\] chain done in [0-9.]*s: \(.*\) ([0-9]* -> [0-9]* gates, .*$/\1/p' "$RUN/stage3.log" | tail -1)
  # The stage-3 settings resolved at run time (FROZEN_FILTER under auto).
  printf 'stage3_settings=1\nfrozen_filter_mode=%s\nfrozen_filter=%s\np_db=%s\nstop_reason=%s\n' \
    "$FILTER" "${FF:-unset}" "${DBFLAGS[1]}" "${STOP_REASON:-unknown}" > "$RUN/stage3.settings"
  case "$STOP_REASON" in
    "moves budget spent"|"profile complete (size schedule finished)") ;;
    "stop flag"|"")
      mv -f "$DB_MIXING" "$RUN/db_mixing.stopped.mpmct1"
      rm -f "$RUN/stage3.stop"
      fatal "stage 3 ended by ${STOP_REASON:-an unrecognised stop (no chain-done line in $RUN/stage3.log)}, below the recipe's dose: kept as $RUN/db_mixing.stopped.mpmct1 (NOT a stage-3 artifact); this run has no deliverable; a resume re-runs stage 3 from scratch" ;;
    *) note "WARNING: stage 3 ended by \"$STOP_REASON\", not by its move budget or profile; see $RUN/stage3.log" ;;
  esac
  [ "$FORCE_FROM" -le 4 ] || FORCE_FROM=4
  note "stage 3 done: $(gates_of "$DB_MIXING") gates"
else
  note "stage 3: $DB_MIXING exists, skipping"
fi
if [ "$STOP_AFTER" -le 3 ]; then
  wait_side; note "PIPELINE STOPPED after stage 3: $DB_MIXING"; exit 0
fi

# ---- stage 4: tdp_packing run (windowed ANF delivery) ----
PACK_DIR=$RUN/packing; MARKER=$RUN/packing.recipe
stage4_exit_meaning() {
  case "$1" in
    2) echo "usage or parameter error" ;; 3) echo "input error" ;;
    4) echo "verification failure" ;; 5) echo "exposure floor missed" ;;
    6) echo "over budget" ;; 7) echo "I/O error writing outputs" ;;
    8) echo "internal invariant violated" ;; 101) echo "panic" ;; *) echo "unknown exit code" ;;
  esac
}
expected_marker() {
  echo "packing_recipe=1"
  echo "tdp_packing_sha256=$(sha_of "$BIN/tdp_packing")"
  echo "params=$(IFS=' '; echo "${PACKFLAGS[*]:-default}")"
  mac stage2="$GADGET" stage3="$DB_MIXING" source_c="$GADGET.source_c.g57"
}
stage4_done=0
if [ "$FORCE_FROM" -gt 4 ] && [ -s "$MARKER" ]; then
  exp=$(expected_marker)
  nexp=$(printf '%s\n' "$exp" | wc -l | tr -d ' ')
  if [ "$(head -n "$nexp" "$MARKER")" = "$exp" ] && [ -s "$PACK_DIR/deliverable.tdp4" ] &&
     [ "$(sed -n "$((nexp + 1))p" "$MARKER")" = "$(mac deliverable_tdp4="$PACK_DIR/deliverable.tdp4")" ]; then
    stage4_done=1
    note "stage 4: $PACK_DIR/deliverable.tdp4 matches its completion marker, skipping"
  else
    note "stage 4: completion marker does not match inputs/parameters/binary/deliverable; re-running stage 4"
  fi
fi
if [ "$stage4_done" -eq 0 ]; then
  rm -f "$MARKER" "$RUN/stage4.failed"; rm -rf "$PACK_DIR"
  THREADFLAGS=(); [ -z "$THREADS" ] || THREADFLAGS=(--threads "$THREADS")
  note "stage 4: tdp_packing run (window gates $PACK_WINDOW_GATES, split budget $PACK_SPLIT_BUDGET, caps $PACK_CAP_SEM${PACKFLAGS[*]:+; flags ${PACKFLAGS[*]}})"
  RECIPE_ENV=()
  set +e
  stage_exec "$BIN/tdp_packing" run --stage2 "$GADGET" --stage3 "$DB_MIXING" --source-c "$GADGET.source_c.g57" \
      --n "$N" --out-dir "$PACK_DIR" ${PACKFLAGS[@]+"${PACKFLAGS[@]}"} ${THREADFLAGS[@]+"${THREADFLAGS[@]}"} \
      > "$RUN/stage4.log" 2>&1
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    # the managed driver attaches the tdp_packing exit meaning only when this
    # marker names its exit code (a 2 from gen, the mixer or bash is not stage 4's)
    printf 'exit=%s\n' "$rc" > "$RUN/stage4.failed"
    wait_side
    note "stage 4 FAILED: exit $rc ($(stage4_exit_meaning "$rc")); see $RUN/stage4.log"
    exit "$rc"
  fi
  status=$(sed -n 's/^status=//p' "$PACK_DIR/packing.report")
  { expected_marker; mac deliverable_tdp4="$PACK_DIR/deliverable.tdp4"; echo "status=$status"; } > "$MARKER.tmp"
  mv "$MARKER.tmp" "$MARKER"
fi
wait_side
status=$(sed -n 's/^status=//p' "$MARKER")
if [ "$status" = ok ]; then
  rm -f "$PACK_DIR/NOT_RECOMMENDED"
  note "PIPELINE COMPLETE (stages 1-4): $PACK_DIR/deliverable.tdp4"
else
  : > "$PACK_DIR/NOT_RECOMMENDED"
  note "PIPELINE FINISHED WITH AN ALLOWED FAILURE (status=$status): $PACK_DIR/deliverable.tdp4 is NOT a recommended deliverable"
fi
