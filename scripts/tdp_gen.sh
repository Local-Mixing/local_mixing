#!/bin/bash
# TDP generation: the end-to-end six-stage pipeline. Manual: docs/tdp_pipeline.md;
# design and the reference recipe of stages 1-3 and the packer: docs/STAGE4_RUST_DESIGN.md;
# the delivery stage: docs/DELIVERY_STAGE.md.
#
# Stages (every stage after the first is optional: --run-skip-stages; the
# default runs all six):
#   1+2  gen_sandwich_gadget: sliced sandwich S of a source C, then embedded
#        masking (preset L8 by default; A = no bounded chains) or nonlinear291.
#        Skipping stage 2 hands the sandwich itself (tdp.mpmct1.sandwich.mpmct1)
#        to the next stage.
#   3    circuit_mixer db_mixing: --tdp --db-mixing, schedule `validated`
#        (--profile 2,2,2,2,2, --moves int(G*2*2*1.3)), regular frozen store
#   4    circuit_mixer --split --split-stop: every complemented g57 gate split
#        into plain conjunctions, joined to distant brackets (archive flags)
#   5    circuit_mixer crossing walk (resumes split.state when stage 4 ran):
#        target = round(G * factor), temperature target/divisor, 2*target moves
#   6    delivery (RC, 2026-10-12: the former stages 6 packing and 7 unpacking
#        as ONE stage): tdp_packing run (pack G=512, repair, ANF serialization
#        with budget 65536) -> packing/deliverable.tdp4; then tdp_unpacking
#        run --refine: every window resynthesized from its ANF alone, a window
#        that does not resynthesize cut in two and both halves repacked and
#        unpacked (up to --delivery-max-rounds rounds), the result concatenated
#        and canonically reordered -> unpacking/deliverable.tdp4 (the refined
#        lattice) + unpacking/unpacked.mpmct1. --delivery-anf-only ends the
#        stage after the packer's deliverable.
#
# DEFAULTS of stages 1-3 and the packer reproduce scripts/windowed_delivery/
# waist/val_chain.sh CONFIG=L8 as run for the chosen validation runs vL64a/b,
# vL128a/b: same gen argv and stage-2 environment, same mixer argv and
# environment (serial), the same packer (bit-exact port of the Python
# oracle); `--run-skip-stages 4,5 --delivery-anf-only` is that four-stage
# chain exactly. Stages 4 and 5 run with the flags of the archive-era driver
# (ea2e516f). Seeds are fresh CSPRNG draws kept in the private $RUN/SEEDS
# (format 3: gen, mix, split, cross and unpack seeds; the unpack seed is the
# delivery stage's); explicit seeds (--seed-file) are for calibration and
# reproduction only. Unlike val_chain.sh, the seeds reach the stage binaries
# on inherited descriptors (@/dev/fd/N, --seed-file /dev/fd/3), never on
# argv where ps shows them; their banners print "(file)".
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
  --crossing-target-size-factor R   stage-5 target = round(G * R), R in [1, 16]  [2]
  --crossing-width-penalty-base B   wide-split acceptance penalty base B >= 1     [3]
  --crossing-width-penalty-threshold C  width below which the penalty is off     [1]
  --crossing-size-tolerance-divisor D   temperature = max(64, round(target / D)) [25]
  --crossing-move-budget-factor F   move budget = round(F * target), F in (0, 1000]     [2]
  --crossing-move-attempts M        absolute move budget instead of the factor (exclusive)
                                    (either is added to the resumed split counter)
  --delivery-anf-only               stage 6 ends after the packer's deliverable
                                    (no resynthesis, no lattice refinement)
  --delivery-time-limit S           synthesis wall limit per window per round, seconds [30]
  --delivery-restarts R             synthesis restarts per window              [16]
  --delivery-beam B                 synthesis lookahead width                   [6]
  --delivery-max-rounds N           lattice refinement rounds (a window that does not
                                    resynthesize is cut in two and repacked)   [16]
  --run-skip-stages LIST   comma-separated stages to skip (2..6)             [none]
  --run-stop-after-stage K   stop after stage K (2..6)                      [6]
  --run-rerun-from-stage K   rebuild from stage K (2..6)                [unset]
  --parallel-pieces P      stage-3 piecewise-parallel mixing [1 = serial; 1..64]
                           (changes the stage-3 bytes; not the validated recipe)
  --parallel-target-piece-gates B   automatic piece sizing (2..1000000000)
  --parallel-threads T     piece pool threads (1..1024; needs pieces > 1 or auto)
env (inputs only; every other output-affecting variable is refused):
  FROZEN_DB_DIR       the regular frozen store (required for stage 3)
  TDP_GEN_ALLOW_EMPTY_STORE=1  testing only: run stage 3 with no store
  TDP_BIN_DIR         directory with gen_sandwich_gadget, circuit_mixer, tdp_packing,
                      tdp_unpacking (not needed under --delivery-anf-only)
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
STOP_AFTER=6; FORCE_FROM=99; SKIP_LIST=""
XR=2; XB=3; XC=1; XTDIV=25; XMF=2; XMOVES=""
DELIV_ANF_ONLY=0
DELIV_TIME_LIMIT=30; DELIV_RESTARTS=16; DELIV_BEAM=6; DELIV_MAX_ROUNDS=16
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
    --skip-stages) set -- --run-skip-stages "${@:2}" ;;
    --xr) set -- --crossing-target-size-factor "${@:2}" ;;
    --xb) set -- --crossing-width-penalty-base "${@:2}" ;;
    --xc) set -- --crossing-width-penalty-threshold "${@:2}" ;;
    --xtdiv) set -- --crossing-size-tolerance-divisor "${@:2}" ;;
    --xmoves) set -- --crossing-move-attempts "${@:2}" ;;
    --xmf) set -- --crossing-move-budget-factor "${@:2}" ;;
    --pieces) set -- --parallel-pieces "${@:2}" ;;
    --min-block-size) set -- --parallel-target-piece-gates "${@:2}" ;;
    --piece-threads) set -- --parallel-threads "${@:2}" ;;
  esac
  case "$1" in
    -s|--calibration-seed)
      fatal "seeds on argv are no longer accepted (they appear in ps); use --seed-file (lines gen_seed= and mix_seed=)" ;;
    --unpacking-no-fallback)
      fatal "--unpacking-no-fallback is gone: stages 6 and 7 are one delivery stage, and a window that does not resynthesize is cut in two and repacked (--delivery-max-rounds), not emitted verbatim" ;;
    --unpacking-*)
      fatal "$1 is now --delivery-${1#--unpacking-} (stages 6 and 7 are one delivery stage; see --help)" ;;
  esac
  case "$1" in
    --*)
      if [ "${SEEN[$1]+present}" ]; then fatal "duplicate setting $1"; fi
      SEEN[$1]=1 ;;
  esac
  case "$1" in
    -h|--help|--leakage-repair|--packing-allow-floor-miss|--packing-allow-over-budget|--delivery-anf-only) ;;
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
    --crossing-target-size-factor) XR=$2; shift 2 ;;
    --crossing-width-penalty-base) XB=$2; shift 2 ;;
    --crossing-width-penalty-threshold) XC=$2; shift 2 ;;
    --crossing-size-tolerance-divisor) XTDIV=$2; shift 2 ;;
    --crossing-move-attempts) XMOVES=$2; shift 2 ;;
    --crossing-move-budget-factor) XMF=$2; shift 2 ;;
    --delivery-anf-only) DELIV_ANF_ONLY=1; shift ;;
    --delivery-time-limit) DELIV_TIME_LIMIT=$2; shift 2 ;;
    --delivery-restarts) DELIV_RESTARTS=$2; shift 2 ;;
    --delivery-beam) DELIV_BEAM=$2; shift 2 ;;
    --delivery-max-rounds) DELIV_MAX_ROUNDS=$2; shift 2 ;;
    --threads) THREADS=$2; shift 2 ;;
    --run-skip-stages) SKIP_LIST=$2; shift 2 ;;
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
STOP_AFTER=$(bounded_uint --run-stop-after-stage "$STOP_AFTER" 2 6)
[ "$FORCE_FROM" = 99 ] || FORCE_FROM=$(bounded_uint --run-rerun-from-stage "$FORCE_FROM" 2 6)
# ---- optional stages (every stage after the first; 6 = delivery, the last) ----
SKIP=","
if [ -n "$SKIP_LIST" ]; then
  [[ $SKIP_LIST =~ ^[0-9]+(,[0-9]+)*$ ]] || fatal "--run-skip-stages must be a comma-separated list of stages in 2..6"
  IFS=, read -r -a skip_items <<< "$SKIP_LIST"
  for item in "${skip_items[@]}"; do
    item=$(bounded_uint --run-skip-stages "$item" 2 6) || exit 2
    [[ $SKIP == *",$item,"* ]] || SKIP="$SKIP$item,"
  done
fi
stage_on() { [ "$1" -le "$STOP_AFTER" ] && [[ $SKIP != *",$1,"* ]]; }
# the unpacking sub-step of stage 6 runs unless --delivery-anf-only
unpacking_on() { stage_on 6 && [ "$DELIV_ANF_ONLY" = 0 ]; }
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

# ---- stage-5 (crossing walk) parameters ----
XR=$(bounded_number --crossing-target-size-factor "$XR" 1 16 0) || exit 2
XB=$(bounded_number --crossing-width-penalty-base "$XB" 1 1000000 0) || exit 2
XC=$(bounded_uint --crossing-width-penalty-threshold "$XC" 0 1000000000)
XTDIV=$(bounded_uint --crossing-size-tolerance-divisor "$XTDIV" 1 1000000000)
XMF=$(bounded_number --crossing-move-budget-factor "$XMF" 0 1000 1) || exit 2
[ -z "$XMOVES" ] || XMOVES=$(bounded_uint --crossing-move-attempts "$XMOVES" 1 1000000000000)
if [ -n "$XMOVES" ] && [ "${SEEN[--crossing-move-budget-factor]+x}" = x ]; then
  fatal "--crossing-move-attempts (an absolute budget) and --crossing-move-budget-factor are exclusive"
fi

# ---- stage-6 (delivery) unpacking sub-step parameters ----
DELIV_TIME_LIMIT=$(bounded_number --delivery-time-limit "$DELIV_TIME_LIMIT" 0 86400 1) || exit 2
DELIV_RESTARTS=$(bounded_uint --delivery-restarts "$DELIV_RESTARTS" 1 1000)
DELIV_BEAM=$(bounded_uint --delivery-beam "$DELIV_BEAM" 1 1000)
DELIV_MAX_ROUNDS=$(bounded_uint --delivery-max-rounds "$DELIV_MAX_ROUNDS" 1 1000)
# --refine and its round bound are the stage's definition and always on the
# command line; the synthesizer's knobs only when not at their defaults
DELIVFLAGS=(--refine --max-rounds "$DELIV_MAX_ROUNDS")
dfl() { [ "$2" = "$3" ] || DELIVFLAGS+=("$1" "$2"); }
dfl --time-limit "$DELIV_TIME_LIMIT" 30
dfl --restarts "$DELIV_RESTARTS" 16
dfl --beam "$DELIV_BEAM" 6

# ---- stage-6 (delivery) packing sub-step parameters ----
PACK_WINDOW_GATES=$(bounded_uint --packing-window-gates "$PACK_WINDOW_GATES" 1 1000000)
PACK_INTERVAL_CAP=$(bounded_uint --packing-interval-cap "$PACK_INTERVAL_CAP" 1 1000000)
# monomial and split budgets are resource bounds (memory grows with them):
# at most 2^18 = MAX_MONOMIAL_BUDGET in src/stages/packing/types.rs
PACK_MONOMIAL_BUDGET=$(bounded_uint --packing-monomial-budget "$PACK_MONOMIAL_BUDGET" 1 262144)
PACK_SPAN_CAP=$(bounded_uint --packing-span-cap "$PACK_SPAN_CAP" 1 1000000000)
PACK_SPLIT_BUDGET=$(bounded_uint --packing-split-budget "$PACK_SPLIT_BUDGET" 1 262144)
PACK_FIX_ITERATIONS=$(bounded_uint --packing-fix-iterations "$PACK_FIX_ITERATIONS" 0 1000)
PACK_FALLBACK_ITERATIONS=$(bounded_uint --packing-fallback-iterations "$PACK_FALLBACK_ITERATIONS" 0 1000)
case "$PACK_CAP_SEM" in cpython39|canonical) ;; *) fatal "--packing-anf-cap-semantics must be cpython39 or canonical" ;; esac
! stage_on 6 || [ "$N" -le 256 ] || fatal "stage 6 (delivery) supports source wires up to 256"
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
if [ "$PREPROCESSING_MODE" = nonlinear291 ] && stage_on 6; then
  # nonlinear_tdp_wire_count(2n, 2|C|+2s+n): 12 per logical wire, 12 per gate, 29 shared
  NL_WIRES=$(( 12 * 2 * N + 12 * (2 * M_CD + 2 * S_SL + N) + 29 ))
  [ "$NL_WIRES" -le 1024 ] ||
    fatal "nonlinear291 makes a $NL_WIRES-wire stage-2 circuit; stage 6 (delivery) supports at most 1024 state wires; stop before stage 6 or use --preprocessing-mode embedded-masking"
fi

# Renamed stage artifacts must not make an existing legacy run look fresh.
for legacy_artifact in gss_command.conf gss.mpmct1; do
  [ ! -e "$RUN/$legacy_artifact" ] ||
    fatal "$RUN contains legacy GSS artifact $legacy_artifact; use a fresh run directory or the checkout that created it"
done
[ ! -e "$RUN/SEED" ] ||
  fatal "$RUN was created by recipe v9 (a one-line SEED file); resume it with the checkout that created it"

BIN=${TDP_BIN_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)/target/release}
need=(gen_sandwich_gadget circuit_mixer)
! stage_on 6 || need+=(tdp_packing)
! unpacking_on || need+=(tdp_unpacking)
for b in "${need[@]}"; do
  [ -x "$BIN/$b" ] || { echo "FATAL: $BIN/$b missing — cargo build --release first" >&2; exit 1; }
done

# ---- seeds (§8): $RUN/SEEDS, format 3, private ----
# Five seeds: gen (stages 1+2), mix (3), split (4), cross (5, only when stage 5
# starts from a circuit file; a resume of split.state continues that RNG) and
# unpack (6: the delivery stage's seed, read by tdp_unpacking run --refine;
# the name is unchanged from the seven-stage driver so its SEEDS files
# resume). A format-2 SEEDS (gen and mix only, written by the four-stage
# driver) is upgraded in place on resume: the three new seeds are drawn fresh.
U63=9223372036854775792
SEED_NAMES=(gen mix split cross unpack)
random_seed() { python3 -I -c "import secrets; print(secrets.randbelow($U63 + 1))"; }
write_seeds() { # FILE SOURCE -> writes GEN_SEED..UNPACK_SEED and RUN_KEY_HEX, mode 600, never printing values
  ( umask 077; set -o noclobber
    printf 'seeds_format=3\ngen_seed=%s\nmix_seed=%s\nsplit_seed=%s\ncross_seed=%s\nunpack_seed=%s\nrun_key=%s\nseed_source=%s\n' \
      "$GEN_SEED" "$MIX_SEED" "$SPLIT_SEED" "$CROSS_SEED" "$UNPACK_SEED" "$RUN_KEY_HEX" "$2" > "$1.tmp" ) || return 1
  mv -f "$1.tmp" "$1"
}
parse_seeds() { # FILE -> sets GEN_SEED MIX_SEED SPLIT_SEED CROSS_SEED UNPACK_SEED RUN_KEY_HEX SEED_SOURCE; never prints values
  local f=$1 lines upgraded=0
  { [ -f "$f" ] && [ "$(wc -c < "$f")" -le 1024 ]; } || fatal "$f is not a valid SEEDS file"
  mapfile -t lines < "$f"
  if [ "${#lines[@]}" -eq 5 ] && [ "${lines[0]}" = seeds_format=2 ]; then
    # format 2 (four-stage driver): gen, mix, run_key, source
    { [[ ${lines[1]} =~ ^gen_seed=([0-9]+)$ ]] && GEN_SEED=${BASH_REMATCH[1]} &&
      [[ ${lines[2]} =~ ^mix_seed=([0-9]+)$ ]] && MIX_SEED=${BASH_REMATCH[1]} &&
      [[ ${lines[3]} =~ ^run_key=([0-9a-f]{64})$ ]] && RUN_KEY_HEX=${BASH_REMATCH[1]} &&
      [[ ${lines[4]} =~ ^seed_source=(random|explicit)$ ]] && SEED_SOURCE=${BASH_REMATCH[1]}; } ||
      fatal "$f is malformed (format 2: seeds_format=2, gen_seed=, mix_seed=, run_key=<64 hex>, seed_source=)"
    # the three new seeds: the explicit seed file's when it names them, else fresh draws
    SPLIT_SEED=${EXPL_SPLIT:-$(random_seed)}; CROSS_SEED=${EXPL_CROSS:-$(random_seed)}; UNPACK_SEED=${EXPL_UNPACK:-$(random_seed)}; upgraded=1
  else
    { [ "${#lines[@]}" -eq 8 ] && [ "${lines[0]}" = seeds_format=3 ] &&
      [[ ${lines[1]} =~ ^gen_seed=([0-9]+)$ ]] && GEN_SEED=${BASH_REMATCH[1]} &&
      [[ ${lines[2]} =~ ^mix_seed=([0-9]+)$ ]] && MIX_SEED=${BASH_REMATCH[1]} &&
      [[ ${lines[3]} =~ ^split_seed=([0-9]+)$ ]] && SPLIT_SEED=${BASH_REMATCH[1]} &&
      [[ ${lines[4]} =~ ^cross_seed=([0-9]+)$ ]] && CROSS_SEED=${BASH_REMATCH[1]} &&
      [[ ${lines[5]} =~ ^unpack_seed=([0-9]+)$ ]] && UNPACK_SEED=${BASH_REMATCH[1]} &&
      [[ ${lines[6]} =~ ^run_key=([0-9a-f]{64})$ ]] && RUN_KEY_HEX=${BASH_REMATCH[1]} &&
      [[ ${lines[7]} =~ ^seed_source=(random|explicit)$ ]] && SEED_SOURCE=${BASH_REMATCH[1]}; } ||
      fatal "$f is malformed (expected seeds_format=3, gen_seed=, mix_seed=, split_seed=, cross_seed=, unpack_seed=, run_key=<64 hex>, seed_source=)"
  fi
  local name var t
  for name in "${SEED_NAMES[@]}"; do
    var=${name^^}_SEED
    t=$(bounded_uint "${name}_seed" "${!var}" 0 "$U63" 2>/dev/null) || fatal "$f: ${name}_seed out of range"
    printf -v "$var" '%s' "$t"
  done
  if [ "$upgraded" = 1 ]; then
    write_seeds "$f" "$SEED_SOURCE" || fatal "cannot upgrade $f to seeds format 3"
    SEEDS_UPGRADED=1
  fi
}
EXPL_GEN=""; EXPL_MIX=""; EXPL_SPLIT=""; EXPL_CROSS=""; EXPL_UNPACK=""
read_seed_file() { # PATH -> sets EXPL_GEN EXPL_MIX EXPL_SPLIT EXPL_CROSS EXPL_UNPACK (the last three may stay empty: drawn fresh)
  local f=$1 lines l mode name
  [ -f "$f" ] || fatal "--seed-file $f is not a regular file"
  [ "$(wc -c < "$f")" -le 256 ] || fatal "--seed-file $f is too large (maximum 256 bytes)"
  mode=$(python3 -I -c 'import os,sys; print(oct(os.stat(sys.argv[1]).st_mode & 0o777)[2:])' "$f")
  [ $((8#$mode & 8#077)) -eq 0 ] || fatal "--seed-file $f must not be accessible by group/other (chmod 600)"
  mapfile -t lines < "$f"
  if [ "${#lines[@]}" -eq 1 ] && [[ ${lines[0]} =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]]; then
    fatal "seed file format 1 holds one seed; recipe v12 needs gen_seed= and mix_seed= lines (the old seed was the gen seed; the old mix seed was gen+2)"
  fi
  EXPL_GEN=""; EXPL_MIX=""; EXPL_SPLIT=""; EXPL_CROSS=""; EXPL_UNPACK=""
  for l in "${lines[@]}"; do
    if [[ $l =~ ^gen_seed=([0-9]+)$ ]] && [ -z "$EXPL_GEN" ]; then EXPL_GEN=${BASH_REMATCH[1]}
    elif [[ $l =~ ^mix_seed=([0-9]+)$ ]] && [ -z "$EXPL_MIX" ]; then EXPL_MIX=${BASH_REMATCH[1]}
    elif [[ $l =~ ^split_seed=([0-9]+)$ ]] && [ -z "$EXPL_SPLIT" ]; then EXPL_SPLIT=${BASH_REMATCH[1]}
    elif [[ $l =~ ^cross_seed=([0-9]+)$ ]] && [ -z "$EXPL_CROSS" ]; then EXPL_CROSS=${BASH_REMATCH[1]}
    elif [[ $l =~ ^unpack_seed=([0-9]+)$ ]] && [ -z "$EXPL_UNPACK" ]; then EXPL_UNPACK=${BASH_REMATCH[1]}
    elif [ -n "$l" ]; then fatal "--seed-file $f: unexpected line (only gen_seed=, mix_seed=, split_seed=, cross_seed=, unpack_seed=)"
    fi
  done
  { [ -n "$EXPL_GEN" ] && [ -n "$EXPL_MIX" ]; } || fatal "--seed-file $f needs both gen_seed= and mix_seed="
  local v t
  for name in GEN MIX SPLIT CROSS UNPACK; do
    v=EXPL_$name
    [ -n "${!v}" ] || continue
    t=$(bounded_uint "${name,,}_seed" "${!v}" 0 "$U63" 2>/dev/null) || fatal "--seed-file: ${name,,}_seed out of range 0..$U63"
    printf -v "$v" '%s' "$t"
  done
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
SEEDS_UPGRADED=0
if [ -e "$SEEDS" ]; then
  parse_seeds "$SEEDS"
  if [ -n "$SEED_FILE" ]; then
    conflict=0
    [ "$EXPL_GEN" = "$GEN_SEED" ] || conflict=1
    [ "$EXPL_MIX" = "$MIX_SEED" ] || conflict=1
    [ -z "$EXPL_SPLIT" ] || [ "$EXPL_SPLIT" = "$SPLIT_SEED" ] || conflict=1
    [ -z "$EXPL_CROSS" ] || [ "$EXPL_CROSS" = "$CROSS_SEED" ] || conflict=1
    [ -z "$EXPL_UNPACK" ] || [ "$EXPL_UNPACK" = "$UNPACK_SEED" ] || conflict=1
    [ "$conflict" = 0 ] || fatal "--seed-file conflicts with the existing $SEEDS; use the original seeds or a fresh run directory"
  fi
  SEED_SRC="RESUMED from $SEEDS ($SEED_SOURCE)"
  [ "$SEEDS_UPGRADED" = 0 ] || SEED_SRC="$SEED_SRC, upgraded from format 2: split, cross and unpack seeds drawn fresh"
else
  [ ! -s "$GADGET" ] || fatal "existing stage-2 artifact has no SEEDS; use a fresh run directory"
  RUN_KEY_HEX=$(python3 -I -c 'import secrets; print(secrets.token_hex(32))')
  if [ -n "$SEED_FILE" ]; then
    GEN_SEED=$EXPL_GEN; MIX_SEED=$EXPL_MIX; SEED_SOURCE=explicit
    SPLIT_SEED=${EXPL_SPLIT:-$(random_seed)}; CROSS_SEED=${EXPL_CROSS:-$(random_seed)}; UNPACK_SEED=${EXPL_UNPACK:-$(random_seed)}
  else
    GEN_SEED=$(random_seed); MIX_SEED=$(random_seed); SPLIT_SEED=$(random_seed); CROSS_SEED=$(random_seed); UNPACK_SEED=$(random_seed)
    SEED_SOURCE=random
  fi
  write_seeds "$SEEDS" "$SEED_SOURCE" || fatal "cannot create $SEEDS"
  unset RUN_KEY_HEX
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
for v in ${ignored[@]+"${ignored[@]}"}; do note "inherited $v ignored: cache sizes are set by the driver (exact memo caches; stage-3 bytes matched the reference runs under the built-in sizes)"; done

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
# The circuit each stage reads: the output of the last stage that ran (stage 2
# skipped: the sandwich S itself). STAGE2_INPUT is what stage 6 hides.
CUR=$GADGET; STAGE2_INPUT=$GADGET
if ! stage_on 2; then
  CUR=$GADGET.sandwich.mpmct1; STAGE2_INPUT=$CUR
  [ -s "$CUR" ] || fatal "stage 2 skipped but $CUR (the sandwich) is missing"
  note "stage 2: SKIPPED (the sandwich $CUR goes on)"
fi
RAN="1"; ! stage_on 2 || RAN="1,2"
ran_text() { # "1,2,3,5,6" -> "1-3,5-6"
  python3 -I -c '
import sys
xs = [int(x) for x in sys.argv[1].split(",")]
out, i = [], 0
while i < len(xs):
    j = i
    while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
        j += 1
    out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
    i = j + 1
print(",".join(out))' "$RAN"
}
stop_here() { # K: stop after stage K when asked
  if [ "$STOP_AFTER" -le "$1" ]; then
    wait_side; note "PIPELINE STOPPED after stage $1 (stages $(ran_text)): $CUR"; exit 0
  fi
}
stop_here 2

# Stage completion markers for stages 3-5 (stage 6 writes delivery.recipe):
# the recipe text, then the keyed tag of the output, then extra lines. A resume
# skips the stage only when every line of the recipe text and the output's tag
# still match, so an interrupted or partially written output, or a changed input,
# parameter or binary, re-runs the stage.
marker_ok() { # MARKER EXPECTED_TEXT ARTIFACT MACKEY
  local marker=$1 expected=$2 artifact=$3 key=$4 nexp
  [ -s "$marker" ] && [ -s "$artifact" ] || return 1
  nexp=$(printf '%s\n' "$expected" | wc -l | tr -d ' ')
  [ "$(head -n "$nexp" "$marker")" = "$expected" ] &&
    [ "$(sed -n "$((nexp + 1))p" "$marker")" = "$(mac "$key=$artifact")" ]
}
write_marker() { # MARKER EXPECTED_TEXT ARTIFACT MACKEY EXTRA_LINES...
  local marker=$1 expected=$2 artifact=$3 key=$4; shift 4
  { printf '%s\n' "$expected"; mac "$key=$artifact"; [ $# -eq 0 ] || printf '%s\n' "$@"; } > "$marker.tmp"
  mv "$marker.tmp" "$marker"
}
# The mixer's chain-done reason: "[circuit_mixer] chain done in Ts: <reason> (...)".
stop_reason_of() { sed -n 's/^\[circuit_mixer\] chain done in [0-9.]*s: \(.*\) ([0-9]* -> [0-9]* gates, .*$/\1/p' "$1" | tail -1; }
MIXER_CACHE_ENV=(FROZEN_REGULAR_VALUE_CONVENTION=native LOOKUP_CACHE_MB=2048 CANON_CACHE_MB=256 XPOLY_CANON_CACHE_MB=1024)

# ---- stage 3: circuit_mixer db_mixing ----
if stage_on 3; then
G_IN=$(gates_of "$CUR")
if [ "$SCHEDULE" = validated ]; then
  A_MOVES=$(python3 -I -c "import sys; G=int(sys.argv[1]); print(int(G*2*2*1.3))" "$G_IN")
else
  A_MOVES=$(python3 -I -c "import sys; n0,h,r=map(float,sys.argv[1:4]); G=int(sys.argv[4]); print(int((n0+h)*r*G*1.3))" "$GROWTH" "$HOLD" "$EXPAND" "$G_IN")
fi
QCFLAGS=()
if [ "$QC" = 1 ]; then
  QCFLAGS=(--leakage-repair --leakage-repair-seed "$QC_SEED" --leakage-repair-report "$RUN/db_mixing.leakage_repair.txt")
  [ -z "$QC_REF" ] || QCFLAGS+=(--leakage-repair-reference "$QC_REF")
fi
# Stage-3 completion marker: written only after the mixer finished with an
# accepted stop reason, keyed to the stage-3 input, the schedule and the output
# bytes. The store is deliberately not part of it: a resume after stage 3 needs
# no store.
STAGE3_MARKER=$RUN/stage3.recipe
expected_stage3_marker() {
  echo "tdp_stage3_recipe=1"
  echo "stage3_args=--profile $PROFILE --moves $A_MOVES${PIECEFLAGS[*]:+ ${PIECEFLAGS[*]}}${QCFLAGS[*]:+ ${QCFLAGS[*]}}"
  mac stage2="$CUR"
}
stage3_done=0
if [ "$FORCE_FROM" -gt 3 ] && [ -s "$DB_MIXING" ]; then
  if [ -s "$STAGE3_MARKER" ]; then
    exp3=$(expected_stage3_marker)
    nexp3=$(printf '%s\n' "$exp3" | wc -l | tr -d ' ')
    if [ "$(head -n "$nexp3" "$STAGE3_MARKER")" = "$exp3" ] &&
       [ "$(sed -n "$((nexp3 + 1))p" "$STAGE3_MARKER")" = "$(mac db_mixing="$DB_MIXING")" ]; then
      stage3_done=1
    fi
  fi
  # A run from before this marker: adopt its output only when stage 6 already
  # consumed exactly these bytes (the delivery marker's keyed tags of both
  # inputs; packing.recipe for a run of the seven-stage driver) and
  # stage3.settings records an accepted stop.
  if [ "$stage3_done" -eq 0 ] && [ ! -e "$STAGE3_MARKER" ] &&
     grep -Eqx 'stop_reason=(moves budget spent|profile complete \(size schedule finished\))' "$RUN/stage3.settings" 2>/dev/null; then
    for vouching in delivery.recipe packing.recipe; do
      [ -s "$RUN/$vouching" ] && grep -qxF "$(mac stage2="$CUR")" "$RUN/$vouching" &&
        grep -qxF "$(mac stage3="$DB_MIXING")" "$RUN/$vouching" || continue
      { expected_stage3_marker; mac db_mixing="$DB_MIXING"; sed -n 's/^stop_reason=/stop_reason=/p' "$RUN/stage3.settings"; echo "adopted=$vouching"; } > "$STAGE3_MARKER.tmp"
      mv "$STAGE3_MARKER.tmp" "$STAGE3_MARKER"
      stage3_done=1
      note "stage 3: adopted $DB_MIXING (no stage-3 marker; stage 6's completion marker $vouching covers these exact bytes)"
      break
    done
  fi
  [ "$stage3_done" -eq 1 ] ||
    note "stage 3: $DB_MIXING has no matching completion marker ($STAGE3_MARKER missing or stale: an interrupted or partial output, or a changed stage-2 input or schedule); re-running stage 3"
fi
if [ "$stage3_done" -eq 0 ]; then
  if [ -z "${FROZEN_DB_DIR:-}" ] && [ "${TDP_GEN_ALLOW_EMPTY_STORE:-0}" != "1" ]; then
    echo "FATAL: stage 3 needs FROZEN_DB_DIR (or TDP_GEN_ALLOW_EMPTY_STORE=1 for a plumbing test)" >&2; exit 1
  fi
  rm -f "$STAGE3_MARKER" "$DB_MIXING.tmp" "$RUN/stage3.settings"; FF=""
  RECIPE_ENV=("${MIXER_CACHE_ENV[@]}" MIXER_STOP_FLAG="$RUN/stage3.stop" MIXER_DUMP_FLAG="$RUN/stage3.dump")
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
  [ "$QC" != 1 ] || note "stage 3: leakage repair enabled; report $RUN/db_mixing.leakage_repair.txt"
  note "stage 3: circuit_mixer --tdp --db-mixing --profile $PROFILE --moves $A_MOVES (schedule $SCHEDULE, $PIECE_DESCRIPTION)$FILTER_NOTE"
  # val_chain.sh's argv, in its order (the seed as --seed-file /dev/fd/3, off argv); optional flags only when enabled.
  stage_exec "$BIN/circuit_mixer" --input "$CUR" --output "$DB_MIXING" --tdp --db-mixing \
      --profile "$PROFILE" --moves "$A_MOVES" "${DBFLAGS[@]}" \
      --db-max-degree 9 --db-max-span 30 --db-wire-terms 1024 --db-total-terms 2048 \
      --no-local-verify --seed-file /dev/fd/3 \
      ${QCFLAGS[@]+"${QCFLAGS[@]}"} ${PIECEFLAGS[@]+"${PIECEFLAGS[@]}"} 3<<< "$MIX_SEED" > "$RUN/stage3.log" 2>&1 ||
    { note "stage 3 FAILED; see $RUN/stage3.log"; exit 1; }
  # Why the chain ended. An operator stop flag ($RUN/stage3.stop) ends it
  # "cleanly" with exit 0 but below the recipe's dose: that output is never
  # handed on under the canonical name, so neither this run nor a resume ships it.
  STOP_REASON=$(stop_reason_of "$RUN/stage3.log")
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
  { expected_stage3_marker; mac db_mixing="$DB_MIXING"; echo "stop_reason=$STOP_REASON"; } > "$STAGE3_MARKER.tmp"
  mv "$STAGE3_MARKER.tmp" "$STAGE3_MARKER"
  [ "$FORCE_FROM" -le 4 ] || FORCE_FROM=4
  note "stage 3 done: $(gates_of "$DB_MIXING") gates"
else
  note "stage 3: $DB_MIXING matches its completion marker, skipping"
fi
CUR=$DB_MIXING; RAN="$RAN,3"
else
  note "stage 3: SKIPPED"
fi
stop_here 3

# A mixer stage (4 or 5) run by the driver: STAGE LOG OUTPUT ACCEPTED_REASON_REGEX ARGS...
# Runs circuit_mixer under the stage-3 environment (no store: --p-db 0), reads
# the chain-done reason, and refuses a stop-flag ending like stage 3 does.
run_mixer_stage() {
  local stage=$1 log=$2 output=$3 accepted=$4; shift 4
  RECIPE_ENV=("${MIXER_CACHE_ENV[@]}" MIXER_STOP_FLAG="$RUN/stage$stage.stop" MIXER_DUMP_FLAG="$RUN/stage$stage.dump")
  rm -f "$RUN/stage$stage.stop" "$output.tmp"
  stage_exec "$BIN/circuit_mixer" "$@" > "$log" 2>&1 || { note "stage $stage FAILED; see $log"; exit 1; }
  STOP_REASON=$(stop_reason_of "$log")
  if [[ $STOP_REASON =~ ^($accepted)$ ]]; then
    return 0
  fi
  case "$STOP_REASON" in
    "stop flag"|"")
      mv -f "$output" "${output%.mpmct1}.stopped.mpmct1"
      fatal "stage $stage ended by ${STOP_REASON:-an unrecognised stop (no chain-done line in $log)}, below the recipe's dose: kept as ${output%.mpmct1}.stopped.mpmct1 (NOT a stage-$stage artifact); a resume re-runs stage $stage" ;;
    *) note "WARNING: stage $stage ended by \"$STOP_REASON\"; see $log" ;;
  esac
}

# ---- stage 4: circuit_mixer --split (the archive-era stage 4, verbatim flags) ----
SPLIT=$RUN/split.mpmct1; SPLIT_STATE=$RUN/split.state; STAGE4_MARKER=$RUN/stage4.recipe
SPLIT_RAN=0
if stage_on 4; then
  G_IN4=$(gates_of "$CUR"); B_MOVES=$((G_IN4 + 1000000))   # one move per split twist; comp count < G
  SPLIT_ARGS=(--split --split-stop --p-join 0.8 --split-reach-k 2 --split-fail-limit 100 --split-canaries 256
              --p-db 0 --p-comp 0 --p-any 0 --p-twist 0 --k-max 12 --moves "$B_MOVES" --report-every 1000000)
  expected_stage4_marker() {
    echo "tdp_stage4_recipe=1"
    echo "circuit_mixer_sha256=$(sha_of "$BIN/circuit_mixer")"
    echo "stage4_args=${SPLIT_ARGS[*]}"
    mac --seed split
    mac stage4_input="$CUR"
  }
  stage4_done=0
  if [ "$FORCE_FROM" -gt 4 ] && marker_ok "$STAGE4_MARKER" "$(expected_stage4_marker)" "$SPLIT" split; then
    stage4_done=1
    note "stage 4: $SPLIT matches its completion marker, skipping"
  elif [ "$FORCE_FROM" -gt 4 ] && [ -s "$SPLIT" ]; then
    note "stage 4: $SPLIT has no matching completion marker (interrupted output, or a changed input, parameter or binary); re-running stage 4"
  fi
  if [ "$stage4_done" -eq 0 ]; then
    rm -f "$STAGE4_MARKER" "$SPLIT_STATE"
    note "stage 4: circuit_mixer --split --split-stop (moves ceiling $B_MOVES) on $(basename "$CUR")"
    run_mixer_stage 4 "$RUN/stage4.log" "$SPLIT" 'split stage complete \(stopped at the stage boundary\)|moves budget spent' \
      --input "$CUR" "${SPLIT_ARGS[@]}" --seed-file /dev/fd/3 --state-out "$SPLIT_STATE" --output "$SPLIT" 3<<< "$SPLIT_SEED"
    write_marker "$STAGE4_MARKER" "$(expected_stage4_marker)" "$SPLIT" split "stop_reason=$STOP_REASON"
    [ "$FORCE_FROM" -le 5 ] || FORCE_FROM=5
    note "stage 4 done: $(gates_of "$SPLIT") gates"
  fi
  CUR=$SPLIT; RAN="$RAN,4"; SPLIT_RAN=1
else
  note "stage 4: SKIPPED"
fi
stop_here 4

# ---- stage 5: the crossing walk (archive-era stage 5; resumes split.state when stage 4 ran) ----
CROSSING=$RUN/crossing.mpmct1; STAGE5_MARKER=$RUN/stage5.recipe
if stage_on 5; then
  G_IN5=$(gates_of "$CUR")
  read -r X_TGT X_TEMP X_BUDGET <<< "$(python3 -I - "$G_IN5" "$XR" "$XTDIV" "$XMF" "${XMOVES:-0}" <<'PY'
import sys
g, xr, tdiv, xmf, xmoves = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
tgt = round(g * xr)
temp = max(64, round(tgt / tdiv))
# The walk reaches its target in about 2*target moves (RC, 2026-10-11: the
# default budget is 2*target; the archive-era driver's 6*target also held the
# fragments under the thermostat for a while, eroding them).
print(tgt, temp, xmoves if xmoves > 0 else max(1, round(xmf * tgt)))
PY
)"
  X_ARGS=(--target-size "$X_TGT" --temp "$X_TEMP" --split-base "$XB" --split-damp "$XC"
          --p-twist 0 --p-db 0 --p-comp 0 --p-any 0 --k-max 12 --report-every 500000)
  if [ "$SPLIT_RAN" = 1 ] && [ -s "$SPLIT_STATE" ]; then
    # --moves is ABSOLUTE on a resume: the split stage's counter plus the budget.
    X_SOURCE="resume split.state"
    state_moves=$(awk '$1=="moves"{print $2; exit}' "$SPLIT_STATE")
    [[ $state_moves =~ ^[0-9]+$ ]] || fatal "$SPLIT_STATE has no moves counter (truncated state); rerun with --run-rerun-from-stage 4"
    X_MOVES=$(( state_moves + X_BUDGET ))
    X_INPUT=(--resume "$SPLIT_STATE"); X_SEED=()
  else
    [ "$SPLIT_RAN" = 0 ] || fatal "stage 4 ran but $SPLIT_STATE is missing; rerun with --run-rerun-from-stage 4"
    X_SOURCE="input $(basename "$CUR")"
    X_MOVES=$X_BUDGET
    X_INPUT=(--input "$CUR"); X_SEED=(--seed-file /dev/fd/3)
  fi
  expected_stage5_marker() {
    echo "tdp_stage5_recipe=1"
    echo "circuit_mixer_sha256=$(sha_of "$BIN/circuit_mixer")"
    echo "stage5_args=${X_ARGS[*]} --moves $X_MOVES"
    echo "stage5_source=$X_SOURCE"
    if [ "$SPLIT_RAN" = 1 ]; then mac split_state="$SPLIT_STATE"; else mac --seed cross; fi
    mac stage5_input="$CUR"
  }
  stage5_done=0
  if [ "$FORCE_FROM" -gt 5 ] && marker_ok "$STAGE5_MARKER" "$(expected_stage5_marker)" "$CROSSING" crossing; then
    stage5_done=1
    note "stage 5: $CROSSING matches its completion marker, skipping"
  elif [ "$FORCE_FROM" -gt 5 ] && [ -s "$CROSSING" ]; then
    note "stage 5: $CROSSING has no matching completion marker (interrupted output, or a changed input, parameter or binary); re-running stage 5"
  fi
  if [ "$stage5_done" -eq 0 ]; then
    rm -f "$STAGE5_MARKER"
    note "stage 5: crossing walk ($X_SOURCE) target=$X_TGT temp=$X_TEMP b=$XB c=$XC budget=$X_BUDGET ($([ -n "$XMOVES" ] && echo absolute || echo "${XMF}*target")) moves=$X_MOVES"
    run_mixer_stage 5 "$RUN/stage5.log" "$CROSSING" 'moves budget spent' \
      "${X_INPUT[@]}" "${X_ARGS[@]}" --moves "$X_MOVES" ${X_SEED[@]+"${X_SEED[@]}"} --output "$CROSSING" 3<<< "$CROSS_SEED"
    write_marker "$STAGE5_MARKER" "$(expected_stage5_marker)" "$CROSSING" crossing "stop_reason=$STOP_REASON"
    [ "$FORCE_FROM" -le 6 ] || FORCE_FROM=6
    note "stage 5 done: $(gates_of "$CROSSING") gates"
  fi
  CUR=$CROSSING; RAN="$RAN,5"
else
  note "stage 5: SKIPPED"
fi
stop_here 5

# ---- stage 6: delivery = tdp_packing run (windowed ANF delivery), then
# tdp_unpacking run --refine (resynthesis with lattice refinement) ----
# One stage, one completion marker (delivery.recipe), one log (stage6.log,
# the two sub-steps appended with a separator), one failure marker
# (delivery.failed: exit=<rc> and step=packing|unpacking). The packer's
# outputs stay in packing/ (rep<G>.* is what the parity tooling reads), the
# refine sub-step's in unpacking/: deliverable.tdp4 (the refined lattice),
# bounds.final, unpacked.mpmct1, windows.tsv, rounds.tsv, unpacking.report.
PACK_DIR=$RUN/packing; UNPACK_DIR=$RUN/unpacking; MARKER=$RUN/delivery.recipe; FAILED=$RUN/delivery.failed
# the repaired circuit the packer writes, named by its window size (rep<G>.*): the lattice the refinement starts from
PACKED=$PACK_DIR/rep$PACK_WINDOW_GATES.mpmct1; PACKED_BOUNDS=$PACK_DIR/rep$PACK_WINDOW_GATES.bounds
packing_exit_meaning() {
  case "$1" in
    2) echo "usage or parameter error" ;; 3) echo "input error" ;;
    4) echo "verification failure" ;; 5) echo "exposure floor missed" ;;
    6) echo "over budget" ;; 7) echo "I/O error writing outputs" ;;
    8) echo "internal invariant violated" ;; 101) echo "panic" ;; *) echo "unknown exit code" ;;
  esac
}
unpacking_exit_meaning() {
  case "$1" in
    2) echo "usage or parameter error" ;; 3) echo "input error" ;;
    4) echo "verification or synthesis failure" ;; 7) echo "I/O error writing outputs" ;;
    8) echo "internal invariant violated" ;; 101) echo "panic" ;; *) echo "unknown exit code" ;;
  esac
}
# The stage's final output: the packer's deliverable under --delivery-anf-only,
# else the resynthesized circuit (next to the refined deliverable.tdp4).
if [ "$DELIV_ANF_ONLY" = 1 ]; then FINAL_DIR=$PACK_DIR; FINAL=$PACK_DIR/deliverable.tdp4
else FINAL_DIR=$UNPACK_DIR; FINAL=$UNPACK_DIR/unpacked.mpmct1; fi
if stage_on 6; then
expected_marker() {
  echo "delivery_recipe=1"
  echo "tdp_packing_sha256=$(sha_of "$BIN/tdp_packing")"
  echo "packing_params=$(IFS=' '; echo "${PACKFLAGS[*]:-default}")"
  echo "anf_only=$DELIV_ANF_ONLY"
  if [ "$DELIV_ANF_ONLY" = 0 ]; then
    echo "tdp_unpacking_sha256=$(sha_of "$BIN/tdp_unpacking")"
    echo "delivery_params=${DELIVFLAGS[*]}"
    mac --seed unpack
  fi
  mac stage2="$STAGE2_INPUT" stage3="$CUR" source_c="$GADGET.source_c.g57"
}
# The outputs tagged after the recipe text: the packer's deliverable, and
# without anf_only the refined deliverable and the resynthesized circuit.
DELIV_OUTPUTS=(deliverable_tdp4="$PACK_DIR/deliverable.tdp4")
[ "$DELIV_ANF_ONLY" = 1 ] || DELIV_OUTPUTS+=(refined_tdp4="$UNPACK_DIR/deliverable.tdp4" unpacked="$UNPACK_DIR/unpacked.mpmct1")
stage6_done=0
if [ "$FORCE_FROM" -gt 6 ] && [ -s "$MARKER" ]; then
  exp=$(expected_marker)
  nexp=$(printf '%s\n' "$exp" | wc -l | tr -d ' ')
  outputs_present=1
  for o in "${DELIV_OUTPUTS[@]}"; do [ -s "${o#*=}" ] || outputs_present=0; done
  if [ "$outputs_present" = 1 ] && [ "$(head -n "$nexp" "$MARKER")" = "$exp" ] &&
     [ "$(sed -n "$((nexp + 1)),$((nexp + ${#DELIV_OUTPUTS[@]}))p" "$MARKER")" = "$(mac "${DELIV_OUTPUTS[@]}")" ]; then
    stage6_done=1
    note "stage 6: $FINAL matches its completion marker, skipping"
  else
    note "stage 6: completion marker does not match inputs/parameters/binaries/outputs; re-running stage 6"
  fi
fi
if [ "$stage6_done" -eq 0 ]; then
  rm -f "$MARKER" "$FAILED" "$RUN/stage6.log"; rm -rf "$PACK_DIR" "$UNPACK_DIR"
  THREADFLAGS=(); [ -z "$THREADS" ] || THREADFLAGS=(--threads "$THREADS")
  note "stage 6 (delivery) packing: tdp_packing run on $(basename "$CUR") (window gates $PACK_WINDOW_GATES, split budget $PACK_SPLIT_BUDGET, caps $PACK_CAP_SEM${PACKFLAGS[*]:+; flags ${PACKFLAGS[*]}})"
  RECIPE_ENV=()
  set +e
  stage_exec "$BIN/tdp_packing" run --stage2 "$STAGE2_INPUT" --stage3 "$CUR" --source-c "$GADGET.source_c.g57" \
      --n "$N" --out-dir "$PACK_DIR" ${PACKFLAGS[@]+"${PACKFLAGS[@]}"} ${THREADFLAGS[@]+"${THREADFLAGS[@]}"} \
      > "$RUN/stage6.log" 2>&1
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    # the managed driver attaches the binary's exit meaning only when this
    # marker names the sub-step and its exit code (a 2 from gen, the mixer or
    # bash is not stage 6's)
    printf 'exit=%s\nstep=packing\n' "$rc" > "$FAILED"
    wait_side
    note "stage 6 FAILED: packing exit $rc ($(packing_exit_meaning "$rc")); see $RUN/stage6.log"
    exit "$rc"
  fi
  pstatus=$(sed -n 's/^status=//p' "$PACK_DIR/packing.report")
  ustatus=skipped
  if [ "$DELIV_ANF_ONLY" = 0 ]; then
    for f in "$PACK_DIR/deliverable.tdp4" "$PACKED" "$PACKED_BOUNDS"; do
      [ -s "$f" ] || fatal "the packer did not write $f (needed by the unpacking sub-step); see $RUN/stage6.log"
    done
    note "stage 6 (delivery) unpacking: tdp_unpacking run ${DELIVFLAGS[*]} (time limit $DELIV_TIME_LIMIT s/window/round, restarts $DELIV_RESTARTS, beam $DELIV_BEAM, max rounds $DELIV_MAX_ROUNDS)"
    printf -- '---- unpacking: tdp_unpacking run %s ----\n' "${DELIVFLAGS[*]}" >> "$RUN/stage6.log"
    set +e
    stage_exec "$BIN/tdp_unpacking" run --deliverable "$PACK_DIR/deliverable.tdp4" \
        --packed "$PACKED" --bounds "$PACKED_BOUNDS" \
        --source-c "$GADGET.source_c.g57" --n "$N" --out-dir "$UNPACK_DIR" --seed-file /dev/fd/3 \
        "${DELIVFLAGS[@]}" ${THREADFLAGS[@]+"${THREADFLAGS[@]}"} 3<<< "$UNPACK_SEED" \
        >> "$RUN/stage6.log" 2>&1
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
      printf 'exit=%s\nstep=unpacking\n' "$rc" > "$FAILED"
      wait_side
      note "stage 6 FAILED: unpacking exit $rc ($(unpacking_exit_meaning "$rc")); see $RUN/stage6.log"
      exit "$rc"
    fi
    for f in "$UNPACK_DIR/unpacking.report" "$UNPACK_DIR/deliverable.tdp4" "$UNPACK_DIR/unpacked.mpmct1"; do
      [ -s "$f" ] || fatal "tdp_unpacking --refine exited 0 but did not write $f (docs/DELIVERY_STAGE.md §2); see $RUN/stage6.log"
    done
    ustatus=$(sed -n 's/^status=//p' "$UNPACK_DIR/unpacking.report")
    [ -n "$ustatus" ] || fatal "$UNPACK_DIR/unpacking.report has no status= line; see $RUN/stage6.log"
  fi
  # status: the sub-step statuses other than ok, joined by +; ok when both are
  bad=(); [ "$pstatus" = ok ] || bad+=("$pstatus")
  [ "$ustatus" = ok ] || [ "$ustatus" = skipped ] || bad+=("$ustatus")
  status=ok; [ "${#bad[@]}" -eq 0 ] || status=$(IFS=+; echo "${bad[*]}")
  { expected_marker; mac "${DELIV_OUTPUTS[@]}"
    printf 'status=%s\npacking_status=%s\nunpacking_status=%s\n' "$status" "$pstatus" "$ustatus"; } > "$MARKER.tmp"
  mv "$MARKER.tmp" "$MARKER"
  if [ "$DELIV_ANF_ONLY" = 1 ]; then
    note "stage 6 done (anf_only): $FINAL"
  else
    note "stage 6 done: $UNPACK_DIR/deliverable.tdp4 (refined lattice, $(sed -n 's/^windows_final=//p' "$UNPACK_DIR/unpacking.report") windows after $(sed -n 's/^rounds=//p' "$UNPACK_DIR/unpacking.report") rounds); $FINAL $(gates_of "$FINAL") gates"
  fi
fi
status=$(sed -n 's/^status=//p' "$MARKER")
rm -f "$PACK_DIR/NOT_RECOMMENDED" "$UNPACK_DIR/NOT_RECOMMENDED"
if [ "$status" != ok ]; then
  # an allowed packer failure (floor_miss, over_budget) or a window emitted
  # verbatim by the refinement (verbatim_used): reported, not fatal
  grep -E '^(status|packing_status|unpacking_status)=' "$MARKER" > "$FINAL_DIR/NOT_RECOMMENDED"
  note "stage 6 FINISHED WITH AN ALLOWED FAILURE (status=$status): $FINAL is NOT a recommended deliverable"
fi
CUR=$FINAL; RAN="$RAN,6"
else
  note "stage 6: SKIPPED"
fi
wait_side
if stage_on 6 && [ -e "$FINAL_DIR/NOT_RECOMMENDED" ]; then
  note "PIPELINE FINISHED WITH AN ALLOWED FAILURE (stages $(ran_text); status=$(sed -n 's/^status=//p' "$MARKER")): $CUR is NOT a recommended deliverable"
else
  note "PIPELINE COMPLETE (stages $(ran_text)): $CUR"
fi
