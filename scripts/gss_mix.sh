#!/bin/bash
# GSS-MIX — the end-to-end mixing pipeline for GSS (gadgetized sliced
# sandwich) circuits. Manual: docs/GSS_MIX.md.
#
# Stages:
#   1+2  generate the sliced sandwich S and Gray-fold gadgetize it
#        (gen_sandwich_gadget; |C| = |D| = round(n·(log2 n)^2), the library
#        convention; the production preset — mask plan [2,2,2,3] + Gray fold —
#        is the tool default)
#   3    fmix phase A: --gss --phase-a --profile 3,(3+HOLD),(+20),R1,R2
#        with R2 = 1 + (R1-1)/2
#   4    fmix phase B part 1: the split stage, current defaults, to exhaustion
#   5    fmix phase B part 2: the crossing walk (resumes the split state;
#        defaults are PROVISIONAL until the X-panel pins them)
#   6    fcompress
#
# Every stage writes its artifact + log into the run dir and is skipped when
# the artifact already exists (rerun with --force-from K). bash only — do not
# port to zsh (word-splitting changes silently).
set -euo pipefail

usage() {
  cat <<'EOF'
usage: gss_mix.sh -n N -o RUNDIR [options]
  -n N           wires of the source computation C (required)
  -o DIR         run directory (created; all artifacts + logs land here)
  -s SEED        master seed (default 1; stages derive their own from it)
  --mcd M        override |C| = |D| gate count            [0 = round(n(log2 n)^2)]
  --expand R     phase-A max expansion factor R1          [2]
  --hold E       phase-A hold duration in effs            [30]
  --xr R         stage-5 crossing target factor           [2; 2.5 = max-spread point]
  --xb B         stage-5 width-damper base                [3   — X-panel calibrated]
  --xc C         stage-5 width-damper threshold           [1   — X-panel calibrated]
  --xtdiv D      stage-5 temperature = target/D           [25]
  --xmoves M     stage-5 move budget (default 6 x target: STOP AT ARRIVAL —
                 median spread peaks there and the hold erodes it)
  --stop-after K stop after stage K (2, 3, 4, 5 or 6)
  --force-from K rebuild from stage K even if artifacts exist
env:
  FROZEN_DB_DIR       required for stage 3 (the frozen replacement store)
  FROZEN_CURATED_DIR  recommended for stage 3 (curated-first cascade)
  GSS_MIX_ALLOW_EMPTY_STORE=1  testing only: run stage 3 with no store
EOF
  exit 1
}

N=""; RUN=""; SEED=1; EXPAND=2; HOLD=30; MCD=0
XR=2; XB=3; XC=1; XTDIV=25; XMOVES=""
STOP_AFTER=6; FORCE_FROM=99
while [ $# -gt 0 ]; do
  case "$1" in
    -n) N=$2; shift 2 ;;
    -o) RUN=$2; shift 2 ;;
    -s) SEED=$2; shift 2 ;;
    --mcd) MCD=$2; shift 2 ;;
    --expand) EXPAND=$2; shift 2 ;;
    --hold) HOLD=$2; shift 2 ;;
    --xr) XR=$2; shift 2 ;;
    --xb) XB=$2; shift 2 ;;
    --xc) XC=$2; shift 2 ;;
    --xtdiv) XTDIV=$2; shift 2 ;;
    --xmoves) XMOVES=$2; shift 2 ;;
    --stop-after) STOP_AFTER=$2; shift 2 ;;
    --force-from) FORCE_FROM=$2; shift 2 ;;
    *) echo "unknown arg $1"; usage ;;
  esac
done
[ -n "$N" ] && [ -n "$RUN" ] || usage

BIN=$(cd "$(dirname "$0")/.." && pwd)/target/release
for b in gen_sandwich_gadget fmix fcompress; do
  [ -x "$BIN/$b" ] || { echo "FATAL: $BIN/$b missing — cargo build --release first"; exit 1; }
done
mkdir -p "$RUN"; RUN=$(cd "$RUN" && pwd)
LOGALL=$RUN/gss_mix.log
note() { echo "[gss-mix] $*" | tee -a "$LOGALL"; }

# Derived sizes (the library conventions, computed here so they are pinned in
# the log): |C| = |D| = round(n (log2 n)^2), s = round(n log2 n),
# slice_gates = 10 * 2n, rg_freq = 1.
read -r M_CD S_SL SLICE_G <<< "$(python3 - "$N" "$MCD" <<'EOF'
import math, sys
n, mcd = int(sys.argv[1]), int(sys.argv[2])
l = math.log2(n)
print(mcd if mcd > 0 else round(n * l * l), max(n, round(n * l)), 10 * 2 * n)
EOF
)"
gates_of() { python3 -c "import sys; print(sum(1 for _ in open(sys.argv[1])) - 1)" "$1"; }
state_moves() { awk '$1=="moves"{print $2; exit}' "$1"; }

note "run=$RUN n=$N seed=$SEED |C|=|D|=$M_CD s=$S_SL slice_gates=$SLICE_G expand=$EXPAND hold=${HOLD}effs x=(r=$XR b=$XB c=$XC tdiv=$XTDIV)"

GADGET=$RUN/gss.mpmct1
PHASEA=$RUN/phaseA.mpmct1
SPLITB=$RUN/splitB.mpmct1
CROSSB=$RUN/crossB.mpmct1
FINAL=$RUN/final.mpmct1

# ---- stages 1+2: sandwich + Gray-fold gadgetization ----
if [ "$FORCE_FROM" -le 2 ] || [ ! -s "$GADGET" ]; then
  note "stage 1+2: gen_sandwich_gadget (production preset, Gray fold default)"
  "$BIN/gen_sandwich_gadget" "$GADGET" "$N" "$M_CD" "$M_CD" "$S_SL" 1 "$SLICE_G" \
      "$SEED" "$((SEED + 1))" "$SEED" > "$RUN/stage12.log" 2>&1
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
  [ -z "${FROZEN_CURATED_DIR:-}" ] && note "WARNING: FROZEN_CURATED_DIR unset — curated-first cascade OFF for phase A"
  G_IN=$(gates_of "$GADGET")
  read -r PROFILE A_MOVES <<< "$(python3 - "$EXPAND" "$HOLD" "$G_IN" <<'EOF'
import sys
r1, hold, g = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
n0, comp = 3.0, 20.0
n1 = n0 + hold; n2 = n1 + comp
r2 = 1.0 + (r1 - 1.0) / 2.0
# move ceiling: effs x peak size x margin
print(f"{n0:g},{n1:g},{n2:g},{r1:g},{r2:g}", round(n2 * r1 * g * 1.3))
EOF
)"
  note "stage 3: fmix phase A --gss --phase-a --profile $PROFILE (moves ceiling $A_MOVES)"
  export CANON_RULE_L_BRANCH_CAP=${CANON_RULE_L_BRANCH_CAP:-512}
  export CANON_MONOMIAL_CAP=${CANON_MONOMIAL_CAP:-200000}
  export FMIX_STOP_FLAG=$RUN/stage3.stop FMIX_DUMP_FLAG=$RUN/stage3.dump
  rm -f "$FMIX_STOP_FLAG"
  # Plumbing-test mode: fmix hard-requires the store whenever any DB channel
  # is armed, so the bypass must zero all three coins (a true null plant —
  # the profile still walks its eff schedule, no re-encoding happens).
  DBFLAGS=(--p-db 1.0 --p-comp 1.0 --p-any 0.1)
  if [ -z "${FROZEN_DB_DIR:-}" ]; then
    DBFLAGS=(--p-db 0 --p-comp 0 --p-any 0)
    note "WARNING: empty-store plumbing mode — phase A performs NO re-encoding"
  fi
  "$BIN/fmix" --input "$GADGET" --gss --phase-a --profile "$PROFILE" \
      --moves "$A_MOVES" --seed "$((SEED + 2))" \
      "${DBFLAGS[@]}" \
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
  note "stage 4: fmix --split (current defaults), moves ceiling $B_MOVES"
  export FMIX_STOP_FLAG=$RUN/stage4.stop FMIX_DUMP_FLAG=$RUN/stage4.dump
  rm -f "$FMIX_STOP_FLAG"
  "$BIN/fmix" --input "$PHASEA" --split --split-stop \
      --p-join 0.8 --split-reach-k 2 --split-fail-limit 100 --split-canaries 256 \
      --p-db 0 --p-comp 0 --p-any 0 --p-twist 0 --k-max 12 \
      --moves "$B_MOVES" --seed "$((SEED + 3))" --report-every 1000000 \
      --state-out "$RUN/splitB.state" --output "$SPLITB" > "$RUN/stage4.log" 2>&1
  grep -E "split stage ENDED|split spans|canary deciles" "$RUN/stage4.log" | tee -a "$LOGALL" || true
  note "stage 4 done: $(gates_of "$SPLITB") gates"
else
  note "stage 4: $SPLITB exists, skipping"
fi
[ "$STOP_AFTER" -le 4 ] && { note "stopped after stage 4"; exit 0; }

# ---- stage 5: the crossing walk (phase B part 2) — params TBD by the X-panel ----
if [ "$FORCE_FROM" -le 5 ] || [ ! -s "$CROSSB" ]; then
  G_S=$(gates_of "$SPLITB")
  read -r X_TGT X_TEMP X_MOVES_ABS <<< "$(python3 - "$G_S" "$XR" "$XTDIV" "${XMOVES:-0}" "$(state_moves "$RUN/splitB.state")" <<'EOF'
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
  note "stage 5: crossing walk (resume) target=$X_TGT temp=$X_TEMP b=$XB c=$XC moves(abs)=$X_MOVES_ABS — TBD params, see manual"
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
  "$BIN/fcompress" --input "$CROSSB" --output "$FINAL" --seed "$((SEED + 5))" \
      > "$RUN/stage6.log" 2>&1
  G_X=$(gates_of "$CROSSB"); G_F=$(gates_of "$FINAL")
  note "stage 6 done: $G_X -> $G_F gates (residual $(python3 -c "print(f'{100*$G_F/$G_X:.1f}%')"))"
else
  note "stage 6: $FINAL exists, skipping"
fi

note "PIPELINE COMPLETE: $FINAL"
