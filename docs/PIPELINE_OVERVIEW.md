# The GSS Mixing Pipeline — Components, Parameters, Rationale

*2026-08-05. For readers familiar with the modules who want to understand
what the knobs are and why they are set the way they are. Deep dives are
referenced by filename (all in `docs/` unless noted). The runnable
packaging of everything here is `scripts/gss_mix.sh` — see `GSS_MIX.md`.*

```
n ──▶ [I. construction: sliced sandwich + gadgetization] ──▶ GSS (4n wires)
        ──▶ [II. phase A: DB re-encoding mix] ──▶ [III. phase B: structure breaking]
              ──▶ [IV. fcompress] ──▶ deliverable
```

The pipeline's one free input is **n**, the wire count of the secret
computation C. Everything else is either a library convention (a function
of n), a validated preset, or a measured calibration — this document says
which, and why.

---

## I. Circuit construction — sliced sandwich + gadgetization

**What.** Sample a fresh random g57 circuit C on n wires; build the sliced
sandwich A on 2n wires (random D, keyed slice interleave, N-column float)
with `A(x,0) = (junk, C(x))` on the zero slice; then gadgetize to 4n wires
behind a zero-slice preblock, using the diversified CG menu, nonlinear
{RG1,RG2,RG3}, the product-share mask encoding, and a final commuting
shuffle. (`SLICED_SANDWICH.md`, `NEW_GADGETIZE.md`,
`NONLINEAR_RG_CG_MENU.md`.)

**Parameters and why:**

| knob | value | rationale |
|---|---|---|
| `|C| = |D|` | `round(n·(log₂n)²)` | library convention (`sandwich_default_m`); D sized like C so neither side of the sandwich is the thin one |
| slicing budget s | `round(n·log₂n)` | `sandwich_default_s`; enough keyed slice mixing to decorrelate block boundaries |
| slice gates | `20n` | zero-slice preblock size, tool default |
| mask plan | `[2,2,2,3]` + Gray fold | **measured**: statistical leak is LINEAR in the piling-up product ε (R² = 0.996 across plans); [2,2,2,3]+fold cuts ε 3× vs the old [2,3,3] at −14% gates. Disjoint deg-2 piling-up is PROVED optimal for sparse-ANF masks (`CORRELATING_TWO_COMPUTATIONS.md`, `TWISE_INDEPENDENT_MASKS.md`, `GRAY_FOLD_CG.md`) |
| Gray fold | on (default) | every fold gate ≤2 controls → per-gate DB reachability 31.6% → 95.6% (`GRAY_FOLD_CG.md`; judge digestibility by reachability, never match rate) |
| carrier decode | single-carrier | the shipped [3,3] was really [1,1,3,3] with a second linear carrier FREE to an affine adversary (`SINGLE_CARRIER_CONSTRUCTION.md`, `SINGLE_CARRIER_DECODE.md`) |
| band | rolling, nonlinear cascaded fill, retire-refill epochs | band-in-slice exact disturbance theorem; rolling +12%; fill must be jointly uniform (triangular pivot); band lifetime, not width, is the recovery channel (`BAND_HARDENING.md`, `BAND_INDEPENDENCE.md`) |

No `PROD_*` overrides: the production preset IS the default
(`PRODUCT_SHARE_ENCODING.md`, `PRODUCT_SHARE_UPDATE.md`). The gadget
sample-verifies `low-2n ≡ A` on the zero slice at build time.

---

## II. Phase A — the DB re-encoding mix (`fmix --gss --phase-a --profile`)

**What.** A move-based walk whose slot-2 DB moves re-spell windows of the
circuit against the frozen replacement store (MIX grows-if-needed, COMP
strictly shrinks), driven by a layer-2 size profile; g57-word twists supply
state-frame rotation. Phase A is deliberately **g57-preserving** — its job
is re-encoding depth (generation dose) and spelling diversity, not
structure breaking. (`POSTMIX_MANUAL.md` §2, `FMIX_PHASE_A.md`,
`FMIX_LAYER1.md`, `FMIX_LAYER2.md`.)

**Parameters and why:**

| knob | value | rationale |
|---|---|---|
| size profile | `3, 3+H, 3+H+20, R, 1+(R−1)/2` (default `3,33,53,2,1.5`) | expansion leg fixed at 3 effs — transport happens during growth, so the ramp is short and steep; hold H=30 effs at R=2 is the re-encoding workhorse; compression leg fixed at 20 effs sheds back to halfway (R2=1.5), keeping spelling diversity that a full return to 1× would spend (`FMIX_LAYER2.md`) |
| exposed knobs | `--expand R`, `--hold H` | the only two the pipeline exposes: dose (hold) and peak size (expand) |
| DB knobs | the `--gss` profile | per-mode: COMP descent from s_db 12 (a START, walks 12…1), MIX uniform draw ≤6; curated-first cascade ON both modes; precedence rules in `FMIX_PARAM_PRECEDENCE.md`. Calibrated by the mode×geometry×curation cube (`DB_CAMPAIGN_20260805.md`, `CURATED_DB_COMPARISON.md`) |
| twists | `--phase-a` block: twist-g57 at p 0.0005 | brackets are adaptive all-g57 words that ABSORB neighborhood gates (`G57_TWIST_BRACKETS.md`); rate kept tiny — one twist rewrites O(window) gates |
| store guards | degree 9, span 30, terms 1024/2048 | degree = the store's max ANF degree; span/terms cap canonicalization cost |
| env | `FROZEN_DB_DIR`, `FROZEN_CURATED_DIR` | the runtime is frozen-store only; read at startup, never from rc files |

---

## III. Phase B — structure breaking (split stage + crossing walk)

Phase B's reframed job (2026-08-04) is **anti-inversion**: break the g57
structure with absolute spread — not state mixing (fmix alone hides
nothing; hiding is sandwich-borne). It runs with **no DB and no
swap-family twists**; both parts are pure walk machinery.

**Part 1 — the split stage** (`--split`, defaults as shipped). One move
to exhaustion: presplit a random g57 into (CNOT/NCNOT, 2-control AND);
with p_join = 0.8 wrap an **absorbed pure-NOT twist** on its target wire —
both brackets flip a control polarity (zero synthetic gates), the segment's
w-reading pins flip, segment g57s force-split — then one cross from the
AND piece. Bracket side drawn ∝ remaining circuit length (kills the
short-span spike; squared suppression), bracket = farthest of
`reach-k = 2` samples. Growth = 1 + comp-fraction (measured ×1.7–2.0);
output is dead-even CNOT/NCNOT + 1:2:1 ANDs, zero comp; coverage is
absolute (canary-verified, mid-humped span geometry only). Full spec
`FMIX_SPLIT_TWIST.md`; measurements and the three-arm bracket-draw A/B in
`SPLIT_TWIST_REPORT.pdf` and `reports/split_trials_20260805/RESULTS.md`.

**Part 2 — the crossing walk** (resumes the split state). Pure crossings
under the thermostat. The 2026-08-05 X-panel set every knob:

| knob | value | rationale |
|---|---|---|
| damping | b=3, c=1 | heavy damping EQUALIZES: median descendants up, tail down — the only knob that lifts the median (spread must be judged by MEDIAN absolute descendants/span; the Yule tail makes means dishonest). b>3 under test |
| temperature | target/25 | second-order; medium keeps arrival clean |
| target `--xr` | 2 (2.5 = max-spread) | frontier at the arrival peak: realized 1.63/1.84/2.01/2.25× → frac(≥3 desc) 0.49/0.56/0.60/0.62, median span 360/464/552/649 — decelerating; xr 2 clears median ≥ 3 with margin |
| budget | 6 × target, **stop at arrival** | both median metrics PEAK when size reaches equilibrium and the constant-size hold ERODES them (frac≥3 0.555→0.498 over +20M moves) — the hold moves spread from the median to the tail. There is no moves-for-size trade past arrival |

The 2/3-of-inputs ≥3-descendants bar is out of expansion's reach; the next
lever is min-dgen cross-shot bias (planned), not more growth.

---

## IV. fcompress — the final pass and honesty check

**What.** The attacker-computable greedy compressor (gather → group-cap →
ANF reduce), run whole-function (`--live-wires all`), as both the final
size pass and the honesty metric: the residual (output/input gates) is
incompressibility EARNED by mixing — healthy mixed material lands ≳90%,
vs 83% for raw split artifacts. (`POSTMIX_MANUAL.md` §3.) Parameters are
the tool defaults; nothing here is tuned per run.

---

## Quick reference — every free parameter in one place

| stage | knob | default | change when… |
|---|---|---|---|
| all | `-n` | (required) | it's the problem size |
| all | `-s` seed | 1 | independent replicate wanted |
| II | `--expand` | 2 | more peak size for more re-encoding room |
| II | `--hold` | 30 effs | more/less generation dose |
| III.1 | (none exposed) | shipped split-stage defaults | — |
| III.2 | `--xr` | 2 | 2.5 for max median spread; frontier decelerates above |
| III.2 | `--xb/--xc/--xtdiv` | 3 / 1 / 25 | X-panel-calibrated; b>3 probe pending |
| III.2 | `--xmoves` | 6×target | never linger — arrival is the peak |

Ops: run production sizes on the server; export the store env vars in the
launch environment; every fmix launch exports `FMIX_STOP_FLAG`/
`FMIX_DUMP_FLAG`; outputs promoted to `circuits/` need a
`CIRCUIT_GENERATION_INFO` entry.
