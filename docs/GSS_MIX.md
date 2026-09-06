# GSS-MIX — the end-to-end mixing pipeline for GSS circuits

`scripts/gss_mix.sh` packages the whole mixing pipeline for the special case
of **GSS inputs** (gadgetized sliced sandwich): one command from "a wire
count" to a compressed, mixed circuit, with every stage's artifact, state
file and log left in the run directory. Companion docs:
`SLICED_SANDWICH.*` (stage 1), `NEW_GADGETIZE.md` (stage 2),
`POSTMIX_MANUAL.md` §2.1.2–2.1.3 (stage 3), `FMIX_SPLIT_TWIST.md` +
`SPLIT_TWIST_REPORT.pdf` (stage 4), `POSTMIX_MANUAL.md` §3 (stage 6).

```
n ──1─▶ sliced sandwich S ──2─▶ GSS gadget ──3─▶ phase A ──4─▶ split stage ──5─▶ crossing walk ──6─▶ fcompress
        (2n wires)              (4n default)    (DB mixing)   (g57 → pairs)     (pure growth)      (final)
```

## Editable `gss` command configuration

`local_mixing_bin gss` reads only the marked block below. Edit values on the
right side of `=` and leave any value blank to use its documented default.
Unknown or duplicate keys are errors, so a typo cannot silently start a long
run with the wrong setting. Values are literal: do not add shell quotes,
variable expansion, or inline comments.

<!-- GSS_MIX_CONFIG_BEGIN -->
```ini
# Runner and artifacts. Blank n = 128. Blank run_dir makes a fresh unique
# runs/gssmix_n128_<id> directory; fill an existing directory only to resume.
n =
run_dir =
build_release =
build_target_dir =
adopt_existing_run =

# Replacement stores. A blank path inherits the matching FROZEN_* environment
# variable. The regular store has no unsafe guessed default and is required
# from stage 3 onward. The curated store is optional but recommended.
frozen_db_dir =
frozen_curated_dir =
curated_value_convention =
frozen_filter =
allow_empty_store =

# Stages 1-2. Blank mode = product-2223. Preset and post_fragment apply only
# to product-2223; leave them blank for nonlinear modes.
gadgetization_mode =
production_preset =
post_fragment =
mcd =

# A raw seed must never be written in this tracked file. For a paired
# CALIBRATION ONLY run, set calibration_only=true and point at a private
# chmod-600 file containing one decimal seed. Leave both blank for production.
calibration_only =
calibration_seed_file =

# Stage 3 Phase A. Blank values use expand=2 and hold=30.
expand =
hold =

# Stage 5 crossing walk. Blank values use xr=2, xb=3, xc=1, xtdiv=25,
# and xmoves=6*target.
xr =
xb =
xc =
xtdiv =
xmoves =

# Lifecycle. Blank values run through stage 6 and do not force a rebuild.
stop_after =
force_from =
```
<!-- GSS_MIX_CONFIG_END -->

Defaults and accepted values:

| key | blank/default | accepted nonblank values |
|---|---|---|
| `n` | `128` | `3..=4095` |
| `run_dir` | a fresh unique directory under `runs/` | a new directory, or the exact prior directory when resuming |
| `build_release` | `true` | `true`, `false` |
| `build_target_dir` | inherit `CARGO_TARGET_DIR`, otherwise repository `target/` | a dedicated Cargo target directory |
| `adopt_existing_run` | `false` | `true` once to acknowledge unverifiable provenance when first wrapping an older direct-script run; also requires `calibration_only=true` |
| `frozen_db_dir` | inherit `FROZEN_DB_DIR`; otherwise stage 3 fails | existing directory |
| `frozen_curated_dir` | inherit `FROZEN_CURATED_DIR`; otherwise regular-only | existing directory |
| `curated_value_convention` | `native` | `native`, `legacy-swapped-controls` |
| `frozen_filter` | `auto` (the driver's RAM gate) | `auto`, `on` (requires `filters.bin` in every selected store), `off` |
| `allow_empty_store` | `false` | `true` only for a no-re-encoding plumbing test |
| `gadgetization_mode` | `product-2223` | `product-2223`, `nonlinear193`, `nonlinear291`; `2223` is accepted as a compatibility alias and normalized to `product-2223` |
| `production_preset` | `production` | product-2223 only: `production`, `no-gray-phase-a`, `micro-gray`, `sentinel-gray`, `no-gray-post-exact`, `no-gray-post-native`, `five-carrier`, `strong-five-carrier`, `six-carrier`, `strong-six-carrier`, `seven-carrier` |
| `post_fragment` | preset behavior | product-2223 only: `off`, `exact`, `native-deep` |
| `mcd` | `round(n(log2 n)^2)` | integer `1..=1000000000` |
| `calibration_only` | `false` | `true`, `false` |
| `calibration_seed_file` | CSPRNG for a fresh run; `<run>/SEED` for resume | private mode-600 file; requires `calibration_only=true` |
| `expand`, `hold` | `2`, `30` | expansion `(1,16]`; hold `[0,10000]` |
| `xr`, `xb`, `xc`, `xtdiv` | `2`, `3`, `1`, `25` | `xr` in `[1,16]`, `xb` in `[1,1000000]`, integer `xc` in `[0,1000000000]`, integer `xtdiv` in `[1,1000000000]` |
| `xmoves` | `6 * crossing target` | integer `1..=1000000000000` |
| `stop_after` | `6` | stage `2..=6` |
| `force_from` | no forced rebuild | stage `2..=6` |

The block contains every **supported GSS run parameter**. Values described
later as pinned—DB probabilities and guards, controller legs, seed offsets,
split invariants, verification settings, and compressor limits—define the
tested recipe rather than an interchangeable configuration surface. Use a
separate experimental driver when studying those constants; arbitrary
low-level overrides must not be mistaken for a production GSS run. The `gss`
command removes inherited `PROD_*`, rewrite/debug, SAT-scoring, and compressor
override variables and pins the documented canonicalization/cache defaults.
The direct Bash interface remains available for explicitly labeled experiments.

`product-2223` with `production_preset=production` is the production-accepted
default (the existing `[2,2,2,3]` product construction). Other presets inside
that family remain study arms. `nonlinear193` and `nonlinear291`
are experimental, capacity-limited integration modes: selecting either does
not imply production acceptance, and the generator may reject sizes that
exceed its current wire/workspace bounds. `production_preset` and
`post_fragment` must remain blank for those nonlinear modes and are rejected
when set rather than silently ignored. The resolved canonical mode is recorded
in logs and in the resume manifest. These modes establish functional
zero-slice integration, but their deterministic ingress masks and auxiliary
endpoint state have not yet been requalified by the GSS gauntlet; evidence for
the standalone 193/291 bodies must not be treated as evidence for the complete
adapter.

## Quick start

```bash
# First fill frozen_db_dir (and preferably frozen_curated_dir) in the block,
# or export FROZEN_DB_DIR/FROZEN_CURATED_DIR while leaving those lines blank.
cargo run --release -- gss --dry-run
cargo run --release -- gss
```

The command validates the block, incrementally builds exactly the three
production executables into the selected target directory, and then runs the
existing Bash orchestrator. A dry
run performs no build, creates no run directory, and launches no pipeline
process. An already-built command can also be invoked as
`target/release/local_mixing_bin gss`. Use `--config PATH` to read the same
marked block from another Markdown file.

On launch, the wrapper writes a seed-free `<run>/gss_command.conf` containing
the complete resolved recipe plus XXH3 fingerprints of the three production
binaries and `scripts/gss_mix.sh`. A managed resume compares that manifest
*before building* and reuses the exact recorded binaries; a changed recipe,
script, or binary is refused rather than mixed with skipped artifacts.
`stop_after`, `force_from`, filter selection, and build-location controls are
lifecycle settings and are not locked. All eventual stores and future-stage
knobs should nevertheless be chosen before the first partial run, because the
whole future recipe is locked immediately. Frozen stores are identified by
their canonical path and are operationally required to remain immutable.

An older run made directly by the Bash script has no manifest. To bring it
under the wrapper, set both `adopt_existing_run = true` and
`calibration_only = true` once. Its original seed provenance cannot be
verified, so the wrapper will not relabel it as a deliverable. The manifest
records the unverified adoption and every later resume repeats that warning.
Calling the Bash script directly bypasses these provenance guards.

For low-level experiments or automation, the Bash interface remains available:

```bash
cargo build --release --bin gen_sandwich_gadget --bin fmix --bin fcompress
FROZEN_DB_DIR=/absolute/path/to/frozen_regular \
FROZEN_CURATED_DIR=~/frozen_curated_m1_m11_native \
  bash scripts/gss_mix.sh -n 128 -o runs/gssmix_n128_experiment \
    --gadgetization-mode product-2223
```

The driver uses three production entry points:
`src/preprocessing/bin/gen_sandwich_gadget.rs`,
`src/db_mixing/bin/fmix.rs`, and
`src/postprocessing/bin/fcompress.rs`. Use `bash`, not `sh` or `zsh`;
calling the script through `bash` also works in a checkout that did not retain
its executable bit.

Both frozen directories use `tables.bin` plus `shard_00.frz` through
`shard_ff.frz`; `filters.bin` is optional. New regular and curated stores use
the native value convention. Only when serving a historical curated store
built before commit `2ed0222a`, also set:

```ini
curated_value_convention = legacy-swapped-controls
```

When invoking the Bash driver directly, the equivalent is
`FROZEN_CURATED_VALUE_CONVENTION=legacy-swapped-controls`. Leave the config
blank for a new native store. The rebuild and value-convention contracts are
in `src/db_generation/README.md`.

If the aggregate Gray fold is outside the threat model, select the measured
no-Gray Phase-A preset at generation time:

```bash
export PROD_PRESET=no-gray-phase-a
bash scripts/gss_mix.sh -n 128 -o runs/gssmix_n128_safe
```

It expands each source product atom-by-atom and selectively narrows fragments
through width four before Phase A. It never gathers an operand's complete mask
sum onto one dirty accumulator. Do not pre-compress this stage: in the paired
n=16 frozen-store experiment, `fcompress` reduced regular-store reach from
92.22% to 77.96% even though the control-width profile remained narrow. This is
a security/throughput alternative, not the default; it is larger and its GSS
MIX dry-run hit rate was 48.0% versus 61.2% for aggregate Gray.

Three additional study arms are available without changing the production
default:

```bash
export PROD_PRESET=micro-gray             # four shares, 16 restored rectangles
export PROD_PRESET=sentinel-gray          # gather lower degrees, not max degree
export PROD_PRESET=no-gray-post-native    # no aggregate; native dirty ladder
```

`micro-gray` removes the single before/after accumulator interval that exposes
the aggregate operand, but its four public share deltas still XOR back to that
operand. `sentinel-gray` keeps every maximum-degree atom out of the aggregate;
it is experimental and intentionally leaves the widest high--high products as
fossils. `no-gray-post-native` performs the no-Gray cap-four construction and
then fragments every remaining wide gate after the final shuffle, keeping rung
zero exact and spelling deeper dirty-ladder rungs in the frozen-store-native
g57/CNOT vocabulary. The equivalent explicit override is
`PROD_POST_FRAGMENT=native-deep`; `exact` selects all-plain rungs and `off`
disables the post pass. All helpers are arbitrary dirty values and are restored.

The paired n=16 results and metric definitions are in
`docs/CARRIER_GADGETIZATION_SUMMARY.tex`. In brief, verified live Phase-A MIX
splice rates were 9.52% aggregate Gray, 7.91% no-Gray cap four, 8.41% native
post-fragmentation, and 12.15% four-share micro-Gray. Micro-Gray started at
4.56 times the Gray circuit size, so the higher splice rate is not free.

To generate the five-carrier nonlinear representation instead, select its
standalone-generator preset:

```bash
export PROD_PRESET=five-carrier
bash scripts/gss_mix.sh -n 128 -o runs/gssmix_n128_five
```

This is the `gen_sandwich_gadget` counterpart of
`sss --cnot --gadgetize --five-carrier`. Each of the sandwich's `2n` logical
values uses five carrier wires plus the existing auto band, so the stage-2
artifact is `12n` wires (`1536` at source `n=128`) rather than the default
`4n`. The single-carrier production preset remains the default when
`PROD_PRESET` is unset.

Here “endpoint degree” means exact recovery of the local target-flip bit from
an identified before/after carrier tuple. The production single-carrier
snapshot is already degree three because its unchanged external mask plan is
`[2,2,2,3]`; nevertheless its endpoint flip is degree one because those masks
cancel, leaving `carrier_before XOR carrier_after` (up to a known ledger
constant). The supplied five raises this boundary to degree two, strong five
and either six raise it to degree three, and seven raises it to degree four.
This endpoint ladder is separate from aggregate Gray's internal degree-one
space-time witness.

The experimental cubic five-carrier sibling is selected with
`PROD_PRESET=strong-five-carrier`, or through
`sss --cnot --gadgetize --strong-five-carrier`. It keeps the same wire count,
but moves exact endpoint recovery from supplied five's degree two to degree
three (versus degree one for production single-carrier) and lowers
the first weight-three correlation from 75% to 62.5% agreement. Its compact
six-gate update leaves two carrier-tail lanes fixed, so it is an explicit
algebraic experiment rather than a silent replacement for the supplied map.

The stronger six-carrier variant is selected analogously:

```bash
export PROD_PRESET=six-carrier
bash scripts/gss_mix.sh -n 128 -o runs/gssmix_n128_six
```

An experimental structural sibling is selected with
`PROD_PRESET=strong-six-carrier`, or through
`sss --cnot --gadgetize --strong-six-carrier`. It retains the same cubic
decode, exact degree-three endpoint boundary, and zero raw endpoint parity
through weight three. Its 21-gate update has full affine graph rank and moves
every carrier lane; the compact ten-gate six-carrier update has frozen lanes.
This is a structural trade, not a blanket statistical dominance: the first
weight-four bias is unchanged, and some higher-weight coefficients are larger.

Its endpoint trace has exactly zero Walsh correlation with every parity of up
to three before/after carrier wires, and its gate-firing bit is outside the
span of every degree-two endpoint monomial (degree three is the first exact
recovery). It uses `6*(2n) + 2n = 14n` wires with the auto band (`1792` at
source `n=128`). The Gray fold is representation-aware: it gathers the full
six-carrier decode, including both cubic atoms, and restores all three dirty
borrowed carriers. As in five-carrier mode, this endpoint claim does not
remove the stronger space-time witness obtained by observing a Gray
accumulator immediately before and after a complete gather.
The compact exact U0 also leaves carrier lanes c4 and c5 unchanged and has an
affine non-c0 update. That is a structural distinguisher; the guarantees above
are specifically about static low-weight decode leakage and recovery of the
gate-firing bit, not general S-box nonlinearity.

The seven-carrier variant is selected with:

```bash
export PROD_PRESET=seven-carrier
bash scripts/gss_mix.sh -n 128 -o runs/gssmix_n128_seven
```

This is also the retained strong-seven/Pareto preset. A bounded search found
lower-bias updates (56.25% rather than 62.5% first-detector agreement), but
every such candidate regressed exact recovery from degree four to degree
three. No degree-four candidate improved the shipped map's weight-four
multiplicity or higher spectrum, so no misleading `strong-seven-carrier`
alias was added.

Its selected update is a fixed-point-free nonlinear permutation within each
decode class. The endpoint firing trace has exactly zero Walsh correlation
with every parity detector of weight at most three; the first nonzero detector
has weight four. There is no perfect affine before/after trace relation, and
the firing bit is outside the degree-three endpoint span (the decode identity
recovers it at degree four). The Gray fold gathers the full seven-carrier
decode, including its quartic atom, using a restored dirty helper. With the
auto band this mode uses `7*(2n) + 2n = 16n` wires (`2048` at source `n=128`).
As with the other nonlinear representations, the endpoint guarantees do not
claim to hide an accumulator observed in the middle of a complete Gray gather.

Artifacts land in the run dir: `gss.mpmct1` (+ `.sandwich.mpmct1`,
`.source_c.g57`), `phaseA.mpmct1`(+`.state`), `splitB.mpmct1`(+`.state`),
`crossB.mpmct1`(+`.state`), `final.esop1` (packed; `--no-pack` for a cube `final.mpmct1`), per-stage logs, and
`gss_mix.log` (the pinned-parameter narrative). The direct driver also writes
`stage12.recipe`, which binds `gss.mpmct1` to its normalized gadgetization
mode and dimensions. A stage whose artifact already exists is skipped only
when that recipe still matches. To continue a killed pipeline, invoke the driver
again with the same `-n`, `-o`, `gadgetization_mode`, preset, and tuning
options, and **omit `-s`**:
the driver reloads the original secret from `<run>/SEED`. Do not reuse a run
directory with a different explicit seed or configuration; use a fresh run
directory instead. `--force-from K` deliberately rebuilds stage K and every
later stage; `--stop-after K` ends early. Because the skip check is based on
the main `.mpmct1` artifact, confirm that `splitB.state` is present before
resuming into stage 5. Each fmix stage arms its own `stageN.stop` /
`stageN.dump` flag files (touch to stop cleanly / snapshot).
The shell driver currently treats a nonempty stage artifact as complete; after
an unclean kill during a file write, inspect the stage log/state and use
`force_from` at that stage rather than trusting the skip check.

## The stages and their parameters

**1+2 — generate + gadgetize** (`gen_sandwich_gadget`). The source-size
parameter is **n**, the wires of the source computation C, and
`gadgetization_mode` selects the representation family. The remaining size
conventions are computed and logged by the driver:
`|C| = |D| = round(n·(log₂n)²)`, slicing budget `s = round(n·log₂n)`,
`slice_gates = 20n`, `rg_freq = 1`. The default `product-2223` gadgetization
runs the **production preset**, which since 2026-08-21 is the
**swap-refresh symmetric construction** (`docs/SWAP_REFRESH_REDESIGN.md`):
mask plan [2,2,2,3] on a single-carrier decode, per-gate mask
swap-with-refresh (the target and one control of every fold retire one
monomial and gain a fresh draw — the Gray fold is declined; its gather is
a linear operand recovery), expanded fold with the selective ladder at
cap 3 (≈82% of the stream in pure g57+CNOT vocabulary; scratch and
discharge helpers come from live band variables, never carriers),
independently drawn **junk-half zero-slice guards at BOTH ports**,
nonlinear band fill sourced from the low data half, band roll, and
retire-refill epochs with split channels (band-only linear pivot;
carrier sources only inside product terms). Contract changes from the
Gray era: the gadget preserves S on the **upper data half only** (the
payload contract — the closing guard deliberately perturbs the junk
half), and the construction is **reverse-honest** — the reversed gadget
on `(a, 0, zero band)` yields `D⁻¹(a)` on the upper half, so one
artifact evaluates C forward and D⁻¹ backward, mirroring the sandwich.
Measured at n=128: no cross-circuit linear or (windowed) degree-2
segment relations beyond the public port boundary, up to 0–2
tail-seam windows per seed; phase-A digestibility at or above the Gray
baseline (56.5% splice rate, 88.7% curated). Env knobs: `PROD_SWAP` /
`PROD_CLOSE_SLICE` (0/0 restores the prior Gray stream byte-for-byte),
`PROD_LADDER_CAP` (0 = 0.68× the Gray size but 48% wide fossils;
3 = default, 1.31×; 4 = store-weak conj-2 at 2.55×), and the size
levers `PROD_K` (2 → −18.5%, at 2× the piling-up ε) and
`PROD_CG_JITTER` (0 → −8.8%; both → −27.3%, e.g. 1.17M gates full /
639k half at n=128). Alternate `PROD_PRESET`s remain
(`no-gray-phase-a`, a fold/fragmentation study arm, `five-carrier`,
`strong-five-carrier`, `six-carrier`, `strong-six-carrier`, or
`seven-carrier`); the multi-carrier presets take the fold-level swap but
not the output-port hygiene.
The `nonlinear193` and `nonlinear291` families are experimental and
capacity-limited; their availability at a requested size is not a production
acceptance claim.
The sandwich S (2n wires) and the source C are dumped beside the gadget. In
`product-2223`, the gadget uses 4n wires by default, 12n under either
five-carrier mode, 14n under either six-carrier mode, or 16n under
`seven-carrier`. The nonlinear modes use two five-wire E shares per logical
value plus per-gate fresh regions; their exact capacity is checked before any
artifact is written. With derived `mcd`/slice defaults they currently fit only
through source n=63 under the u16 wire ABI (a smaller explicit `mcd` can permit
larger n). The tool
sample-verifies the gadget against S on the zero slice (256 bit-sliced
samples): under `product-2223` the upper data half forward (the payload
contract) **and** the reversed gadget against the reversed sandwich
(the D⁻¹ contract); other modes verify the full low range forward.
Seeds: C and the sandwich use the master seed, the gadgetization
`seed+1`. Omit `-s` to obtain a fully fresh production pipeline. Use an
explicit `-s N` only when paired calibration arms must share the same input;
those outputs are calibration-only.

**3 — fmix phase A** (`--gss --phase-a --profile`). The DB-mixing stage,
using the GSS DB profile (curated-first, per-mode s_db, g57-preserving) and
the phase-A block (twist-g57, db-advance, pay-random) with the layer-2 size
profile as the single size authority. The two knobs exposed, per the design:

- `--expand R` — the **max expansion factor** R1 (default 2);
- `--hold E` — the **stable-stage duration** in effs (default 27).

The profile is then `N0,N1,N2,R1,R2 = 3, 3+E, 3+E, R, R`: the expansion
leg is fixed at 3 effs and the run ends at the held size — there is **no
compression leg** (removed 2026-08-17: it only simplified the circuit,
which made sense when phase A stood alone but is wasted work now that
phase B follows). The defaults give `--profile 3,30,30,2,2`, i.e. 30 effs
overall ending at 2× the GSS size. The move budget is a ceiling
(`N2 × R1 × gates × 1.3`); the finished profile ends the run
(`ProfileDone`). Needs `FROZEN_DB_DIR` (hard error
without it; `GSS_MIX_ALLOW_EMPTY_STORE=1` bypasses for plumbing tests
only — a null store means zero re-encoding). DB guards are pinned:
degree 9, span 30, terms 1024/2048; CANON caps exported.

**4 — the split stage** (phase B part 1, `--split --split-stop`), at the
current shipped defaults: `p_join 0.8`, `split-reach-k 2` (bracket side ∝
remaining length, farthest of 2), fail-limit 100, 256 canaries, `k_max 12`,
no DB, no swap-family twists. Runs to g57 exhaustion; expect growth ≈
1 + comp-fraction (≈2× on a typical phase-A output, i.e. ≈4× the GSS
size), zero comp gates out, and the stage summary + canary deciles echoed
into `gss_mix.log`. The production Stage-4 algorithm is in
`src/postprocessing/splitting.rs`; it operates on the shared `engine::Mixer`
state and uses the shared crossing primitive for each joined move's final
shot. The older `experimental/split_engine.rs` implementation is a standalone
research walk and is not called by GSS.

**5 — the crossing walk** (phase B part 2). Resumes `splitB.state` — the
split form's per-gate directions and litters carry over — and runs the pure
crossing economy (no twists, no DB) under the thermostat. The production
crossing, exact-undo, and merge-contraction operations are in
`src/postprocessing/cross_walk.rs`; the scheduler and resumable state remain in
`src/engine/mix.rs`. Parameters
**calibrated by the 2026-08-05 X-panel** (`reports/split_trials_20260805`)
and shipped as the current numerical defaults. The repository has not yet
resolved whether those calibrated defaults are promoted for deliverables;
until that policy decision is recorded, stage-5 outputs remain calibration
material:

| knob | flag | default | why |
|---|---|---|---|
| target factor over the split size | `--xr` | 2 | see the frontier below |
| width-damper base B | `--xb` | 3 | heavy damping equalizes: median descendants up, tail down |
| width-damper threshold c | `--xc` | 1 | (with b=3: a width-w split passes w.p. 3^-(w-1)) |
| temperature (target/D) | `--xtdiv` | 25 | second-order |
| move budget | `--xmoves` | 6 × target | **STOP AT ARRIVAL** — see below |

Panel findings that set these: (i) spread must be judged by MEDIAN absolute
descendants/span — the Yule tail makes means dishonest; (ii) both medians
PEAK when size reaches its damped equilibrium and the hold then ERODES them
(frac(≥3): 0.555 at arrival → 0.498 after 20M more moves; the tail keeps
its reach — the hold moves spread from the median to the tail), so budgets
stop at arrival; (iii) knobs b/c/temp are second-order for spread except
heavy damping's equalization; (iv) the spread-vs-size frontier under
b=3/c=1, at the arrival peak (sizes = realized, over the split form):

| target r | realized | peak frac(≥3 desc) | peak median span |
|---|---|---|---|
| 1.75 | 1.63× | 0.49 | 360 |
| 2.0 | 1.84× | 0.56 | 464 |
| 2.25 | 2.01× | 0.60 | 552 |
| 2.5 | 2.25× | 0.62 | 649 |

`--xr 2` is the smallest point clearing median ≥ 3 with margin; `--xr 2.5`
is the measured max-spread point; gains decelerate above it, and a 2/3
frac(≥3) bar is out of reach of expansion alone — the next lever there is
biasing cross-shot selection toward never-crossed (min-dgen) gates.

`--moves` is ABSOLUTE on a resume; the driver reads the state's move
counter and adds the budget — do not pass raw fmix moves yourself.

**6 — fcompress + pack**, whole-function (`--live-wires all`), the
attacker-computable greedy compressor as the final pass and honesty check,
followed by packing. The driver logs the cube residual (compressed cubes /
pre-compress gates) and the packed gate count. Healthy mixed material
historically landed ≳ 90% under the 2026-08-22 pass. ⚠️ The 2026-09-05 pass
(in-gather transport = Toffoli sliding at any distance, separated reads,
reversed-list gather; `POSTMIX_MANUAL.md` §3) removes far more: a delivered
K2 final that was a fixed point of the old pass shrinks to 57.8%, so
residuals and "effective size" figures quoted before that date are ~1.7× too
high, and the ≳ 90% health bar must be recalibrated (2-eff finals land at
~52%, full hold-10 at ~58% of the old finals).

**The deliverable is `final.esop1`, the packed canonical form** (see
`docs/FCOMPRESS_TRANSPORT_AND_PACKING.md`): one generalized gate per maximal
same-target run of the compressed circuit, its activation function first
brought to algebraic normal form (the unique representation of a Boolean
function) and then compacted into a mixed-polarity ESOP by the deterministic
reducer strategies, from the ANF alone — so the file still has one spelling
per function and carries nothing of how the mixer happened to spell it as
cubes, at about the cube count (~+5%) rather than the ANF's ~2.4×. Format:
header `esop1 <wires> <gates>`, then `<target> <n> [<width> <wire> <pol>…]*`
per line, terms sorted by (size, literals), an empty term = the constant 1.
Every mpmct1 reader in the tree (`format::read_mpmct`) loads an esop1 file
transparently as its term expansion, so hmap_affine, the censuses and
fcompress itself accept it; `fcompress --no-pack` writes the mpmct1 cube
circuit instead (an intermediate that stays server-side).

## Sizing expectations (defaults, n = 128)

| stage | size |
|---|---|
| sandwich S | ~2·|C| + slicing ≈ 13–15k gates, 256 wires |
| GSS gadget | ~1.3M gates, 512 wires (production preset, measured 2026-08-17) |
| phase A out | 2× GSS ≈ 2.6M |
| split out | ≈ 2× phase A ≈ 5.2M (zero comp) |
| crossing out | `--xr` × split ≈ 10M at the default 2 |
| final (cubes) | ≳ 90% of crossing out under the 2026-08-22 pass; ~50–60% of that again under the 2026-09-05 pass (recalibrate) |
| final.esop1 (packed) | ~22% as many gates as the compressed cubes (one per same-target run), ~1.05× as many terms as cubes |

Runtimes are dominated by stages 3 and 5 (tens of millions of moves);
run production sizes on the server, exporting the store paths in the
launch environment (they are read at startup, not from any rc file).

## Notes and contracts

- **Seeds — the seed IS the secret.** Stages 1+2 regenerate bit-identically
  from `(n, seed)`, so anyone holding the seed reconstructs C and the whole
  gadget: **always run with the default random seed** (a fresh CSPRNG draw,
  written to `<run>/SEED` mode 600 and never echoed into the shared log).
  Never a constant or a counter. The seed of a deliverable is secret — keep
  it out of chat, reports, and `CIRCUIT_GENERATION_INFO`. Explicit `-s N` is
  for CALIBRATION arms that must share an input (paired knob comparisons);
  anything produced that way is calibration material, never a deliverable.
  fmix stages are deterministic per (input, flags, seed) at fixed binary.
  The current three stage executables still receive master-derived seeds in
  their process arguments. On a multi-user Linux host those arguments may be
  visible through `ps` or `/proc`; run production only under a dedicated
  single-user account/host or with an administrator-enforced `hidepid` policy.
  The executables also print their received seed into the mode-600 per-stage
  logs. Treat `stage12.log`, `stage3.log`, `stage4.log`, `stage5.log`, and
  `stage6.log` as secret material alongside `<run>/SEED`; never attach them to
  a public issue or report without redacting the seed. A future file-descriptor
  seed interface and seed-redacted executable logging should replace these
  limitations.
- **State v2**: stages 4–5 write resume states; stage 5's resume of the
  ended split stage never re-arms it (the tri-state contract).
- **Provenance**: a pipeline output promoted into `circuits/` or the
  mixing challenge must get a `CIRCUIT_GENERATION_INFO` entry — record the
  run dir, n, binary commit, stage-5 knobs, and that the seed was generated
  randomly. **Never publish or copy the production seed value** into
  `CIRCUIT_GENERATION_INFO`, a report, chat, or another shared artifact; keep
  `<run>/SEED` as protected secret material.
- **Do not** run stage 3 with an empty store outside plumbing tests, and
  do not port the driver to zsh (word-splitting silently changes flag
  passing).
- Stage-5's numerical defaults reflect the X-panel results, but their
  deliverable-promotion status is still unresolved. Until that policy is
  explicitly recorded, treat any stage-5 output as calibration material, not
  a deliverable.
- **Stage-5 calibration objective** (2026-08-05): minimize the expansion
  factor `--xr` subject to decent ABSOLUTE spread — descendants per input
  gate and farthest-descendant distance in gates, not circuit fractions
  (`xpanel_spread.py` computes both from an arm's state file via the origin
  labels; merges attribute conservatively, the July convention). Runtime is
  expendable: if longer walks (or thermostat breathing) at a smaller target
  buy the same spread, prefer the smaller circuit — the linger-extension
  arms test exactly that trade.
