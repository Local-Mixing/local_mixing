# The TDP pipeline, step by step (v15)

This document walks through the six stages of the TDP pipeline as it stands after
2026-11-02 ("v15"), one step at a time. Each stage lists what it does to the circuit,
in order, with the sizes of one complete run: the n = 128 pipeline run of
2026-11-02 (`port_runs/tdp128_s6`, delivered as `circuits/tdp128_s6_20261102.mpmct1`).

**What v15 is.** Recipe v15 (`tdp_command_recipe=15`) is the managed driver's
default (`local_mixing_bin tdp_gen`, `configs/tdp.toml`) since 2026-11-02. Against
recipe v14 (`main` at `2b4b6374`) it changes three stages:
- stage 2: the opening guard is the two-layer guard keyed on the register and the band
  (§2, step 2), and the band seed and reseed are random g57 gates (steps 3 and 5);
- stage 3: wire-swap twists at rate 0.001 over spans of at least 100,000 gates;
- stage 6: the redesigned adaptive stage (`docs/notes/stage6_config_search_20261102.md`),
  with the leak guard on every block acceptance, every cleanup, phase C and the
  canonical order.

The changes and how they were chosen: `docs/PIPELINE_V15_CHANGES.md`. Every default:
the table at the end.

**About the worked example.** The n = 128 run below was made on 2026-11-02 before the
stage-2 change: its stage 2 opened with recipe v14's band-keyed guard (2,560 gates) and
seeded the band with about 2,860 gates at each end. In v15 the guard is 9,380 gates
and each band seed 7,168 at n = 128, so a v15 stage 2 is about 15,400 gates (1.4 %)
larger; the compute is unchanged. Its stage 6 used a bounded synthesizer effort (2
restarts, 12 s budgets) and 16 census batches, which is what the v15 census batch rule
gives at n = 128, and took 4.0 h, 2.8 h of it in the census. With the default effort
(16 restarts, 60 s budgets) stage 6 at n = 128 is estimated at about 7 h on this Mac
(about 2.8 h census, 3 h synthesis, 0.7 h cleanups), the whole pipeline at about 9 h;
at n = 64 about 2-2.5 h in all (estimates; the default effort has not been run at
n = 128).

**Conventions.** C is the secret source circuit on n wires. A *g57 gate* is
`a ^= b OR NOT c`. The *honest slice*: inputs x on wires 0..n-1, every other wire
zero. Every circuit the pipeline produces computes, on the honest slice, C(x) on
wires n..2n-1. Gate counts are raw counts unless a step says otherwise; a gate
with k controls costs 4k - 8 two-control gates when lowered (k > 2).

| stage | binary | what it produces | n = 128 example | time (Mac, 2026-11-02) |
|---|---|---|---|---|
| 1 sandwich | `gen_sandwich_gadget` | C, copy, D, V gadget on 2n wires | 6,761 gates, 256 wires | stages 1-2 together: a few seconds |
| 2 three-mask preprocessing | (same process) | the masked circuit on 4n wires | 1,141,924 gates, 512 wires (v14 guard and seeds; v15: about +15,400) | (with stage 1) |
| 3 database mixing | `circuit_mixer` | DB-respelled, ~2x | 2,566,339 | 58 min |
| 4 splitting | `circuit_mixer` | complemented gates split, ~1.9x | 4,739,227 | 5 min |
| 5 crossing walk | `circuit_mixer` | gates crossed, wide gates, ~1.85x | 8,746,933 | 39 min |
| 6 delivery | `tdp_adaptive` | **the published circuit**, every gate <= 2 controls | **4,427,214** | 4.0 h (12 threads) |

---

## Stage 1: the sandwich (2n wires)

Low half = wires 0..n-1, register = wires n..2n-1.

**1. Draw C.** A random g57 circuit on the low half: each gate a uniform ordered
triple of distinct wires, no two adjacent gates equal; |C| = 1,000 at n = 64,
3,000 at n = 128.

**2. Draw D.** An independent random g57 circuit of the same size and design on the
low half. After the copy it turns the low half into junk, so C's input does not
survive to the output.

**3. Copy C's outputs into the register** (the V gadget's copy). C's n outputs are
split by a random permutation into groups of three (the last group 3-5). Each
register wire receives a balanced degree-2 function of three of its group's values,
built from two two-control conjunctions, so no register wire is ever a single C
output. The copy gates float backward into C's tail as far as commutation allows.

**4. V beside D, then V\* beside D.** V is t = n random gates on the register, run
interleaved with the first half of D; V\* (V reversed) runs with D's second half.

**5. Recovery.** Register-only gates (a CNOT, polarity CNOTs, controlled swaps per
group) undo the copy's bijection, leaving C(x) on the register.

Result on the honest slice: (x, 0) -> (junk, C(x)). n = 128: 6,761 gates = C 3,000 +
copy 256 (42 groups) + V 128 + V\* 128 + recovery 249 + D 3,000. Options, off by
default: slice blocks S1/S2 (`sandwich.slicing`), layered sampling of C, D and V
(`sandwich.layer`, default 1 = one gate at a time).

---

## Stage 2: three-mask preprocessing (4n wires)

The frame: **opening guard ; band seed ; masked compute ; band reseed.**

**1. Widen.** The sandwich's 2n data wires plus a **band** of 2n extra wires (256 at
n = 128), 4n wires in all. The band is zero on the honest slice.

**2. Opening guard** (the two-layer guard, RC 2026-11-02). Write A for wires 0..n-1
(the input x), B for wires n..2n-1 (the register) and C, D for the band. The guard is
the identity on A and B exactly when B and the band are zero (the honest slice). It
has two blocks:
- **Block 1 randomizes B (and A) whenever the band is nonzero.**
  - *Key spreading, universal-hashing style:* 3·⌈log2 2n⌉ rounds of CNOTs between
    disjoint random pairs of band wires (24 rounds, 3,072 gates at n = 128). This is a
    random linear map on the band: positive controls only, so it does nothing to the
    zero band, and it sends a key with a single nonzero bit to about half of the band.
    It is never undone (the band is junk).
  - *Three-control gates* t ^= s·u·v, with s a band key wire, t in B (2·n·⌈log2 n⌉ =
    1,792 at n = 128) or in A (n·⌈log2 n⌉ = 896), and u, v data wires of either
    polarity. Each is emitted as h ^= u·v; t ^= h·s, with h one of a few band helper
    wires (n/8 of them, 16 at n = 128), not restored. A helper is therefore never a key
    afterwards, and a short prologue first makes each helper wire a key once (one gate
    into B, one into A).
  - *Coverage:* for every one-hot band key and whatever B holds, every B bit must be
    liable to flip. Each key is checked on 256 random inputs, and a missed B wire gets
    one more gate keyed on a wire the spread key sets (a handful of gates in all).
- **Block 2 randomizes A whenever B is nonzero:** on every B wire b a partition triple
  t1 ^= b·u·v, t2 ^= b·u·¬v, t3 ^= b·¬u on three different A wires (u, v two other A
  wires). For a one-hot B key exactly one of the three fires on every input, so one A
  bit flips whatever A holds, and which one depends on A; C's ripple then changes every
  output bit. The two three-control gates are emitted as h ^= u·v; t ^= h·b; h ^= u·v
  with the helper h another B wire, restored, since B carries the payload: 7 gates per B
  wire (896 at n = 128).

No gate has more than two controls. n = 128: 9,380 gates (block 1: 3,072 + 32 +
5,376, block 2: 896, plus 4 repair gates); n = 64: 4,120. Measured on 256 random
inputs per key: a one-hot band key changes all 128 B wires (and 127-128 of the A
wires), a one-hot B key flips one A bit on every input (one of 3 wires). Every draw
is checked (exhaustively up to 20 wires, by sampling above: the slice exact, the
one-hot coverage, 512 weight-2 and 512 random keys disturbing A) and redrawn if it
fails. v15 keeps no reverse contract, so there is
no closing guard. The guard is not a setting: recipe v14's guard was keyed on the band
only, so with the band zero every value of B passed it and the circuit evaluated the
sandwich off its slice.

**3. Band seed.** 8·n·⌈log2 n⌉ random g57 gates (7,168 at n = 128, 3,072 at n = 64),
each with its target a band wire and its two controls any two other wires, data or
band. The band becomes a pseudorandom-like function of the data, F(A, B) = (A,
R(A, B)), so masks are not constants on the honest slice. One exclusion: no gate's two
controls are a pair of input wires that a first-layer gate of the sandwich reads raw
(that closed the first-layer "fringe" leak). The seed is redrawn until every band wire
comes out non-constant on the honest slice, behind the guard. **Pre-growth** then
rewrites the band coordinates that will serve as mask sources before any mask
references them.

**4. Masked compute.** Every sandwich gate becomes a **fire**:
- each data value is carried as d + M1 + M2 + M3: a physical carrier plus three
  **cubic masks**, each the product of three band coordinates (a mask is a stored
  triple of band indices with a unique label, not a register);
- a logical gate x_t ^= x_u x_v is computed without decoding its controls:
  d_t ^= (d_u + S_u)(d_v + S_v) expands into at most 16 products of up to six
  controls, each emitted as a chain of two-control gates on borrowed helper wires
  (from per-fire lanes of a reserve);
- per fire: a temporary cover mask goes on the target, the product pieces run in
  shuffled order, the target's oldest mask is replaced by a fresh one (a read-only
  cubic refresh, 15 gates), the cover comes off, and growth gates rewrite the
  growing part of the band;
- band **turnover** per wire, in cohorts of 16: serving (supplies mask coordinates,
  never written) -> retiring (drained of references) -> growing (rewritten), so masks
  are a renewable supply;
- **persistent migration** moves data roles across physical carrier wires;
- at the end of the compute every mask is XORed out;
- **interleaving**: the finished compute is reordered randomly, respecting
  dependencies, with no gate more than 4,096 positions ahead of the earliest waiting
  gate, so a 250-gate window holds pieces of ~13 fires (n = 128).

**5. Band reseed and check.** The band is reseeded at the output end by an
independent draw of the same 8·n·⌈log2 n⌉ random g57 gates (7,168 at n = 128), and the
circuit is checked on bit-sliced samples of the honest slice: on (x, 0, 0) wires
n..2n-1 hold C(x).

n = 128 (the example run, v14 guard and seeds): 1,141,924 gates on 512 wires, every
gate at most two controls; the compute is 1,133,642 of them (730,606 fire + 349,419
mask + 45,568 growth + 8,049 migration gates), the rest the guard, the seed and the
reseed. A v15 stage 2 at n = 128 has the same compute and outer blocks of about
23,700 gates: guard 9,380, seed 7,168, reseed 7,168 (a fresh run: 1,159,814 gates,
compute 1,136,098).

---

## Stage 3: database mixing

`circuit_mixer --tdp --db-mixing` against the frozen store `frozen_m1_m11` (small
g57 circuits indexed by the canonical form of their function). Each move:

**1. Sample a window** of at most 12 gates, contiguous or convex (commuting gates
moved aside).

**2. Look it up**: compute the canonical key of the window's function and find the
stored spellings of that function.

**3. Splice** a randomly chosen stored spelling in place of the window, verified
exhaustively, and push the new gates outward through the gates they commute with,
so that later replacements overlap earlier ones.

**4. Steer the size.** A size controller alternates growing (MIX) and shrinking
(COMP) rounds to follow the profile `2,2,2,2,2`: grow to 2x the input, then hold.

**5. Wire-swap twists.** With probability 0.001 per round, two wires swap labels
over a window drawn uniformly between 100,000 gates and the whole circuit, with
brackets at both ends.

A global sampled equality check runs every 10,000 moves; the budget is 5.2 x the
input's gate count in moves. n = 128: 1,141,924 -> 2,566,339 gates, mostly in g57
form.

---

## Stage 4: splitting

`circuit_mixer --split --split-stop`. Each move:

**1. Split** a complemented g57 gate `a ^= b OR NOT c` exactly into the two plain
conjunctions `a ^= b` and `a ^= NOT b AND NOT c`.

**2. Join** (probability 0.8): the one-control piece is joined to a distant bracket
on the same wire by an absorbed NOT twist (both brackets flip polarity, every read
of the wire in between is complemented; no extra gates).

**3. One crossing shot** (as in stage 5, step 1-2) ends the move.

The stage runs until no complemented gate is left (k-max 12) and writes
`split.state`, where stage 5 resumes. n = 128: 2,566,339 -> 4,739,227 gates (x1.85).

---

## Stage 5: the crossing walk

`circuit_mixer` resuming `split.state`. Each move:

**1. Shoot** a random gate in its stored direction until it meets its first
collider (the first gate it does not commute with).

**2. Cross** the two gates by one of three exact case-split rules; the residue terms
can carry more controls than either gate.

**3. Penalize width**: a split of width c is accepted with probability 3^-(c-1).

**4. Hold the size.** A thermostat keeps the size near 2x the input (temperature
max(64, target / 25)), contracting by exact undo of journaled crossings and by
pairwise merges.

The budget is 3 x target moves. n = 128: 4,739,227 -> 8,746,933 gates, with gates
of up to 12 controls (5.27M of them wide).

---

## Stage 6: delivery (v15)

The managed driver runs `tdp_adaptive run ... --stop-fraction 0.2 --pre-cleanup
--cost two-control --descent 128,32,16 --descent-stop-fraction 0.2 --lower-controls 2
--max-block-gates 512 --census-batches auto --forward-only` (`tdp_adaptive`'s own
defaults are recipe v14's stage 6, so the v15 settings are passed explicitly);
everything else is at its default: 2 synthesizer seeds per block, 16 restarts, 60 s
budgets, 10 fcompress iterations per cleanup, no round cap. `auto` census batches are
the **census batch rule**, ⌈(W + 512) / 64⌉ batches of 64 honest inputs for a W-wire
circuit (16 at n = 128, 12 at n = 64): an exact exposure fits every sample, so fewer
samples never miss one, and with a 512-sample margin over the width a value that is
not affine fits them by chance with probability about 2^-511; every hit is re-tested
on fresh and structured inputs anyway. The census time grows with the batch count
(the former default of 96 batches would make it 6 times longer at n = 128). The n = 128
example used the bounded effort `--restarts 2 --time-limit 12` (and 16 census batches,
the rule's value); the default effort compresses about 6 % further at about 3.4x the
synthesis time (measured at n = 64).

**1. Pre-cleanup** (one pass, exact)
- **NOT-push:** absorb the interior NOT gates into the polarity of the downstream
  controls, and drop gates that never fire.
- **fcompress:** merge and cancel nearby gates on the same target wire, sliding them
  past each other where that is exact, for 10 iterations.
- **Equivalence check** on random full states.
- **Leak guard:** census before and after; the cleanup is rejected if it exposes
  more of C (deeper values, affinely or directly, or C output bits before 90 % depth).
- n = 128: 8,746,933 -> 3,332,296 gates. This single step removes most of the
  crossing walk's redundancy.

**2. Phase A: ANF resynthesis rounds at block cap 512.** Each round:
1. **Census:** which of C's values are recoverable, directly or affinely, at which
   cuts.
2. **Plan:** cut the circuit into blocks of up to 512 gates, with boundaries placed
   away from exposed cuts.
3. **Synthesize** each block from its ANF (2 seeds per block), with output gates of
   at most 2 controls, each block checked against the original.
4. **Refine:** blocks that fail to synthesize are split in two and retried, twice at
   most.
5. **Accept, per block:** keep the synthesis only if it is **smaller in 2-control
   cost** and **not leakier** ("leakier" counts C's deeper values plus any C output
   bit exposed before 90 % depth); otherwise the original block is kept.
6. **Assemble** the blocks and verify the round's output equivalent to its input.

Rounds repeat until two consecutive rounds compress jointly by less than 20 %. n =
128: 3 rounds, 2-control cost 8,532,324 -> 5,246,403.

**3. Phase B: the descent.** The same rounds at shrinking block caps, 128, then 32,
then 16. These blocks are planned without looking at C, but acceptance still applies
the leak guard. Each cap runs to the same stall rule, and after each one comes a
cleanup (as in step 1, guarded, kept only if smaller). n = 128: 2 rounds per cap,
ending at 4,401,829 gates.

**4. Phase C: lowering to <= 2 controls.**
1. **Lower:** rewrite every remaining wide gate exactly as a chain of 2-control
   gates (Barenco, with borrowed helper wires that are restored).
2. **Clean:** NOT-push and fcompress.
3. **Lower again**, because fcompress can re-widen gates.
4. **Hard check** that every gate has at most 2 controls, then verify equivalence.
5. **Leak guard:** census before and after. If the result exposes more (deeper
   births or early C outputs), retry the lowering with random helper wires (up to
   8 times, NOT-push only) and publish the first that exposes no more; if none
   does, publish the least exposed one and report `status=leakier_lowering` (not
   recommended).

n = 128: 7,065 wide gates were left by this point (most had been resynthesized
away); the result was 4,427,214 gates.

**5. Finish**
- **Canonical gate order:** reorder commuting gates into a canonical order, kept
  only if it exposes no more than the circuit it reorders: deeper births (affine
  or direct) and early C outputs.
- **Zero slice** (sanity check): x on the input wires, zeros elsewhere, gives C(x)
  on the output wires.
- **Payload contract** (sanity check) against the sandwich (forward only: the v15
  stage 2 has no reverse contract).
- **Final census and report**, including the early-output count.

n = 128 result: 4,427,214 gates on 512 wires, every gate at most 2 controls; 0 deeper
C values exposed in the census, 0 early C-output copies.

---

## Default parameters (recipe v15)

Every value below is what `tdp_gen` resolves with no setting (`configs/tdp.toml`
leaves them commented; the manifest `tdp_command.conf` of a run records them). The
TOML key and the script flag of each setting: `docs/tdp_pipeline.md`.

**Run and source**

| setting | default |
|---|---|
| `source.wires` (n) | 128 in `configs/tdp.toml`; 3..4095, at most 256 when stage 6 runs |
| `source.gates` (\|C\| = \|D\|) | 1,000 at n = 64, 3,000 at n = 128 (rule `validated`; other n by formula, marked unvalidated) |
| `source.construction` | `random` (uniform g57 gates; `balanced` = every internal value read exactly twice) |
| `run.skip_stages` | none: all six stages run |
| seeds | fresh random 63-bit seeds, kept in the private `SEEDS` file (format 3) |
| `[leakage_repair]` | off |

**Stage 1 (sandwich)**

| setting | default |
|---|---|
| `sandwich.slicing` | false (no slice blocks S1/S2) |
| `sandwich.layer` | 1 (one gate at a time) |
| `sandwich.v_gates` (t) | n (128 at n = 128; 0 = the N column of copy CNOTs) |
| sandwich variant | classic |

**Stage 2 (three-mask preprocessing)**

| setting | default |
|---|---|
| `preprocessing.mode` | `three-mask` (the compute between the guard and band seeds below) |
| `preprocessing.reverse_contract` | false (no closing guard; forward payload only) |
| opening guard | the two-layer guard (not a setting): block 1 = band spread + 3·n·⌈log2 n⌉ three-control gates keyed on the band (2/3 into B, 1/3 into A), block 2 = a partition triple keyed on every B wire (7n gates); 9,380 gates at n = 128 |
| band seed, band reseed | 8·n·⌈log2 n⌉ random g57 gates each (not a setting): 7,168 at n = 128, 3,072 at n = 64 |
| `three_mask.band_wires` | 0 = auto: the larger of 2n and 128, a multiple of 32 (256 at n = 128) |
| `three_mask.banks` | 0 (auto; must be 0 under wire turnover) |
| `three_mask.turnover` | `wire` (cohorts of 16 wires) |
| `three_mask.interleave` | 4,096 (the reorder window, gates) |
| `three_mask.pregrow` | true |
| `three_mask.retire_budget`, `max_retiring`, `rotation`, `maturity`, `visit_sweeps` | 2, 0, 1, 2, 2 |
| `three_mask.growth`, `warmup`, `segments`, `mask_count` | 8, 4, 6, 3 |
| `three_mask.random_helpers`, `unique_cover` | true, true |

**Stage 3 (database mixing)**

| setting | default |
|---|---|
| `database.regular_dir` | the frozen store `frozen_m1_m11` (required; set in the local config or `FROZEN_DB_DIR`) |
| `db_mixing.schedule` | `validated`: profile 2,2,2,2,2, moves int(G · 2 · 2 · 1.3) = 5.2 G for a stage-2 circuit of G gates |
| `db_mixing.twist_rate` | 0.001 per mixer round |
| `db_mixing.twist_min_span` | 100,000 gates |
| `db_mixing.twist_spans` | `uniform` (between the minimum and the circuit size, every window inside the circuit) |
| verification / report interval | every 10,000 / 50,000 moves |
| `parallel.pieces` | 1 (serial) |
| caches | lookup 2,048 MB, canon 256 MB, xpoly 1,024 MB |

**Stage 4 (splitting)**: no settings; `circuit_mixer --split --split-stop --p-join 0.8
--split-reach-k 2 --split-fail-limit 100 --split-canaries 256 --k-max 12` with the
DB, twists and compression off.

**Stage 5 (crossing walk)**

| setting | default |
|---|---|
| `crossing.target_size_factor` | 2 (target = 2 x the stage-4 size) |
| `crossing.width_penalty_base`, `width_penalty_threshold` | 3, 1 (a split of width c accepted with probability 3^-(c-1)) |
| `crossing.size_tolerance_divisor` | 25 (temperature max(64, target / 25)) |
| `crossing.move_budget_factor` | 3 (budget = 3 x target moves) |

**Stage 6 (delivery, adaptive mode)**

| setting | default |
|---|---|
| `delivery.mode` | `adaptive` |
| `adaptive.pre_cleanup` | true |
| `adaptive.cost` | `two-control` (a gate with k > 2 controls costs 4k - 8) |
| `adaptive.max_block_gates` | 512 (phase A) |
| `adaptive.stop_fraction` | 0.2 (phase A's two-round stop rule) |
| `adaptive.descent` | 128,32,16 (phase B block caps) |
| `adaptive.descent_stop_fraction` | 0.2 |
| `adaptive.cleanup_fcompress_iters` | 10 |
| `adaptive.lower_controls` | 2 (phase C: every published gate has at most 2 controls) |
| `adaptive.max_rounds` | 0 (until the plateaus) |
| `adaptive.seeds_per_block` | 2 |
| `adaptive.time_limit`, `restarts`, `beam` | 60 s per block of at most 1,000 gates (x2 up to 4,000, x4 above), 16, 6 |
| `adaptive.term_cap` | 131,072 ANF terms |
| `adaptive.segments`, `avoid_radius` | 16, 64 |
| `adaptive.refinement_levels`, `min_refine_gates` | 2, 128 |
| `adaptive.census_batches` | auto: ⌈(W + 512) / 64⌉ for the W-wire stage-2 circuit, 16 at n = 128 (1,024 honest inputs), 12 at n = 64; an integer overrides it |
| `adaptive.accept` | `smaller-not-leakier` (the leak guard) |
| `adaptive.canonical_order` | true |

The windowed mode's settings (`[packing]`, `[delivery]`) apply only with
`delivery.mode = "windowed"`.

---

## Correctness checks along the way

Every stage is function-preserving on all 4n wires after stage 2 (stage 2 itself is
checked on the honest slice): stage 2 verifies its payload on samples of the honest
slice; stages 3-5
verify every DB splice exhaustively and run a global sampled check every 10,000
moves; stage 6 checks every block, every round, every cleanup and phase C against
their inputs on random full states. The zero-slice and payload-contract checks at
the end (and on the stage-6 input, before round 1) compare against the original
specification, C and the sandwich, rather than the previous step. Separately from
equivalence, stage 6's leak guard censuses both sides of every block acceptance,
cleanup, phase C and the canonical order, and refuses (or, for phase C, retries)
any step that exposes more of C.

## What is published and what stays secret

Published: the stage-6 output (`adaptive/final.mpmct1`). Secret, kept in the run
directory and `~/.ran-private`: C, D, the sandwich, every intermediate stage output
and every seed.
