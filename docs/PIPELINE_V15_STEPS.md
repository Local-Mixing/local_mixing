# The TDP pipeline, step by step (v15)

This document walks through the six stages of the TDP pipeline as it stands after
2026-11-02 ("v15"), one step at a time. Each stage lists what it does to the circuit,
in order, with the sizes of one complete run: the n = 128 pipeline run of
2026-11-02 (`port_runs/tdp128_s6`, delivered as `circuits/tdp128_s6_20261102.mpmct1`).

**What v15 is.** Stages 1-5 are the recipe-v14 defaults (pushed to `main` at
`2b4b6374`) with the stage-3 twist settings of 2026-11-01; stage 6 is the redesigned
adaptive stage of 2026-11-02 with its recommended settings
(`docs/notes/stage6_config_search_20261102.md`). The code is on branch
`ran/stage6-deep`. **Status:** the managed driver (`local_mixing_bin tdp_gen`) runs
v15's stages 1-5 by default, but its default stage 6 is still v14's (large blocks,
raw-gate acceptance); v15's stage 6 is selected with `tdp_adaptive` flags (given in
§6) until the recipe is bumped and the flags are wired into the driver.

**Conventions.** C is the secret source circuit on n wires. A *g57 gate* is
`a ^= b OR NOT c`. The *honest slice*: inputs x on wires 0..n-1, every other wire
zero. Every circuit the pipeline produces computes, on the honest slice, C(x) on
wires n..2n-1. Gate counts are raw counts unless a step says otherwise; a gate
with k controls costs 4k - 8 two-control gates when lowered (k > 2).

| stage | binary | what it produces | n = 128 example | time (Mac, 2026-11-02) |
|---|---|---|---|---|
| 1 sandwich | `gen_sandwich_gadget` | C, copy, D, V gadget on 2n wires | 6,761 gates, 256 wires | stages 1-2 together: < 2 min |
| 2 three-mask preprocessing | (same process) | the masked circuit on 4n wires | 1,141,924 gates, 512 wires | (with stage 1) |
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

**2. Opening guard.** A Feistel-style block of 20n gates (2,560 at n = 128) over all
2n data wires that is the identity exactly when the band is zero. v15 keeps no
reverse contract: the guard covers all data wires and there is no closing guard.

**3. Band seed.** Fill the band from the input x, so that masks are not constants on
the honest slice; **pre-growth** then rewrites the band coordinates that will serve
as mask sources before any mask references them.

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

**5. Band reseed and check.** The band is reseeded at the output end (kept at both
ends), and the circuit is checked on bit-sliced samples: on (x, 0, 0) wires n..2n-1
hold C(x).

n = 128: 1,141,924 gates on 512 wires, every gate at most two controls; the compute
is 1,133,642 of them (730,606 fire + 349,419 mask + 45,568 growth + 8,049 migration
gates), the rest the guard, the seed and the reseed.

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

`tdp_adaptive run ... --pre-cleanup --cost two-control --max-block-gates 512
--descent 128,32,16 --seeds-per-block 2 --lower-controls 2 --cleanup-fcompress-iters 10
--stop-fraction 0.2 --descent-stop-fraction 0.2 --max-rounds 0 --forward-only`
(the n = 128 run also used `--restarts 2 --time-limit 12 --census-batches 16`; the
production synthesizer settings, 16 restarts and 60 s budgets, compress about 6 %
further at about 3.4x the time).

**1. Pre-cleanup** (one pass, exact)
- **NOT-push:** absorb the interior NOT gates into the polarity of the downstream
  controls, and drop gates that never fire.
- **fcompress:** merge and cancel nearby gates on the same target wire, sliding them
  past each other where that is exact, for 10 iterations.
- **Equivalence check** on random full states.
- **Leak guard:** census before and after; the cleanup is rejected if it exposes
  more of C.
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

n = 128: 7,065 wide gates were left by this point (most had been resynthesized
away); the result was 4,427,214 gates.

**5. Finish**
- **Canonical gate order:** reorder commuting gates into a canonical order, kept
  only if it does not increase exposure.
- **Zero slice** (sanity check): x on the input wires, zeros elsewhere, gives C(x)
  on the output wires.
- **Payload contract** (sanity check) against the sandwich (forward only: the v15
  stage 2 has no reverse contract).
- **Final census and report**, including the early-output count.

n = 128 result: 4,427,214 gates on 512 wires, every gate at most 2 controls; 0 deeper
C values exposed in the census, 0 early C-output copies.

---

## Correctness checks along the way

Every stage is function-preserving on all 4n wires after stage 2 (stage 2 itself is
checked on the honest slice): stage 2 verifies its payload on samples; stages 3-5
verify every DB splice exhaustively and run a global sampled check every 10,000
moves; stage 6 checks every block, every round, every cleanup and phase C against
their inputs on random full states. The zero-slice and payload-contract checks at
the end (and on the stage-6 input, before round 1) compare against the original
specification, C and the sandwich, rather than the previous step.

## What is published and what stays secret

Published: the stage-6 output (`adaptive/final.mpmct1`). Secret, kept in the run
directory and `~/.ran-private`: C, D, the sandwich, every intermediate stage output
and every seed.
