# Pipeline changes of 2026-11-01 (recipe v14)

This document is for everyone on the project who knows the October release (recipe v13, `main` at ec8474c4) and
needs to know what changed in this update, why, and how to get the old behaviour back. It covers every commit
between ec8474c4 and this push: the review fixes of 2026-10-28, the stage-2 interleaving work of 2026-10-28 to
2026-11-01, and the stage-1/2 update decided by RC on 2026-11-01. The pipeline itself is described from scratch in
[PIPELINE_INTRODUCTION.md](PIPELINE_INTRODUCTION.md); the settings are in [tdp_pipeline.md](tdp_pipeline.md).

## Summary

| Area | Recipe v13 (main ec8474c4) | Recipe v14 (this push) | Why (evidence) |
|---|---|---|---|
| Stage 1 shape | the sliced sandwich: slice blocks S1, S2 interleaved with C and D; the N column of n copy CNOTs | the same sandwich construction, updated: no slice blocks by default; the **V gadget** on the register (nonlinear copy in groups of three, V and V* of t = n gates beside D, register-only recovery) | degree-2 midpoint replica; dead-slice re-encodings; fire concurrency (sections 1.1-1.3) |
| Random C, D, V | fully random gate sequences | the same by default; `sandwich.layer = w` samples layers of w wire-disjoint gates | sandwich dependency width (section 1.4) |
| Stage-2 turnover | bank rotation | per-wire turnover (cohorts of 16) | removes bank limbo and the positional bank structure (section 2.1) |
| Stage-2 fire order | each fire contiguous | fires interleaved (window 4,096 gates), with read-only refresh, helper lanes and grouped migration | narrow-window exposure threefold lower (section 2.1) |
| Stage-2 warm-up | grows the growing half of the band only | also pre-grows the serving half before the first masks | half of the remaining recoveries sat on raw band-seed coordinates (section 2.2) |
| Stage-2 port guards | two junk-half guards: the reverse contract | by default no reverse contract: one opening guard over all data wires, no closing guard; band seed and reseed unchanged at both ends | RC, 2026-11-01 (section 2.3) |
| Stage 6 | payload contract checked in both directions | `tdp_adaptive --forward-only` checks the forward direction only; the driver passes it when the reverse contract is off | follows from 2.3 (section 3) |
| Recipe bookkeeping | recipe v13, stage-1/2 marker 8 | recipe v14, marker 9, config schema still 5; v13 runs are refused on resume | section 4 |

The default stage 2 is smaller than before: 400,935 gates at n = 64 and 1,146,183 at n = 128, against 454,272 and
1,285,912 for the v13-shaped (rust2-like) stage 2 built with the same seeds, i.e. 12 % and 11 % less
([notes/stage12_trials_20261101.md](notes/stage12_trials_20261101.md)). Stages 3 to 6 have not yet been run on
the v14 defaults.

## 1. Stage 1: the sandwich, updated

### 1.1 The V gadget (default t = n)

**What.** After C, its n output wires are split by a secret random permutation into groups of three (the last
group takes 3 to 5 wires). For a group (a; x_1 .. x_m) the copy XORs into the register wires H(·) the bijection
H(x_1) ^= mux(a; x_1, x_2), H(a) ^= mux(a; x_1, ¬x_2), H(x_i) ^= mux(a; x_i, x_π(i)), with mux(a; u, v) =
(¬a ∧ u) ⊕ (a ∧ v) and random literal polarities. Every register wire receives a balanced degree-2 function of
three of the group's values from two two-control conjunctions, so no wire ever holds a single copied value and the
copy uses no CNOT. V is t random g57 gates on the register, run beside D's first half; V* (V reversed) and then the
recovery run beside D's second half. The recovery acts on the register alone (one CNOT, per-wire polarity CNOTs,
controlled swaps on a) and inverts the copy on the honest slice, so the register ends with C(x). Code:
`v_gadget_sandwich`, `group_copy` in [src/stages/sandwich/construct.rs](../src/stages/sandwich/construct.rs).

**Why.** The preimage campaign found that the sandwich's mid-circuit copy of C survives the full pipeline at
degree 2: up to 50 of 64 output bits are sparse degree-2 functions of the live wires at a mid-circuit cut, 62 of 64
over two circuits ([notes/preimage_campaign_report_20261029.md](notes/preimage_campaign_report_20261029.md) §4).
RC observed the replicas returning after ANF packing and resynthesis even with the three-mask stage 2, which by
itself had shown no degree-2 replica at the scanned depths in the variants campaign
([notes/pipeline_variants_report_20261101.md](notes/pipeline_variants_report_20261101.md) §3.7). The interleaving
investigation proposed a secret bijection V on the register after the copy and its inverse before the end: the
register then holds V(copy(C(x))) through D, and V's gates run in parallel with D's at the midpoint where fires
are most pinned ([notes/stage2_interleaving_20261101.md](notes/stage2_interleaving_20261101.md), "Sandwich
redesign for gate mobility" item 5 and "The second half of the sandwich").

**Design decisions.** RC asked for a copy with no midway CNOTs. The first proposal, the pair copy c' = c + a(b+1) +
1, d' = d + b(a+1) + 1, cannot be undone from the register: it maps (a, b) = (0, 0) and (1, 1) to the same value,
and every invertible map on two bits is affine, so groups of three are the smallest nonlinear choice. The copy's
inverse is itself of degree 2 (a controlled permutation), so V must scramble the register for the gadget to remove
a degree-2 replica; RC set t = n as a low default (t = 1 was too optimistic) until the degree-2 midpoint search on
v14 circuits reports. The trials measured t = n log2 n / 2 and n log2 n too
([notes/stage12_trials_20261101.md](notes/stage12_trials_20261101.md)).

**Cost.** About 340 stage-2 gates per V gate; the copy and recovery add about 3 gates per output wire against 1
CNOT. At t = n the default stage 2 is still smaller than v13's because the slice blocks are gone (section 1.2).

### 1.2 Slice blocks optional, off by default

**What.** `sandwich.slicing = false` (default) leaves out S1 and S2; `true` draws them exactly as before.

**Why.** A dead S1 gate on a live C value re-encodes it in stage 2 and counts like an extra read: in rust2's
attribution the recovery rate of a C birth rises from 8 % with no such re-encoding to 38 % with three, and a
quarter of births have one ([notes/stage2_interleaving_20261101.md](notes/stage2_interleaving_20261101.md),
"Effect on exposure"). The slice blocks also cost 17-20 % of stage 2 and give back part of a layered C's width
(block 1 at w = 21: 23.0 -> 19.6) ([notes/stage12_trials_20261101.md](notes/stage12_trials_20261101.md)). Without
them, the reverse direction's junking of the low half is done by the stage-2 input guard. The price is the chaff
S1 provided among C's fires, which the interleaving note lists as unmeasured against the structural attacks.

### 1.3 Terminology

Stage 1 is still the sandwich: C, a copy of its outputs into the register, and a random decoy D, now without slice
blocks by default and with the V gadget on the register. gen's banner says "sandwich" and adds "with slice
blocks" only when they are on.

### 1.4 Layered sampling (`sandwich.layer`)

**What.** The random C (under `source.construction = "random"`), D and V are drawn `w` wire-disjoint gates at a
time; w = 1 (default) keeps the old draws exactly. A supplied or balanced C is not sampled gate by gate.

**Why.** The fires stage 2 can interleave are bounded by the sandwich's dependency width, 11 at n = 64 for the v13
sandwich ([notes/stage2_interleaving_20261101.md](notes/stage2_interleaving_20261101.md), "Sandwich redesign for
gate mobility" item 1). **Measured:** the width grows as intended (n = 64: 10.2 at w = 1 to 23.3 at w = 21; with V
at t = 384 up to 31.2), but the fires a 250-gate window of stage 2 holds stay at about 10 (n = 64) and 14 (n = 128)
because the helper lanes bind; with the lane cap lifted and a 16k interleave window, layering adds only about 1 to
1.5 fires per window ([notes/stage12_trials_20261101.md](notes/stage12_trials_20261101.md), high level item 2).
w stays 1 by default.

## 2. Stage 2: the three-mask compute

### 2.1 Wire turnover and fire interleaving (now the default)

**What.** `three_mask.turnover = "wire"`: band coordinates turn over per wire, promotion cohorts of 16 wires taking
the roles of the banks, and a drained wire grows again at once. `three_mask.interleave = 4096`: the finished
compute is reordered by a dependency-preserving random topological order with a 4,096-gate window. Interleaving
switches on three compute changes that removed the chaining between consecutive fires: the differential cubic
refresh writes no band coordinate (`emit_cubic_delta_readonly`, the same 15 gates), dirty helpers come from
per-fire lanes of a reserve that growth never writes (a quarter of the band, four wires per lane: 8 lanes at
n = 64, 16 at n = 128), and the carrier migration moves roles in one fire box per lane cycle. Function-preserving,
no gates added. Code: [src/stages/preprocessing/three_mask/](../src/stages/preprocessing/three_mask/); details in
[THREE_MASK_STAGE2.md](THREE_MASK_STAGE2.md), "Turnover modes and fire interleaving".

**Why.** The review of the three-mask prototype found that a whole fire (about 158 gates) factors as x_u·x_v inside
a 125-250-gate window, with interleaving as the zero-gate counter
([notes/three_mask_review_20261021.md](notes/three_mask_review_20261021.md), finding 5). Measured in the variants
campaign on fully mixed circuits: the narrow factor attack exposes 89 deeper C births instead of 254 (per
1000-gate window 81 instead of 231), the attacker's shortened circuit 129 instead of 251, and the per-read recovery
probability falls from 18 % to 7 %. Wire turnover alone (arm `rust`) changed nothing; interleaving (arm `rust2`) is
what matters ([notes/stage2_interleaving_20261101.md](notes/stage2_interleaving_20261101.md), "Effect on
exposure"; [notes/pipeline_variants_report_20261101.md](notes/pipeline_variants_report_20261101.md)).

**Cost.** None in gates; the migration's gates fall from 50k to 6.5k at n = 64; stage 6 compresses the interleaved
circuit less (0.866 vs 0.778).

### 2.2 Pre-growth of the whole band (`three_mask.pregrow`, default on)

**What.** Before the first mask references the band, every serving wire is rewritten `warmup` times (one
two-control band gate per serving wire per sweep: 256 gates at n = 64, 512 at n = 128); the warm-up then grows the
growing half as before.

**Why.** Half of rust2's remaining recoveries are in the first tenth of the circuit, where the initial serving
banks still carry the raw band seed for 100 to 190 fires
([notes/stage2_interleaving_20261101.md](notes/stage2_interleaving_20261101.md), high level item 4 and
clear-cut suggestion 2). Its effect is being measured by the paired arms `s2v_n64_w1_t384` and
`s2v_n64_w1_t384_nopre` of the trial battery.

### 2.3 The reverse contract becomes optional (default off)

**What.** `preprocessing.reverse_contract = false` (the default in the guarded modes): the opening guard covers all
2n data wires with the same gate count (its Feistel rounds alternate halves, so each round's uncompute stays
exact) and there is no closing guard, so only the forward payload is kept; the band seed and reseed stay at both
ends. `true` keeps the two junk-half guards: the reversed circuit carries (a bijection of) D⁻¹ of its input on the
register, and gen verifies it. nonlinear291 has no guards and keeps its behaviour. Code:
`preprocess_sandwich_ports` in [src/stages/preprocessing/construct.rs](../src/stages/preprocessing/construct.rs),
`slice_zero_feistel_guard_dims_all` in
[src/stages/preprocessing/slice_guards.rs](../src/stages/preprocessing/slice_guards.rs).

**Why.** RC's decision of 2026-11-01. The interleaving note had found no forward downside to dropping the output
guard with the reverse contract
([notes/stage2_interleaving_20261101.md](notes/stage2_interleaving_20261101.md), clear-cut suggestion 3 and "The
second half of the sandwich"); the band reseed is kept on RC's instruction.

### 2.4 Review fixes of 2026-10-28

RC's code review of the October release found four real defects, fixed in bb67e0dc and f701e3c1: the packer's
oracle `*.log` fixtures were git-ignored (CI on a clean checkout); the auto band now rounds to a multiple of 32 (an
odd n gave 9 banks) and the resolved geometry is validated up front; a three-mask stage 2 wider than 1,024 wires is
refused when stage 6 runs; the adaptive census batches and term cap are checked against the stage-2 width.

## 3. Stage 6: `--forward-only`

`tdp_adaptive run --forward-only` checks the sandwich payload contract (on the input before round 1 and on the
published circuit) in the forward direction only and records `check.payload_directions=forward` in
`adaptive.report`. The managed driver passes it whenever stage 2 keeps no reverse contract, i.e. by default;
nothing else in stage 6 changed. Code: `payload_contract_dirs` in
[src/stages/adaptive/driver.rs](../src/stages/adaptive/driver.rs).

## 4. The managed driver (recipe v14)

- New settings: `[sandwich]` `slicing` (false), `layer` (1), `v_gates` (default n, 0 = the N column);
  `preprocessing.reverse_contract` (false in the guarded modes); `three_mask.pregrow` (true); the
  `three_mask.turnover` and `three_mask.interleave` defaults become `"wire"` and 4096. Script flags
  `--sandwich-slicing`, `--sandwich-layer`, `--sandwich-v-gates`, `--preprocessing-reverse-contract`,
  `--three-mask-pregrow`; gen variables `SANDWICH_SLICING`, `SANDWICH_LAYER`, `SANDWICH_V_GATES`,
  `STAGE2_REVERSE_CONTRACT`, `THREE_MASK_PREGROW`, each passed only when it differs from gen's default.
- `tdp_command_recipe=14` and `tdp_stage12_recipe=9`; the config schema stays 5, so an existing v5 file without
  the new keys now gets the v14 stages 1-2. A v13 run directory is refused on resume, with a message naming the
  settings that rebuild v13's stages 1-2.
- The validated-chain pin (`val_chain-L8/A-20261007`) now also requires the legacy stage 1 and the reverse
  contract.
- nonlinear291 takes the V gadget only with slicing (its templates cover the zero-preserving copy only).

To build v13's stages 1-2 in a fresh run:

```toml
[sandwich]
slicing = true            # the slice blocks
v_gates = 0               # the N column instead of the V gadget
[preprocessing]
reverse_contract = true   # both junk-half guards
[three_mask]              # three-mask mode only
turnover = "bank"
interleave = 0
pregrow = false
```

This reproduces the old circuits byte for byte (`stage2_three_mask_parity`, `stage2_typed_params` and
`tests/shell/tdp_stage12_parity.bash` pin it).

## 5. Measurements and open questions

- **Structure** of the 51 trial arms (n = 64 and 128, w from 1 to n/3 in steps of 5, t = 1, n log2 n / 2, n log2 n,
  slicing on for four arms, one arm without pre-growth, two with a 256-wire band, two v13-shaped baselines):
  [notes/stage12_trials_20261101.md](notes/stage12_trials_20261101.md).
- **Exposure:** the stage-2 attack battery (validated census, the 11-case narrow portfolio, per-read attribution)
  is running on the server; the first 125/250-gate ledgers do not separate the arms yet (same note, section (d)).
- **To decide later:** the V size once the degree-2 midpoint search on v14 circuits (after stages 3-6) reports;
  whether to lift the 64-wire helper-reserve cap and widen the interleave window, which is what would let stage 2
  use the wider sandwiches (lanes experiment in the trial note); whether the chaff of the slice blocks matters to
  the structural attacks.
- **Related:** inversion attempts on the plain rust2 stage-2 output
  ([notes/stage2_inversion_attempts_20261101.md](notes/stage2_inversion_attempts_20261101.md)).

## 6. Documents added in this push

Campaign and analysis notes that had only lived in a working copy, now under `docs/notes/`:
`anf_compress_survey_20261014.md`, `compress_iter_20261014.md` (+ figures and tables),
`c_design_report_20261026/` (report and the scripts behind it), `campaign_summary_20261026/` (PDF and TeX),
`pipeline_variants_report_20261101.{md,tex,pdf}`, `pipeline_variants_status.{md,html}` (the campaign status table
as of this push), `preimage_campaign_report_20261029.md`, `preimage_summary_20261026.{md,tex,pdf}`,
`stage2_inversion_attempts_20261101.md`, `stage12_trials_20261101.md`, and the current versions of
`c_design_exposure_20261021.md` and `pipeline_variants_20261026.md`. New documents: this one, and the updated
[PIPELINE_INTRODUCTION.md](PIPELINE_INTRODUCTION.md); [PIPELINE_V2_CHANGES.md](PIPELINE_V2_CHANGES.md) keeps the
recipe v13 changes and a short recipe v14 section.
