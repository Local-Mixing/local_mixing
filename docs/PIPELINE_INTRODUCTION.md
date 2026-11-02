# The TDP Pipeline: An Introduction

*Release of 2026-11-02 (recipe schema `config_version = 5`, saved recipe
v15; the changes since recipe v14 are in `docs/PIPELINE_V15_CHANGES.md`, those
since the October release of recipe v13 in `docs/PIPELINE_V14_CHANGES.md`;
step by step with every default: `docs/PIPELINE_V15_STEPS.md`): stage 1 is the
sandwich without slice blocks, with the V gadget on its register; stage 2 is
the three-mask compute with wire turnover, fire interleaving and pre-growth,
by default without the reverse contract and with the two-layer opening guard;
stage 3 twists twice as often over spans of at least 100,000 gates; stage 6
cleans, resynthesizes in small blocks judged by the 2-control cost and
publishes a circuit whose gates all have at most two controls. This is the entry point for a reader who has not seen the
pipeline before. The operational manual (`docs/tdp_pipeline.md`), the full
reference (`docs/TDP_PIPELINE_REFERENCE.md`) and the design notes listed in
section 6 hold the details and the evidence. Every number below is taken from
one of those documents; the campaign figures were measured on the server
campaigns of October 2026 at n = 64, |C| = 1,000, and are cited as such.*

## 1. What the pipeline produces, and the threat model

### 1.1 The published object

The input is a secret reversible circuit C on n wires. C may use any
reversible gate with up to two controls (NOT, CNOT, Toffoli with either
control polarity); the engine takes it in the **g57 form**, in which every
gate is a triple `[a, b, c]` meaning `a ^= b OR NOT c`, that is
`a ^= 1 + c + b·c` over GF(2), a form that expresses that whole gate class
(`docs/tdp_pipeline.md`; `docs/TDP_PIPELINE_REFERENCE.md` §1.2).

The output is one published circuit, the **TDP**: a plain gate list on 4n
wires in the `mpmct1` text format (`docs/TDP_PIPELINE_REFERENCE.md`,
Glossary). It computes a permutation 𝒢 of {0,1}^{4n} with the public-slice
behaviour (`docs/local_mixing_documentation.md`, "The construction and its
contract"):

```text
𝒢(x, 0^n, 0^{2n})    = (junk, C(x), junk)         forward:  C(x) appears on wires n..2n
𝒢⁻¹(a, 0^n, 0^{2n})  = junk (the default since recipe v14: no reverse contract)
                     with preprocessing.reverse_contract = true: a fixed bijection
                     of D⁻¹(a) on wires n..2n, an unrelated random function

wires    0 .......... n-1 | n .......... 2n-1 | 2n ................... 4n-1
input    x  (free)        | 0^n   (pinned)    | 0^{2n}   (pinned)
output   junk             | C(x)  (answer)    | junk
         data half        | answer register   | band
```

The **honest slice** (also "zero slice" or "public slice") is the set of
inputs with x on wires 0..n and zeros everywhere else; only there is the
circuit asked to reproduce C, and only the middle n output bits are the
answer. Fixing the auxiliary inputs and projecting the outputs are part of
the application contract. For a preimage problem the target is an n-bit t
and the witness an x with C(x) = t: whoever knows C reverses its gates on t,
while reversing the published gate list on the slice yields junk (or, with
the reverse contract kept, the random decoy D⁻¹), not C⁻¹, and the junk
outputs that would allow reversing 𝒢 are not part of the target (sections
3.1 and 3.2).

### 1.2 What the adversary sees

The adversary holds the published circuit and nothing else: not C, not the
intermediate circuits of stages 1-5, not the seeds, and none of the
per-round outputs of stage 6. It can cut the circuit anywhere, compute the
function of any segment, compose segments and resynthesize them into
circuits of its own (`docs/tdp_pipeline.md`, "What the adversary sees").
The question the pipeline is engineered and measured against is whether
such rewriting lets the adversary recover internal values of C, and from
them C's gates.

### 1.3 Exposure, the measured quantity

Let c_j(x) be the value written by the j-th gate of C on input x (a
**birth**). At a **cut** t of a circuit (the state after its first t gates),
with w_i(t, x) the wire values on honest inputs, the birth c_j is **directly
exposed** if some wire satisfies w_i(t, x) = c_j(x) ⊕ b for a constant b, and
**affinely exposed** if c_j(x) = b ⊕ XOR_{i in S} w_i(t, x) for a set S of
wires live at the same cut (`~/Codex/output/docs/stage6_adaptive.md` §1).
The **census** checks every cut and the whole affine span of the live wires
on 6,144 honest inputs and re-tests every identity it finds on 8,192 fresh
and about 8,000 structured inputs
(`docs/notes/campaign_summary_20261026/summary.md` §1).

The last value born on each wire of C is an output and visible by
definition, so at n = 64 with |C| = 1,000 there are 936 internal births; 60
of them are polynomials of degree at most two in the public inputs or
outputs (**boundary** values) and the remaining 876 **deeper** values are
the ones that count (`docs/notes/stage2_stage6_campaign_20261021.md`).
Counts are of births, not independent bits, and are not an inversion
probability; they are the quantity the pipeline is optimized on.

One fact drives the design: the raw outputs of stages 1-5 expose no deeper
value at any cut (raw census 0/0 in every campaign arm), but any
resynthesis, the adversary's or the pipeline's own, recreates some, because
a synthesizer that re-implements a gate multiplying a C value needs that
value in the clear (`docs/notes/c_design_exposure_20261021.md` §1). Stage 6
is built around this (section 3.6).

## 2. The source circuit C

Stage 1 obtains C in one of three ways (`configs/tdp.toml`, `[source]`):

| setting | C |
| --- | --- |
| default (`construction = "random"`) | Each gate is a uniform ordered triple of distinct wires; no two adjacent gates are equal (`docs/TDP_PIPELINE_REFERENCE.md` §1.2). |
| `construction = "balanced"` | RC's balanced construction: every internal value is read exactly twice as a control, no value more than twice, and the write counts of the wires stay within one of each other (`~/Codex/baseline64_fanout2_pipeline_20261019/FINAL_REPORT.md`). Exclusive with `path`. |
| `path = "circuits/source.g57"` | A supplied C in the g57 file form; `gates` must be given. This is the route to a C with more than 64 bits of entropy, since a generated C is a deterministic function of a 64-bit seed (`docs/tdp_pipeline.md`, "Seeds and calibration"). |

The gate count defaults to the validated table, 1,000 at n = 64 and 3,000 at
n = 128; any other width needs `source.gates` or `gates_rule = "formula"`
(round(n·log2(n)²), reported as an unvalidated size).

Why a balanced C exists. In a random 1,000-gate C about 36 % of the births
are never read as a control and stay hidden through every resynthesis, while
253 births are read three times or more and were 62-97 % exposed after the
baseline pipeline; each read is roughly an independent chance of about 0.3
(affine) or 0.2 (direct) of being materialized
(`docs/notes/c_design_exposure_20261021.md` §1-2). The balanced C removes
the hubs, so no exposed value serves more than two gates and half as many
gates have both operands in the clear (248 against 462 under the baseline
stage 2), at the price of more exposed births (455 against 319), because it
also removes the unread values; within g57 circuits of this size redesigning
C redistributes exposure rather than reducing it. Under the three-mask
stage 2 with the adaptive v2 stage 6, every C tested gave 0 deeper
exposures in the census (§3, §5 of the same note).

## 3. The six stages

```text
C (n wires, g57)
 |  stage 1  sandwich                  gen_sandwich_gadget   tdp.mpmct1.sandwich.mpmct1  2n wires; 2,390 gates at n=64 (v14; 2,832 in v13)
 |  stage 2  preprocessing             (same process)        tdp.mpmct1                  4n wires; 777k (L8) / 505k (three-mask v13) / 401k (v14)
 |  stage 3  database mixing   [opt]   circuit_mixer         db_mixing.mpmct1            ~2x: 1.60M / 1.06M
 |  stage 4  splitting         [opt]   circuit_mixer         split.mpmct1 + split.state  ~1.9x: 3.02M / 1.89M
 |  stage 5  crossing walk     [opt]   circuit_mixer         crossing.mpmct1             ~1.8x: 5.51M / 3.50M
 |  stage 6  delivery          [opt]   tdp_adaptive          adaptive/final.mpmct1       THE PUBLISHED CIRCUIT: 667k / 2.71M
 v                                     (windowed mode: tdp_packing, tdp_unpacking -> unpacking/unpacked.mpmct1)
```

Gate counts are the n = 64 campaign figures for the embedded-masking baseline
and the three-mask modified arm
(`docs/notes/stage2_stage6_campaign_20261021.md`), made with recipe v13's
stages 1-2 (slice blocks and the N column; bank turnover, no interleaving);
the v14 defaults give a 2,390-gate sandwich and a 400,935-gate stage 2 at
n = 64 (1,146,183 at n = 128; `docs/notes/stage12_trials_20261101.md`), and
the later stages have not yet been run on them. Every stage after the
first is optional (`run.skip_stages`); each stage reads the output of the
last stage that ran (`docs/tdp_pipeline.md`). Stages 1 and 2 run in one
process, `gen_sandwich_gadget`; `circuit_mixer` runs stages 3, 4 and 5 as
three of its modes; stage 6 is its own binary.

### 3.1 Stage 1: the sandwich (2n wires)

Purpose: fix the problem shape (which inputs are free, which are pinned,
where the answer appears) and remove the trivial "reverse the gate list"
attack. It adds no hiding: on its slice the sandwich runs C gate by gate in
the clear (`docs/TDP_PIPELINE_REFERENCE.md` §1).

Construction (`docs/tdp_pipeline.md`, "Stage 1"; `[sandwich]`; RC,
2026-11-01, `docs/notes/stage2_interleaving_20261101.md`): still the
sandwich, updated, with the slice blocks optional and off by default. With
x the low half (wires 0..n) and y the high half, the register (n..2n), in
execution order (recipe v14)

```text
A = C ; copy ; [ D on the low half beside V, V*, recovery on the register ]
```

D is an independently drawn random g57 circuit of C's size and design. The
**V gadget** replaces the plain copy: the copy writes a nonlinear bijection
of C's outputs into the register in groups of three wires, each register
wire receiving a balanced degree-2 function of three of its group's values
from two two-control conjunctions (never a single value on a wire); V is
t random gates on the register and V* its inverse, run beside D; the
register's own inverse of the copy then restores C(x). Forward on the
slice, (x, 0) → (C(x), 0) → (C(x), copy(C(x))) → (junk, V(copy(C(x)))) and,
after V* and the recovery, (junk, C(x)). The default is t = n, a low count
(RC, 2026-11-01): degree-2 payload replicas returned at the midpoint after
ANF packing and resynthesis even with three masks, and the copy's inverse is
itself of degree 2 (a controlled permutation), so V has to scramble the
register; larger t (n·log2 n / 2 and n·log2 n were trialled) waits for the
degree-2 midpoint search.
Backward the register carries the copy's bijection of D⁻¹(a). The copy
gates float backward into C as far as commutation allows. |A| = |C| + |D| +
2n + 2t + the recovery (3n − 5⌊n/3⌋ gates plus drawn polarity
corrections): 2,343 to 2,429 gates on 128 wires at n = 64, t = n.

Two options restore parts of the earlier construction. `slicing = true`
interleaves the slice blocks S1 with C and S2 with D: s = max(n,
round(n·log2 n)) gates each, every one targeting the low half and reading a
y wire, so each block is the identity when y = 0 (V and the copy then
preserve zero, keeping S2 dead in reverse). `v_gates = 0` is the N column of
n CNOTs `y_i ^= x_i`. Both together are recipe v13's stage 1:
|A| = |C| + |D| + 2s + n, 2,832 gates at n = 64, 7,920 at n = 128
(`docs/TDP_PIPELINE_REFERENCE.md` §1.4-1.5). `layer = w` draws the random C,
D and V as layers of w wire-disjoint gates (default 1), which roughly
doubles their dependency width (17 to 21 against 9 for C at n = 64).

### 3.2 Stage 2: preprocessing (4n wires)

Purpose: hide every sandwich value behind nonlinear masks keyed on a **band**
of 2n extra wires, so that no value of C exists in the clear on any wire.
Stage 2 decides which values ever appear bare; the stages after it respell
and spread the material but cannot hide what stage 2 exposes
(`docs/ANF_WINDOW_DELIVERY.md` §3.3). Every later stage preserves the
stage-2 output as a full 4n-wire permutation.

Both constructions share the same five-module frame
(`docs/TDP_PIPELINE_REFERENCE.md` §2a.3):

```text
opening guard ; band seed ; masked compute ; band reseed ; closing guard
```

The band is zero on the slice, so the **band seed** fills it from the input
x (otherwise every mask would be a slice constant), the **reseed** does the
same for the backward direction, and the two **slice guards** (20n gates
each) are the identity exactly when the band is zero. At the end, on input
(x, 0^n, 0^{2n}), wires n..2n hold C(x) and everything else is junk
(`docs/tdp_pipeline.md`, "Stage 2"). With `preprocessing.reverse_contract =
true` the guards keep the **reverse contract**: each junks only the low
half, so the reversed circuit still carries D⁻¹ of its input on wires n..2n.
The default since recipe v14 (RC, 2026-11-01) is false: there is no closing
guard, so only the forward payload is kept, and since recipe v15 (RC,
2026-11-02) the opening guard is the **two-layer guard**, keyed on the
register (wires n..2n-1) as well as the band, because the honest slice has
both at zero: as a map (A, B, band) -> (f(A), h(B), band'), the identity on
the data exactly on the honest slice, A scrambled whenever B or the band is
nonzero, B whenever the band is. Layer 1 (keyed on the band) and layer 2
(keyed on B) each hold n·⌈log2 n⌉ three-control gates opened into
two-control gates through a helper wire (restored in B, not in the junk
band); 4,512 gates at n = 128. Recipe v14's guard was keyed on the band only,
so an input with the band zero and any value of B evaluated the sandwich off
its slice; it is no longer built and the guard is not a setting. The band
seed and reseed stay at both ends. The adaptive stage 6 checks the sandwich
payload contract forward only (`tdp_adaptive --forward-only`).

**Embedded masking** (`preprocessing.mode = "embedded-masking"`, preset
`L8`; the default before recipe v13). Each data wire carries two or three open balanced
quadratic masks `M = 1 + y + x·y + z` over band wires x, y, z (the gate pair
`g57(w,x,y); CNOT(w,z)` opens a mask and the same pair closes it). A
sandwich gate is fired through the still-quadratic decodes of its operands
("quadratic fire", product degree ≤ 4), the cubic and quartic terms being
added by dirty-ancilla brackets of four and eight two-control gates; a
temporary **fire cover** and a persistent **straddle** mask opened mid-fire
hide which gate fired; the band is refreshed by rerandomization bursts; and
**persistent carrier migration** (dose 8) moves data roles across physical
wires so that no wire class is recognizable
(`docs/TDP_PIPELINE_REFERENCE.md` §2b.3, §2b.5, §2c;
`docs/local_mixing_documentation.md`, "Embedded Masking"). Preset L8 adds
`PREMASK`, `SEED_AVOID_C1` and bounded bracket chains (`CHAIN=2,
CHAIN_LEN=8`), which closed the first-layer "fringe" leaks and cut bracket
rebirths (`docs/STAGE2_LINKABILITY.md` §3, §6). Every gate has at most two
controls, the vocabulary the stage-3 store digests. Size: 0.777-0.779 M
gates on 256 wires at n = 64, 2.22-2.24 M on 512 wires at n = 128
(`docs/TDP_PIPELINE_REFERENCE.md`, "Data flow").

**Three-mask compute** (`preprocessing.mode = "three-mask"`, knobs in
`[three_mask]`; new in this release and its default stage 2). The same outer blocks (opening guard,
band seed, band reseed, closing guard, preset L8) surround a different
compute block (`~/Codex/output/pdf/stage2_three_mask_20261020/stage2_three_mask_design.md`):

- Each data value is carried as `x_i = d_i + M_{i,1} + M_{i,2} + M_{i,3}`,
  where d_i is the physical carrier and each M_{i,j} = b_p·b_q·b_r is a
  cubic monomial in three band coordinates; a mask is a stored triple of
  band indices, not a register, and cubic labels are unique across roles.
- The band (`band_wires = 0` = the larger of 2n and 128, rounded up to a
  multiple of 32) passes its coordinates through **serving** (supplying
  mask coordinates, never written), **retiring** (drained of references)
  and **growing** (freely rewritten by 8 nonlinear growth gates per fire)
  phases, so masks are a renewable supply. Since recipe v14 the turnover is
  per wire (`turnover = "wire"`): promotion cohorts of 16 wires take the
  roles of the design's banks, and a drained wire grows again at once
  instead of waiting for its bank-mates; `turnover = "bank"` keeps the
  design's banks (`banks = 0` = banks of 16 wires: 8 at n = 64, 16 at
  n = 128). **Pre-growth** (`pregrow`, on since v14) rewrites the serving
  half `warmup` times before the first masks, so no mask starts on raw
  band-seed coordinates.
- A logical gate `x_t ^= x_u·x_v` is compiled without decoding its controls:
  `d_t ^= (d_u + S_u)(d_v + S_v)` expands to at most 16 products of up to six
  controls, each emitted as a non-restoring **dirty-helper chain** of 2k−3
  Toffolis on helpers borrowed from a growing bank (the chain adds the
  product whatever the helper held, and the helper may stay dirty because no
  live mask references it).
- Per fire: a temporary cover mask is XORed into the target, the product
  units are emitted in shuffled order, the target's oldest persistent mask is
  replaced by a fresh one through the **differential cubic refresh**
  `abc + def = (a+d)bc + d(b+e)c + de(c+f)` (15 gates), the cover is
  removed, up to two idle roles are refreshed to drain the retiring bank,
  and the growth gates are emitted. A **persistent migration** pass (6
  segments) spreads target writes over physical carriers, and all masks are
  XORed out at the end of the compute.
- Since recipe v14 the finished compute is **interleaved**
  (`interleave = 4096`): a dependency-preserving random reordering in which
  no gate runs more than 4,096 positions ahead of the earliest waiting gate,
  so a 250-gate window holds pieces of 9.6 fires on average at n = 64
  (2.4 without interleaving, where a fire is contiguous;
  `docs/THREE_MASK_STAGE2.md`). Interleaving switches on three changes that
  remove the chaining between consecutive fires: the cubic refresh becomes
  read-only on the band (the same 15 gates in bracket form), dirty helpers
  come from per-fire lanes of a reserve that growth never writes (a quarter
  of the band, four wires per lane: 8 lanes at n = 64, 16 at n = 128), and
  the carrier migration moves roles in one fire box per lane cycle. The
  fires a window holds are then bounded by the lanes, not by the sandwich
  (`docs/notes/stage2_interleaving_20261101.md`). Against the narrow factor
  attack on the fully mixed circuit this cut the exposed deeper C births
  threefold (254 to 89) and the per-read recovery probability from 18 % to
  7 % (same note, from the variants campaign,
  `docs/notes/pipeline_variants_report_20261101.md`).

The defaults are the campaign configuration "three-mask modified / v2"
(plus, since recipe v14, wire turnover, interleaving and pre-growth):
`random_helpers = true` (helpers drawn from a shuffled pool per fire, which
drops the share of band writes landing on the four busiest wires from
14.8 % to 4.5 % at zero gate cost) and `unique_cover = true` (a cover whose
cubic label is live is rejected); `false` for both gives the generator's
"as-is" package (`docs/notes/three_mask_review_20261021.md`, findings 1 and
4). The three-mask circuit is about 35 % smaller than embedded masking:
505 k against 777-779 k gates at n = 64 with recipe v13's stage 1 (401 k with
v14's), 177 against 272 gates per logical fire; 77 % of that saving comes from not restoring the helpers, and a
serving bank's coordinates are constants for about 103 fires, roughly
16,300 gates (review, findings 2-3). The construction claims circuit-level
concealment under the tested rewriting observers, not information-theoretic
masking.

What stage 2 protects: no affine relation to C beyond the outputs was found
at any scanned stage-2 cut in 29 variants at both scales
(`docs/STAGE2_LINKABILITY.md` §3.1). The third mode, `nonlinear291`, is a
wide research construction not described here.

### 3.3 Stage 3: database mixing (optional)

Purpose: un-invertible respelling, dilution and "digestion" of the stage-2
vocabulary into the g57 form; it is not a hiding stage
(`docs/TDP_PIPELINE_REFERENCE.md` §3). It runs `circuit_mixer --tdp
--db-mixing --p-db 1.0` against the regular **frozen store** (`frozen_m1_m11`,
small g57 circuits keyed by the canonical form of their function;
`database.regular_dir` or `FROZEN_DB_DIR`). One move: sample a window of at
most 12 gates (contiguous, or convex after moving commuting gates aside),
compute the canonical key of its function, look it up, splice in a randomly
chosen stored spelling of the same function after verifying it, and push the
new gates outward through the gates they commute with so that replacements
overlap (`docs/local_mixing_documentation.md`, "Database mixing"). A size
controller alternates growing (MIX) and shrinking (COMP) rounds; a g57 twist
swaps two wire labels across a long window. Since 2026-11-01 (RC) a round is
a twist with probability 0.001 (circuit_mixer's preset: 0.0005), and each
twist window is uniform between 100,000 gates and the whole circuit and lies
wholly inside it (preset: log-uniform from 64 gates, truncated at the ends):
the doubled rate makes up for the carrier hops the interleaved stage 2 gave
up, and no twist fits inside one stage-6 block (at most 65,536 gates)
(`db_mixing.twist_*`, `docs/tdp_pipeline.md`). The validated schedule
(profile `2,2,2,2,2`, int(G·2·2·1.3) moves for G stage-2 gates) doubles the
circuit (ratio 2.04-2.13) and leaves it about 88 % g57 gates, in about 34
minutes at n = 64 (`docs/TDP_PIPELINE_REFERENCE.md` §3.8). Campaign sizes:
1.60 M (baseline) and 1.06 M (three-mask) gates.

### 3.4 Stage 4: splitting (optional)

Purpose: remove the single repeated complemented gate shape that stage 3
leaves, which rainbow tables and algebraic attacks lean on
(`docs/TDP_PIPELINE_REFERENCE.md` §4.5). Every complemented g57 gate
`a ^= b OR NOT c` is rewritten exactly into the two plain conjunctions
`{a ^= b, a ^= NOT b AND NOT c}`; with probability 0.8 the one-control piece
is joined to a distant bracket on the same wire by an **absorbed NOT twist**
(both brackets flip polarity and every read of the wire in between is
complemented, at zero extra gates), and each move ends with one crossing
shot. The stage runs to exhaustion, has no settings, multiplies the gate
count by about 1.9 (1.60 M → 3.03 M in about 70 s at n = 64;
`docs/tdp_pipeline.md`, "Stage 4") and writes `split.state`, the resume
point of stage 5. It adds no hiding by itself, and because it acts on the
g57-form gates that stage 3 produces, skipping stage 3 makes it nearly a
no-op.

### 3.5 Stage 5: the crossing walk (optional)

Purpose: the only transformation that leaves the two-control vocabulary. The
walk resumes `split.state`, shoots a random gate in its stored direction to
its first collider and rewrites the collision by one of three exact
case-split rules so that the two gates cross; a width penalty accepts a
split of width c with probability 3^−(c−1). A thermostat holds the size near
target = round(2 × input) with temperature max(64, target/25), contracting
by exact undo of journaled crossings and pairwise merges; the move budget is
3 × target (`docs/TDP_PIPELINE_REFERENCE.md` §5; `[crossing]` in
`configs/tdp.toml`). The output has multi-control gates (median 3 controls,
up to 12) and about 1.8 × its input: 3.03 M → 5.49 M in 46 min at n = 64
with the archive-era 6 × budget; the campaign's stage-5 outputs were 5.51 M
(baseline) and 3.50 M (three-mask) gates. The raw crossing output carries no
C value at any cut, but re-cutting and resynthesizing it uniformly at
2,000-gate windows recovers 244 of 936 interior C values at n = 64
(`docs/notes/stage45_revival_20261011.md`, cited in the reference §5.6).

### 3.6 Stage 6: delivery

Purpose: produce the published circuit by re-spelling the mixed circuit from
its function alone, so that the mixer's spelling never ships, while
controlling the exposure that re-spelling creates. Stage 6 reads C: it is a
construction-time, C-informed procedure, not an attack.

**Adaptive mode** (`delivery.mode = "adaptive"`, the default; binary
`tdp_adaptive`, parameters in `[adaptive]`). Designed by RC
(`~/Codex/output/docs/stage6_adaptive.md` §1-4), revised into the "v2"
variant by the review
(`docs/notes/stage6_adaptive_review_20261021.md`) and extended on 2026-11-02
into recipe v15's stage 6 (`docs/notes/stage6_config_search_20261102.md`,
`docs/ADAPTIVE_STAGE.md`). It first runs one exact **pre-cleanup** of the
stage-5 circuit (NOT-push, then fcompress for 10 iterations), which alone
removes about 62 % of the crossing walk's gates. Then each round, on the
current circuit G:

1. **Census.** The every-cut exposure census (96 × 64 = 6,144 honest inputs)
   locates where each internal value of C is directly or affinely
   recoverable from the live wires, as intervals of cuts.
2. **Planning.** The circuit is partitioned into large variable-length
   blocks: from a start cut the gates are composed into exact ANF until the
   total reaches 131,072 terms at an eligible endpoint or the block reaches
   512 gates (`term_cap`, `max_block_gates`; 65,536 before recipe v15). The boundary is chosen in
   the upper half of the feasible interval by, in order: fewest deeper C
   values affinely exposed at the cut, fewest internal values exposed,
   largest clearance from the exposure intervals, fewest direct proxies,
   distance of at least `avoid_radius` = 64 from the previous round's
   boundaries, and the latest cut. The 16 `segments` are planned in
   parallel with seams at exposure-free cuts.
3. **Synthesis.** Every block is resynthesized from its ANF by the ANF
   synthesizer (the bidirectional search of the unpacking kernel,
   `docs/STAGE7_UNPACKING.md` §2: 16 restarts, lookahead width 6) with
   `seeds_per_block` = 2 seeds under a size-scaled budget: `time_limit` =
   60 s for a block of at most 1,000 gates, 120 s up to 4,000, 240 s above.
   A block that does not synthesize is split at exposure-aware cuts, two
   levels deep (`refinement_levels`), never below 128 gates.
4. **Acceptance.** Per block the cheapest candidate that exposes no more
   deeper C values, and no more C output bits before 90 % of the circuit,
   than the original is kept (`accept = "smaller-not-leakier"`; `"smaller"`
   judges by cost only). Cost is the 2-control cost (`cost =
   "two-control"`): a gate with k > 2 controls counts as its 4k - 8 Toffolis,
   so a synthesis that removes wide gates is not rejected as growing. This is sound
   because the honest-slice state at every retained boundary is fixed, so a
   block's exposure depends only on its own candidate.
5. **Assembly.** The assembled circuit is checked equivalent to its input
   and its exposure measured.

These large-block rounds (phase A) end when two consecutive rounds compress
jointly by less than `stop_fraction` = 20 % or after two rounds without
decrease. **Phase B**, the descent, runs the same rounds at block caps 128,
32 and 16 (`descent`; planned without reading C, the leak guard still
applied), each to the same stall rule (`descent_stop_fraction` = 0.2) and
followed by an exact cleanup. **Phase C** lowers every gate with more than 2
controls (`lower_controls` = 2) by the exact Barenco decomposition and cleans
again, so every gate of the published circuit has at most two controls.
Every cleanup, like every block, is rejected if it raises the exposure;
`max_rounds` (0 = until the plateaus) caps all phases together. The result,
in canonical commuting gate order (`canonical_order = true`, so no block
boundary survives), is written as `adaptive/final.mpmct1` after a zero-slice
check against C and the sandwich payload contract, in both directions, or
forward only (`--forward-only`, which the driver passes when stage 2 keeps no
reverse contract, the default since recipe v14) (`configs/tdp.toml`,
`[adaptive]`; `scripts/tdp_gen.sh`, stage 6). Recipe v15's stage 6 on the
default runs: at n = 64 the 2-control cost of the stage-5 circuit, 17.3 M,
went to 1.25 M (ratio 0.072, 2.35 h on 6 threads), at n = 128 41.6 M went to
4.43 M (0.106, 4.0 h on 12 threads with a bounded synthesizer effort), each
with 0 deeper C values and 0 early C outputs exposed in the census; the v2
stage 6 had reached 0.250 at n = 128 in 4.6 days and left 8 early outputs.

Why v2 rather than boundary choice alone: in RC's matched experiment a
round whose 3,370 boundaries were all proxy-free still exposed 319 internal
births, because the proxies are created by the synthesizer inside the
blocks; the exposure-aware acceptance, the parallel planner, the avoidance
radius and the size-scaled budget each answer a measured shortcoming of the
first driver (review, findings 1-5).

The v2 stage 6 (recipes v13 and v14) as measured on the server campaigns of
October 2026 (`docs/notes/stage2_stage6_campaign_20261021.md`;
`docs/notes/campaign_summary_20261026/summary.md` §4-5), n = 64, one
master seed, arms differing only in stage 2:

| stage 2 / stage 6 | rounds | stage 5 → published gates | deeper direct / affine exposed |
| --- | --- | --- | --- |
| embedded masking (baseline) / asis | 24 | 5,508,346 → 666,745 | 384 / 424 of 876 |
| three-mask as-is / v2 | 14 | 3,516,629 → 2,368,032 | 0 / 0 |
| three-mask modified / v2 | 9 | 3,497,436 → 2,706,063 | 0 / 0 |
| three-mask modified / asis | 12 | 3,497,436 → 2,761,726 | 14 / 14 |

("asis" = RC's frozen policy with one seed; the all-internal affine count of
the v2 outputs is 1, a boundary value.) Stage 6 compresses the baseline
8.3-fold and exposes half of C while doing so; it compresses the three-mask
outputs only 1.3-1.5-fold and the v2 acceptance keeps their census at zero.
A residual remains: the C-blind **narrow portfolio** (rewrites of 125- and
250-gate windows) still recovers 157 deeper values at two or more locations
(76 once) from the modified/v2 output and 264 (63) from the as-is/v2
output; C-blind resynthesis of the output to its own plateau raises its
census to 6-9, a stage-3 remix followed by repacking to 19-39; and every
compression step, the defender's or the attacker's, raised exposure. The
summary therefore recommends judging stage 6 by exposure rather than size
and treating a plateau as the fixed attacker's result, not a floor. The
same week's pipeline-variants campaign found the three-mask circuit without
stage 3 ANF-incompressible in stage 6.

**Windowed mode** (`delivery.mode = "windowed"`, the previous release's
stage 6; `docs/DELIVERY_STAGE.md`). The packer (`tdp_packing run`,
`[packing]`) reorders the circuit along commutations so that values are
born and consumed inside one window, cuts it into windows of at most 512
gates, repairs boundary exposures of C's values and serializes each window
as the canonical ANF of its permutation (`packing/deliverable.tdp4`); the
unpacker (`tdp_unpacking run --refine`, `[delivery]`) resynthesizes every
window from its ANF alone, bisects a window that fails at an exposure-aware
cut for up to 16 rounds, and emits `unpacking/unpacked.mpmct1` in canonical
order. With `delivery.anf_only = true` the packer's deliverable is the
published object and the recipient unpacks it. Measured at n = 64: the
unpacked circuit exposes 5 of 936 interior C values at its own cuts
(`docs/DELIVERY_STAGE.md` §2).

## 4. What is published, what stays secret

Only one file leaves the run: `adaptive/final.mpmct1` (windowed mode:
`unpacking/unpacked.mpmct1`; under `anf_only`: `packing/deliverable.tdp4`),
a gate list with no block lattice, no ANF and no stage provenance.
Everything else is secret or a secret intermediate (`docs/tdp_pipeline.md`,
"Quick start"): the source `tdp.mpmct1.source_c.g57` and the sandwich; the
stage-2 to stage-5 circuits (they regenerate or expose C); the per-round
outputs of stage 6; the private `SEEDS` file, which regenerates everything;
the logs, the recipe manifest `tdp_command.conf` and the reports (counts and
parameters only, but not part of the TDP). Because stage 6 reads C for its
census, all six stages run in the secret-holding environment.

Seeds: a fresh run draws five seeds (gen, mix, split, cross, unpack = the
stage-6 seed) from the OS CSPRNG plus a 256-bit `run_key` into `$RUN/SEEDS`
(mode 0600) before anything else is written. No seed is ever on a command
line: the driver hands them to the stage binaries on inherited file
descriptors and the binaries print `(file)` in place of the value, so
nothing appears in `ps`, the logs or the manifest; completion markers use
HMAC-SHA256 keyed by `run_key`. A generated C is a function of at most 64
bits of seed, so production deliverables should supply C through
`source.path` (`docs/tdp_pipeline.md`, "Seeds and calibration").

## 5. Running it

The supported entry point is the managed driver `tdp_gen`, which reads a
TOML recipe, builds and fingerprints the stage binaries
(`gen_sandwich_gadget`, `circuit_mixer`, `tdp_adaptive`; `tdp_packing` and
`tdp_unpacking` for the windowed mode), draws the seeds, writes the manifest
and runs `scripts/tdp_gen.sh`, which executes the stages
(`docs/tdp_pipeline.md`, "Two ways to run"):

```bash
cargo build --release --locked
cp configs/tdp.toml configs/local.toml      # set database.regular_dir (or FROZEN_DB_DIR) and source.wires
cargo run --release --locked -- tdp_gen --config configs/local.toml --dry-run
cargo run --release --locked -- tdp_gen --config configs/local.toml
```

`configs/tdp.toml` is both the default recipe and the annotated template;
unknown keys and invalid combinations (a stage-2 knob next to a preset, an
`[adaptive]` key under the windowed mode, a `[three_mask]` key under another
mode) are rejected before a run starts. The settings a first user touches:

| setting | meaning |
| --- | --- |
| `source.wires`, `source.gates`, `source.construction`, `source.path` | n (3..4095; at most 256 when stage 6 runs), |C|, random/balanced, or a supplied C |
| `[sandwich]` | the stage-1 sandwich's shape: `slicing` (false), `layer` (1), `v_gates` (n; `slicing = true` with `v_gates = 0` is recipe v13's stage 1) |
| `preprocessing.mode`, `preprocessing.preset`, `[three_mask]` | `three-mask` (default) with its knobs, or `embedded-masking` (preset `L8`); `preprocessing.reverse_contract` |
| `run.skip_stages`, `run.stop_after_stage`, `run.rerun_from_stage`, `run.threads` | stages 2..6 not run; where to stop; what to redo; thread pin |
| `database.regular_dir` | the frozen store (required when stage 3 runs) |
| `db_mixing.schedule`, `[crossing]` | `validated` or `custom`; the stage-5 target, penalty, temperature and budget |
| `delivery.mode`, `[adaptive]` | `adaptive` (default) with its rounds, seeds, budgets and acceptance rule; `windowed` with `[packing]` and `[delivery]` |
| `calibration.seed_file` | an explicit private seed file, for calibration and reproduction only |

The bash driver takes the same settings as flags (`scripts/tdp_gen.sh
--source-wires N --run-directory DIR [--preprocessing-mode three-mask]
[--source-construction balanced] [--delivery-mode adaptive|windowed]
[--adaptive-* ...]`; `--help` lists them all) and needs bash 5.1 or newer.

A run directory holds, per stage, the artifact, a log and a keyed completion
marker: `tdp.mpmct1.source_c.g57` and `tdp.mpmct1.sandwich.mpmct1` (stage
1); `tdp.mpmct1`, `stage12.log`, `stage12.recipe` (stage 2);
`db_mixing.mpmct1`, `stage3.log`, `stage3.recipe` (stage 3);
`split.mpmct1`, `split.state`, `stage4.log`, `stage4.recipe` (stage 4);
`crossing.mpmct1`, `stage5.log`, `stage5.recipe` (stage 5); `adaptive/` with
the per-round directories, `boundary.txt`, `progress.txt`, `final.mpmct1`,
`adaptive.report` (`status=`, `rounds=`, `stop_reason=`) and
`adaptive.log`, plus `stage6.log` and `delivery.recipe` (stage 6); and for
the run `tdp_command.conf`, `tdp_gen.log` and the private `SEEDS`. A stage
is skipped on a resume only when its marker matches its inputs, parameters,
binaries and output; an interrupted adaptive attempt of the same recipe
resumes with its completed rounds kept. Every stage binary starts from an
empty environment plus an allow-list; an inherited variable that could
change the output is a fatal error.

Checks: stage 2 verifies the payload on the honest slice (and backward
under the reverse contract), stage 3 runs periodic global equivalence
checks, and stage 6 checks every round, cleanup and phase-C output against
its input and the final circuit against C on the zero slice and against the
sandwich payload contract (forward only without the reverse contract, the
default). The run ends
with `PIPELINE COMPLETE (stages 1-6): <run>/adaptive/final.mpmct1`, or with
`PIPELINE FINISHED WITH AN ALLOWED FAILURE` and a `NOT_RECOMMENDED` file
when `adaptive.report` records a status other than `ok` (a failed stage
leaves `delivery.failed` with the exit code and step). Read
`adaptive.report` first: the rounds, the stop reason and the exposure counts
of the published circuit.

## 6. Where to read more

- `docs/tdp_pipeline.md`: the operational manual: every setting, every file
  a run writes, seeds and environment policy (its stage-6 text is the
  windowed mode).
- `docs/TDP_PIPELINE_REFERENCE.md`: the full reference of stages 1-5 and the
  windowed stage 6: constructions, parameters, measured behaviour, glossary.
- `docs/ADAPTIVE_STAGE.md`: the adaptive stage 6 (referenced by the recipe
  template and the driver).
- `~/Codex/output/docs/stage6_adaptive.md` and
  `docs/notes/stage6_adaptive_review_20261021.md`: RC's design of the
  adaptive resynthesis and the review that produced the v2 variant.
- `~/Codex/output/pdf/stage2_three_mask_20261020/stage2_three_mask_design.md`
  and `docs/notes/three_mask_review_20261021.md`: the three-mask stage 2 and
  its review.
- `~/Codex/baseline64_fanout2_pipeline_20261019/FINAL_REPORT.md` and
  `docs/notes/c_design_exposure_20261021.md`: the balanced C and why
  exposure is per control read.
- `docs/notes/stage2_stage6_campaign_20261021.md` and
  `docs/notes/campaign_summary_20261026/summary.md`: the matched campaign,
  the resilience attacks and the final tables.
- `docs/PIPELINE_V14_CHANGES.md`: what changed in recipe v14 and why;
  `docs/notes/stage2_interleaving_20261101.md` (the interleaving
  investigation and the stage-1/2 recommendations it led to),
  `docs/notes/pipeline_variants_report_20261101.md` (the variants campaign
  that measured them), `docs/notes/preimage_campaign_report_20261029.md` (the
  degree-2 midpoint replica) and `docs/notes/stage12_trials_20261101.md` (the
  stage-1/2 trial sweep).
- `docs/STAGE2_LINKABILITY.md`: the embedded-masking redesign and the
  metrics behind preset L8.
- `docs/DELIVERY_STAGE.md`, `docs/STAGE7_UNPACKING.md`,
  `docs/ANF_WINDOW_DELIVERY.md`: the windowed delivery and the ANF
  synthesizer.
- `docs/local_mixing_documentation.md`: the general documentation: gate
  vocabulary, sandwich, embedded masking, frozen store, database mixing,
  attack tooling.
- Code: `src/stages/{sandwich,preprocessing,db_mixing,post-processing,adaptive,packing,unpacking}/`,
  the binaries under `src/programs/`, the driver `scripts/tdp_gen.sh` and
  its Rust front end `src/tdp/`.

## Glossary

- **g57 gate**: the triple `[a, b, c]` meaning `a ^= b OR NOT c`
  (`a ^= 1 + c + b·c`); the form in which C is taken and the vocabulary of
  the frozen store.
- **honest slice** (zero slice, public slice): inputs with x on wires 0..n
  and zeros on wires n..4n; there the circuit computes C(x) onto wires n..2n
  forward (and, with the reverse contract kept, a bijection of D⁻¹ backward).
- **sandwich**: the stage-1 circuit on 2n wires; since recipe v14
  `C ; copy ; [D beside V, V*, recovery]` (the slice blocks S1, S2 of
  `[C ⧢ S1]; N; [D ⧢ S2]` optional); **D** is the random decoy of C's size and
  design.
- **V gadget**: the nonlinear copy of C's outputs into the register in groups
  of three wires, V and V* (t = n random gates each by default) on the
  register beside D, and the register's own inverse of the copy.
- **band**: the 2n extra wires of stage 2, zero on the slice, filled from the
  input by the band seed, the source of every mask; in the three-mask
  construction organized in serving, retiring and growing banks.
- **fire**: the masked evaluation of one sandwich gate in stage 2.
- **birth**, **internal value**, **deeper value**: a value written by a gate
  of C; internal if not an output (936 at n = 64, |C| = 1,000); deeper if
  also not a public-quadratic boundary value (876).
- **exposure**, **direct / affine proxy**: a birth equal, on honest inputs,
  to one live wire (up to complement) or to an affine combination of live
  wires at some cut.
- **census**: the every-cut exposure scan of a circuit against C (6,144
  honest inputs, witnesses re-tested on fresh and structured inputs).
- **block** (adaptive stage 6): a variable-length stretch of the circuit
  grown by exact ANF composition and resynthesized from its ANF; **window**
  is the fixed-size analogue of the windowed mode.
- **plateau**: the end of the adaptive rounds, when two consecutive rounds
  compress by less than the stop fraction; a result for the fixed procedure,
  seeds and budgets, not a size floor.
- **narrow portfolio**, **C-blind resynthesis**, **remix**: the attack
  families of the resilience campaigns: rewrites of 125/250-gate windows,
  keep-smaller resynthesis to the attacker's own plateau, and a stage-3 remix
  of the published circuit followed by resynthesis.
- **frozen store**: the regular replacement store `frozen_m1_m11` of small
  g57 circuits keyed by canonical function key, used only by stage 3.
- **mpmct1**: the gate-list text format of every circuit the pipeline writes:
  a header `mpmct1 <wires> <gates>`, then one line per gate with its target,
  a complement bit and its control literals.
