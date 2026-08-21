# **Band hardening: the band joins the slice, and starts moving**

*2026-07-25*

> Markdown rendering of `docs/BAND_HARDENING.tex`. The PDF is authoritative for figures, diagrams and display math.

*Assumes* `docs/PRODUCT_SHARE_ENCODING` *and its
update* `docs/PRODUCT_SHARE_UPDATE`. *For the construction from
scratch, see* `docs/NONLINEAR_GADGETIZATION`.

## Summary

Three weaknesses of the product-share gadgetizer, all of them
**syntactic**—visible by reading the gate list, without executing
anything, and therefore entirely outside the reconstruction heatmap's field of
view. Two of the three were known and listed as open; one was a gap in a fix we
believed was already in place.

| p{0.30}p{0.30}}

weakness | before | now |
|---|---|---|
| Fold fragments emitted in deterministic odometer order: a per-gate progress
clock needing no execution. | Shuffled on the *narrow* path only; the production *wide* path
still emitted in odometer order. | Shuffled on both paths. |
| [4pt]
The band was outside the input slice: setting band wires to nonzero
*provably* changed nothing, so two queries separate "band" from "aux". | Slice = aux wires only. | Slice = every non-data wire, and the guarantee is now **exact for all
slices** rather than exact-for-some / heuristic-for-the-rest. |
| [4pt]
Band wires body-static: never written between the ports, so trivially
identifiable—which is exactly what a restriction adversary needs. | Write census separates band from carriers perfectly (band 10–23 writes,
carriers 175–928). | **Rolling band**: distributions fully overlap, no threshold separates
them, and no wire is unwritten in any 50k-gate window. |

Measured security is unchanged: at n=128 the deg-1 ridge is identically dead
with and without rolls (§7). Rolling costs +12.4% gates at one roll per
inter-SG gap. A fourth question—whether the rolling band lets us *shrink*
the band to the minimum correctness needs—is answered in §6, and the answer
is no: the minimal band is broken by an 800-regressor attack.

## Fold-order randomization, actually applied

The previous pass reported that fold fragments are shuffled within each fold.
That was true of `fold_cg_narrow` only. The wide path—the production
path—still walked the odometer over the cartesian product of the operands'
atom sets and emitted in that order, so consecutive fragments shared long atom
prefixes and each source gate's fold was a legible, self-delimiting block.

The wide path now materializes the fragment list, shuffles it, and then emits.
This is free, and it is sound for the same reason the narrow shuffle is: every
fragment of one fold XORs into the *same* value's two carriers, and reads
only *other* values' carriers plus band wires. No fragment reads what
another writes, so they commute pairwise and the emission order carries no
information the function depends on.

The stronger version—interleaving fragments *across* adjacent folds,
random ESOP re-covers, k-jitter so the per-fold count is not constant—is
still designed and not built.

## The band is part of the slice

### The weakness

The gadget is correct on its low n wires for *arbitrary* junk on every
other wire; the slice-zero preblock is what turns that into "computes C only
on the all-zero aux slice". But the preblock only ever read the *aux*
half. The band was outside it, so a band wire could be set to anything and the
circuit still computed C exactly.

That is a one-query distinguisher, and it needs no knowledge of C: compare
G(x,s) with G(x,0) for a random x. Wires where a flip changes nothing are
band; wires where it junks the output are aux. An adversary who can partition
the non-data wires that cheaply has the band handed to it—and the band is
where every mask source lives.

### The fix

The preblock now spans ${data} [0,n) | {aux} [n,2n) |
{band} [2n,2n{+}b)$ and the slice is **every non-data wire**. The
gate vocabulary is unchanged (positive-polarity CNOTs x_t ⊕= s_w
and CCNOTs x_t ⊕= x_c ∧ s_w, so every gate is individually
dead on the all-zero slice), but the construction is new, and it upgrades the
guarantee.

**Old guarantee.** The CNOT parity matrix over the aux columns was
resampled until invertible, which made the disturbance exact on slices that fire
no CCNOT, and *heuristic* elsewhere—a slice firing CCNOTs was fixed only
if its fired subsequence composed to the identity *and* the conjugated
translations cancelled. Vanishingly unlikely, measured at 0/50k by
n=8, but not a theorem. With b extra columns it could not even have been
stated: an n×(n+b) parity matrix always has a kernel, so a CNOT-only
block *provably* fixes 2_b nonzero slices. The degree-2 gates are what
make the guarantee expressible at all.

**What is built.** The block stays unstructured: the target and the data
controls of every gate are drawn freely over all n data wires, so a wire is a
target of one gate and a control of another, no wire set is exempt from
disturbance, and no input subcube switches the nonlinearity off. Only two
regularities are imposed. Slice wires are covered by a balanced round-robin —
a slice wire that no gate reads is a provably fixed slice, and free draws leave
one uncovered a few percent of the time at these budgets. And half of the
two-control budget becomes *three*-control gates
x_t ⊕= x_a ∧ x_b ∧ s_w, which is what keeps the
disturbance from being affine in x: with one data control per gate, fixing any
slice turns every gate into a constant flip or a transvection, and those compose
to an affine map — degree 1, pinned by n+1 queries.

"No nonzero slice is fixed" is then a property of the draw, so it is
**checked and redrawn** rather than argued: exhaustively over every slice
and every input where the space fits (n+b ≤ 20, which is exactly the regime
where wrong-slice fixes were ever observed), and otherwise by sampling every
single-wire slice, many weight-2 slices and random slices against 64 bit-sliced
random inputs each. The sampled version is a spot check, not a proof; the honest
statement at production width remains the measured one above.

### An exactly-pinning variant, and why it is not used

It *is* possible to make the property a theorem, and the construction is
worth recording because the reason to decline it is the interesting part.

Split the data wires into disjoint pools: targets T and data controls R,
with every gate targeting T and reading only R. On a fixed slice each fired
gate adds e_t m(x_R) for a monomial m ∈ {1, x_c, x_a x_b}, and since no
gate can change another's monomial they all commute:

```tex
S(x,s) = x + ∑_{(t,m)} q_{(t,m)}(s) e_t m(x_R),
```

with every q a *linear* form in s and the monomials independent
functions of x_R. So "identity on slice s" is a linear system in s,
certifiable by a rank check on the emitted gates and seeded by one pin gate per
slice wire. Order-free, so the uniform shuffle stays free.

One layer of that has two defects. Every wire in R is never a target, hence
never disturbed; and the whole disturbance is a function of x_R *alone*
—nine bits of the input at production sizing—so an adversary who sets
x_R=0 collapses the block to a translation. **Two layers repair both**:
run pools (T_1,R_1) entirely, then (T_2,R_2)=(R_1,T_1). Writing F_i for
layer i's disturbance, layer 2 never writes T_1, so

```tex
z_{T_1} = x_{T_1} + F_1(x_{R_1}),
z_{R_1} = x_{R_1} + F_2(x_{T_1} + F_1(x_{R_1})),
```

and z=x forces F_1 0 from the first line and then F_2 0 from
the second. The conditions decouple, the rank check is just the union of both
layers' rows, the pins split across layers, and the gate budget is
**unchanged**. Every data bit now influences the disturbance, and the input
set that kills the nonlinearity shrinks from a 2_ n-|R| subcube to a single
point.

**Why it is not used: the mixing barrier.** Layer-2 gates read exactly what
layer-1 gates write, so the two halves do not commute and *cannot* be
shuffled into each other. That ordering survives the final
`commuting_shuffle` and every downstream mixing move that preserves the
relative order of colliding pairs: the block carries a permanent two-phase seam
that no amount of mixing can dissolve. We would be buying it to exclude one
failure—a nonzero slice that also computes C—whose payoff to an adversary
is unclear, since C is computable on the honest slice anyway. Structure that
mixing cannot erase is the more expensive side of that trade, so the
unstructured block stands and the guarantee is checked rather than proved.

## The rolling band

### The weakness

Band wires were written by the input fill, read constantly by the body, and
never written again. A census of the shipped n=128 gadget makes the
signature unmistakable:

| n=128 gadget, 568 wires | carriers (0–511) | band (512–567) |
|---|---|---|
| writes per wire (min/median/max) | 175 / 542 / 928 | 10 / 16 / 23 |
| reads per wire (min/median/max) | 228 / 487 / 714 | 9,413 / 11,639 / 14,560 |

A single threshold on either statistic separates the two populations perfectly.
Any adversary that wants to condition on a mask source—a restriction
attack—first has to find one, and this hands over the whole list.

### The fix

The role of "band variable" now moves around the physical wires, by extending
RG2's move—which swaps carriers between two logical values—across the
carrier/band boundary. One **roll** picks a band variable and a uniformly
random other wire, and swaps their contents with the same 3-CNOT network RG2
uses, so the move adds no new gate shape. The bookkeeping then re-points:

- if the partner was another band wire, the two variables trade locations;
- if it was a carrier of some value, `GadgetState::pairs` is
re-pointed at the vacated band wire—the carrier and the band variable have
*swapped roles*, and the wire that was band is now a live carrier written
by subsequent folds and RGs.

Nothing about the encoding changes: the band variable keeps its value, the
carrier keeps its value, so every decode is invariant. What changes is where the
frozen band lives. Two invariants make this safe and are asserted in tests:
carriers and band wires remain a **partition** of the wire space (so a fold
target is never also a mask literal), and mask slots name band
**variables**, not wires—physical literals are resolved through a
location map at emission time, so the dedup set, re-sourcing, and the final
strip are all invariant under rolling.

Two consequences elsewhere. The band fills take a wire *list* rather than a
range, and the output-side mirror fill targets the band's current locations
together with its home range (minus anything below n, which the decode has
already turned into an output wire). And the narrow ladder's borrow pool widens
from the carriers to the whole wire space: borrows are restored exactly within
the fragment, so a band wire is as safe to borrow as a carrier, and the wider
pool removes a real exhaustion risk once a fragment's literals can sit anywhere.

### What it costs and what it buys

At n=128 with the production plan [2,3,3] and the nonlinear fill:

|  | `–prod-roll 0` | `–prod-roll 1` |
|---|---|---|
| gates | 283,561 | 318,813 (+12.4%) |
| rolls | 0 | 7,919 |
| writes/wire, carriers | 209–1004 | 284–969 |
| writes/wire, band | 10–22 | **282–867** |
| reads/wire, carriers | 272–804 | 625–3287 |
| reads/wire, band | 9725–14712 | **702–3421** |
| width-1 share of writes, carriers | 0.033–0.115 | 0.050–0.165 |
| width-1 share of writes, band | 0.556–0.826 | **0.078–0.155** |
| separable by any of the above? | **yes** | no |
| unwritten wires, 50k window | **exactly 56** | 0 |
| unwritten wires, 10k window (median) | 68 | 13 |

(Four rolls per gap was also measured: it cuts the 10k-window figure
to about 6 at roughly four times the gate cost.)

One roll per inter-SG gap is enough to erase the global signature completely:
the band's write and read distributions land inside the carriers', and over any
50k-gate window every wire is written. Four rolls per gap only halves the
residual *local* statistic (a band variable still sits still for
2000 gates at roll 1) and costs four times as much, so **one roll
per gap is the recommendation**; `–prod-roll` is off by default so the
legacy build stays bit-identical.

**Write counts are not the only census.** A first version emitted each
roll as three plain CNOTs, which matched the write *counts* but not their
*shape*: nothing but rolls ever writes a wire while it holds a band
variable, so those wires were written almost exclusively by width-1 gates
(fraction of width-1 writes: band 0.556–0.826, carriers 0.033–0.115
—separable again, by a different statistic). Each transvection is now drawn
from a mixed vocabulary: either the plain CNOT or the pair
t ⊕= s ∧ u, t ⊕= s ∧ u over a
random helper, which sums to exactly s, adds no constant, and—carrying
opposite literals on u—does not collide, so the commuting shuffle drives
the two halves apart. After the change the same statistic reads band
0.078–0.155 against carriers 0.050–0.165: overlapping.

**What this does not fix.** A windowed census still narrows the band down:
in a 10k-gate window about a dozen wires go unwritten, and those are the band
variables that happened not to roll. An adversary willing to stitch windows
together and follow the swaps can still track the band; rolling raises the cost
of identification from a single whole-circuit census to that, and no further.
This is why band *width* (§6) still matters—the two defenses multiply.

## Two follow-ups: quadratic slicing, and the missing instrument

### The preblock was affine in x

With one data control per gate, fixing any slice s turns every preblock gate
into a constant flip or a transvection, and those compose to an *affine*
map: S(·,s) has degree 1 in x, fully determined by n+1 queries. The
high degree was entirely in s (the fired set is linear in s, but the product
of the fired transvections makes the entries of L_s high-degree in s). If
the intent is "S looks like a random function of both arguments", that held
in one argument only.

The block now also emits x_t ⊕= x_a ∧ x_b ∧ s_w with
both data controls in R. The exactness theorem survives *unchanged in
form*: every gate still writes only T and reads only R, so they all still
commute, and the composite is

```tex
S(x,s) = x + ∑_{(t,m)} q_{(t,m)}(s) e_t m(x_R),
m ∈ {1, x_c, x_a x_b},
```

with every q still linear in s. The monomials are independent functions of
x_R, so "identity map" still forces every coefficient to vanish, and the
rank check simply gains rows keyed by data *pairs* alongside the singles.
It costs nothing (the gate budget is fixed) and a new test confirms a genuine
second-order term S(a⊕ b)⊕ S(a)⊕ S(b)⊕ S(0)≠ 0 on some slice.

### A statistical readout

Every measurement in this project has been made with `hmap_affine`,
which asks whether C_i's bit is an **exact** GF(2) function of
G_j's wires and scores 0.5 whenever it is not. That is blind to bias — and
bias is precisely what the mask-term count k controls. The naive predictor
c_0 ⊕ c_1 is wrong only when the mask sum fires, so piling-up over
monomials firing with probability 2_- predicts a best-predictor
agreement of 57%, 64%, 78%, 87.5% for
[2,2,3,3], [2,3,3], [3,3], [3] — while the exact-span plate reports
meanH≈ 0.498 for all of them.

`red_team_tests/bin/heatmaps/hmap_stat.rs` measures that directly: per cell and target bit,
the best agreement over every single wire and every XOR of two wires (the family
containing the carrier pair), against a null floor from the same search on a
random target. At n=128, 31 rows × 64 columns, 8192 samples:

| build | floor | median | p90 | piling-up theory |
|---|---|---|---|---|
| plain gadget | 0.525 | **0.907** | 0.970 | 1.000 |
| [2,3,3] (shipped) | 0.526 | **0.609** | 0.624 | 0.641 |
| [3,3] | 0.525 | **0.703** | 0.741 | 0.781 |
| [3] | 0.526 | **0.803** | 0.845 | 0.875 |

(Per-row max over columns, interior only: both axes have unencoded
ends, and cell (0,0) — C's input state against G's input wires — reads
1.0 in *every* build, encoded or not. Trimming the outer tenth of both
axes moves the tails but not the medians, which read $0.907 / 0.615 / 0.718 /
0.803$ untrimmed.)

It tracks the theory a few points low, as it must: the prefix grid is coarse, so
no column lands exactly on the matched snapshot, and the predictor family stops
at two wires — every number is a lower bound. Validation: on the plain gadget
the interior reaches 0.970, the pair search finding the carrier XOR, which
there simply IS the value.

This settles what the exact-span measure could not. The ridge scores [2,3,3],
[3,3] and the b=12 build identically dead; the statistical measure
separates them and prices the k:3→2 cut at 0.609 → 0.703
predictability for its 25% of the gates. And [3], at 0.803, sits only
0.10 below the *unencoded* gadget — consistent with its exact-ridge
failure (ρ = 0.995), and confirming two terms as the floor.

### One product term closes most of that gap

`–and-wires` gives the predictor a single
(w_p ⊕ a)(w_q ⊕ b) over a declared wire set on top of its best XOR
predictor: the *degree-2-capable* adversary, and exactly what cancels a
degree-2 mask term. On the non-rolling builds, where the band sits on its home
wires and the adversary knows where to look:

| build | XOR only | + one AND | gain |
|---|---|---|---|
| [2,3,3], roll 0 | 0.602 | **0.693** | +0.090 |
| [3,3], roll 0 | 0.718 | **0.728** | +0.010 |

The base degree-2 term *is* a product of two band wires, so one AND removes
it and leaves the two degree-3 terms — which is where [3,3] already sits.
The gap between the plans falls from 0.115 to 0.036: the term that the
[3,3] cut removes is worth far less than the linear-only measure implies, and
the 25% saving is correspondingly cheaper than §5.2 alone suggests.

Rolling changes the picture again. The same search on a `–prod-roll 1`
build, restricted to the home range, buys almost nothing (+0.002 and +0.001)
because the band is no longer there. So the rolling band does not only defeat
the syntactic census of §4; it also raises the price of this statistical
adversary, who must either locate the band per snapshot or widen the AND search
to all 5682 pairs and pay the higher noise floor. Every number here
is a lower bound.

## How wide should the band be?

Rolling raises the natural question: if the band's role wanders anyway, why
carry b=56 variables instead of the minimum correctness needs? The slot-space
assertion is the only hard constraint—bd2_d distinct factor
tuples must cover the n· k_total live slots with headroom—and
at n=128 (gadget width 256, plan [2,3,3]) it bottoms out at b=12. That
build is 266,650 gates on 524 wires: 44 wires and about 5% of the gates
cheaper than b=56.

**It is also broken.** Band width is not a correctness parameter, it is
the *dimension* of the space the mask terms live in, and therefore the
cost of the degree-d adversary the tower deliberately admits. A reconstructor
that can form every degree-3 product over the band's wires has every mask term
in its regressor span, and the encoding collapses back to affine. That costs
b3 regressors: **220 at b=12, 27,720 at b=56**.

Measured, same C, same c-step, an adversary with an identical and cheap
budget (all triples over 12 wires, 800 regressors in total):

| build | depth | depthMed | ρ | meanH | verdict |
|---|---|---|---|---|---|
| b=12, roll 0 | 0.259 | 0.266 | **1.000** | 0.478 | ALIVE—broken |
| b=12, roll 1 | 0.044 | 0.000 | 0.018 | 0.499 | dead |
| b=56, roll 0 | 0.044 | 0.000 | 0.162 | 0.498 | dead |

The minimal band with a static layout hands the diagonal straight back—a
perfect ρ=1 progress ridge, for an attack that fits in a few hundred
regressors. Rolling saves it, but only against *this* adversary, which
guessed a fixed wire set and never tracked the swaps; §4 shows a windowed
census still narrows the band to about a dozen candidate wires, and at b=12
a dozen candidates is the whole band.

So the two mechanisms defend different things and multiply: width makes the
attack expensive *once you know where the band is*, rolling makes knowing
where it is expensive. Dropping to the minimum keeps only the second factor,
which is not airtight, and saves 7.7% of the wires. **Keep the band
wide**—if anything the auto size (4nk_total, chosen for
slot space, not for this) is worth revisiting upward, since b3
grows cubically while the cost is a handful of fill gates per wire.

## Validation

n=128 real sliced sandwich (source C = 3000 g57 gates, sandwich 7920
gates / 256 wires), identical C across all three builds, deg-1 all-wire
adversary, matched c-step 30 / 101 rows and 152 columns:

| build | gates | depth | depthMed | ρ | meanH |
|---|---|---|---|---|---|
| plain gadget (control) | 85,354 | 0.356 | 0.363 | **1.000** | 0.475 (ALIVE) |
| [2,3,3]+nlfill, roll 0 | 281,436 | 0.043 | **0.000** | 0.018 | 0.498 (dead) |
| [2,3,3]+nlfill, roll 1 | 316,926 | 0.043 | **0.000** | 0.018 | 0.499 (dead) |

The rolling build is statistically indistinguishable from the non-rolling one
and both are dead, against a control that reads a perfect diagonal on the same
C. A band-restricted degree-2 adversary is dead on both as well (depthMed
0.000, meanH 0.498). Endpoint verification passes at n=4 (exhaustive over
the full input domain, rolls on, both wide and narrow) and at n=128.

Note what the ridge measure can and cannot say here: it reads all of a
snapshot's wires, so it is invariant under permuting them, and a roll is
exactly such a permutation. The rolls therefore *cannot* move this number,
and the fact that they do not is a consistency check, not evidence of benefit.
The benefit is the syntactic one measured in §4.

Library suite **177/177**, five new tests: the wide fold no longer emits
in odometer order; a roll preserves every value's decode over the whole input
domain while keeping carriers and band disjoint; a rolling gadget round-trips
and actually writes the band in the body (compared against the same seed with
rolls off, since the fills alone already write every band wire); the preblock
disturbs every nonzero slice exhaustively with a band present; and the
constructor builds across the widths the gadget paths reach, at both the
default and the minimum gate budget.

## Open

- The statistical readout now exists (§5.2) but only measures a
**two-wire** predictor family. A stronger adversary—greedy or LMS fits
over more terms, or products—would give a tighter lower bound; the current
numbers are floors, not estimates.
- With the instrument in hand, re-price the whole mask plan. [2,3,3]
sits at 0.615 predictability; [2,2,3,3] should sit near 0.57 and has never
been measured this way. The cheapest plan meeting a chosen predictability
target is now a decidable question rather than a guess.
- Deeper fold randomization: cross-fold interleaving, random ESOP re-covers,
k-jitter.
- The CG fold is 76% of the gadget and costs
(2+k_total+)^arity per source gate. The largest
untried lever is dropping the XOR carrier pair for a single carrier
((1+k)^2 instead of (2+k)^2, about -44%), which trades away gate-local
non-completeness and needs the balance-obstruction argument revisited.
- Roll-rate tuning against a *windowed* adversary rather than the
global census, and whether a cheaper non-uniform schedule (roll a variable just
before it is next read) buys the same invisibility.
- Static/syntactic distinguishers are still only instrumented ad hoc: the
census in §4 was written for this pass, not by a standing tool. Every other
measurement we have is execution-based.
