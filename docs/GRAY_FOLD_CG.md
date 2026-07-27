# The Gray-code CG fold

`--prod-gray-fold 1`. Implemented in `ProdLedger::fold_cg_gray`
(`src/replace/gadgets.rs`), with the atom-level realization in
`emit_atom_onto`.

## The problem

Under the product-share encoding a value is carried as

```
v = w_v + x1x2 + x3x4x5 + x6x7x8            (the [1,2,3,3] plan)
```

and the CG for a source gate `a' = a + bc` must add `(w_b + B)(w_c + C)` onto
`w_a` alone, leaving every aux wire as it found it. The shipped fold expands
that product into one gate per term of the cartesian product: 16 fragments at
`[1,2,3,3]`, of width up to `arity * max_deg = 6`.

Everything above two controls is invisible to the frozen store — width-3 gates
hit at 0.41% against ~99% for narrow material, and widths 4–6 are absent from
the store at any control cap or canonicalization budget — and that material is
56.2% of the shipped n=128 gadget. The existing remedy, laddering each wide
fragment down with the Barenco double sweep (`--prod-max-width 2`), costs ~6.2x
the fold, because every fragment re-derives the same mask products from scratch.

## The construction

Write `S_w = L_w + M_w` for each operand: `L_w` the width-≤1 atoms (the carrier
literal, and the ledger's constant atom when the control is negative or the
value carries a constant), `M_w` the mask terms. Borrow two wires `u`, `z` and
toggle them around the Gray cycle over `(u holds M_b, z holds M_c)`:

```
  A=(0,0) --gather b--> B=(1,0) --gather c--> C=(1,1)
     ^                                           |
     +---- strip c ---- D=(0,1) <--- strip b ----+
```

Reading `u` once from each column and `z` once from each row gives

```
  A_u = U1 + U2 = M_b        A_z = Z1 + Z2 = M_c
```

— the borrows' unknown incoming values cancel between the two readings, exactly
as the ladder's double sweep cancels its own. So emitting

* `L_b x L_c` once each (anywhere),
* `L_b x A_z` once per z-column, `A_u x L_c` once per u-column,
* `A_u x A_z` once at every one of the four phases,

sums to `(L_b + A_u)(L_c + A_z) = S_b S_c`, and every one of those gates is at
most two controls by construction. At `[1,2,3,3]` the block is 45 fragments
against the wide fold's 16, but **none of them is wide**.

## Why the accumulators must stay dirty

`w_b + M_b` *is* `b`. An accumulator that started clean would hold the full mask
sum next to a live carrier, putting the operand one XOR away from a wire pair —
the same use-point re-exposure that sank the deferred-mask peek. Borrowed, the
wire holds `u0 + M_b`, and `u0` is off the wire set for the whole dirty window.

This is audited, not argued. Over all 46 prefixes of a `[1,2,3,3]` block, the
exact maximum correlation over **all** affine predictors in the live wire values
(Walsh transform per residual component, masks restricted to variables still
live) peaks at

```
  0.28125 = (1/2)(3/4)^2
```

for each of `a`, `b`, `c`, `a'` — which is the encoding's own steady-state
piling-up bound. The block's interior gives an affine adversary nothing its
endpoints do not. No secret enters the GF(2) span of the wires and their
pairwise products at any prefix, and a beam search over quadratic mirror
features peaks at the `(3/4)^2` design bound. The same audit against a
CLEAN-accumulator variant recovers `b` at correlation 1.0 and flags 35 prefixes
on the span test, so the instrument is sensitive to the failure it is guarding.

The audit was re-run over a sample of the phase-assignment x pivot family the
implementation actually draws from, not just one member.

Admissibility is read in the SPIRIT here, deliberately: `u += x1x2` does park a
whole degree-2 atom's literal pair in one gate's support, the same way the
shipped fold's carrier x mask2 fragments already do. It sits on top of `u0`,
which is what the measurement above is measuring. Degree-3 atoms are never
named whole in any one gate — strictly better than the width-4..6 fragments
this replaces. Hiding the pair as well would cost about +12 gates per block via
a CNOT-copy detour; it is not implemented.

## The residual-constant trap

`emit_g57_form` leaves a complement on its target, so a gather actually lands
`M_b + delta`. Left alone that is not a leak but a **wrong function**: the
four-phase sum becomes `(M_b + delta)(M_c + eps)`, i.e. the block silently
acquires `delta*M_c + eps*M_b`. It is absorbed for free by toggling the
operand's constant atom — `L'_w = L_w + delta` restores `L'_w + A_w = S_w` —
which is the same ledger mechanism the wide fold already uses for a negative
control.

Worse, and caught only by the exactness test: `emit_narrow_fragment` chooses its
ladder pivot **at random** among equally-scored candidates, so routing a gather
and its strip through it independently leaves DIFFERENT residuals and the
accumulator comes back off by one — a corrupted wire, not a wrong constant.
`emit_atom_onto` therefore takes the pivot as a parameter and the caller draws
it once per atom, reusing it for both passes. Spellings still vary between the
passes, since every spelling of one function carries the same constant.

## Cost and what it buys

Measured on the n=128 sliced sandwich (|C|=|D|=3000, seed 1, production preset
otherwise, same sandwich and same source C on both sides — the two runs differ
only in the fold). Both verified: `verify PASSED (200 samples, low 256 wires)`.

| | wide fold | gray fold | gray + `--prod-ladder-cap 3` |
|---|---|---|---|
| gadget gates | 339,786 | 805,245 (2.37x) | 1,021,244 (3.01x) |
| fold fossils (>2 controls) | 153,421 | **0** | **0** |
| gray blocks | 0 | 7,209 | 7,209 |
| **store-reachable gates** | **31.55%** | **95.59%** | **99.87%** |
| policy-blocked (>2 ctrl) | 54.96% | 4.18% | 0.08% |
| store-blocked | 13.49% | **0.24%** | **0.04%** |
| CNOT reachable | 70.6% | 99.8% | — |
| g57 reachable | 69.5% | 99.8% | — |

Read reachability, not match rate: the two move in opposite directions, and
reachability is the per-gate question of whether ANY window containing this gate
resolves to something the store holds. 95.59% is above the previous best on
record (82.40%, at `--prod-ladder-cap 3`, which cost 2.2x for 5.7x reachable
material and peaked there — deeper laddering manufactures unreachable material).

The residual 4.18% of wide gates is **not** the fold, which emits none. They are
mask-slot emissions: a degree-3 mask term is a 3-control conjunction every time
it is injected, re-sourced, retired or stripped (`emit_slot`), and that path
still honours `--prod-ladder-cap` alone. Adding `--prod-ladder-cap 3` ladders
those too — single-rung, exactly the measured cap-3 optimum — and takes the
circuit to 99.87% reachable, 0.04% store-blocked, for 3.01x the baseline.

Note what happened to the old cap-3 caution. Laddering was measured to PEAK at
cap 4 and decline after, because deep multi-rung ladders manufacture material the
store cannot reach. That finding is intact and is why the fold does not ladder:
the Gray fold removes the wide fragments instead of laddering them, so cap 3 is
left doing only what it is good at — single-rung width-3 slot emissions.

Wire count is unchanged: accumulators and sandwich helpers are borrowed from the
existing carrier roles and restored.

## The duplicate-gate question, and why it is not paid for here

The Gray structure emits the same function on the same wires several times by
construction — `A_u * A_z` at four phases, each accumulator gate and rung twice
per pass and twice again in the strip. That is exactly the shape the `sort |
uniq -c` census once used to locate every ladder, so it was measured.

Read it against a NULL, not raw. A circuit with 2.4x the gates drawn from the
same shape space collides far more by chance: at 379k CNOTs over 512 wires the
pigeonhole floor alone is ~76%. The null keeps every gate's shape (comp, arity,
polarity pattern) and the per-shape counts and resamples only the wire labels.

Measured at n=128 (`dup_census.py`, 2 null trials):

| n=128 | in duplicate groups (obs / null / excess) | in exact pairs |
|---|---|---|
| wide fold | 18.5% / 7.1% / **+11.4 pts** | 14.9% / 6.0% / **+8.9 pts** |
| gray fold | 82.9% / 36.1% / **+46.9 pts** | 45.1% / 16.0% / **+29.0 pts** |
| gray + cap 3 | 79.9% / 41.4% / **+38.5 pts** | 30.4% / 13.9% / **+16.5 pts** |

Note the third row: laddering the slot emissions dilutes the pair excess as well
as clearing the wide gates, because a ladder's own material is drawn from a
larger effective space.

So there is a real excess, and two attempts to remove it were measured and
REJECTED, both recorded at their call sites so they are not re-tried:

* Forcing the accumulator gate same-polarity (free algebraically — the helper
  literal's polarity cancels in the sweep) costs **+5.7% gates** and improves
  the census by **nothing**.
* Steering the pivot toward a same-polarity rung: same result.

The reason both fail is that the four equal-size spellings of a same-polarity
conjunction are only TWO gate multisets reordered, and the duplicate census —
like `commuting_shuffle`, which runs afterwards anyway — is order-blind.
Removing the excess would need DIFFERENT WIRES per emission, which the
cancellation identity forbids.

What makes this acceptable is the reachability result above: at 95.6% (99.9%
with cap 3) phase A can re-encode essentially every gate in the circuit, where
the wide fold left 55% of it permanently outside the store's reach. The
duplicate structure is a property of the RAW gadget, and the fmix pass is now in
a position to churn it. The honest place to check it is a post-fmix census —
which is an open measurement, not a claim made here.

## Scope and fallbacks

The fold declines, and the block falls through to the odometer with laddering
forced on, when:

* the source gate is not arity 2 (an X or CNOT source: nothing to amortize),
* an operand has no mask terms (the plain expansion is already ≤ `1 + max_deg`),
* a mask term has degree > 3 (a helper chain rather than one sandwich; the
  production plan is `[1,2,3,3]` and the generalization is unaudited),
* the carrier pool leaves no room to borrow two accumulators and a helper —
  which happens only at toy widths.

Borrows are drawn by ROLE through `pairs`/`loc`, never by wire index: the home
index range stops describing the carrier set the moment `--prod-roll` is on, and
borrowing by index leaves a static index-shaped trace that rolling cannot
average away (measured write-count AUC 0.518 -> 0.875 when that was last got
wrong).

## Tests

* `prod_gray_fold_is_share_native_and_two_control` — over the whole input domain
  and six source-gate shapes: the target's decode transitions by exactly the
  virtual gate, every other value is untouched, every borrow is restored, and no
  gate exceeds two controls. This is what catches a residual-parity slip, which
  is otherwise a silent wrong-function bug.
* `prod_gray_fold_keeps_the_accumulators_dirty` — the borrows are drawn from the
  carrier roles, are distinct from the target and every operand literal, and are
  each written an even number of times so their incoming junk survives to cancel.
