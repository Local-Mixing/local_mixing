# Gadgetization

Our current GSS gadgetizer uses **quadratic masking**. We begin with the sliced
sandwich on $2n$ wires and add $2n$ band wires, giving us $4n$ wires in total.
The idea is to keep the intermediate values under nonlinear masks while still
being able to compute on them. All random choices are made when we generate
the circuit. Evaluating the resulting circuit requires only its gate list.

The six-stage order is described in [the pipeline guide](GSS_PIPELINE.md). Here,
we describe what happens in its first two stages, including the shuffles which
are already present before database mixing begins.

## The circuit entering gadgetization

Let $C$ be our source circuit on $n$ wires. The default, classic sandwich is

$$
A=[C\text{ interleaved with }S_1]\ ;\ N\ ;\
  [D\text{ interleaved with }S_2].
$$

Both $C$ and the independent random circuit $D$ operate on the first $n$ wires,
which we call $x$. The second half is a slice register $y$. Every gate in $S_1$
and $S_2$ targets the first half and reads a positive literal from $y$, so these
gates are dead when $y=0$. The middle block $N$ consists of the copy gates
$y_i\mathrel{\oplus{=}}x_i$.

Namely, on the public slice we get

$$
(x,0)\longrightarrow(C(x),0)\longrightarrow(C(x),C(x))
\longrightarrow(\text{junk},C(x)).
$$

The sandwich is already shuffled in two ways. We randomly interleave each
slice block with its computation while preserving the order within each list.
We then assign every copy CNOT a random direction, shuffle the order in which
we process those CNOTs, and move each one as far as it can go by commuting
swaps. Thus, the middle copy column is spread through the circuit before
gadgetization. These moves preserve the complete function of the sandwich.

There is also a `balanced` sandwich variant which places $D$ on the high half
and keeps $C(x)$ on the low half. This is a separate choice from **balanced
masks**, which are enabled by default even with the classic sandwich.

## The five parts of preprocessing

The gadgetizer wraps the masked computation in the following order:

```
opening guard -> band seed -> masked computation -> band reseed -> closing guard
```

These are five parts within stage 2, not five additional GSS stages.

1. **Opening guard.** We add a slice block which reads the outer band and
   targets the low $n$ wires. It is dead when the band is zero. Its slice
   controls are distributed across the band and shuffled; the sampled gates
   are then shuffled as well.
2. **Band seed.** Starting from a zero band, we fill each band wire with
   $B_j=x_a\oplus x_b$, using two distinct wires sampled from the original
   $n$-wire input prefix. This is emitted as two CNOTs. In particular, the
   band is derived from the input; it is not an independent random string.
3. **Masked computation.** We compute the sandwich while opening and closing
   masks, emitting its gates through masked reads, and refreshing the band.
   At the end of this part, we close the remaining masks so that the low
   $2n$ wires contain the sandwich output.
4. **Band reseed.** We apply another collection of two-CNOT updates to the
   band, now reading the current low $n$ wires. This uses a different seed
   from the first fill. It is not the inverse of that fill and does not
   restore the band to zero.
5. **Closing guard.** We apply an independently sampled slice block against
   the resulting band. In the classic sandwich it targets only the low
   junk half, leaving the high-half answer intact. With the balanced sandwich
   it instead targets the high junk half.

The source, sandwich, and gadgetizer have separate random streams. The fill
and reseed also use distinct derived seeds. Thus, we can hold the source and
sandwich fixed while changing the gadgetization.

## Open quadratic masks

Write $V_w$ for a logical sandwich value and $W_w$ for the physical wire which
currently carries it. A mask is *open* after its injection and before its
removal. If $\mathcal O_w$ is the collection of masks currently open on wire
$w$, we maintain

$$
V_w=W_w\oplus\bigoplus_{j\in\mathcal O_w}M_j(B).
$$

At the default mask size, one balanced mask has the form

$$
M_j(B)=1\oplus B_y\oplus B_xB_y\oplus B_z.
$$

We inject it with `g57(w, x, y)` followed by `CNOT(w, z)`. Recall that the g57
increment is $1\oplus y\oplus xy$. Applying the same mask again removes it,
provided its band values have not changed. The code calls these open/close
mask pairs LGIs, for locally geodesic identities.

Why include $B_z$? The g57 term alone is one on three of its four inputs.
Adding a separate uniform bit makes the mask balanced while keeping its
quadratic term. This is a statement about the mask as a function of its band
variables; it does not assert that the input-derived band variables are all
independent.

The default parameters are `mask_pair_wires = 2`, `max_open_masks = 3`, and
`min_open_masks = 2`. The first setting counts the two wires in the quadratic
pair; the balancing wire is additional. We try to choose fresh masks whose
band wires are disjoint from the masks already open on that carrier. Very
small bands can force the sampler to relax this disjointness condition.

The lower bound applies during the masked interior. Before a control is read,
we bring it up to at least two open masks and keep those masks open. The
three-mask setting is a rolling cap for the ordinary mask schedule; temporary
cover masks and refresh replacements can exceed it. Public inputs and outputs
still have unmasked fringes.

We sample filler masks throughout the computation, choosing their target
wires according to the remaining per-wire budget. Each logical write also
opens a mask on its target during the write itself. These two kinds of opens
serve different purposes: fillers cover the values between uses, while the
mid-write mask hides the complete firing increment of that write.

## Computing through the masks

Suppose a source gate is

$$
V_t\mathrel{\oplus{=}}1\oplus V_b\oplus V_aV_b.
$$

Let $P_a=W_a\oplus M_a(B)$ and $P_b=W_b\oplus M_b(B)$ be the full decodes of
its controls. We implement the update by expanding

$$
W_t\mathrel{\oplus{=}}1\oplus P_b\oplus P_aP_b.
$$

This is **quadratic fire**. Each operand is quadratic in the current physical
wires, so their product has degree at most four. We emit its contributions
without first placing $V_a$ or $V_b$ on a wire. The earlier linear read
temporarily changed a control's mask into an affine expression. Those
intervals could then grow when later stages reordered the gates. The current
read keeps the quadratic masks in place.

The degree-three and degree-four terms are implemented using dirty band wires:
we borrow their current values and restore them at the end of each small
block. For example, the sequence

```
t ^= h & c
h ^= a & b
t ^= h & c
h ^= a & b
```

adds $abc$ to $t$ and restores $h$, regardless of the value with which $h$
started. Degree-four terms use an eight-gate block with two such helpers.
Thus, the fire itself uses gates with at most two controls and adds no clean
scratch wires. The surrounding slice guards can still have three controls.

We treat each of these complete blocks as one *fire unit*. The order inside a
unit matters. Its net action restores its helpers and only updates the target,
so the complete units commute. We divide the fire into two halves and shuffle
the units within each half independently. The resulting order is

```
open temporary target cover
    shuffled first half of the fire
    open a persistent target mask
    shuffled second half of the fire
close temporary target cover
```

The temporary cover is sampled away from the band wires in the operand
polynomials where space permits. The persistent mask stays open after the
fire. Therefore, looking at the target immediately before and after this
whole block gives the logical update together with a new mask.

## Refreshing the band and preserving the order

The band changes during the computation as well. We spread refresh slots
through the mask and gate placements. At the default pair size $k=2$, a
sandwich with $m$ gates gets $\lfloor m/(4k)\rfloor$ slots, each containing
$8k=16$ updates to one sampled band wire. Each update has the form

$$
B_j\mathrel{\oplus{=}}\ell_a\ell_b,
$$

where each literal is sampled from the live low data prefix or the band.
The current default uses both pools. The helper wires used inside quadratic
fire, by contrast, are drawn only from the band.

We cannot change $B_j$ while leaving its old masks unexplained. In the default
refresh, we first open replacement masks where needed to retain coverage,
then close the masks which read $B_j$, and only then apply its burst. The
optional repair refresh instead removes the affected terms before the burst
and reapplies them using the new band value afterward. Repair slots are off
by default. When both types are requested, their slot plan is shuffled before
it is spread through the computation.

These interior bursts do not replace the separate reseed after the compute.
The final order remains: finish the computation, close its remaining masks,
reseed the band, then apply the closing guard.

The sandwich gates themselves are placed using a dependency-ready queue.
Reads remain after the earlier writes they depend on, and writes remain after
earlier reads of that wire. Gates which only XOR into the same target can
commute. The queue is FIFO; the randomness comes from the sampled masks,
refreshes, and fire-unit shuffles. This construction keeps the carrier and band
wire labels fixed. It does not insert the physical wire-swap network used by
the earlier paired gadgetizer.

## What the output promises

For the default classic sandwich, the layout is

| Wires | Input | Output on the public slice |
|---|---|---|
| $0,\ldots,n-1$ | $x$ | junk |
| $n,\ldots,2n-1$ | zero sandwich register | $C(x)$ |
| $2n,\ldots,4n-1$ | zero band | junk |

Thus,

$$
G(x,0^n,0^{2n})=(\text{junk},C(x),\text{junk}).
$$

The circuit remains reversible on all its wires. The zero inputs are the
public slice used by the caller, and the junk outputs are unconstrained. With
the classic layout, the symmetric guards also preserve the reverse slice:
$G^{-1}(p,0,0)$ carries $D^{-1}(p)$ on the high sandwich half. With the balanced
sandwich, the forward payload is instead on the low half.

The generator checks the forward payload against the sandwich on 256
bit-sliced samples, and checks the classic reverse payload on another 256.
The algebraic identities explain why the updates preserve the function;
these runtime checks catch construction mistakes.

## The optional nonlinear291 mode

`preprocessing.mode = "nonlinear291"` selects a different supported
construction. Each logical value uses two five-wire shares decoded by

$$
E(s)=s_0\oplus s_1\oplus\operatorname{maj}(s_2,s_3,s_4),
\qquad V=E(S_1)\oplus E(S_2).
$$

It uses fixed gate templates, fresh per-operation storage, an ingress slice
guard, and an egress which writes the decoded answer to the public wires.
For $q$ sandwich wires and $m$ sandwich gates, its width is $12q+12m+29$,
so the $4n$ width of quadratic masking does not apply. After construction,
eight bounded passes randomly swap adjacent gates only when they commute.
Its complete stage-2 output has at most two controls per gate.

The earlier product-share construction with Gray folding is historical. It is
not the implementation selected by either of these two current mode names.

## Where this lives in the code

| File | What to look for |
|---|---|
| [`sandwich/construct.rs`](../src/stages/sandwich/construct.rs) | Slice blocks, interleaving, and floating the copy column. |
| [`preprocessing/construct.rs`](../src/stages/preprocessing/construct.rs) | The five-part envelope and guard placement. |
| [`preprocessing/quadratic_masking.rs`](../src/stages/preprocessing/quadratic_masking.rs) | `production`, `emit_lgi`, `product_fire`, band seeding, dependency order, and the refresh/open/close loop. |
| [`preprocessing/slice_guards.rs`](../src/stages/preprocessing/slice_guards.rs) | Guard sampling and its zero-slice checks. |
| [`preprocessing/verify.rs`](../src/stages/preprocessing/verify.rs) | Forward and reverse payload verification. |
| [`preprocessing/nonlinear291.rs`](../src/stages/preprocessing/nonlinear291.rs) | The optional E-encoding adapter, resource plan, and final commuting passes. |
| [`gen_sandwich_gadget.rs`](../src/programs/gen_sandwich_gadget.rs) | Source loading, separate seeds, mode selection, and writing the artifacts. |

See the [README](../README.md) for running GSS and selecting preprocessing
parameters.
