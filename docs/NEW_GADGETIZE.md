# New Gadgetize

**"New gadgetize"** is the canonical name for the current ×2 gadgetization
recipe (2026-07-20; RG policy + final shuffle revised 2026-07-20b). Use it
for all new gadget-path runs:

```
local_mixing_bin sss --cnot --gadgetize --slice-zero-ccnot \
    -n <n> -s <source> -d <dest> -r <rounds> -m <m> -x <x> \
    [--slice-zero-ccnot-gates <G>]   # default 10n
    [--rg-frequency <K>]             # default 1 on this path
```

It differs from the plain `--gadgetize` path in two ways: a **zero-slice
preblock** (S1) at the input, and the **RG randomization policy** in the
gadget body (nonlinear g57 networks, one per SG). The whole output is
finished with a **commuting-order rerandomization** (see below).

## Structure

The emitted circuit lives on 2n wires (`x` = wires `0..n`, second half =
wires `n..2n`) and is CONSTRUCTED, left to right, as:

```
[ S1 slice preblock ] [ bookend ] [ W_i encode ] [ SG/RG body ] [ decode ] [ bookend ]
      10n gates         2n·ln n      7n CNOTs                       <=7n      2n·ln n
```

The construction order is not the emitted order: a final commuting shuffle
re-draws the gate order as a random linear extension of the wire-dependency
partial order, so slice-block, bookend, W_i, and body gates interleave
wherever dependencies allow.

Contract: **`A(x, 0) = (C(x), J)`** — on the all-zero slice of the second
half, the low n output wires carry `C(x)` and the high wires carry junk.
On every other slice the circuit computes `C` of an affinely disturbed
input. The slice is the canonical zero vector (public; no metadata sidecar).

## S1 — the zero-slice preblock (`slice_zero_ccnot_preblock`)

`--slice-zero-ccnot-gates` gates (default 10n) in **one uniform random
order**, drawn from exactly two shapes, all targets on the first half:

- CNOT `x_i ^= a_j` (control on the second half), ~1/3 of the gates;
- CCNOT `x_i ^= x_j & a_k` (one control per half), ~2/3 of the gates.

Every gate reads at least one second-half wire with positive polarity, so on
the zero slice every gate is individually dead: S1 is the identity there,
gate by gate, independent of order. Off-slice, for fixed `a != 0`, S1 is an
invertible affine map on `x`.

Guarantees:
- The CNOT pin set is resampled until its target-by-control parity matrix
  `C` is invertible, which makes the off-slice disturbance **exact** on any
  slice that fires no CCNOT (`x ^= C·a`, nonzero).
- For slices firing CCNOTs the "no other slice is fixed" property is
  **heuristic**: measured wrong-slice-fixed rates at the 10n default are
  ~4e-3 (n=3), 4e-4 (n=4), ~2-4e-5 (n=5-6), 0/50k (n=8) — decaying fast in
  n, negligible at production widths.
- Inverse protection: in the inverse circuit S1⁻¹ runs last and fires on the
  gadget's mask residue, junking the low half — without it, a bare gadget's
  inverse hands out `C⁻¹` on the low wires for **any** junk input.

Design rationale: the block is pure positive-polarity CNOT/Toffoli material —
indistinguishable in vocabulary from ordinary mixed-circuit content, with no
complemented gates and no polarity pattern encoding a slice.

## RG policy — nonlinear {RG1, RG2, RG3}, one per SG

The gadget body interleaves re-randomization gadgets between the SG gadgets.
The policy (revised 2026-07-20b) reinstates the legacy NONLINEAR g57
networks, drawn uniformly:

- **RG1** (6 g57s, ANF degree 3): swaps the virtual values of pairs *i*, *j*.
- **RG2** (6 g57s, degree 2/3): re-pairs pairs *i*, *j* crosswise while
  keeping both virtual values.
- **RG3** (2 g57s, degree 2): XORs `r1 OR NOT r2` of two random foreign
  wires into both carriers of one pair — a nonlinear cross-value mask
  refresh.

Rate: `--rg-frequency` uniform draws between every two consecutive SGs (none
after the last SG); the sss `--cnot` driver defaults this to **1** — one
random RG per SG. The feistel and legacy paths keep the old flag meaning
("one RG every K SGs", default 2).

Why the reversal from the earlier linear {RG2, RG3} CNOT basis: the
`hmap_affine` degree-1/degree-2 reconstruction maps showed the affine RGs
leave the body's re-randomization transparent to low-degree predictors — the
old gadget's non-affine encoding was precisely what blocked degree-2
readout. The trade is deliberate: RG1/RG2 gates read both carriers of a
value, so gate-local non-completeness (first-order probe masking) is given
up in exchange for low-degree opacity. The linear emitters remain in the
codebase (`emit_rg{1,2,3}_x`) for the feistel path and for comparison runs.

## Final commuting-order rerandomization (`commuting_shuffle`)

After assembly (including S1), the gate order is re-drawn preserving only
the relative order of pairs that actually collide per `XGate::collides` —
gates commute unless proven otherwise. In particular a read/write crossing
alone does not pin a pair: two conjunction gates sharing a control at
opposite polarities have disjoint firing supports and reorder freely (the
separation exemption; complemented g57s get no such exemption).
Implementation: alternating-direction randomized insertion passes — each
gate lands uniformly within its maximal commuting span, giving it global
reach up to its nearest truly-colliding neighbors.

Motivation: the heatmaps of gadgetized circuits showed an S-shape — no
mixing between the computation block and the bookends, with the linear W_i
curtain sitting between the Z gates and the body. The shuffle dissolves the
construction-time block layout: W_i CNOTs, slice-block gates, and bookend
g57s migrate into any position their wire dependencies allow, at zero
functional cost (the emitted circuit computes the identical function, so
all slice/inverse guarantees are inherited verbatim).

## Verification

The driver checks the pinned contract (`GadgetLow` view with the second half
forced to zero) after transformation and after every mixing round. Unit
tests: `ccnot_preblock_fixes_exactly_the_zero_slice`,
`ccnot_preblock_uses_only_the_agreed_gate_shapes`,
`slice_zero_ccnot_gadgetize_matches_only_on_the_zero_slice`,
`slice_block_stops_the_inverse_from_revealing_c_inverse`, plus the
zero-round driver test. The 2026-07-20b revision adds
`commuting_shuffle_preserves_function_and_relocates_gates` and
`gadget_body_carries_nonlinear_rg_material`.

## Open thread

S1's off-slice map is a shallow affine randomizer at the default budget; its
strength may matter for post-mixing inversion resistance. Knob:
`--slice-zero-ccnot-gates`. A planned measurement: avalanche/dislocation of
an isolated S1 vs gate budget at production n.
