# New Gadgetize

**"New gadgetize"** is the canonical name for the current ×2 gadgetization
recipe (2026-07-20). Use it for all new gadget-path runs:

```
local_mixing_bin sss --cnot --gadgetize --slice-zero-ccnot \
    -n <n> -s <source> -d <dest> -r <rounds> -m <m> -x <x> \
    [--slice-zero-ccnot-gates <G>]   # default 10n
    [--rg-frequency <K>]             # default 2
```

It differs from the plain `--gadgetize` path in two ways: a **zero-slice
preblock** (S1) at the input, and a leaner, faster **RG randomization
policy** in the gadget body.

## Structure

The emitted circuit lives on 2n wires (`x` = wires `0..n`, second half =
wires `n..2n`) and reads, left to right:

```
[ S1 slice preblock ] [ bookend ] [ W_i encode ] [ SG/RG body ] [ decode ] [ bookend ]
      10n gates         2n·ln n      7n CNOTs                       <=7n      2n·ln n
```

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

## RG policy — basis {RG2, RG3}, two per gap

The gadget body interleaves re-randomization gadgets between the SG gadgets.
As of new gadgetize, only the orthogonal basis is used:

- **RG2** (3 CNOTs): swaps one carrier of pair *i* with one of pair *j* and
  re-pairs crosswise — the sole re-pairing move.
- **RG3** (2 CNOTs): XORs a foreign carrier into both carriers of one pair —
  the sole cross-value mask injector.

RG1 (the 6-CNOT value-swap) is dropped from this path: it is a composite of
two generalized RG2s plus a mask stir that RG3 covers.

Rate: `--rg-frequency` (default **2**) independent uniform RG2/RG3 draws are
emitted **between every two consecutive SGs** (none after the last SG). Note
the flag's meaning changed for this path — it used to mean "one RG every K
SGs"; the feistel and legacy paths keep the old meaning.

Both RGs are gate-locally non-complete (no physical gate ever reads both
carriers of one logical value), so first-order prefix masking is preserved.

## Verification

The driver checks the pinned contract (`GadgetLow` view with the second half
forced to zero) after transformation and after every mixing round. Unit
tests: `ccnot_preblock_fixes_exactly_the_zero_slice`,
`ccnot_preblock_uses_only_the_agreed_gate_shapes`,
`slice_zero_ccnot_gadgetize_matches_only_on_the_zero_slice`,
`slice_block_stops_the_inverse_from_revealing_c_inverse`, plus the
zero-round driver test.

## Open thread

S1's off-slice map is a shallow affine randomizer at the default budget; its
strength may matter for post-mixing inversion resistance. Knob:
`--slice-zero-ccnot-gates`. A planned measurement: avalanche/dislocation of
an isolated S1 vs gate budget at production n.
