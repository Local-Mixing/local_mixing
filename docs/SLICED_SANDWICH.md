# Sliced Sandwich

**"Sliced sandwich"** is a keyed obfuscation construction on `2n` wires,
distinct from the gadget/feistel paths. It is a slicing mechanism specific to
the sandwich structure: the transformed circuit computes the source function
`C` only on one designated input slice, and its inverse reveals only a random
function on the same slice.

```
sss --cnot --sliced-sandwich -n <n> -s <source> -d <dest> -r <rounds> -m <m> -x <x> \
    [--sandwich-m M]   # gates in the random D computation, default n*(log2 n)^2
    [--sandwich-s S]   # gates in each slice block S1, S2, default n*log2 n
```

## Wires

`2n` wires in two halves of `n`:

- first half = wires `0..n-1` (the input `x`),
- second half = wires `n..2n-1` (the auxiliary `y`, the slice register).

## Structure

```
A = [ C interleaved with S1 ] ; N ; [ D interleaved with S2 ]
```

- **C** — the source circuit (g57 gates), placed directly on wires `0..n-1`.
  No secret sharing; this is a plaintext computation whose obfuscation comes
  from the subsequent mixing.
- **D** — a fresh random circuit of `m` g57 gates on wires `0..n-1`, the same
  design as the C block. Default `m = n(log2 n)^2`.
- **N** — the copy step `y ^= x`: `n` CNOTs, wire `n+i ^= wire i`.
- **S1, S2** — two independent slice blocks of `s` gates each (default
  `s = n log2 n`). Every gate targets the first half and reads at least one
  second-half wire with positive polarity:
  - CNOT `x_i ^= y_j` (control in the second half), ~1/3 of the gates;
  - CCNOT `x_i ^= x_j & y_k` (one control per half), the rest.
  So each block is **dead when the second half is zero**. S1 is randomly
  interleaved with C, S2 randomly interleaved with D (a uniform interleaving
  that preserves each computation's internal order).

**Final float stage** (2026-07-20): the N column is the sandwich's most
structure-revealing feature (the C|N|D seam). Each N CNOT is **assigned an
independent random direction, registered up front**, and then floats in
that direction as far as commutation allows — until it meets a gate that
truly collides per `XGate::collides` (a write to its control wire `i`, or a
read of its target `n+i`), commuting freely past everything else, including
the other N gates. The registered direction ensures each gate only ever
travels one way — no oscillation — and one float pass provably reaches the
fixpoint of same-direction floating: the blockers are static (only N gates
move) and the floaters mutually commute, so further passes could only
permute commuting floaters among themselves, which is travel-free. The
contiguous column dissolves into a band
(measured at n=16, m=150: span 15 -> ~76 positions) whose width is set by
how densely C and D write the low wires. Every hop is an adjacent commuting
swap, so A's function — and with it every slice/inverse guarantee — is
unchanged.

## Slice semantics

On the zero slice `y = 0` the answer exits on the **second half**:
\[
  A(x, 0) = (\text{junk}, C(x)).
\]
S1 is dead throughout the C stage (the second half is still zero), so C
computes correctly; the N step copies `C(x)` into the second half; and though
S2 is live during D (the second half now holds `C(x) \neq 0`), it only targets
the junk first half, so it cannot corrupt the answer.

The inverse is sliced on the **same** slice, revealing the random `D`:
\[
  A^{-1}(p, 0) = (\text{junk}, D^{-1}(p)).
\]
Running `A` backwards, the `[D || S2]` stage inverts first with the second
half zero, so S2 is dead and `D^{-1}` computes cleanly; then `[C || S1]^{-1}`
runs with a nonzero second half, so S1 fires and junks the first half. Each
slice block guards one direction's correctness (S1 forward, S2 inverse) and
scrambles the other direction's junk.

Off-slice (`y \neq 0`) the second-half output is `y ^ C'(x, y)` for a
disturbed computation `C'` (S1 fires during C), so it differs from `C(x)`;
the `y ^` masking from the N step guarantees the output is not the clean
answer on any nonzero slice.

## Parameters

| symbol | meaning | default |
|--------|---------|---------|
| `n` | wires of the source function (circuit is `2n` wires) | — |
| `m` | gates in the random D computation | `n (log2 n)^2` |
| `s` | gates in each slice block S1, S2 | `n log2 n` |

Requires `n >= 3` (g57 gates need three distinct wires).

## Verification

The driver checks the pinned contract (`SandwichSecond` view: second half
forced to zero, second-half output must equal `C(x)`) after transformation
and after every mixing round. Unit tests in `src/preprocessing/gadgets.rs`:
`sandwich_slice_gates_are_dead_on_the_zero_slice`,
`sliced_sandwich_computes_c_on_the_second_half_on_the_zero_slice`,
`sliced_sandwich_inverse_is_dead_slice_and_reveals_d_inverse`.

## Relation to the other constructions

Unlike **new gadgetize** (`docs/NEW_GADGETIZE.md`), the sliced sandwich uses
no secret-sharing gadget and no bookends: C and D are plaintext g57 circuits
on the low `n` wires, and the slice blocks are interleaved through the
computation rather than prepended. Both share the CNOT/CCNOT slice-gate
vocabulary (positive controls, dead on a zero slice, no polarity signature).
The sandwich's two-sided slicing (forward reveals `C`, inverse reveals
`D^{-1}`) is its distinguishing feature.

## Provenance

Code: `sliced_sandwich_cnot`, `random_g57_xgates`, `sandwich_slice_gates`,
`random_interleave`, `sandwich_default_m`, `sandwich_default_s` in
`src/preprocessing/gadgets.rs`; driver branch and `FunctionView::SandwichSecond`
in `src/db_mixing/main_mix_cnot.rs`; flags in `src/main.rs` /
`src/commands/sss.rs`.
