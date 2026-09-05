# Sliced Sandwich

**"Sliced sandwich"** is a keyed obfuscation construction on `2n` wires,
distinct from the gadget/feistel paths. It is a slicing mechanism specific to
the sandwich structure: the transformed circuit computes the source function
`C` only on one designated input slice, and its inverse reveals only a random
function on a designated slice.

It comes in two variants, selected by a choice bit (`SandwichVariant`):

| variant | N column | D block | S2 block | forward | inverse |
|---------|----------|---------|----------|---------|---------|
| `classic` (default) | `y ^= x` | low half, beside C | targets low, reads high | `A(x,0) = (junk, C(x))` | `A^-1(p,0) = (junk, D^-1(p))` |
| `balanced` | `x ^= y` | **high half** | **targets high, reads low** | `A(x,0) = (C(x), junk)` | `A^-1(0,q) = (junk, D^-1(q))` |

In the classic variant the low half is a pure workspace and the high half a
pure answer register: both directions are sliced at `y = 0` and both read out
on the high half. In the balanced variant each half hosts one computation:
the forward direction is sliced at `y = 0` and reads out on the low half, the
inverse is sliced at the mirrored `x = 0` and reads out on the high half.

```
sss --cnot --sliced-sandwich -n <n> -s <source> -d <dest> -r <rounds> -m <m> -x <x> \
    [--sandwich-balanced]  # build the balanced variant instead of the classic one
    [--sandwich-m M]   # gates in the random D computation, default n*(log2 n)^2
    [--sandwich-s S]   # gates in each slice block S1, S2, default n*log2 n
```

## Wires

`2n` wires in two halves of `n`:

- first half = wires `0..n-1` (the input `x`),
- second half = wires `n..2n-1` (the auxiliary `y`, the slice register).

## Structure

```
classic:   A = [ C ⧓ S1 ] ; N(y ^= x) ; [ D  ⧓ S2  ]
balanced:  A = [ C ⧓ S1 ] ; N(x ^= y) ; [ D' ⧓ S2' ]
```

(`⧓` = random interleaving.)

- **C** — the source circuit (g57 gates), placed directly on wires `0..n-1`
  in both variants. No secret sharing; this is a plaintext computation whose
  obfuscation comes from the subsequent mixing.
- **D** — a fresh random circuit of `m` g57 gates, the same design as the C
  block. Default `m = n(log2 n)^2`. Classic places it on wires `0..n-1`
  beside C; **balanced places it on wires `n..2n-1`** (`D'`).
- **N** — the copy step, `n` CNOTs. Classic copies UP (`y ^= x`, wire `n+i ^=
  wire i`); **balanced copies DOWN** (`x ^= y`, wire `i ^= wire n+i`).
- **S1, S2** — two independent slice blocks of `s` gates each (default
  `s = n log2 n`). Every gate targets one half and reads at least one wire of
  the other half with positive polarity:
  - CNOT `t_i ^= r_j` (control in the read half), ~1/3 of the gates;
  - CCNOT `t_i ^= t_j & r_k` (one control per half), the rest.

  So each block is **dead when the read half is zero**. S1 always targets the
  first half and reads the second (dead when `y = 0`). S2 does the same in the
  classic variant; in the **balanced** variant it mirrors with D — targets the
  second half, reads the first, **dead when `x = 0`** (`S2'`). S1 is randomly
  interleaved with C, S2 with D (a uniform interleaving that preserves each
  computation's internal order).

**Final float stage** (2026-07-20): the N column is the sandwich's most
structure-revealing feature (the C|N|D seam). Each N CNOT is **assigned an
independent random direction, registered up front**, and then floats in
that direction as far as commutation allows — until it meets a gate that
truly collides per `XGate::collides` (for classic, a write to its control
wire `i` or a read of its target `n+i`; mirrored for balanced), commuting
freely past everything else, including the other N gates. The registered
direction ensures each gate only ever travels one way — no oscillation — and
one float pass provably reaches the fixpoint of same-direction floating: the
blockers are static (only N gates move) and the floaters mutually commute
in either variant (distinct targets, and every control sits in the opposite
half from every target), so further passes could only permute commuting
floaters among themselves, which is travel-free. The contiguous column
dissolves into a band (measured at n=16, m=150: span 15 -> ~76 positions)
whose width is set by how densely the surrounding blocks write the wires the
column touches. Every hop is an adjacent commuting swap, so A's function —
and with it every slice/inverse guarantee — is unchanged.

The column is tracked by its recorded insertion range rather than recovered
from the finished gate list: "the gates targeting the second half" identifies
it in the classic layout only. In the balanced layout the N gates target the
low half and read the high one — exactly the shape of S1's CNOTs — while the
high-targeting gates are D and S2.

## Slice semantics — classic

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

## Slice semantics — balanced

Forward, on the same zero slice `y = 0`, the answer stays on the **first
half**:
\[
  A(x, 0) = (C(x), \text{junk}).
\]
Block 1 is unchanged, so S1 is again dead through the C stage (nothing writes
the second half before N) and `x = C(x)` at the seam. The flipped N step is
then a no-op on the slice, and block 2 writes only the second half — D' lives
there and S2' targets it — so the answer is frozen from the seam onwards and
never needs to be moved out of harm's way. The second half meanwhile evolves
from `0` through `D'` under S2' injections and comes out as junk.

The inverse is sliced on the **mirrored** slice `x = 0`, revealing `D`:
\[
  A^{-1}(0, q) = (\text{junk}, D^{-1}(q)).
\]
Backwards, `[D' || S2']^{-1}` runs first with the first half zero, so S2' is
dead and `D^{-1}` computes cleanly on the second half; N copies it down into
the first half; then `[C || S1]^{-1}` runs with a nonzero second half, so S1
fires and — together with `C^{-1}` — junks the first half, leaving `D^{-1}(q)`
untouched on the second.

Off-slice (`y \neq 0`) the first-half output is `C'(x, y) ^ y` for the
S1-disturbed `C'`: the flipped N step supplies exactly the same `y ^` masking
as the classic one, now on the answer half, so no nonzero slice yields the
clean answer.

**Why S2 must mirror.** With D on the second half, a low-targeting S2 would
fire as soon as D makes the second half nonzero and would overwrite the
forward answer, which now stays in the first half. Mirroring S2 with D is
what restores a slice contract, and it lands the block exactly where the
inverse direction needs it (dead on `x = 0`).

**Two slices, not one.** The balanced variant's two directions are sliced at
*different* points (`y = 0` forward, `x = 0` backward), where the classic
variant slices both at `y = 0`. Equivalently: the classic layout has a half
(the low one) that is junk in both directions, and the balanced layout has
none — every half is somebody's payload. That is the point of the variant.

## Parameters

| symbol | meaning | default |
|--------|---------|---------|
| `n` | wires of the source function (circuit is `2n` wires) | — |
| `m` | gates in the random D computation | `n (log2 n)^2` |
| `s` | gates in each slice block S1, S2 | `n log2 n` |
| variant | `classic` or `balanced` | `classic` |

Requires `n >= 3` (g57 gates need three distinct wires). The size before
mixing is `|C| + m + n + 2s` gates in both variants.

## Invocation

The variant is chosen per run, never per build, and defaults to `classic`
(unbalanced) everywhere:

- `sss --cnot --sliced-sandwich --sandwich-balanced ...`
- `gen_sandwich_gadget ... <gadgetization_mode> <sandwich_variant>` — the final
  positional argument, or `SANDWICH_VARIANT=classic|balanced` in the
  environment, which avoids spelling out the ten positionals ahead of it. An
  explicit positional wins over the environment.

## Verification

The driver checks the pinned contract after transformation and after every
mixing round: the `SandwichSecond` view (classic — second half of the input
forced to zero, second-half output must equal `C(x)`) or the `SandwichFirst`
view (balanced — same pinned slice, first-half output must equal `C(x)`).
Unit tests in `src/preprocessing/gadgets.rs`:
`sandwich_slice_gates_are_dead_on_the_zero_slice`,
`mirrored_sandwich_slice_gates_are_dead_on_the_zero_first_half`,
`sliced_sandwich_computes_c_on_the_second_half_on_the_zero_slice`,
`sliced_sandwich_inverse_is_dead_slice_and_reveals_d_inverse`,
`balanced_sliced_sandwich_computes_c_on_the_first_half_on_the_zero_slice`,
`balanced_sliced_sandwich_inverse_reveals_d_inverse_on_the_mirrored_slice`
(this one checks `D(D^-1(q)) = q` against the explicit D, not just
bijectivity), and `sliced_sandwich_floats_the_middle_column_into_a_band`,
which covers both variants and pins the classic column to the old
"targets the second half" identification.

## Relation to the other constructions

Unlike **new gadgetize** (`docs/NEW_GADGETIZE.md`), the sliced sandwich uses
no secret-sharing gadget and no bookends: C and D are plaintext g57 circuits
on the data wires, and the slice blocks are interleaved through the
computation rather than prepended. Both share the CNOT/CCNOT slice-gate
vocabulary (positive controls, dead on a zero slice, no polarity signature).
The sandwich's two-sided slicing (forward reveals `C`, inverse reveals
`D^{-1}`) is its distinguishing feature.

## Provenance

Code: `SandwichVariant`, `sliced_sandwich_cnot`, `sliced_sandwich_with_d`,
`sliced_sandwich_build`, `random_g57_xgates`, `sandwich_slice_gates`,
`shift_xgate_wires`, `random_interleave`, `sandwich_default_m`,
`sandwich_default_s` in `src/preprocessing/gadgets.rs`; driver branch and
`FunctionView::SandwichSecond` / `FunctionView::SandwichFirst`
in `src/db_mixing/main_mix_cnot.rs`; flags in `src/main.rs` /
`src/commands/sss.rs`; variant argument in
`src/preprocessing/bin/gen_sandwich_gadget.rs`.

Classic circuits are unaffected by the addition of the variant: the choice bit
defaults to `classic` everywhere, and the builder draws the same random
values in the same order as before, so a classic seed still reproduces its
circuit bit for bit.
