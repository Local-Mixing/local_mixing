# The GSS Pipeline

GSS means gadgetized sliced sandwich. We start with a reversible circuit $C$
on $n$ wires and pass it through six steps. The first two build the
representation we want to mix. The next three change its local structure, and
the last step removes redundancy and packs the circuit.

```text
C → sliced sandwich → gadgetization → DB mixing → splitting → crossing → final.esop1
        step 1           step 2         step 3       step 4      step 5       step 6
```

This is a guide to the current code. For commands and configuration, start
with the [README](../README.md).

## Where a run starts

The `gss` command enters through [src/main.rs](../src/main.rs) and
[src/commands/gss.rs](../src/commands/gss.rs). From there,
[src/gss/config/](../src/gss/config/) reads and checks the recipe, and
[src/gss/runner.rs](../src/gss/runner.rs) prepares the run, checks any saved
manifest, and selects the stage binaries.

Then [scripts/gss_mix.sh](../scripts/gss_mix.sh) runs the six steps below.
This is the easiest file to read when we want to see the exact stage order,
arguments, or output names. Steps 1 and 2 share one generator process;
`fmix` runs steps 3–5, and `fcompress` runs step 6.

## 1. Build the sliced sandwich

We put $C$ inside a circuit on $2n$ wires. In execution order, the classic
sandwich is

$$
[C\text{ interleaved with }S_1]\ ;\ N\ ;\
[D\text{ interleaved with }S_2].
$$

$D$ is a separately sampled circuit, $N$ copies the first half into the second
with CNOTs, and the slice blocks $S_1,S_2$ use the second half as controls.
On inputs $(x,0^n)$, the second output half is $C(x)$; the first half may
contain junk. We also move the copy CNOTs through gates they commute with so
they do not remain in one obvious column.

Read [src/stages/sandwich/construct.rs](../src/stages/sandwich/construct.rs),
starting with `prepare_source`, `construct_seeded_sandwich`, and
`sliced_sandwich_cnot`.

## 2. Gadgetize the sandwich

We now hide each sandwich value behind a nonlinear mask. The default is
quadratic masking: one carrier and one band wire per sandwich value, taking
us from $2n$ to $4n$ physical wires. The construction surrounds the masked
computation with slice guards and input-derived band seed/reseed blocks.

At the end, on input $(x,0^n,0^{2n})$, wires $n,\ldots,2n-1$ contain $C(x)$.
The remaining outputs are junk. The full circuit is reversible, but this is
the public slice on which we ask it to reproduce the source computation.

Start at `preprocess_sandwich` in
[src/stages/preprocessing/construct.rs](../src/stages/preprocessing/construct.rs).
The masking and shuffles are in
[quadratic_masking.rs](../src/stages/preprocessing/quadratic_masking.rs);
the guard and seed blocks are in
[slice_guards.rs](../src/stages/preprocessing/slice_guards.rs).
[verify.rs](../src/stages/preprocessing/verify.rs) checks the promised outputs.
The optional `nonlinear291` mode has its own adapter in the same directory.
See [gadgetization](GADGETIZATION.md) for the mask and shuffle construction.

The generator's command handling is in
[src/programs/gen_sandwich_gadget.rs](../src/programs/gen_sandwich_gadget.rs).
Steps 1–2 write `gss.mpmct1`, its source/sandwich sidecars, and `stage12.log`.

## 3. Mix with the frozen database

We sample small circuit windows and replace them with different spellings of
the same function. To find these spellings, we compose the window's
polynomials, canonicalize its wire labels, and query the frozen tables.
A candidate must be mapped back to the window and checked before insertion.

The default schedule grows toward twice the incoming size and then holds it
there while continuing replacements. Its profile is `3,30,30,2,2`. There is no
final shrinking leg in this GSS stage. Optional leakage repair runs here too.
The output is `db_mixing.mpmct1`, with `db_mixing.state` and `stage3.log`.

Start with [src/programs/fmix/mod.rs](../src/programs/fmix/mod.rs) for how the
mode is selected. Follow the mixer loop in
[src/engine/mixer/runtime.rs](../src/engine/mixer/runtime.rs).
[src/engine/mixer/replacement.rs](../src/engine/mixer/replacement.rs) samples,
verifies and splices the replacement; it calls
[src/stages/db_mixing/replacement.rs](../src/stages/db_mixing/replacement.rs)
for lookup and candidate selection.
[src/canonicalization/](../src/canonicalization/) builds the keys;
[src/database/](../src/database/) reads and decodes the stored values. The
[frozen DB](FROZEN_DATABASE.md) and
[canonicalization](POLYNOMIAL_CANONICALIZATION.md) guides explain these parts.

## 4. Split the remaining complemented gates

The database still gives us many gates with the same r57 shape. We split
their complemented firing conditions into plain conjunctions. For instance,

$$
a\mathrel{\oplus{=}}b\lor\neg c
\quad\longrightarrow\quad
\{a\mathrel{\oplus{=}}b,\quad a\mathrel{\oplus{=}}\neg b\wedge\neg c\}.
$$

The two conditions are disjoint, and their XOR is the original condition.
The stage also tries absorbed-NOT twists between compatible brackets and
shoots fragments through the circuit. It stops when the complemented-gate
pool is exhausted or repeated bracket searches reach the failure limit.
It then writes `split.mpmct1`, `split.state`, and `stage4.log`.

Read `split_twist_move` in
[src/stages/post-processing/splitting.rs](../src/stages/post-processing/splitting.rs).
The local rewrite identities live in
[src/engine/moves/rules.rs](../src/engine/moves/rules.rs).

## 5. Run the crossing walk

Now we move fragments toward collisions and rewrite the colliding gates so
their pieces can cross. These moves can introduce wider conjunctions than the
original r57 gates. Undo and merge moves remove compatible fragments again,
while the size target and width penalty control growth.

This step resumes `split.state`. It writes `crossing.mpmct1`,
`crossing.state`, and `stage5.log`. The current default target is twice the
split circuit's size, with six times that target as the move-attempt budget.

Read `cross_move_on`, `undo_move`, and `merge_move` in
[src/stages/post-processing/crossing.rs](../src/stages/post-processing/crossing.rs).
The scheduler and acceptance logic are in
[src/engine/mixer/scheduling.rs](../src/engine/mixer/scheduling.rs).

## 6. Compress and pack

Finally, we gather compatible gates with the same target and simplify their
combined firing function. We repeat this with transport and reduction passes,
then pack each consecutive same-target run into a generalized gate. Its
activation function is reduced through ANF and a deterministic ESOP spelling.

The final file is `final.esop1`, with `stage6.log`. A packed-gate count and an
expanded cube count measure different things, so keep both in mind when
reading the compression log.

Read [src/programs/fcompress.rs](../src/programs/fcompress.rs), then
`compress_anc` in
[src/stages/post-processing/compression/mod.rs](../src/stages/post-processing/compression/mod.rs).
The same directory separates transport, reduction, and
[packing](../src/stages/post-processing/compression/packing.rs).

## The shared pieces

We do not need to understand every file to change one stage. The circuit
representation is in [src/circuit/xgate.rs](../src/circuit/xgate.rs), and its
readers/writers are in [src/circuit/formats.rs](../src/circuit/formats.rs).
The mutable gate tape used by `fmix` is in
[src/engine/arena.rs](../src/engine/arena.rs). Mixer state, sampling, checkpoint
I/O, and reporting live together in [src/engine/mixer/](../src/engine/mixer/).

Stages 3–4 can run on contiguous pieces in parallel; the coordinator is
[piecewise.rs](../src/engine/mixer/piecewise.rs). Correctness checks live in
[tests/](../tests/), while attacks and leakage measurements live in
[security_tests/](../security_tests/). The main
[research document](Local_Mixing_Documentation.pdf) gives the motivation and
earlier experiments behind this order.
