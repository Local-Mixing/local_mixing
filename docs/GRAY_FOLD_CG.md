# The Gray-code CG fold

`--prod-gray-fold 1`. Code: `ProdLedger::fold_cg_gray` and `emit_atom_onto` in
`src/preprocessing/gadgets.rs`.

**The full write-up is `docs/GRAY_FOLD_CG.tex` / `.pdf`** — construction, the
per-prefix exposure audit, the two silent failure modes, the measurements, and
the tests. This file is a pointer, deliberately: an earlier full markdown copy
drifted from the LaTeX within a day (it kept a superseded gate count), so there
is one source now.

## One-paragraph version

The fold expands `PROD_w (carrier_w + masks_w)` into one gate per term of the
cartesian product, giving fragments of width up to `arity * max_deg`. Anything
above two controls is material the frozen store cannot digest, and at n=128
that was 56% of the gadget. The Gray fold instead gathers each operand's mask
sum ONCE onto a dirty borrowed accumulator and reads it back four times, walking
the Gray cycle over `(u holds M_b, z holds M_c)`; the borrows' unknown values
cancel between readings. Every emitted gate is then at most two controls with
nothing laddered.

The accumulators must stay DIRTY — `carrier + masks` IS the operand value, so a
clean accumulator re-exposes it, and the audit confirms a clean variant recovers
the operand at correlation 1.0.

## Measured, n=128 sliced sandwich (same sandwich and source C, all verified)

| | wide fold | Gray fold | Gray + `--prod-ladder-cap 3` |
|---|---|---|---|
| gadget gates | 339,786 | 808,618 (2.38x) | 1,021,244 (3.01x) |
| fold fossils (>2 controls) | 153,421 | **0** | **0** |
| **store-reachable gates** | **31.55%** | **95.47%** | **99.87%** |
| store-blocked | 13.49% | 0.25% | 0.04% |

Exposure is at the encoding's own bound (affine 0.28125, degree-2 0.5625) at
every prefix and across the emitted family. `hmap_affine` reads both builds dead
at degree 1 — depthMed 0.0000, zero genuine interior rows. The stress battery
reads both ALIGNED-LEAK on the single-bit target: **the Gray fold buys
digestibility, not hiding**, and must not be sold as a fix for the diagonal.
