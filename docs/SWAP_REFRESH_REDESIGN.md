# The Swap-Refresh Gadgetization (2026-08-20, symmetric-ports revision 2026-08-21)

Redesign of the single-carrier production gadgetizer — the
`gen_sandwich_gadget` sandwich path. The aggregate-Gray fold is replaced
by a per-gate **mask swap-with-refresh**; both ports carry an
independently drawn **junk-half zero-slice guard**; the tail is
leak-hygienic; and the composite is **reverse-honest**: the same artifact
evaluates `C(x)` forward and `D⁻¹(a)` backward, each on its own zero
slice — the gadget-level mirror of the sandwich's symmetry.

## 1. Goal

No GF(2)-linear equation should hold in which some variables are wire
segments of the source (sandwich) circuit and others are wire segments of
the gadgetized circuit — except through the public port boundary
(functions of the shared input at the input port, of the shared outputs
at the output port), which any correct implementation exhibits. The
prior construction failed this three ways, all measured: the endpoint
cancellation (time-invariant masks cancel in every fold's before/after
XOR; flip_match: 100% of linear source gates), the aggregate-Gray
operand witness (`B = c_b ⊕ u⁻ ⊕ u⁺ ⊕ κ`), and the one-sided zero-slice
port.

Additional requirements adopted during the redesign: the construction
must be **symmetric under reversal** (backward evaluation on the honest
reverse slice yields `D⁻¹`), and the emitted stream must remain
**predominantly g57/CNOT** for phase-A digestion.

## 2. The construction

Layout (all under `ProdConfig::production_single()`; `PROD_SWAP=0
PROD_CLOSE_SLICE=0` restores the prior Gray stream byte-for-byte):

```
[P junk-guard] [W0 fill] [inject] [folds + swap/reloc churn] [route home] [strip ALL] [F' fill] [Q junk-guard]
```

- **Per-gate mask swap-with-refresh**: at every fold the target and one
  random control each retire one base-degree monomial and gain a fresh
  draw (fresh band positions — a verbatim move leaks by GF(2)
  conservation, a polarity re-roll by low-degree difference). The
  target-side inject is strictly interior to the fold's fragment stream;
  a target-stable commuting shuffle preserves per-wire write order.
  Every relocation also refreshes one monomial of each moved value.
  Gray modes are declined (the gather is a linear operand recovery).
- **Symmetric ports**: `P` and `Q` are independent draws of one
  generator — identity exactly on the zero band slice, every nonzero
  slice perturbs the data, targets confined to the LOW (forward-junk)
  data half. `Q` fires forward into the junk half (band is junked by
  then); `P⁻¹` fires backward into the same half. Neither touches the
  half where the live payload of its direction emerges.
- **Reverse honesty** (the telescope lemma): a fold reading value *v*
  compensates *v*'s mask fragments against the carrier's reverse-time
  content — the XOR of the emissions *after* the fold — which telescopes
  to *v*'s fold-time slots exactly when the final registry is stripped.
  So EVERY value's registry is discharged at the tail; keeping the junk
  half masked (an earlier revision) corrupted `D⁻¹`. Verified: the
  reversed gadget on `(a, 0, 0-band)` reproduces the reversed sandwich's
  upper half, bit-sliced, at every size, plus a permanent
  `reverse verify` in `gen_sandwich_gadget`.
- **Tail hygiene** (each item closed a measured flip_match class):
  route-home precedes the strip and carries the band-placement map
  (`loc`); the strip's constant-discharge helpers are live band
  VARIABLES via `loc` (the raw band wire range holds displaced carrier
  states after rolls/route); both band fills source from the low data
  half; epoch refills split their channels — the linear pivot is
  band-sourced only (a carrier-sourced pivot linearly copies a masked
  payload-era state into the band, the measured exact leak, since the
  low half is the payload's birthplace mid-circuit) while the product
  terms readmit carriers at `refill_data`% (degree-2 in the values:
  nothing linearly peelable, refill clusters read across the
  band/carrier partition, genuinely new algebra enters the band, no
  rank drift); the ladder borrows band variables, never live carriers.

## 3. Vocabulary: the ladder and the g57 share

The expanded fold's arity-2 product fragments are wide; fmix's store
speaks g57/CNOT. The selective ladder re-spells them; its scratch is the
band pool (live-carrier borrows exposed data states — the measured cap-4
leak). Measured menu at n=128 (production parameters):

| ladder      | gates      | g57-form | CNOT  | conj-2 | wide  | vs Gray size |
|-------------|-----------|----------|-------|--------|-------|--------------|
| cap 0       | 882,768   | 28.7%    | 22.7% | 0.2%   | 48.4% | 0.68× |
| **cap 3** (default) | **1,687,450** | **45.4%** | **36.2%** | 0.1% | 18.3% | **1.31×** |
| cap 4       | 3,297,951 | 34.7%    | 26.6% | 35.4%  | 3.3%  | 2.55× |

Cap 4's extra narrowness arrives as store-weak plain conj-2 gates at
twice cap 3's size, so cap 3 is the default (81.6% pure g57+CNOT);
`PROD_LADDER_CAP` is the campaign A/B lever. (The old Gray build was
97.6% g57+CNOT at 1,292,677 gates — but carries the operand witness.)

## 4. Verification status (n=128, fresh random seeds)

- Function forward AND reverse: bit-sliced verifies pass; 318/318 tests.
- Inputs↔outputs: affine rank of `{1, x, y}` full (257/257) — no linear
  relation between inputs and outputs (conclusive).
- segment_deduce (degree 1 and restricted degree-2 windows, both
  directions, fresh-sample verified): **zero interior-cut equations**;
  all deducible segments on the shared input/output cuts.
- Affine-predictor heatmap vs C: flat at the 0.5 ceiling, port corners
  only; indistinguishable from the Gray control plate.
- flip_match (kwin 12/30/60, 8192 samples): body and nonlinear classes
  at zero in both orientations (forward and reversed). Residual: 0–2
  exact **seam windows** per build (seed-dependent; 6/2/0/5 matched
  source gates over four seeds, all ≥99.4% depth), formed by mid-group
  cuts across the route/strip seam composing mask cancellations; plus
  occasional port-boundary coincidences (early W0/late F' fill products
  over public port values equal to early-C/late-D gate deltas). The seam
  is the region phase-A mixing rewrites most heavily; driving the
  gadget-level flicker to zero at every seed would need per-wire
  alternation of fresh material through the seam — an open refinement.

## 5. History of removed leak channels

Endpoint cancellation (swap-refresh); Gray operand witness (expanded
fold); shuffle window isolation (stable-target shuffle + interior
interleave + zero-term prefilter); bare-value route swaps (route before
strip); bare helper reads (band-variable helpers); payload-through-refill
(band-only refills); relocation mask recurrence (relocation-coupled
refresh); ladder scratch exposure (band borrows); junk-half masking
(removed — it broke reverse honesty; its leak was re-closed by the
above). Each was found and confirmed dead by re-running flip_match.
