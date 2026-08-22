# Gadgetize Update — Change Summary (2026-08-20 → 2026-08-22)

What changed in the gadgetization step (the `gen_sandwich_gadget`
single-carrier production path) across this update cycle, commit series
`fa80bc96 … HEAD` on `ssg-gen-mix-clean`. The companion design note with
the full rationale and algebra is `docs/SWAP_REFRESH_REDESIGN.{md,pdf}`;
the operational surface is documented in `docs/GSS_MIX.md` (stage 1+2).

## 1. What was replaced

The aggregate-Gray fold era construction: time-invariant `[2,2,2,3]`
masks whose cancellation in every fold's before/after XOR left the exact
identity `carrier⁺ ⊕ carrier⁻ = src⁺ ⊕ src⁻` (100% of linear source
gates), the Gray gather's linear operand-recovery witness, and a
zero-slice guard at the input port only.

## 2. New mechanisms (in emission order of the layout)

1. **Symmetric junk-half port guards** — both ports carry independent
   draws of one slice-guard generator (identity exactly on the zero band
   slice; every nonzero slice perturbs the data; targets confined to the
   low, forward-junk data half).
2. **Per-gate mask swap-with-refresh** — at every fold, the target value
   and one randomly chosen control each retire one base-degree mask
   monomial and gain a freshly drawn one (fresh band positions; verbatim
   moves and polarity re-rolls are provably unsound). The target-side
   inject is placed strictly interior to the fold's fragment stream.
3. **Gray declined; expanded fold + selective ladder (cap 3)** — the
   Gray gather materializes an operand's complete mask sum as an
   accumulator segment pair (a linear operand recovery no mask shuffle
   removes), so swap mode takes the expanded fold; the ladder re-spells
   its wide product fragments into the g57/CNOT vocabulary, borrowing
   scratch from live band variables (live-carrier borrows leaked data
   states through chain deltas).
4. **Target-stable commuting shuffle** — same-target XOR writes keep
   emission order, so the interior interleave survives reordering
   exactly rather than probabilistically.
5. **Relocation-coupled refresh** — every relocation also refreshes one
   monomial of each moved value (a payload value is a fold target
   exactly once; its mask function must not recur across stops).
6. **Tail hygiene** — route-home precedes the strip and carries the band
   placement map; every value's registry is discharged (the telescope
   lemma: an undischarged registry breaks reverse evaluation); the
   strip's constant-discharge helpers are live band variables resolved
   through `loc`; both band fills source from the low data half.
7. **Split-channel epoch refills** — the linear pivot is band-sourced
   only (a carrier pivot linearly copies a masked payload-era state into
   the band — the measured leak), while product terms readmit carrier
   sources at `refill_data`% (degree-2 in the values; restores
   band/carrier dataflow homogeneity, fresh band algebra, and rank
   independence).
8. **Peephole cleanup** — commutable identical gate pairs (coinciding
   residue CNOTs) are cancelled to a fixpoint after the final shuffle
   (−2 to −3.5%); the deliberate mask redundancy is collision-guarded
   and untouchable by any function-preserving local pass.

## 3. Contract changes

- The composite preserves the sandwich on the **upper data half only**
  (the payload contract; the closing guard deliberately perturbs the
  junk half). `gen_sandwich_gadget`'s bit-sliced verify checks this.
- **Reverse honesty**: the reversed gadget on `(a, 0, zero band)` yields
  `D⁻¹(a)` on the upper half — one artifact evaluates `C` forward and
  `D⁻¹` backward, mirroring the sandwich's own symmetry. A permanent
  reverse verify runs at generation time.
- Legacy escape hatch: `PROD_SWAP=0 PROD_CLOSE_SLICE=0` reproduces the
  prior Gray stream byte-for-byte.

## 4. Parameters

| knob | values | effect (n=128, full sandwich) |
|---|---|---|
| `PROD_SWAP` / `PROD_CLOSE_SLICE` | 1/1 default | the new construction; 0/0 = legacy Gray |
| `PROD_LADDER_CAP` | **3** default | 1.62M gates, 82% g57+CNOT; 0 → 883k but 48% wide; 4 → 3.3M, conj-2-heavy |
| `PROD_K` | **3** default / 2 | 2 → −18.5% gates at 2× piling-up ε |
| `PROD_CG_JITTER` | **50** default / 0 | 0 → −8.8% gates, spends fragment-count entropy |
| both size levers | | −27.3%: 1,174,130 full / 639,317 half |

## 5. Measured results (n=128, production parameters, fresh seeds)

- **Size**: 1,616,025 full / 879,196 half (with cleanup pass), vs
  1,292,677 (Gray) — 1.25× for the full security upgrade; down to
  1.17M / 639k with both size levers.
- **Relations**: flip_match 0 body matches in both orientations
  (residual: 0–2 exact tail-seam windows per seed, ≥99.4% depth, plus
  occasional port-boundary fill coincidences); segment_deduce zero
  interior-cut equations at degree 1 and windowed degree 2, both
  directions; input↔output affine rank full (257/257); affine-predictor
  heatmap flat at the 0.5 ceiling, indistinguishable from the Gray
  control plate.
- **Function**: forward + reverse verifies green at every size;
  gadget-family tests 140/140.
- **Phase-A digestibility** (100k-move probe on .242, curated-first
  cascade): splice rate **56.5%** vs 53.7% baseline, curated share
  **88.7%** vs 87.3%, 6.28 bits/splice; the g57 share *rises* 46.6% →
  58.5% through phase A; the compression channel succeeds on only 2.1%
  of attempts (the mask material resists removal).

## 6. Known residuals and cautions

- The 0–2 per-seed tail-seam windows (mid-route-swap cuts composing
  through the strip seam) — accepted; phase A rewrites that region
  first; per-seed zero would need per-wire fresh-material alternation.
- `PROD_POST_FRAGMENT` re-opens broad linear leakage under swap mode —
  off-limits for deliverables until audited.
- The multi-carrier presets inherit the fold-level swap but not the
  output-port hygiene.
- Mid-body refill balance remains a measured property, not a theorem
  (no pristine pivot exists mid-body).
