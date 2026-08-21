# The Swap-Refresh Gadgetization (2026-08-20)

Redesign of the single-carrier production gadgetizer — the
`gen_sandwich_gadget` sandwich path — replacing the aggregate-Gray fold
with a per-gate **mask swap-with-refresh**, adding a **closing zero-slice
block**, and closing every measured output-port channel. The result
eliminates all detected linear (and degree-2) relations between the two
circuits' wire segments **and** is about 30% smaller than the construction
it replaces.

## 1. Goal

No GF(2)-linear equation should hold in which some variables are wire
segments of the source (sandwich) circuit and others are wire segments of
the gadgetized circuit — except through the public boundary (functions of
the shared input at the input port, of the shared output at the output
port), which any correct implementation exhibits.

The prior construction failed this in three ways, all measured:

1. **Endpoint cancellation.** Every logical value decodes as
   `v = c_v ⊕ M₁ ⊕ M₂ ⊕ M₃ ⊕ H ⊕ κ` with the mask monomials `Mᵢ`
   (degree 2) and `H` (degree 3) products of *frozen* band literals. A
   fold writes the control's decode onto the target's carrier and never
   touches the target's own mask set, so the masks cancel in every
   before/after XOR across the gate:
   `carrier(post) ⊕ carrier(pre) = src(post) ⊕ src(pre)` —
   an exact cross-circuit linear identity. flip_match measured it at
   **100%** of linear source gates and 32.4% of nonlinear ones.
2. **The Gray operand witness.** The aggregate-Gray fold gathers an
   operand's complete mask sum onto one accumulator wire and strips it
   back, so the operand's value is linearly reconstructible from three
   gadget segments: `B = c_b ⊕ u⁻ ⊕ u⁺ ⊕ κ` (the space-time recovery
   documented in `CARRIER_GADGETIZATION_SUMMARY.tex`).
3. **Port asymmetry.** The zero-slice guard existed only at the input
   port; a reverse evaluator met a structurally different entry.

## 2. Execution

### 2.1 Per-gate mask swap-with-refresh

At every fold, the **target** value and **one randomly chosen control**
each retire one base-degree mask monomial and gain a freshly drawn one
(`ProdLedger::swap_refresh_side`). The target-side inject is emitted
**strictly interior** to the fold's fragment stream; the control-side pair
follows the fold, pinned by read/write collisions. Consequently no
contiguous window of any carrier's writes XORs to a clean operand decode:
every window that covers the fold picks up at least one non-cancelling
fresh monomial.

Two design lemmas fixed the shape of this mechanism:

- **Verbatim moves are unsound.** Moving a monomial `M` between two mask
  sets emits it once on each touched wire, so over GF(2) the two
  carriers' deltas XOR back to the source delta:
  `Δ(t-carrier) ⊕ Δ(c-carrier) = ΔT`. Conservation makes *any* exact
  move leak; the arriving monomials must be fresh draws.
- **Fresh means fresh band positions.** A polarity re-roll of the same
  product differs from it by a polynomial of degree ≤ deg−1 in the band
  wires, whose values are themselves wire segments — back inside a linear
  adversary's span.

Swap mode **declines every Gray mode** (aggregate, micro, sentinel): the
gathers materialize an operand's whole mask sum as an accumulator segment
pair regardless of mask ownership, and their gather/strip snapshot cannot
tolerate a mid-block registry change. The expanded (odometer) fold is used
instead. A new **target-stable commuting shuffle** preserves per-wire
write order (same-target XOR writes commute, and the standard shuffle
would re-expose a clean window at ~14% of arity-1 folds).

### 2.2 The closing zero-slice block

An independently drawn slice guard with the opening block's
specification — identity exactly on the zero band slice, every nonzero
band slice perturbs the data — is appended after the mirror fill, so a
reverse evaluator meets the same structure at their entry as a forward
evaluator does. Its targets are confined to the low (forward-junk) data
half: the honest forward run reaches the output port with a junked band,
so the block fires there and must only perturb junk. The composite
therefore preserves the sandwich on the **upper data half only** — which
is the payload contract (`verify_zero_slice` checks wires `n..2n`).

### 2.3 Output-port hygiene

Reaching actual zero required closing four further channels, each found
by re-running flip_match after the previous fix:

- **The junk half is never stripped.** Bare junk segments at the tail
  hand their local pair-XORs to the source's own segment XORs; nothing
  downstream decodes a junk value, so its masks stay on forever.
- **Route-home runs before the strip** (carrying the ledger's band
  placement map through the swaps), so the routing swaps move masked
  carriers, never bare payload values.
- **No never-bared wire is read linearly at the tail**: the strip's
  constant-discharge helper, both band fills, and the epoch refills all
  source exclusively from the still-masked junk half and the band space.
  (A refill that copies a payload carrier into a band wire stores the
  payload bit where a half-captured emission window can read it back.)
- **Relocation-coupled refresh.** A payload value is a fold target
  exactly once (its N gate), so its mask function would otherwise recur
  identically at every relocation stop, and a segment pair cutting
  through two matching stops recovers the single value transition
  exactly. Every relocation now also refreshes one monomial of each moved
  value: every representation event is a fresh function.

## 3. Why it ended up *smaller*

The size reduction was not a goal; it fell out of declining Gray. The
Gray fold was never cheap — it was bought for phase-A store-reachability
(narrow fragments; 31.55% → 95.47%) at a measured **2.38× the gates** of
the wide fold. Its per-gate machinery — gathering and stripping every
mask atom of both operands onto two accumulators, plus the four-phase
product schedule — costs more than the expanded fold's fragment list at
the production plan `[2,2,2,3]`. The swap-refresh construction removes
all of that and spends back only: 4 monomial emissions per fold (the
swap), 2 per relocation (the coupled refresh), and one extra slice block.
Net, at every size measured (same seed, same sandwich):

| n   | prior (Gray) | swap-refresh | ratio |
|-----|--------------|--------------|-------|
| 32  | 80,950       | 57,217       | 0.71× |
| 64  | 157,586      | 112,394      | 0.71× |
| 128 | ~1.29M (recorded campaign builds) | **878,863** | ~0.68× |

The n=128 half-sandwich build (|C|=|D|=3000) comes to **481,070** gates.

The price is paid in a different currency: with the ladder off, ~7% of
fold fragments remain wide (3+ controls), material phase A's store cannot
re-encode. The two measured remedies both currently violate the
zero-relation contract (`--prod-ladder-cap 4` re-opens a small linear
tail through its borrowed scratch; `PROD_POST_FRAGMENT` re-opens a broad
one) and are **off-limits for deliverables** until audited. Store
reachability under the expanded fold is the open measurement for the next
fmix campaign.

## 4. Verification (n=128, production parameters, fresh random seed)

- **Function**: bit-sliced verify passed (payload half equals the
  sandwich on the honest slice); 317/317 unit tests, including new tests
  pinning the swap, the closing block's specification, and the
  upper-half contract.
- **Inputs↔outputs**: rank of the 257 columns `{1, x, y}` over 1024
  samples is **full (257/257)** — no affine relation between inputs and
  outputs (conclusive: a true relation forces a rank deficit on every
  sample set).
- **flip_match** (local segment-pair XOR vs per-gate source deltas,
  kwin 12/30/60): **0 / 13,568 source gates matched** — down from 100%
  linear / 32.4% nonlinear on the prior build.
- **segment_deduce degree 1** (8 cuts, 16,384 samples, fresh-sample
  verified): 2.91% of sandwich segments deducible, every equation on the
  shared-input or shared-output cut; **zero interior-cut equations**
  (prior build: ~50% deducible with thousands of interior equations in
  the gadget-exposure direction).
- **segment_deduce degree 2** (four restricted windows up to 12,881
  monomials, both directions, with matched degree-1 controls): degree 2
  adds exactly **one** equation — an early C-gate output that is a
  quadratic of the *public input*, matched on the input cut only. No
  other verified equation uses a product. The reverse direction's
  interior-labeled hits are the public payload C(x) at the sandwich's
  C/D seam matching the gadget's output — present for any correct
  implementation. Caveats: windows are restricted (full-width degree 2 is
  combinatorially out of reach), and products are one-sided (no bilinear
  `s·g` cross-terms in the model).

## 5. Operation

`ProdConfig::production_single()` now sets `swap_refresh: 1,
close_slice: 1`; env overrides `PROD_SWAP` / `PROD_CLOSE_SLICE` (0/0
restores the prior Gray stream byte-for-byte). The gss_mix pipeline picks
the new defaults up unchanged. The two-share (`sss`) and multi-carrier
paths receive the fold-level swap wherever `fold_cg` runs but not the
output-port hygiene; extending them is future work.
