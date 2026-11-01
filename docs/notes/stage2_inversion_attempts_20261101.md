# Inverting the plain stage-2 output: attempts and why they fail

*2026-11-01. Working notes. All measurements are on the `rust2` arm of the pipeline-variants
campaign (wire turnover + effective fire interleaving); the plain stage-2 output is the
least-hardened, most-favorable target (no DB mixing / crossing / packing on top). Results are
empirical observations against specific attacks, **not** security proofs.*

## Target and attacker model

- **Circuit.** The plain stage-2 output, 256 wires, **455,884 gates** (raw, generator order) or
  **407,922 gates** after C-blind compression (NOT-pushing, fcompress, small-block ANF
  resynthesis; equivalent circuit). On the contract input — `x` on wires 0–63, all other wires 0 —
  it produces `payload = C(x)` on wires 64–127 and echoes the input on 0–63.
- **Goal.** Find `C⁻¹(0⁶⁴)` (equivalently, reconstruct `C`). `C` is a secret 64-bit permutation,
  a g57 circuit with 936 internal births.
- **Attacker knows the algorithm, not the coins.** Known structure: stage 1 is a sandwich
  `slice-in · compute-C · copy · uncompute-C⁻¹ · slice-out` (2n-wire keyed slicing; the copy is
  the C/D transition); stage 2 is the three-mask compile, where each g57 gate becomes a ~150-gate
  *fire* operating on masked data (a data wire carries `value ⊕ S`, `S` a GF(2) sum of **cubic**
  products of band wires), with fires interleaved (bounded displacement) and band wires under
  turnover / refresh / carrier migration. Unknown: `C`, the slicing key, the per-fire masks, the
  interleave order, the helper assignments.

## Summary

| # | Attack | Result | Why it fails |
|---|--------|--------|--------------|
| 1 | Generic SAT preimage (CryptoMiniSat + kissat portfolio, data-first cubes) | no preimage; grinds | structure-blind; reduces to PRP inversion of `C` |
| 2 | Structural meet-in-the-middle (`blind_recon`) | 0 gates recovered (both directions) | no value-level handles; only 8,425 of 407,922 values ever repeat |
| 3 | Cone-of-influence / copy-seam cut | keeps **98.6 %** of gates | masking entangles the payload with ~the whole circuit |
| 4 | Proxy census (direct / affine) | **0 / 0** of 936 births | no wire equals a C-internal value (or affine image) at any cut |
| 5 | Factor-window exposure | up to **300 / 936** births (~32 %) | functions are exposed but not in a reconstructible gate form |
| 6 | Degree-2 / degree-3 replicas | **0 / 64** payload outputs | no output is a low-degree poly of mid-circuit wires |
| 7 | DCA-style decapitate + differential | can locate, cannot strip | band wires never clean; masks data-dependent and refreshed |
| 8 | Symbolic peel from the clear input | dies at **0.72 %** depth | masks engage within ~1 fire; degree jumps past 3 at once |
| 9 | Atom-basis reconstruction | 38/64 hidden relations; payload never | masks proliferate to **182,822** atoms (turnover) |
| 10 | Mask identification | 0/21 single-product strips; **vacuous** | sum-of-cubic-products mask space is over-complete |

## Details

### 1. Generic SAT preimage
Encode the circuit on the contract subspace (ancillas 0 at entry, payload 0 at exit) and solve for
the 64 inputs; any model is forward-verified. Run unbounded with CryptoMiniSat (XOR-aware,
Gaussian elimination over 405,933 XORs) and a 16-worker kissat portfolio including **data-first
cube-and-conquer** on the input bits. No preimage found; the same instances timed out on the mixed
arms at 24 h. Even a *positive control* with only 12 free input bits (known answer) did not solve
instantly. **Why:** SAT uses none of the known structure; the problem it actually solves is
inverting the PRP `C` at a point, which is designed to be hard. This line was stopped as pointless
once the structural analysis showed there is nothing low-degree to exploit.

### 2. Structural meet-in-the-middle (`blind_recon`)
Greedy walk backward from the payload outputs and forward from the inputs, accepting a gate only
when the implied neighboring value is present elsewhere in the circuit. **0 gates recovered in
either direction, no meet.** Of 407,922 internal values only 8,425 ever repeat, so there is almost
nothing for the walk to anchor on.

### 3. Cone-of-influence / copy-seam cut
The natural structural shortcut: since the circuit is `… compute-C · copy · uncompute-C⁻¹ …`, the
uncompute half should be dead for the payload, letting us drop it and invert only compute-C.
Measured backward cone from the 64 payload outputs: **98.6 % of gates kept, all 256 wires
involved.** The three-mask masking couples the payload (24k payload-wire writes) through band wires
that the uncompute also touches, so the compute/uncompute separation that would halve the attack is
gone — exactly the masking's purpose.

### 4–6. Exposure and replica tests
- **Proxy census** (`proxy_intervals` vs `C` + witness audit): **0** of 936 internal births appear
  as a wire value (direct) or affine image (affine) over any interval. Not even non-deep births.
- **Factor windows** (ANF rewrite of 125–2000-gate windows): recover **158** births at window 125,
  rising to **300** at window 500–1000 where it saturates (~32 % of 936). This is ~5× the
  fully-mixed `F` (33–97). So a third of C's gate *functions* are individually exposed — but
  `blind_recon` (#2) shows they are not in an assemblable gate sequence, and they give no preimage.
- **Degree-2 replicas** (are payload outputs degree-≤2 in the live wires at a cut?): **0/64** at all
  five interior cuts (0.40–0.60), non-vacuous (span dim 32,897). **Degree-3** (cubic, mask-matched,
  cubic monomials among the top-48 band wires): **0/64** at the measured cuts; the span grew
  32,897 → 50,193 and still captured nothing. (The old pre-three-mask baseline had 36–50/64.)

### 7. DCA-style decapitate + differential fire isolation
Idea: chop the slicing/mask-setup so band wires are clean, then use 0/one-hot-input differentials to
isolate and peel the fires. Measurements:
- **Band writes are not choppable:** the 48 mask-carrier wires are written across *all 20* slices of
  the circuit (970–3,561 per slice). Turnover/refresh/migration reload masks throughout compute, so
  there is no separable setup/teardown to excise; band wires are masked (high-degree) **before** the
  first compute fire.
- **Differential localizes but cannot strip:** first input-dependent gate at 1,280; the first
  compute-fire gate depends on only 4 input bits (early fire signatures are narrow, median ~8, not
  ~32 — so localization *is* feasible, correcting an earlier claim). But the differential reveals
  data-flow, not values; the fires read already-masked band wires, so you can locate a fire without
  recovering the clear g57 gate. This is textbook differential computation analysis; the three-mask
  + turnover is a masking countermeasure against it.

### 8. Symbolic peel from the clear input
Strongest form of "peel the fires": forward-execute tracking each wire as an exact GF(2) polynomial
in the 64 input bits, degree cap D; a wire we can still represent is structure recovered. (This
upper-bounds any differential peel — it has the exact algebra.) Result: recovers the keyed slicing
**exactly** (first ~1,280 gates, all affine) plus the first ~5 g57 gates, then the data register
overflows at **gate 3,281 (0.72 % depth)** — **identical for D=2 and D=3**. Matching the cubic mask
degree recovers more intermediate updates (122 vs 61) but buys **zero** extra depth: when a data
wire gets masked it jumps straight past degree 3, not through a cubic plateau. The payload is never
low-degree, so `C(x)` is never obtained symbolically. An independent sampled degree test (exact span
of all degree-≤3 monomials in x, N2=49,152) confirms the frontier: the data register is all
degree-≤2 at 0.56 % depth and **63/64 wires exceed degree 3 by 1.1 %** (only 1–2 wires sit
transiently at exactly degree 3) — i.e. degree 3 gives essentially the same frontier as degree 2.

### 9. Atom-basis reconstruction
Treat each band-wire value as an abstract *atom* (fresh atom on every write), and track everything
else as polynomials over `{x} ∪ {band atoms}`, hoping the cubic masks become low-degree in this
basis. Outcome: **38 of 64 data wires stay trackable to the end as low-degree-in-x polynomials** —
the atom-tracking discovers, by symbolic mask-cancellation, that 38 data-register wires are
*secretly* degree ≤3 in x (a genuine hidden-low-degree leak). **But** the 26 wires carrying the
masked compute overflow, the payload is never reconstructed, and the masks **proliferate to 182,822
distinct atoms** (the turnover defense, quantified). Atomizing individual band wires leaves each
cubic mask degree-3, so a fire multiplying two masked values is degree-6 → overflow; staying
low-degree would require abstracting each *mask* as a unit, i.e. knowing the coins.

### 10. Mask identification — the fundamental obstacle
To strip masks you must identify them. Two findings:
- **Single products don't fit.** At the first-fire cut, no single band product of size ≤3 (ancilla
  wires only) strips any of the 21 masked data wires to a degree-≤2 residual: **0/21**.
- **Sums of products over-fit.** The three-mask mask is a GF(2) *sum* of cubic products. The span of
  {degree-≤2 monomials in x} ∪ {cubic band-products} **saturates the whole sample space** — it
  reaches full rank at N=4,096 **and** N=8,192 (2,081 degree-2 dims + ~N−2,081 cubic dims), so *all*
  64 data wires fall inside. Hence any masked wire `= (degree-≤2 logical value) ⊕ (cubic-band mask)`
  admits countless decompositions: **the mask is not uniquely identifiable.** There is nothing to
  strip.

This is the deepest reason inversion fails: the mask space is over-complete, so even with full
knowledge of the algorithm you cannot pin down the mask to recover a single logical value.

## Synthesis: visible but not invertible

What an attacker *can* get from the plain stage-2 output:
- the keyed **linear slicing** (exact, by linear algebra / the affine peel through ~0.7 %);
- ~**one third of C's births** as isolated functions (factor windows), but not as a gate sequence;
- **38 hidden low-degree relations** among the data-register wires (the atom-basis leak).

What blocks turning any of that into `C` or `C⁻¹(0)`:
- no wire is a (direct/affine) copy of a C-internal value (census 0/0);
- the exposed births don't assemble (`blind_recon` 0; no replica at degree 2 or 3);
- no structural cut (cone 98.6 %); the payload depends on ~everything;
- the low-degree frontier from the input dies at ~0.7 % (masks engage within one fire);
- the masks can't be identified (over-complete space) or chopped (woven throughout);
- generic SAT is left facing plain PRP inversion.

## Caveats and open directions

- Empirical, one run (`rust2`). The deg-3 peeling frontier has since completed and confirms the
  degree-2 frontier (data register exceeds degree 3 by ~1.1 % depth); the deg-3 mid-circuit replica
  scan read 0/64 at its first cuts.
- **Not tried / could be pushed:** a structural *fire-template parse* (match the known fire shape to
  segment and read mask wires, rather than algebraic stripping) + a template-aware de-interleaver;
  exploiting the 38 hidden relations as constraints; higher-order DCA across mask shares. Each
  targets the turnover/refresh, which is the design's crux defense.
- These results show resistance to the attacks tried, not a proof. The one place the masking visibly
  leaks beyond the slice-in is the 38 low-degree relations — worth a dedicated look.

## Tooling (session scratchpad / server)
`sat_invert.py`, `preimage408.py`, `kissat_df.py`, `emit_dimacs.py` (SAT); `blind_recon.py`;
`proxy_intervals`, `battery408.py`, `degree2_replica_scan.py`, `deg3_replica_scan.py` (exposure);
`peel_frontier.py`, `symbolic_peel.py`, `atom_peel.py`, `mask_id.py`, `mask_rank.py` (peel /
reconstruction); `diff_isolate.py`, `sig_isolate.py` (DCA); `struct_inspect.py` (structure).
