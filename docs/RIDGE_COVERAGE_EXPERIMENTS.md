# Reducing the Progress-Diagonal Ridge: Coverage Experiments and the Use-Point Wall

*2026-07-23 — branch `ssg-gen-mix-clean`. Companion PDF:
`docs/RIDGE_COVERAGE_EXPERIMENTS.pdf`. Tools committed this session:
`fmix --gen-snap-every` (`49cee4c8`) and `src/bin/pairfloat_exp.rs`
(`5a9f449c`). Plates: `reports/heatmaps/gen300_20260722/`,
`reports/heatmaps/pairfloat_20260723/`.*

## 1. The question

Local mixing rewrites a small boolean circuit `C` into a much larger,
functionally identical circuit `G` whose internal structure should not reveal
`C`'s computation. The sharpest known leak is the **progress diagonal**: a
reconstruction heatmap `H(i, j)` — the best-effort GF(2)-affine reconstruction
error of `C`'s state after `i` gates from *all* wires of `G`'s state after `j`
gates — shows a bright ridge along the line where `G`'s computational progress
matches `C`'s. The ridge says an adversary can read *how far through the
original computation* any prefix of the mixed circuit has gotten. We read the
heatmap with the **ridge measure** (`reports/plot_hmap_ridge.py`): `depth`
(ridge prominence, 0 = flat/ideal, ~0.5 = a maximally bright ridge), `rho`
(Spearman correlation of `C`-prefix vs. ridge column; `rho = 1` means "a clean
monotone diagonal"), a permutation `z`-score, and `contrast`.

The ridge has survived every mixing method tried, always at `rho = 1`, with
`depth` floored around 0.23–0.35 depending on method. **This document asks: can
we drive `depth` to 0.05 or below**, and reports the sequence of experiments
that answers it — with a clear negative and, more valuably, a *proof of why*.

## 2. Two prior beliefs cleared first

**Generation dose does not help.** `fmix` phase A drives every gate through a
target number of database re-encodings ("generation"). We ran a fresh
gadgetized `n = 128` sliced sandwich (85,058 gates) to generation 300 —
6× the benchmarked dose — snapshotting at every generation multiple of 50
(the new `--gen-snap-every` flag). The ridge only sharpens toward a floor:

| generation | 0 | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|---|
| gates | 85,058 | 179,437 | 218,809 | 265,096 | 317,289 | 377,912 | 450,318 |
| ridge depth | 0.348 | 0.273 | 0.261 | 0.254 | 0.246 | 0.240 | 0.236 |
| rho | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Depth decrements shrink geometrically toward an asymptote near 0.22–0.23. Six
times the dose past generation 50 bought a 14% depth reduction at 2.5× the
size. Generation is not the lever.

**Affine moves cannot help, by construction.** The reconstruction measure is
invariant to the affine part of any encoding (CNOT, negation, XOR-sharing).
Every conjugation twist in `fmix` — negation, swap, transvection (`CNOT`) — is
affine, so it *provably* cannot move the degree-1 ridge, at any dose. This is
an identity, not a tuning failure.

What is left is genuinely nonlinear re-encoding. Database replacements provide
it, but only inside windows of ≤ 5 gates: at any prefix cut, the handful of
values mid-window are hidden and everything else is affine-readable. The 0.23
floor is the saturation of that thin nonlinear layer. The experiments below
test whether we can add nonlinear coverage *directly*.

## 3. The mechanism: free nonlinear brackets

Insert an identical pair of gates `g · g` (the identity) into `C`, then float
the two copies apart by adjacent swaps through gates they commute with.
Correctness is exact and preserved bit-for-bit (a swap only exchanges commuting
neighbors; the pair began as the identity). But between the two separated
copies, exactly one `g` has been applied, so **every prefix cut in that
interval reads the `g`-image of the true state** — the target wire of `g`
appears at degree 1 + (number of `g`'s controls). A separated pair is thus a
*free nonlinear bracket*: no rewriting, only relocation, and coverage is dialed
by how many brackets straddle each cut and how far each spans.

The tool `pairfloat_exp` builds a fixed `C` (3000 g57 gates, 128 wires) and
tests families of this move, verifying equivalence on 200 random inputs for
every output before measuring its heatmap against `C`.

## 4. What we tried, and what each attempt showed

### 4.1 Insertion count `k` (identical g57 pairs, commuting float only)

Insert `k` pairs before every gate; float each to its commutation extreme.

| k | 0 | 1 | 2 | 3 | 4 | 5 | 8 | 11 | 16 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|
| depth | 0.474 | 0.395 | 0.377 | 0.370 | 0.367 | 0.364 | 0.359 | 0.357 | 0.357 | 0.351 |

(`rho = 1` throughout.) The first pair helps; then the curve flattens hard onto
an asymptote near 0.35. The reason is **self-dilution**: straddling brackets
per cut is `pairs × span / size = 84000 / (1 + 2k)`, which saturates at
`span / 2 ≈ 42` brackets per cut (~27% of 128 wires masked) as `k → ∞`. Each
inserted pair inflates the very circuit it must cover. `k` stops being a lever
around `k = 2`. A calibrated model, `depth ≈ depth₀ · (1 − wire-coverage)`,
fits every point within ~0.01 — the **bracket-coverage law**. (A standing bet
that `k = 11` would reach 0.05 resolved at 0.357, exactly the model's
prediction.)

### 4.2 Cross budget `B` (float past colliders by conjugation)

Free floating stops at the first non-commuting gate (~42 slots). Allow the
floater to *cross* up to `B` colliders using the exact splitting rules
(`postmix::rules` R1/R2/R3; g57 colliders are pre-split), paying fragment
gates. Span grows, but so does size, and half the floaters are consumed as
colliders:

| B (at k = 3) | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| depth | 0.369 | 0.343 | 0.333 | 0.330 | 0.329 |
| mean travel | 41 | 120 | 192 | 302 | 405 |
| gates | 21k | 53k | 77k | 105k | 129k |

Span rose 10×; depth saturated at 0.33. The whole gain is banked by `B = 1`;
beyond that, fragments are pure dilution — net straddle density at `B = 8` is
*below* free floating. Crossing is where the "free" bracket becomes a priced
one, and the price outruns the benefit almost immediately.

### 4.3 Adaptive wire selection (choose mobile wires)

Hypothesis: pick each pair's target/controls to *maximize mobility* — target
from the wires with the longest support-free stretch around the cut, controls
from wires nothing nearby writes. Span should soar. It did — and depth got
**worse**:

| adaptive (k) | 1 | 3 | 5 | +B=4 | +B=8 |
|---|---|---|---|---|---|
| depth | 0.437 | 0.427 | 0.423 | 0.402 | 0.396 |
| mean travel | 150 | 310 | 432 | 1486 | 2039 |

Span rose 7.5–50× and the ridge got *deeper* than doing nothing clever
(0.427 vs. 0.369). The cause is **target concentration**: a wire is mobile
precisely because nothing near the cut touches it, so it stays quiet for
hundreds of consecutive cuts, and every insertion samples the same tiny pool of
quiet wires. Total distinct masked wires collapses to ~10–20 of 128, dropping
per-cut coverage *below* random's. Mobility and diversity are in direct tension
under greedy per-cut choice.

### 4.4 Gap-tiling (the "correct" coverage — and the decisive result)

Invert the assignment: instead of each *cut* choosing its best wire, each
*wire* chooses its cuts. For each wire `w`, every maximal stretch of `C`
between two consecutive touches of `w` is a bracket slot; insert `s` distinct
pairs targeting `w` there, with controls drawn from wires untargeted in that
gap (so the pair commutes with the whole interior — direct correctness), and
float them to the gap ends. This forces diversity across all 128 wires, needs
no crossings, and masks every wire at every cut *except its own use points*.
The material cost is exactly as predicted (×6.9 / ×12.9 / ×24.8 for
`s = 1 / 2 / 4`). The prediction was `depth ≈ 0.25 / 0.12 / 0.05`. The result
was the **opposite**:

| gap-tiling `s` | 1 | 2 | 4 |
|---|---|---|---|
| depth | 0.364 | 0.375 | 0.398 |
| contrast | 3.26 | 3.26 | 3.26 |

Stacking more masks makes the ridge **sharper**, not fainter. The plate shows
why directly: the off-diagonal background hides more and more uniformly, while
the diagonal itself is untouched and, by contrast, stands out *more*.

## 5. The conclusion: the leak lives at the use points

Gap-tiling masks each wire only *strictly between* its uses — by construction
the floaters stop exactly at the touch gates. But the ridge at `C`-row `i`
aligns with the `G`-column where gate `i`'s operands are being **used**, and
those operands are left fully exposed. Between-use masking cannot reach the
diagonal because **the diagonal *is* the operand-use locus**. Worse, hiding the
idle background more thoroughly only increases the aligned column's relative
advantage — hence the rising contrast and depth.

This is the **use-point theorem**, and gap-tiling proves it in isolation: a
construction whose entire design is between-use coverage leaves the ridge
untouched. The small reductions the earlier modes did get (0.47 → 0.33) came
precisely from brackets that *accidentally straddled* use points while floating
blindly; gap-tiling removes those straddles on purpose and therefore does
*worse* than random insertion.

It is the same wall the deferred-mask / nonlinear-share-encoding work hit from
the other side (`docs/NONLINEAR_SHARE_ENCODING`): there, a "peek" un-masked
each operand at the instant it was consumed, and that peek re-exposed exactly
the diagonal. Two independent lines of attack converge on one statement:

> **Between-use coverage — in any form (random `k`, cross budget, adaptive
> mobility, gap-tiling) — cannot reduce the degree-1 progress ridge below
> ~0.33. It is targeting the wrong location. The ridge lives where operands are
> consumed, and no relocation-based masking touches that.**

This is now a demonstrated structural limit, not an engineering shortfall.

## 6. What actually remains

To reduce the ridge one must mask a wire **at the instant it is consumed** —
i.e. compute a gate on the *encoded* operand `w ⊕ mask` without ever
reconstructing `w`. That is share-native / mask-aware computation, exactly what
the legacy g57 gadget does (it computes on shares and never reconstructs an
operand, which is why it hides at degree 2). The open construction is a
**mask-aware CG** that folds the mask correction into the gate rather than
peeking the operand out; its cost is the answer-extraction interface, not gate
count. Coverage-only approaches, and their ~×25-material price tag for full
tiling, are a closed dead end.

## 7. Reproduction

```
# generation-dose progression (new snapshot flag)
fmix --input <gadget>.mpmct1 --target-size 100000 --gen-target 300 \
     --gen-snap-every 50 --gen-split-inherit --gen-stop-frac 0.05 \
     --w-db 1.0 --p-db-ingest 0.5 --p-db-hard 0.05 ...   # -> .gen{50..300}.mpmct1

# bracket-coverage families (fixed C, 3000 g57, 128 wires, seed 20260723)
pairfloat_exp <base> 128 3000 1,2,3,4,5   20260723 g57       0   # section 4.1
pairfloat_exp <base> 128 3000 3           20260723 conj      8   # section 4.2
pairfloat_exp <base> 128 3000 1,3,5       20260723 adaptive  0   # section 4.3
pairfloat_exp <base> 128 3000 1,2,4       20260723 gaptile   0   # section 4.4

# read every heatmap by the ridge, never the mean
hmap_affine --c <base>.source_c.g57 --g <out>.mpmct1 --n 128 --degree 1 --out M
python3 reports/plot_hmap_ridge.py M ...
```

Every `pairfloat_exp` output is equivalence-checked against `C` on 200 random
inputs before it is written.
