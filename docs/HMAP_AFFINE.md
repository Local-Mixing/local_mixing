# Reconstruction heatmaps: the affine and degree-2 predictors

`hmap_affine` measures how much a wider mixed circuit **G** reveals about the
internal states of the smaller original circuit **C** it computes. This note
explains what the degree-1 (affine) and degree-2 predictors compute, why they
are built the way they are, and how to extend to higher degrees — with the
complexity that makes higher degrees hard.

## 1. The measurement problem

C acts on `n` wires; the gadgetized/mixed G acts on `N ≥ n` wires and computes C
in an *encoded* form (shares, ancillas, bookends). We want the analogue of the
prefix-distance heatmap, `H(i,j)` = "how close is state `i` of C to state `j`
of G", but the two states have different widths (`n` vs `N`) and, worse, G
stores each logical value in an *encoding* rather than in the clear. A raw
Hamming comparison of truncated states is meaningless: any individual G wire is,
by gadget design, near-decorrelated from the logical bit, so truncated Hamming
sits at the ~N/2 noise floor everywhere and sees nothing.

Fix the ancilla wires of G to 0 and inject the input `x` on wires `0..n-1`. Then
both `C_i(x)` (state after the first `i` gates of C) and `G_j(x,0,…,0)` (state
after the first `j` gates of G) are **deterministic functions of the same
`n`-bit input `x`**. The question becomes: *is `C_i(x)` recoverable from
`G_j(x,0,…,0)`?* — and "recoverable by what class of function" is exactly the
knob that makes this well posed.

## 2. The affine (degree-1) predictor

For a fixed cell `(i,j)` and each of the `n` logical target bits `t` of `C_i`,
we ask whether `C_i(x)_t` is a **GF(2)-affine function** of G_j's `N` wire
values:

```
C_i(x)_t  =  c0  ⊕  Σ_w  a_w · G_j(x)_w        (a_w, c0 ∈ {0,1})
```

The regressors are the `N` wire columns plus a constant column. Over `S` sampled
inputs `x`, each column is an `S`-bit vector; we solve for the coefficients
`(a_w, c0)` by **GF(2) span membership**: is the target column in the linear
span of the regressor columns? Gaussian elimination over GF(2) answers this and
recovers the coefficients when it is.

We split the samples into **train** and **holdout**:

- Solve on the training rows.
- **Inconsistent** (target not in the regressor span) ⇒ the bit is not an affine
  function of G_j ⇒ error `0.5` (fully unpredictable, by this class).
- **Consistent** ⇒ evaluate the recovered coefficients on the holdout rows and
  report the measured bit-error. For a genuine affine relation this is ≈ 0; if
  the training "fit" was a coincidence of an underdetermined system it is ≈ 0.5,
  and the holdout correctly exposes it.

`H(i,j)` is the mean of this per-bit error over the `n` target bits. **Low H =
`C_i` is linearly recoverable from `G_j`** (leakage / alignment); **H ≈ 0.5 =
hidden** to a linear predictor.

Two properties make this the right degree-1 measure:

- **Invariance to affine encoding.** Gadget encodings are largely affine (CNOT,
  negation, share-XOR). An affine predictor is invariant to that whole class, so
  it looks straight *through* the encoding — this is exactly what dissolves the
  width/encoding mismatch, with no need to know where the answer lives or to
  track share positions.
- **Reduction to the ordinary map.** If C and G coincide (identity embedding),
  the affine fit is the identity and `H` becomes normalized Hamming distance —
  the degree-1 predictor generalizes the original prefix-distance heatmap.

**Why span-membership + holdout rather than "best affine fit".** The truly
Hamming-optimal affine predictor over GF(2) is MAX-XOR-SAT (NP-hard) and its
exact-correlation form needs a `2^N` Walsh search. Span membership is exact
linear algebra: it detects *exact* affine relations (error 0) and, via holdout,
distinguishes them from spurious fits. The low (leaking) side — the side we care
about — is therefore measured exactly; genuinely non-affine bits read as ~0.5.

## 3. The degree-2 predictor

Degree 2 adds, as extra regressors, the **pairwise products** `G_j(x)_a ·
G_j(x)_b` of G's wires. In the bit-sliced sample layout (lane `l` = sample `l`),
a product of two wire values across all samples is simply the **bitwise AND** of
their lane-words — so each degree-2 regressor is free to form. Everything else
(span membership, train/holdout, per-bit error) is unchanged; there are just
more regressor columns.

A degree-2 predictor reconstructs `C_i` by any degree-2 GF(2) function of G_j and
is therefore invariant to any *degree-2* re-encoding, not merely the affine part.
Because the base gate `g57` is itself a degree-2 gate, a degree-2 predictor sees
through g57 structure directly: empirically it broadens and deepens the
reconstruction diagonal relative to degree 1, and it erases the distinction
between an all-g57 gadget and a CNOT-heavy one (both are degree-2 transparent).

`--deg2-wires W` restricts products to wires `0..W`; the linear terms always
cover all wires. A subset makes degree 2 a **lower bound** on degree-2 leakage
(products among excluded wires are unseen), which is the lever used when the full
product set is too large (Section 5).

## 4. Extension to degree d

Degree `d` adds all monomials of degree `≤ d`: products of up to `d` distinct
wires (bitwise AND of `d` lane-words). The regressor count is

```
R(N, d)  =  1  +  Σ_{k=1..d} C(N, k).
```

Nothing else in the pipeline changes — the GF(2) span solve, the train/holdout
split, and the per-bit error are degree-agnostic. Conceptually, **the predictor
degree is the algebraic power of the reconstruction adversary**: degree 1 = a
linear white-box strategy, degree 2 = quadratic, and so on. A reconstruction
diagonal that survives to higher degree is a stronger, more robust leak; each
degree whose map is flat is one more *feasible* reconstruction strategy ruled
out. This is evidence, not proof — the goal is resistance to all feasible
reconstruction, and the degree ladder is a concrete, cheap family within that
class, not a for-all-`d` guarantee (no scheme hides against unbounded degree).

## 5. Complexity, and why high degree is hard

The cost is driven by `R = R(N,d)` regressors:

| quantity | scaling | note |
|---|---|---|
| regressors `R` | `~ N^d / d!` | dominates everything |
| **training samples** | must exceed `R` | else the system is underdetermined and every target is spuriously "consistent"; and the particular solution need not be the true low-degree relation, so genuine degree-`d` bits are *missed* (false "hidden"). Holdout catches overfit but cannot manufacture the signal. |
| basis memory | `O(R · S / 64)` words | up to `R` sample-space vectors of `S/64` words each |
| compute / cell | `O(R² · S / 64)` word-ops | Gaussian elimination; × (grid cells) × (target bits) |

Concrete regressor counts:

| wires N | d = 1 | d = 2 | d = 3 |
|---|---|---|---|
| 48 | 49 | 1,177 | ~18k |
| 128 | 129 | 8,257 | ~350k |
| 512 | 513 | 131,329 | ~22M |

The consequence: cost blows up as `N^d`. **Full degree 2 at N = 512 is ~1.3×10⁵
regressors**, which needs `> 1.3×10⁵` training samples and a basis on the order
of terabytes — intractable. Degree 3 is out of reach except at small `N`.

Mitigations, each with its caveat:

1. **Restrict the wire subset for products** (`--deg2-wires W`, and the analogue
   at higher degree). Cost falls to `~W^d`, but a "hidden" verdict becomes a
   **lower bound** only — leakage carried by excluded wire-tuples is invisible.
2. **Work at small `n`/`N`.** Fully rigorous (all products, samples ≫ regressors)
   but answers the question only at that scale; use it to establish the
   qualitative degree-1-vs-2 gap and whether mixing defeats degree 2, then argue
   scaling separately.
3. **Random-feature / linear-sketch products.** Take products of `K` random
   linear combinations of the wires; this spans a `C(K,2)`-dimensional *subspace*
   of the full degree-2 space — again a lower bound, tunable by `K`.
4. **Target the relevant wires.** Restrict products to wires flagged by the
   degree-1 fit's support; a heuristic that helps only where degree-1 already
   found structure.

There is no free lunch: exact degree-`d` reconstruction over `N` wires is
fundamentally an `R(N,d)`-dimensional GF(2) regression. Any tractable variant at
large `N` explores a subspace of degree-`d` functions and therefore
**lower-bounds** the true degree-`d` leakage — a "flat" result at large `N`
under a restricted predictor is suggestive, never conclusive.

## 6. Reading the maps

- Compare maps by **gradients** — difference maps and 1-D profiles (the diagonal
  floor, the band edges) — not by scalar means, which wash out the structure.
- Low cells = recoverable (leak); `≈ 0.5` = hidden. The forced corners
  (input at the first gate, output at the last) are always recoverable and carry
  no secret; the informative content is the interior.
- A surviving diagonal at degree `d` means a feasible degree-`d` reconstruction
  recovers intermediate values; driving it to `0.5` across increasing `d` (as
  far as feasible) is the evidence sought.

Usage: `hmap_affine --c C --g G --n <n> --degree {1,2} [--deg2-wires W]
[--c-step --g-step --batches --train-batches] --out <prefix>`. Output is a
row-major f32 matrix `<prefix>.bin` plus `<prefix>.meta.json`, same format as
`hmap`.

## Reading the output: the ridge measure (canonical)

Read these maps by the RIDGE, never the mean (which saturates near 0.5). A
diagonal is a low-H valley whose location `argmax_j (0.5 - H(i,j))` advances
monotonically with `i` — C's computational progress still legible in the mix.
`reports/plot_hmap_ridge.py` is the canonical reader: it prints per-map scores
and renders the plates with the detected ridge traced on top.

- **depth** — mean per-row prominence of the ridge cell above its row
  background. The number MIXING moves; a valley may fade mid-plate and stay a
  ridge, so it is a per-row prominence, not a global contrast.
- **rho** — Spearman(i, ridge column). Shape-agnostic monotonicity; `rho ~ 1`
  = clean diagonal, tolerant of fading depth. The "is it a diagonal" number.
- **perm z** — z-score of `rho` vs a null shuffling the per-row ridge columns;
  makes "discernible above chance" quantitative.
- **contrast** — on-band vs off-band mean H in sigmas.

The traced ridge is a depth-weighted smoothing of the per-row argmins, so
shallow rows are bridged by deep neighbours (it follows a fading valley
instead of scattering).

Usage: `python3 reports/plot_hmap_ridge.py --out plates.png <stem1> [stem2 ...]
[--titles "a;b;c"] [--nperm 3000]` where each `<stem>` is an `hmap_affine`
`--out` prefix.
