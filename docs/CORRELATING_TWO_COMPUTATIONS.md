# Correlating two computations

*A tutorial on prediction and classification attacks between a reference
computation and an encoded one, and what each of our protection plans does
against each attack.*

Companion PDF: `docs/CORRELATING_TWO_COMPUTATIONS.pdf`. All measurements are on
the n=128 sliced sandwich with the Gray-code fold (`docs/GRAY_FOLD_CG`).

---

## 1. The general setting

Take two computations run on the same inputs. Call them `C` (the reference —
what we are trying to hide) and `C'` (what an adversary holds). Fix a prefix `i`
of `C` and a prefix `j` of `C'`, and run both on `s` shared random inputs
`x_1..x_s`. You get two binary matrices:

```
  Y = C_i   with  Y[a,b] = (state of C after i gates on x_a)_b     s x n
  Z = C'_j  with  Z[a,b] = (state of C' after j gates on x_a)_b    s x m
```

Every attack in this document is a test of whether `Z` tells you about `Y`,
computed for each `(i,j)` cell and read as a heatmap.

In our case `C'` = `G`, the gadgetized circuit, with `m = 4n` wires against `C`'s
`n`; the extra inputs are pinned to zero (`x` on wires `0..n-1`, zeros
elsewhere).

## 2. The trivial answer, and why security must be class-relative

Ask "are `Y` and `Z` dependent?" and the answer is **always yes**. Both are
deterministic functions of the same `x`, and `G` is an invertible encoding, so
`Z` determines `x` determines `Y`. The mutual information is maximal by
construction.

This is not a technicality — it dictates what can be measured. A test that is
*consistent against all alternatives* (distance correlation, HSIC, any universal
independence test) will simply fire at full strength on every build, secure or
not, and tell you nothing. **There is no test that can certify "no dependence."**

So every meaningful statement has the form:

> *no adversary in class* **A** *can extract feature* **f** *of `C`'s state from
> `G`'s state, at sample budget `s`.*

Every number below is a **lower bound** on what some bounded adversary can do,
and the class must always be named. That is not modesty; it is the only kind of
claim the setting admits.

## 3. The unifying object: the cross-correlation spectrum

Fix `i, j`. For a target mask `α ∈ F_2^n` and a predictor mask `β ∈ F_2^m`, define

```
  M[α, β]  =  E_x [ (-1)^( α·Y  XOR  β·Z ) ]
```

This is the **Walsh (Fourier) cross-spectrum** between the two state vectors.
`M[α,β] = ±1` means `α·Y` is an exact affine function of `Z`; `M[α,β] = 0` means
that particular parity carries nothing; intermediate values are biases.

Every test in this document is a restriction of this one object:

| restriction | gives you |
|---|---|
| `\|M\| = 1` exactly, `β` unrestricted | **algebraic / exact** predictors (§4) |
| `\|M\|` maximized, `β` of bounded weight | **statistical** predictors (§5) |
| replace `β·Z` by any degree-≤D function of `Z` | degree-D versions of both |
| the full singular spectrum, over the reals | **canonical correlation** (§6) |

The matrix is `2^n × 2^m`, so nobody enumerates it. Every practical instrument
picks a restriction, and its blind spots follow from that choice.

## 4. Algebraic (exact) predictors

**The question.** Is `α·Y` *exactly* equal to some degree-≤D function of `Z`, on
every input?

**How it is answered.** Build the design matrix of all monomials of `Z` up to
degree `D` — `N_D = Σ_{k≤D} C(m,k)` columns — and test whether `α·Y` lies in its
GF(2) column span. That is a rank computation, with a held-out split to reject
spurious fits. The answer is *binary per cell*: recoverable or not.

**Our instrument.** `hmap_affine`. It reports `H(i,j)` = mean over `C`'s `n`
target bits of the holdout reconstruction error, scoring an inconsistent bit at
exactly `0.5`. So `H ≈ 0` is total leakage and `H ≈ 0.5` is hidden.

**Why it exists.** It is *invariant to the affine part of the encoding*. `G`
represents each value as a XOR of shares spread over unrelated wires; a raw
Hamming comparison sees only noise, and an early "the diagonal is destroyed"
conclusion was exactly that artifact, corrected once the affine-invariant
measure was built. Degree 2 additionally sees through any quadratic re-encoding.

**Its blind spot.** It cannot see *bias*. A relation holding on 99% of inputs is
scored identically to one holding on 50% — both are "not exact." This is why it
and the statistical measures disagree, and neither subsumes the other.

**How to read it.** Not by the plate mean (which saturates at 0.5) and not by
`rho` (see §9), but by the **ridge**: `depthMed` (median per-row prominence) and
the count of *interior* rows carrying prominence.

## 5. Statistical (bias) predictors

**The question.** How well can a *bounded* predictor **guess** `α·Y` from `Z`?

**How it is answered.** Search a small family of functions of `Z` and report the
best agreement, bias-sensitively (`max(a, 1-a)`, never snapped to ½).

**Our instrument.** `stress_battery`, with two families:

* **F1** — best over {constant, single wire, XOR of two wires}: `|β| ≤ 2`,
  degree 1. This is a GF(2) analogue of *correlation power analysis*.
* **F2** — F1 plus one AND over a declared wire set (for us, the 256 band wires):
  degree 2.

Three design points make the numbers readable:

1. **Lift, not agreement.** Each cell reports `best − base_rate`, where the base
   rate is the best *constant* predictor. A merely skewed target scores zero, so
   sparse targets (like the `traj` gate-activity bit) are not inflated.
2. **An overfit floor.** With `W = 512` predictor wires the search is over
   ~131k candidates, and maximizing over that many buys agreement by chance. The
   floor is that same search run against an unrelated *random* target. Reported
   `prominence = raw − floor`, and the leak gate is `prominence > 2·floor`.
3. **Two gates.** A verdict of `ALIGNED-LEAK` needs both exploitable information
   (`prominence` over floor) **and** progress alignment (`rho`, `perm-z`) — the
   best-predicting `G`-prefix must advance as the `C`-prefix advances. Bias
   without alignment is `bias-only`; neither is `flat`.

**Scaling.** The floor behaves as `√(ln N / 2s)`, so resolving a bias `δ` needs
`s ≳ ln N / δ²`, while cost is *linear* in `s` and *quadratic* in wires:

```
  cost  =  rows × cols × n_targets × ( W²/2 + 2A² ) × s/64
```

Resolving a leak `k×` smaller therefore costs `k²`. Widening `G` from `4n` to
`8n` needs only ~12% more samples but ~4.5× the time.

## 6. Canonical correlation analysis, and the right basis

Classical CCA asks: over **real** vectors `a, b`, maximize `corr(Ya, Zb)`. It is
solved by an SVD of the whitened cross-covariance, and the singular values are
the canonical correlations — the whole spectrum, not just the max.

The translation to our setting is the useful part. Encode bits as `±1`. Then

```
  XOR of a set of bits   <-->   PRODUCT of the corresponding ±1 variables
                         <-->   a single Fourier character
```

So a **GF(2)-linear** relation is a **multiplicative monomial** over `±1`, not an
additive one. Consequently:

* **Real-linear CCA** on the raw bits finds *additive* combinations — essentially
  Hamming-weight-style relations. That is precisely the classical side-channel
  leakage model, and it is the natural basis when your observable is an analog
  quantity like power draw.
* **GF(2) character analysis** finds XOR relations. That is *linear
  cryptanalysis*, and it is the natural basis for a Boolean circuit.

Both are "canonical correlation" — over different function bases. For our
problem the multiplicative/XOR basis is the right one, and real CCA on raw wire
values would mostly measure Hamming-weight structure that the encoding neither
especially protects nor especially exposes. The GF(2) analogue of CCA's full
singular spectrum is the whole Walsh cross-spectrum of §3, which is why the
practical instruments are all *restrictions* rather than decompositions.

## 7. Our case: `G` is a masking scheme, so order matters

The product-share encoding represents each value as

```
  v  =  w  XOR  A_1  XOR ... XOR  A_k          A_r = product of d_r band bits
```

with `w` the carrier wire and each `A_r` a "mask atom" over the frozen band.
This is **Boolean masking**, and the side-channel literature on masked
implementations transfers directly.

**The piling-up lemma.** For a degree-`d` product of uniform independent bits,
`E[(-1)^{A}] = 1 − 2^{1−d}`. A degree-2 atom contributes `0.5`; a degree-3 atom
contributes `0.75`. If the atoms are variable-disjoint (which `draw_slot`
enforces per value), the biases multiply:

```
  ε  =  Π_r ( 1 − 2^{1−d_r} )        corr(carrier, value) = ε
```

so a predictor reading the bare carrier guesses the value with agreement
`(1+ε)/2`. **`ε` is the single number that governs the statistical attack.**

**Order of attack.** To *remove* an atom of degree `d`, an adversary must compute
a degree-`d` monomial of the band wires — a `d`-th order combining function.
After cancelling a set `S`, the residual bias is `Π_{r ∉ S}`. Hence:

* F1 (no AND) sees the full `ε`;
* F2 (one AND) can cancel one **degree-2** atom → sees `ε / 0.5 = 2ε`;
* an F3 (one 3-AND) would cancel a **degree-3** atom → `ε / 0.75`.

**The design tension, in one line.** A degree-2 atom is a *stronger* statistical
masker (factor 0.5 vs 0.75) but a *weaker* algebraic one — it lies inside the
span of a degree-2 exact adversary, whereas a degree-3 atom does not. Low degree
buys statistics; high degree buys algebra. A plan is a choice of mix.

## 8. Results: each plan against each attack

Five mask plans, all with the Gray fold, all at n=128, same source `C` and same
sandwich, `s = 8192`, plate geometry matched per build (`--g-step` scaled by
circuit length — see §9).

### 8.1 Predicted bias, before and after cancellation

| plan | atoms | `ε` (order 1) | after one 2-AND | after one 3-AND |
|---|---|---|---|---|
| `[2,3,3]` *(shipped)* | d2, d3, d3 | 0.28125 | 0.5625 | 0.375 |
| `[2,2,3]` | d2, d2, d3 | 0.1875 | 0.375 | 0.25 |
| `[2,2,3,3]` | d2, d2, d3, d3 | 0.140625 | 0.28125 | 0.1875 |
| `[2,2,2,3]` | d2×3, d3 | 0.09375 | 0.1875 | 0.125 |
| `[2,2,2,3,3]` | d2×3, d3, d3 | 0.0703 | 0.140625 | 0.09375 |

### 8.2 Measured

| plan | `ε` | gates | vs base | store-reachable | F1 raw | F2 raw | verdict F1 / F2 |
|---|---|---|---|---|---|---|---|
| `[2,3,3]` | 0.28125 | 808,618 | — | 95.47% | 0.0817 | 0.1111 | **LEAK / LEAK** |
| `[2,2,3]` | 0.1875 | 634,790 | **−21%** | 97.01% | 0.0536 | 0.0782 | flat / **LEAK** |
| `[2,2,3,3]` | 0.1406 | 864,497 | +6.9% | 96.16% | 0.0435 | 0.0660 | flat / flat |
| `[2,2,2,3]` | 0.09375 | 692,653 | **−14%** | **97.53%** | 0.0318 | 0.0487 | flat / flat |
| `[2,2,2,3,3]` | 0.0703 | 924,284 | +14% | 96.56% | 0.0258 | 0.0409 | flat / flat |

(F1/F2 "raw" = `prominence + floor`; compare raw across arms, because the floor
is a noisy single-sample estimate — §9.)

### 8.3 The law

Regressing the measured signal on the predicted piling-up bias:

```
  F1_raw  =  0.262 · ε  +  0.007        R² = 0.996
  F2_raw  =  0.330 · ε  +  0.018        R² = 0.998
```

Five plans spanning a 4× range of `ε`, and the statistical leak is **linear in
the piling-up product** to within half a percent. This is the strongest
mechanistic result in the series: it identifies the leak as the *carrier's
marginal mask bias* and nothing else.

Two readings fall out of it:

* The **slope ratio** `0.330 / 0.262 = 1.26` is the payoff of second order.
  Theory says a *perfectly targeted* 2-AND cancels a degree-2 atom and doubles
  the effective bias (ratio 2.0); the blind search over `C(256,2)` pairs
  realizes about a quarter of that. A *targeted* F2 using ledger knowledge would
  reach the full factor.
* The **intercepts** (0.007, 0.018) are residual overfit inflation the floor
  subtraction does not fully remove — a reminder that "flat" is a statement
  about a threshold, not about zero.

### 8.4 What the algebraic attack says — and why it is flat across the board

`hmap_affine`, matched geometry, reads every plate measured as dead —
`depthMed = 0.0000`, `depth` 0.044–0.046 against an "alive" calibration of
`depth ≈ 0.35`, and **zero genuine interior rows**:

| plate | degree | depth | depthMed | rho | interior leaks |
|---|---|---|---|---|---|
| wide fold `[2,3,3]` | 1 | 0.0438 | 0.0000 | 0.017 | 0 / 88 |
| Gray fold `[2,3,3]` | 1 | 0.0436 | 0.0000 | 0.016 | 0 / 88 |
| wide fold `[2,3,3]` | 2 | 0.0455 | 0.0000 | −0.042 | 0 / 88 |
| Gray fold `[2,3,3]` | 2 | 0.0439 | 0.0000 | 0.016 | 0 / 88 |
| `[2,2,3]` | 2 | 0.0439 | 0.0000 | 0.016 | **0 / 88** |
| `[2,2,2,3]` | 2 | 0.0444 | 0.0000 | −0.047 | **0 / 88** |

The last two rows are the ones that mattered: `[2,2,3]` and `[2,2,2,3]` are the
plans that trade degree-3 atoms for degree-2 ones, so they are where a degree-2
exact adversary would show up first if the margin were thinner than the theory
says. It is not. (`[2,2,3,3]` and `[2,2,2,3,3]` were not run at degree 2 — they
carry *more* degree-3 material than the plans that passed, so they are covered
a fortiori.)

That is not a coincidence of the plans; it is forced. Exact degree-`D`
reconstruction of `v` requires *every* atom to lie in the degree-`D` span, i.e.
`D ≥ max_r d_r`. All five plans keep at least one degree-3 atom, so all five are
outside a degree-2 adversary's reach — **the algebraic verdict is identical for
every plan**, and the plans differ *only* statistically.

The distinction that does survive is **redundancy, not threshold**: `[2,2,3]` and
`[2,2,2,3]` hold a single degree-3 atom, so if one is ever compromised (a
degenerate source tuple, a re-source the adversary can follow) the value drops
into degree-2 range. `[2,3,3]`, `[2,2,3,3]` and `[2,2,2,3,3]` keep two. A pure
`[2,2,2]` plan would have no algebraic margin at all — statistically the best of
all, and dead at `D=2`.

### 8.5 The Pareto surprise

`[2,2,2,3]` **dominates the shipped `[2,3,3]` on every axis measured at once**:
14% fewer gates, higher store-reachability (97.53% vs 95.47%), 3× lower
statistical leak, and identical algebraic standing.

The reason it is affordable is the Gray fold. Under the old wide fold a block
emits `(1+k)^arity` fragments, so adding a mask term is *multiplicative* — going
from 3 to 4 terms costs +56%. Under the Gray fold the product part of a block is
a fixed ~9 gates regardless of `k`, and a term costs only its own gather: ~1 gate
for degree 2, ~4 for degree 3. Mask-plan cost becomes **additive**, and trading a
degree-3 atom for degree-2 atoms actually makes the circuit *smaller*.

## 9. How to read these numbers without fooling yourself

Five traps, all of which bit during this work:

1. **Never read a heatmap by its mean.** It saturates near 0.5. Read the ridge.
2. **Never read `rho` alone on a plate with ports.** Both axes have unencoded
   ends — cell (0,0) compares `C`'s input against `G`'s input wires and reads
   perfectly in *every* build. On such a plate `rho` is two plateaus ranked
   against noise in a dead middle: two equally-dead artifacts once scored
   `rho = 0.448` and `0.019`. Report **interior rows with prominence**. Our
   degree-1 plates report `perm-z = 4.91`, `p = 0.0003` — and are dead.
3. **The overfit floor is a noisy single-sample estimate, and the verdict rides
   on it.** Measured floors are not monotone in `s`: 0.0454, 0.0205, 0.0242,
   0.0136 at `s` = 2048, 4096, 8192, 16384. The *same circuit* scored `flat` at
   `s=2048` and `ALIGNED-LEAK` at `s=4096`. Compare arms by **raw**, not
   prominence.
4. **A fixed `--g-step` across circuits of different length is a confound.**
   `cols = |G|/g_step`, and row-best is a max over columns, so a longer circuit
   scores leakier for free. A +24%-gate arm ran 52 columns against 42 and looked
   13% worse; matching the geometry removed a third to a half of that.
5. **"Flat" means "below threshold at this `s`."** Since floor `∝ 1/√s`,
   quadrupling the samples halves the floor. `[2,2,2,3]`'s F1 raw of 0.0318 sits
   just above a 0.0239 floor with `rho = 0.996` — the diagonal is still *there*;
   it no longer clears the bar. Nothing here is a death certificate.

## 10. What we did not test

* **A structure-aware adversary.** The mask atoms are *readable off the gate
  list* — the fold's own gates name the band wires of each atom. Such an
  adversary does not search; it reads and cancels exactly, achieving the full
  order-`d` factor rather than the blind search's quarter. This is strictly
  stronger than anything run here, and it is what rolling, re-sourcing and
  epochs exist to complicate.
* **Targeted F3.** Would cancel a degree-3 atom (`ε / 0.75`). Trivial when
  targeted; ~2800× the cost of F2 when blind.
* **Differential correlation.** `ΔC` against `ΔG` under input differences. The
  battery's `traj` target is a *temporal* difference, not an input one — a
  genuinely unexplored axis.
* **Machine-learned predictors** over all `4n` wires, escaping the hand-built
  family entirely.
* **Universal independence tests** (distance correlation, HSIC) are *not* on
  this list, and §2 says why: they answer a question whose answer is already
  known.

## 11. Practical summary

* The statistical leak is the carrier's marginal mask bias, and it is **linear in
  the piling-up product** (R² ≈ 0.996). To reduce it, reduce `ε`.
* Refresh **rate** and refresh **source** do not touch it — measured flat across
  `--prod-refill-data` 0/50/100. They change the band's provenance, not its
  marginal, and the marginal is what F1 sees.
* Low-degree-heavy plans win statistically at *every* attack order; degree-3
  atoms buy the algebraic margin. Keep at least two degree-3 atoms for
  redundancy.
* `[2,2,2,3]` is Pareto-better than the shipped `[2,3,3]` on size, digestibility
  and statistical leak, with identical algebraic standing — but with one fewer
  degree-3 atom of margin. `[2,2,2,3,3]` buys that margin back for +14% gates
  and the lowest leak measured.
