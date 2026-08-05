# **The product-share encoding**

*2026-07-23*

> Markdown rendering of `docs/PRODUCT_SHARE_ENCODING.tex`. The PDF is authoritative for figures, diagrams and display math.

**Superseded in part (2026-07-24).** The production setting is now [2,3,3] with a nonlinear band fill, not the [2,2,3,3] of §8, and the band fill of §5 is no longer linear. See `PRODUCT_SHARE_UPDATE` for what changed and why, and `NONLINEAR_GADGETIZATION` for the whole construction described from scratch.

## The problem: the progress diagonal

The gadgetizer rewrites a small circuit C into a much larger equivalent G
whose gates operate on **secret-shared** logical values. The security goal
is that G leak nothing about C beyond its input/output behaviour—in
particular, that an adversary cannot tell *where in the original
computation* any point of G sits.

The standard sharing is a linear XOR pair: a logical value v is carried on two
wires c_0,c_1 with v=c_0⊕ c_1. This is the root of a persistent leak.
Because the decode is linear, v is a **degree-1 (affine) function of the
wires at every snapshot**. The `hmap_affine` tool measures exactly this:
for each prefix i of C and prefix j of G, it fits the best GF(2)
affine reconstruction of C's state-i from G's wires at j and reports the
holdout error H (0 = perfectly recoverable, 0.5 = hidden). With linear
sharing, at the aligned prefix C_i is recoverable from the shares regardless of
C_i's algebraic complexity in the input, so a low-H **diagonal** runs
down the map. That diagonal reveals the temporal layout of the computation, and
it **survives every mixing method** tried (ssg, fmix, generation
targeting): mixing re-randomises the sharing, but the value stays affine in the
new wires, so the diagonal persists at ρ=1.

The measure is read by the **ridge**, never the mean (which saturates near
0.5): *depth* = mean per-row prominence of the low-H valley above its
row background; ρ = "is it a monotone diagonal"; *perm-z* =
significance vs. a shuffled null.

## The encoding

Replace the pair decode with a **balanced nonlinear** one. Each logical
value carries, on top of its XOR pair, k permanent product mask terms sourced
on a frozen band of extra wires:

```tex
v_i = \underbrace{w_{u(i)}⊕ w_{u'(i)}}_{linear carrier}
⊕ \underbrace{⊕_{j} ∏_{l}(w_{p_{jl}}⊕ a_{jl})}_{k product terms}
⊕ \underbrace{c_i}_{ledger const} .
```

Each product term is a conjunction of band wires with per-factor offsets
a_l; c_i is a ledger constant. The offsets and c are compile-time bits
kept in the gadgetizer's ledger.

## Why the linear carrier is not optional—the balance obstruction

One might hope to drop the XOR pair and carry v as a **pure** two-wire
product v=(w_p⊕ a)(w_q⊕ b). This is impossible to gate.

**Claim.** No exact reversible gadget can conditionally flip a value whose
decode has unequal class sizes—with any ancillae, garbage, or re-encoding.

*Argument.* A pure 2-wire product decode is unbalanced: over the four
(w_p,w_q) states, v=1 has exactly one preimage ( a, b) and v=0
has three (a **3:1** imbalance). Consider realising any gate that must
conditionally flip a value—e.g. CNOT v_t ⊕= v_x.
Exactness forces the gadget F (a reversible map on all wires, ancillae
included, with any new ledger afterward) to biject the state-class
{v_x=α, v_t=β} onto {v_x=α, v_t=β⊕α}.
Counting representations and cancelling common factors, the α=0 equations
give R'(0)/R'(1)=R(0)/R(1) and the α=1 equations give
R'(0)/R'(1)=R(1)/R(0), where R,R' are the target's rep-counts before/after.
Together these force R(0)=R(1): **the target decode must be balanced.**
The pure product is 3:1, so no exact CNOT- (or CCNOT-) gadget exists for it; a
bijection cannot map a 3-class onto a 1-class. Randomising (a,b) flips
*which* side is heavy but never the imbalance itself.

The balanced-implies-affine lemma closes the loop: every nonlinear 2-wire decode
is ungateable, so the nonlinearity must come from **more than the value's
own two wires**—here, a linear carrier pair plus product terms over foreign band
wires. A second, independent reason the pure product fails: even granting a
gadget, its best degree-1 predictor is right on the whole 1-rep side and,
optimising over the three 0-reps, still achieves accuracy 5/6—a
degree-1 leak floor of H≈0.17. The linear carrier symmetrises the classes
and removes this floor.

## The share-native fold-CG—never reconstruct an operand

The predecessor "deferred-mask" design masked values *between* uses but,
at each gate, momentarily **un-masked** the operands so a vanilla gate could
read the true value ("peek"), then re-masked. Validation showed this does not
move the diagonal: the peek re-exposes each gate's operands at exactly the
use-points that form the diagonal. Masking hides between uses; the ridge lives
*at* uses.

The product-share CG removes the peek. It applies v_t ⊕= f(controls)
by **folding f's ANF over the operands' full decodes**: each control value
contributes its summand atoms (two carrier literals, the k product-literal
sets, and a constant when the ledger constant / control polarity applies); every
cross-product of one atom per control operand is emitted as **one
conjunction fragment** written into a random carrier of v_t. A single value's
own product term appears only *inside* a fragment's firing condition,
**never materialised on a wire**. Constants and the gate's complement go to
the ledger constant c_t, so the encoding emits no bare X. No wire is ever
written except the target's carriers—**operands stay masked through the
gate**.

## The frozen source band

The mask sources live on a dedicated band appended after the 2n carriers. At
the input port each band wire is filled with an unbiased data-dependent bit (a
random weight-≥2 XOR of the data wires; on the zero slice, $⟨α,x
⟩$) and then **never written again**. Because the sources are frozen,
every registered product term is **time-invariant**—invariant under all RG
churn and CG traffic. Nothing is ever flushed except the final unshare strip; a
mask is one gate, not seven.

## k versus degree—two different jobs

k (number of terms) and (literals per term) are independent levers.

- **Degree hides algebraically.** A degree-d term is not in the
degree-<d GF(2) span, so against the exact-span measure a single
degree-d term forces H=0.5 for *every* adversary of degree <d,
regardless of k.
- **k hides statistically.** Against a best-approximation adversary
the terms pile up: k stacked degree-2 masks push the best affine readout error
toward 1/2 as 1/2-2_-(k+1). Against the exact-span measure, k>1 is
*redundant* for the degree it already blocks.

A single degree-d mask is a *sparser* perturbation (it flips the value
only 2_-d of the time), so wider masks are individually *weaker*
statistically even as they are stronger algebraically—the case for a mixed
design (§).

## Towers—raising the degree

To hide from a degree-(d-1) adversary you need degree-d masks. The cheapest
realisation is a **wider conjunction**: =d uses d band-wire literals,
still one MPMCT gate, degree d in the inputs. Each +1 of mask degree forces
the adversary from degree D to D+1, multiplying its regressor count by
W (the wire count) and its solve by W^2: one control literal for us, a
factor W for the attacker.

**Cost of a tower level is flat in gate count.** The number of fold
fragments is set by k and the source-gate arity, not the mask degree; widening
only adds control literals to existing fragments. (Measured at n=16:
-3,k=3 is 5 944 gates vs. -2,k=3's 6 161—
slightly *fewer*, as wider atoms hit more literal-contradictions.) The real
price is **fragment width**: a degree-d mask folded through a w-operand
gate yields fragments up to width d· w. Degree-3 reaches width 6 (the
frozen-DB m=6 cliff), degree-4 width 8 (past it, needing scratch
decomposition). Towers are nearly free to *build*; the bill is downstream
**mixing difficulty** (§).

## The mixed design

Because degree and k do different jobs, the production encoding mixes tiers:
k base degree-2 terms (statistical strength; each a strong 1/4 perturbation)
*plus* k_hi degree-3 tower terms (degree-2 algebraic hiding).
The validated setting is [2,2,3,3]—two deg-2 + two deg-3 terms—which kills
the degree-1 *and* degree-2 diagonals with margin (a single deg-3 term
leaves a boundary residual; two do not).

## Validation

All heatmaps read by the ridge measure with verified references (endpoint
H≈0). "ALIVE" = clean diagonal (ρ≈1); "dead" =
median-row prominence at the noise floor and ρ≈0.

**n=16 degree grid** (fixed C, 3 gadget seeds). The plain gadget's
degree-1 diagonal has depth 0.356, ρ 1.0.

| encoding | deg-1 depth | deg-1 ρ | deg-2 | deg-3 |
|---|---|---|---|---|
| plain (k=0) | 0.356 | 1.00 | — | — |
| deg-2 masks | dead (0.11) | 0.5 | **ALIVE 0.35**, ρ 1.0 | — |
| deg-3 masks | dead (0.11) | 0.5 | dead (0.13) | **ALIVE 0.33**, ρ 1.0 |

=d masks kill every adversary of degree <d; the diagonal
returns at degree exactly d (the wall moves up one level per +1 mask degree,
it does not vanish). k=2 hides the degree-2 diagonal as completely as k=3.
All nulls are capability-controlled: a band-restricted degree-2 adversary
*does* recover degree-2 masks (ρ 1.0).

**n=128 sliced sandwich** (real production circuit). Original C=3 000
g57 gates; the mixed [2,2,3,3] gadget is **374 478 gates on 576
wires** (125× C).

| gadget | adversary | depth | depthMed | ρ | meanH |
|---|---|---|---|---|---|
| plain | degree 1 | 0.361 | 0.367 | **1.00** | 0.471 |
| mixed [2,2,3,3] | degree 1 | 0.060 | **0.000** | **-0.02** | 0.498 |
| mixed [2,2,3,3] | degree 2 | 0.060 | **0.000** | **-0.04** | 0.498 |

The mixed encoding kills both diagonals on the real sandwich; the
median original-circuit prefix has zero ridge prominence. Control: the plain
gadget (512 wires) vs. mixed (576, only +64 band) reads a perfect diagonal, so
the mixed flatness is real hiding, not a wire-count artifact.

**Full production pipeline (n=128).** Gadget → phase A (a 3M-move
DB-mixing dose, holding size) → phase B (grown ×3 to 1.12M gates).
Degree-1 hmap vs. the original C at each stage (matched sampling, 101 rows):

| stage | gates | depth | depthMed | ρ | meanH |
|---|---|---|---|---|---|
| plain gadget | 85 k | 0.351 | 0.359 | **1.00** | 0.475 |
| mixed gadget | 374 k | 0.044 | **0.000** | 0.017 | 0.499 |
| phase A (DB dose) | 392 k | 0.044 | **0.000** | 0.018 | 0.499 |
| phase B (×3) | 1.12 M | 0.044 | **0.000** | 0.017 | 0.498 |

The diagonal is killed by the encoding and stays dead
*identically* through mixing and a threefold growth—depth, median-row
prominence, and ρ do not move. Mixing neither helps nor hurts, because the
leak was already gone before mixing began.

## Interaction with mixing—a caveat and a feature

The wide mask fragments (width up to 15 after folding through the
256-wire sandwich's CCNOTs) are **ineligible for frozen-DB re-encoding**
(the narrow m1–m11 store, `--db-ctrl-cap 2`). Operationally,
the all-gates 5th-percentile generation (`Gall=`) is pinned at 0 because
a large fraction of gates can never be re-encoded. *Since* `512ce31c`
*the reported* `G=` *and the dose stop are measured over the
targetable population instead, so "run to generation N" does converge — it
means N over the DB-eligible material.* Defensively, the same
fact means the encoding **resists DB-based simplification**, which is a
property one wants of an obfuscation.

## Implementation

- `src/replace/gadgets.rs`: `ProdConfig` / `ProdLedger`
(draw/inject/strip, `fold_cg`, `emit_band_fill`); wired into
`gadgetize_cnot`, `gadgetize_xgates`, and the
`slice_zero_ccnot` wrappers. `ProdConfig::off` reproduces the
plain gadget bit-identically.
- CLI: `sss --cnot --gadgetize --prod-k/-deg/-k-hi/-deg-hi/-band/-rsrc`;
env-var control in `gen_sandwich_gadget`; `prod_grid.rs` harness.
- `hmap_affine`: `Reg::Triple` so `--degree 3` forms
genuine triples over `--deg2-wire-list` (previously capped at pairs).
- Correctness: library suite 166/166; endpoint mismatches 0 at
n=6,16,128; masked builds pass the pinned zero-slice contract; no bare X.

## Open threads

- The literal "generation 100" phase-A target is infeasible on the
wide-gate gadget (§); the pipeline uses a fixed DB dose + a
×3 growth phase.
- A deg-4 tower needs scratch decomposition of width-8 fragments (the cubic
borrowed-scratch pattern in `emit_poly_add` is the template).
- Quantify the mixing/DB-match cost of width-6 gates directly.
- Production review against non-reconstruction (static gate-statistics)
distinguishers; the heatmaps are execution-based only.
