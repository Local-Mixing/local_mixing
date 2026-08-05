# **Nonlinear Re-Randomization and CG Diversification**

> Markdown rendering of `docs/NONLINEAR_RG_CG_MENU.tex`. The PDF is authoritative for figures, diagrams and display math.

## Abstract
The 2026-07-20b revision of the new-gadgetize recipe makes two
changes to the gadget body: the linear cnot{} re-randomization gadgets
{RG2, RG3} are replaced by the legacy *nonlinear* g57 networks
{RG1, RG2, RG3} (one uniform draw per shared gate), and the fixed
four-fragment realization of the shared computation gadget (CG) is replaced
by a per-gate uniform draw over *seven* structurally different
realizations. This note records the motivation: what the degree-bounded
reconstruction experiments showed about the linear design, what the
reversal buys and what it deliberately gives up, and why implementation
diversity of the CG is itself a security measure. Companion changes (the
final commuting-order rerandomization and the sandwich column float) are
summarized at the end.

## Setting

The two-share gadget maps a circuit C on n wires to an equivalent
circuit on 2n wires. Every logical value v is carried as an XOR pair
v = s_v ⊕ p_v; the construction consists of random Z bookends on the
aux half, a W_i encode/decode curtain of cnot{}s that installs and
removes the sharing, one *computation gadget* (CG) per source gate,
and a stream of *re-randomization gadgets* (RGs) that churn the
sharing between CGs. The zero-slice cnot/ccnot{} preblock guards the
input port.

The original new-gadgetize design chose *linear* RGs (a 3-cnot{}
re-pairing RG2 and a 2-cnot{} mask-refresh RG3) and one fixed
four-fragment CG. Both choices optimized a specific adversary model:
*gate-local non-completeness* — no single physical gate ever reads
both carriers of one value, so every prefix of the circuit is first-order
probe-masked — plus gate-count leanness.

## The adversary that changed the decision

The project's security target is indistinguishability against *all
feasible distinguishers*, evaluated against concrete cheap attack
families. The family that drove this revision is *degree-bounded
algebraic reconstruction* (`hmap_affine`): for prefixes C_i of the
original and G_j of the gadgetized circuit, fit each bit of C_i(x) as
a degree-d GF(2) function of *all* of G_j's wires (train/holdout
split; inconsistency scores 0.5). Low error means the original
computation's progress is legible at that point of the gadget; the ideal
map is flat at 0.5 except the two forced I/O corners. The measure is
invariant to affine re-encodings, which is exactly what makes it the right
lens for a sharing-based gadget: *the XOR sharing itself is free for
this adversary*.

Two facts came out of the measurement campaign:

1. **Degree-2 reconstruction is more persistent than degree-1
under mixing.** Mixing erodes the degree-1 (affine) diagonal quickly, but
the degree-2 diagonal survives much longer. The affine conjugation twists
of the mixing stage cannot address this: an affine change of variables is
transparent at every degree.
1. **The gadget encodings differ sharply.** The old (legacy g57)
gadget's output is *not* recoverable even by restricted degree-2
predictors, while the linear-RG cnot{} gadget is affine-transparent at
its interface. Computing the ANF degree of each component isolated the
cause: the sharing setup (M, W_i) is affine everywhere; the
nonlinearity of the legacy gadget lives in its gate application (degree
2, once per source gate, compounding) and its re-randomizers — RG1
(degree 3), RG2 (degree 2), RG3 (degree 2). The linear replacement kept
the setup and *removed exactly the nonlinear part*.

The structural point is simple: affine maps compose to affine maps. With
linear RGs, the entire re-randomization stream between any two CGs is one
affine transformation of the wire state, so a degree-d predictor sees
through an *arbitrarily long* RG stream at no cost in degree. The
stream churned the sharing but hid nothing from the adversary that
matters. Nonlinear RGs break this composition: every RG multiplies
algebraic degree into the transition, and reconstruction across a stretch
of the body must now pay degree for every RG it crosses.

## The reversal: nonlinear {RG1, RG2, RG3}, one per CG

The revision reinstates the legacy g57 networks, drawn uniformly at one RG
per CG gap (`–rg-frequency`, default 1 on this path):

| RG | effect on the sharing | gates | ANF degree |
|---|---|---|---|
| RG1 | swap the virtual values of two pairs | 6 g57 | 3 |
| RG2 | re-pair two pairs crosswise | 6 g57 | 2 |
| RG3 | XOR r_1 ∨ ¬ r_2 of two foreign wires into both carriers | 2 g57 | 2 |

**What is bought.** Nonlinear transitions of degree 2–3 that compound
along the body; a re-randomization stream that low-degree reconstruction
cannot cross for free; and, as a side effect, nonlinear material (the RG3
reads of two *random* foreign wires) distributed through the entire
body rather than confined to the bookends.

**What is paid, deliberately.** RG1 and RG2 gates read both carriers
of a value, so gate-local non-completeness — first-order probe masking
— is given up. This is a conscious ranking of adversaries: the probing
model was a secondary concern, while degree-bounded reconstruction is a
demonstrated, cheap, working attack on the linear design. (RG3 keeps
single-carrier reads; the masking-safe CG variants below keep the option
of a partially masked body alive.)

**Measured effect** (n=16, pre-mixing, old vs. revised recipe):
the maps flatten (degree-1 σ 0.125 → 0.106, degree-2 $0.161 →
0.149$; the interior column-mean dip that leaked the temporal layout
largely evens out), and the best-case interior degree-1 reconstruction
error roughly doubles (≈ 0.05–0.15 → 0.2–0.3). One honest
caveat: the degree-2 *snapshot* diagonal persists in both versions.
This is inherent to the linear XOR sharing — at any prefix the logical
value *is* s ⊕ p, a degree-1 function of the wires —
independent of how nonlinear the transitions are. Erasing snapshots is the
mixing stage's job (structural DB/split mixing is what erodes it), or, in
a future design, a nonlinear share *encoding*.

## CG diversification: a seven-variant menu

The CG applies A ⊕= B ∨ ¬ C over the shares, i.e.\
f = 1 ⊕ C ⊕ BC with B = b_0 ⊕ b_1, $C = c_0 ⊕
c_1$. The original design emitted one fixed shape — four mixed-polarity
conjunction fragments — for every source gate.

**Why fixed is bad.** A fixed CG is a syntactic beacon: a fixed
period, a fixed local gate-type pattern, and a fragment census that counts
the source gates exactly. Regularity is precisely what static profiling
and SAT-style attacks feed on — a solver that learns one CG shape has
learned them all. And the fragments themselves are cold material for the
mixing stage: the frozen replacement store matches g57 windows at
99% but mpmct fragment windows at 2.8%, so a
fragment-heavy body resists exactly the phase-A mixing that is supposed
to dissolve it.

**The menu.** Each source g57 is now emitted as one of seven
realizations, drawn uniformly per gate, with the carrier roles (which
carrier of A is targeted; share/pad order of B and of C) also
randomized per draw:

| }

# | name | shape | idea |
|---|---|---|---|
| 0 | collapse both | 4 cnot{} + 1 g57 | b_0⊕=b_1, c_0⊕=c_1 put B, C on single wires; one g57 fires f; restore |
| 1 | collapse c | 2 cnot{} + 2 g57 + ¬cnot{} | pair sums to BC; a ⊕= ¬ c_0 adds 1 ⊕ C |
| 2 | collapse b | 2 cnot{} + 1 g57 + 1 frag | [a,B,c_0] plus fragment ¬ B ∧ c_1 |
| 3 | linear tail | ¬cnot{} + cnot{} + 4 g57 | no collapse; quad [a,b_i,c_j] sums to BC; masking-safe |
| 4 | legacy GADGET | 6 g57 | the classic nonlinear network |
| 5 | quad + pair | 5 g57 + 1 frag | quad BC, then (¬ c_0 ∧ c_1) + [a,c_1,c_0] = 1 ⊕ C |
| 6 | 4-cube ESOP | 4 fragments | the previous fixed CG; masking-safe |

Vocabulary is restricted to g57s and pure conjunctions with one or two
controls of any polarity. In particular, *no bare X gates*: several
variants algebraically need a "⊕ 1", but an X census would count
the source gates, so every constant is folded into a negative-control
cnot{} or a complemented fragment.

**Gate-by-gate.** Each realization writes only the target carrier
a_ (one of A's two carriers, chosen per draw); b_0, b_1 and $c_0,
c_1are the two carriers ofBand ofC$ in a per-draw random order.
Notation, all over GF(2) (so + is ⊕ and $w = 1 ⊕
w):t {{+}{=}} wis a cnot (t ← t ⊕ w);t {{+}{=}}
wa negative-control cnot;t {{+}{=}} (x ∨ y)$ a
g57; and t += (ℓ ℓ') a two-control conjunction fragment
(juxtaposition = AND, each literal w or w). Recall $f = 1
⊕ C ⊕ BC = B ∨ C$. Gates execute top to bottom. The
same seven circuits are drawn in standard reversible-gate notation in
Appendix .

0 — collapse both (4 cnot{} + 1 g57):

1. b_0 += b_1 now b_0 = B
1. c_0 += c_1 now c_0 = C
1. a_ += (b_0 ∨ c_0) = B ∨ C = f
1. c_0 += c_1 restore c_0
1. b_0 += b_1 restore b_0

1 — collapse c (2 cnot{}-type + 2 g57):

1. c_0 += c_1 now c_0 = C
1. a_ += (b_0 ∨ c_0)
1. a_ += (b_1 ∨ c_0) the two g57s sum to BC
1. a_ += c_0 adds C = 1 ⊕ C
1. c_0 += c_1 restore c_0

2 — collapse b (2 cnot{} + 1 g57 + 1 fragment):

1. b_0 += b_1 now b_0 = B
1. a_ += (b_0 ∨ c_0) = 1 ⊕ c_0 ⊕ B c_0
1. a_ += (b_0 c_1) = c_1 ⊕ B c_1; total = f
1. b_0 += b_1 restore b_0

3 — linear tail (cnot + cnot{} + 4 g57; masking-safe, no carrier collapsed):

1. a_ += c_0
1. a_ += c_1 gates 1–2 give c_0 ⊕ c_1 = 1 ⊕ C
1. a_ += (b_0 ∨ c_0)
1. a_ += (b_1 ∨ c_0)
1. a_ += (b_0 ∨ c_1)
1. a_ += (b_1 ∨ c_1) the four g57s sum to BC

4 — legacy GADGET (6 g57; c_1 borrowed as restored scratch):

1. c_1 += (b_0 ∨ b_1) scratch on c_1
1. a_ += (c_1 ∨ b_1)
1. a_ += (b_0 ∨ c_1)
1. c_1 += (b_0 ∨ b_1) restore c_1
1. a_ += (b_1 ∨ c_0)
1. a_ += (c_0 ∨ b_0)

5 — quad + pair (5 g57 + 1 fragment; no carrier collapsed):

1. a_ += (b_0 ∨ c_0)
1. a_ += (b_1 ∨ c_0)
1. a_ += (b_0 ∨ c_1)
1. a_ += (b_1 ∨ c_1) the four g57s sum to BC
1. a_ += (c_0 c_1)
1. a_ += (c_1 ∨ c_0) gates 5–6 sum to 1 ⊕ C

6 — 4-cube ESOP (4 fragments; the previous fixed CG, masking-safe):

1. a_ += (c_1 b_0)
1. a_ += (c_1 b_1)
1. a_ += (c_0 b_1)
1. a_ += (c_0 b_0) the four cubes XOR to B ∨ C

Every non-target carrier that a variant touches (the collapses in 0–2, the
scratch in 4) is written an even number of times and so is restored exactly
at the wire level; verified exhaustively over all 7 variants, 8 carrier
role assignments, and 64 inputs.

**What diversification buys.**

- *Irregularity*: no fixed CG period, no repeated shape, varying
local gate-type statistics — less structure for syntactic profiling and
for solvers to latch onto. The carrier-role randomization scrambles pin
patterns even between same-variant draws.
- *DB-warmth*: the body census flips from fragment-heavy to
g57-dominant (at n=16, g57 / cnot{} / fragment counts $822/277/707
→ 1269/486/264$), with cnot{}s only sprinkled — and a few cnot{}s
within many g57s are well tolerated by the replacement store, while the
g57 mass mixes at the 99% match rate.
- *More nonlinearity in the body*: five of seven variants carry
g57 material into the CG itself, compounding with the nonlinear RG stream.

Cost: the menu averages 5.1 gates per CG versus 4, i.e.\
12% body growth at n=16 — accepted.

Every variant is verified exhaustively (all 7 variants × 8 role
draws × 64 carrier assignments: exact update, wire-level
restoration of every non-target carrier, vocabulary compliance), and the
driver's pinned zero-slice functionality check runs on every build.

## Companion changes, briefly

**Commuting-order rerandomization** (`commuting_shuffle`,
`5689c8ea`). The construction-time layout $S_1 | Z | W |
body | W^{-1} | Z$ was legible in the heatmaps as an S-shape
— no mixing between the computation block and the bookends, with the
affine W_i curtain sitting between them. The emitted order is now a
fresh random order that preserves only the relative order of gate pairs
that *provably* collide (`XGate::collides`: "commute unless
proven otherwise", including the opposite-polarity shared-control
exemption). This is a one-time, structure-dependent reorder — consistent
with the principle that local mixing belongs to the mixing stage, whose
job the gadget stage only sets up.

**Sandwich column float** (`9323272a`). In the sliced sandwich,
each middle-column cnot{} is assigned a registered random direction and
floats that way to its commutation extreme, dissolving the C|N|D seam
— the sandwich's most structure-revealing feature — into a band before
gadgetization.

## Open items

The production n=128 A/B (old vs. revised recipe through the
two-phase mixing pipeline, evaluated with `hmap_affine` at degrees 1
and 2) has not yet been run. The persistent degree-2 snapshot legibility
of the linear XOR sharing remains the deepest open design question; the
candidate answers are a nonlinear share encoding (snapshots stop being
degree-1 in the wires) and generalized nonlinear RG3 injections
(compensated pair-writes with arbitrary conjunction fragments).

## Gate-level diagrams of the CG menu

Each realization of Section 4 is drawn below in standard reversible-gate
notation. The five horizontal lines are the carrier wires (top to bottom:
a^ the targeted carrier of A, then b_0, b_1, c_0, c_1); the
untouched second carrier of A is a mask and is omitted. A gate is a
vertical line joining its pins. A filled pin is a positive
control (active when its wire is 1), an open pin a negative
control (active when 0), and ⊕ the target that is flipped when the
gate is active. A plain conjunction gate is active exactly when all its
controls are satisfied. A **g57** gate carries a bar over its
target, ⊕: it is *complemented*, active on the
complement of its control cube — realizing t += (x ∨ y)
from pins on x and on y. Columns are numbered to match
the gate sequences in Section 4 and apply left to right.

**0 — collapse both**

[cgcirc diagram — see the PDF]

**1 — collapse c**

[cgcirc diagram — see the PDF]

**2 — collapse b**

[cgcirc diagram — see the PDF]

**3 — linear tail**

[cgcirc diagram — see the PDF]

**4 — legacy GADGET**

[cgcirc diagram — see the PDF]

**5 — quad + pair**

[cgcirc diagram — see the PDF]

**6 — 4-cube ESOP**

[cgcirc diagram — see the PDF]

Reading these against the algebra: in 0–2 a cnot{} pair collapses a value
onto one carrier (the on b_1/c_1 writing b_0/c_0) and the
mirror pair at the end restores it; variant 4 uses c_1 as scratch,
written and restored by the identical gates in columns 1 and 4. The
masking-safe realizations (3 and 6) are exactly those whose every gate
reads at most one carrier of each logical value.
