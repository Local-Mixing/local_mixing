# Gadgetization — Theory, Security Properties, Constructions

Gadgetization is the stage that turns an ordinary reversible circuit C
into a much larger circuit G computing the same function, such that **no
small piece of G's execution reveals anything about C's intermediate
values**. G is what gets handed to the mixer; if gadgetization leaks, no
amount of mixing repairs it (mixing re-randomizes *which* wires carry a
value — it cannot remove a value that is already exposed).

## The idea

Every logical value `v` of C is **secret-shared**: instead of one wire
carrying `v`, several wires carry shares whose combination (e.g. XOR, or
a product-share decode) equals `v`. Each gate of C is replaced by a
**gadget** — a small circuit that consumes the input shares, computes the
gate's function, re-shares the result with *fresh* randomness, and never
lets any wire, or small tuple of wires, carry information about the
shared values. Randomness enters on dedicated **borrow wires** fed from
an avalanche pool (in testing: either ideal fresh randomness or the
production band pool; see `TESTING_PIPELINE.md` §5).

Why linear sharing (`v = c₀ ⊕ c₁`) is not enough: the decode is affine,
so at every instant every logical value is a degree-1 function of G's
wires, and an adversary who fits affine functions reads C's progress out
of G directly — the famous low-error **diagonal** in `hmap_affine` maps.
The full argument and the nonlinear fix are in
`docs/NONLINEAR_GADGETIZATION.md` (the monotone-function mask trick is
§13 there); the refresh-gadget and CG-menu design is in
`docs/NONLINEAR_RG_CG_MENU.md`.

## The security properties (the checklist)

Stated as attacks that must **fail** against a gadgetized G, per original
(pre-gadgetization) gate, targeting its five logical values
`(a, b, c_old, f, c_new)`:

| # | property | attack name in the gauntlet |
|---|---|---|
| 1 | No trace value ever equals a target bit exactly (no wire carries a raw logical value) | `a1` — direct wire match |
| 2 | No target is an affine function of any single wire state `G_j` | `xrows` — Gaussian elimination per row |
| 3 | No target is an affine function of the **entire execution trace** (all flips/newvals of G) | `xtrace` — Gaussian elimination over the full feature set |
| 4 | No statistically significant **weight-1** correlation between any target and any trace value | `w1` |
| 5 | No statistically significant **weight-2** correlation (xor/and/or of any two trace values) | `w2` |
| 6 | Weight-3 correlations are measured and reported; small known residuals exist in the 193-gate gadget and are tracked, not yet zero | `w3` |

"Significant" is NULL-calibrated: a target is flagged only if its
strongest correlation beats both the NULL column's maximum (a same-class
noise reference) and a 6σ threshold. Properties 1–3 must be *exactly*
zero; properties 4–5 must have *zero flagged targets*; property 6 is the
open residual. The precise definitions, thresholds, and witness format
are in `TESTING_PIPELINE.md`.

Structural properties, verified at build time: G is input-preserving
(equivariance checked by sampled simulation against C), strictly borrowed
randomness never leaves its gadget context, and transitory share borrows
obey the operand duty contract.

## The gadget ladder

Constructions under test, ordered by per-gate cost (names embed the
measured full cost per source gate):

| gadget | construction | cost/gate | docs |
|---|---|---|---|
| `none` | no gadget (control — must fail the battery) | 1 | — |
| `secretshare14` | the paper's paired secret-share `w = s ⊕ r`: one SG from a 7-variant menu ({5,5,4,6,6,6,4} gates) + 2 refresh gadgets (RG1/2/3 = {6,6,2}); expected cost 36/7 + 2·14/3 = 14.48 | ~14.5 (randomized) | `SSG_README.md`, `docs/NONLINEAR_RG_CG_MENU.md` |
| `bandproduct92` | the paper's product-share band `V = C ⊕ M(B) ⊕ κ`, Gray fold, mask plan [2,2,2,3]; randomized gathers/rolls/top-ups | ~92 (randomized) | `docs/PRODUCT_SHARE_ENCODING.md`, `docs/GRAY_FOLD_CG.md`, `docs/BAND_HARDENING.md` |
| `nonlinear193` | folded single-gate gadget: T-conjunction tree with band-pool masks `r·B(s)`, fresh 5-wire re-share R per gate | 193 (exact) | `README_gate_gadget_v2.md` |
| `nonlinear939` | full cascade: same mask structure **and** a re-share gadget whose own 23 mask gates are recursively masked | 939 (exact) | `README_gate_gadget_v2.md` |

Plus `_band0` / `_band16` variants of the last two, where borrow wires
are fed by the input-keyed band pool with 0 or 16 blind layers instead of
ideal randomness (contract and verdict in `TESTING_PIPELINE.md` §5).

The 193-gate gadget passes properties 1–5 exactly and has a measured
weight-3 residual (the mask-cascade leak: three mask flips XOR to
`b·B(s)` with B 3/4-biased) that mixing can demote to weight 2 —
documented in `README_gate_gadget_v2.md` §7 and fixed by construction in
`nonlinear939`.

Retired: the "nonlinear carrier" gadget (an in-repo design note,
`nonlinear-carrier-gadget.md`; not from the paper, superseded by the
folded gadget) — code kept, out of the pipeline.

## Where the pieces live

- Python reference builders (canonical, with their own test suites):
  `gate_gadget.py` (v1, 193), `gate_gadget_v2.py` + `big_gate_gadget.py`
  (v2, 939).
- Rust production gadgetizers: `src/replace/gadgets.rs`
  (`gadgetize_xgates` = secret-share, `gadgetize_xgates_single` =
  band-product, `gadgetize_xgates_gg` = the folded gadget's Rust port).
- The audit battery that checks the property list: `TESTING_PIPELINE.md`.
