# The Folded Gadget — a self-contained, composable, leak-free reversible gate

`gate_gadget_v2.py` implements one gadgetized gate of an obfuscated reversible circuit,

```
c_out = c_in ⊕ V(a, b)
```

over nonlinear encodings, designed against an adversary who sees the **full per-gate
trace**: every input wire, plus after every physical gate its flip bit and the targeted
wire's new value. Two attackers are defended:

* **Exact / affine (total break):** no GF(2) affine combination of trace coordinates may
  equal any of `a`, `b`, `V(a,b)`, `c_in`, `c_out` (found by Gaussian elimination).
* **Weight-≤2 statistical:** no single coordinate, and no pair of coordinates under *any*
  of the 16 two-bit Boolean functions, may have nonzero covariance with those values.

Both hold — to sampling precision — for a single gadget, for chained gadgets (an output
feeding a later input), and for a six-gadget SG spliced together with **no glue at all**.
Weight-3+ combinations are outside the threat model (some are genuinely nonzero, e.g.
stripping both cascade masks takes three coordinates).

## Encoding and representation policy

Every logical value is carried by the 2-resilient nonlinear decode on 5 wires

```
E(x0,x1,x2,x3,x4) = x0 ⊕ x1 ⊕ maj(x2,x3,x4)
```

(minimal: two uniform pads plus a balanced 3-wire nonlinearity — no 4-wire decode is
2-resilient).

* A wire that is **ever written** is 2-shared: `c = E(S1) ⊕ E(S2)`. This is forced:
  writing uses a share's majority as a flip carrier, and a carrier-touched single
  encoding's value lands in the exact attacker's span.
* A wire that is **read-only for its whole life** may stay a single E-encoding — reads
  only ever emit masked monomials, which expose nothing. This is a per-wire compiler
  policy (`a_single` / `b_single`), worth ~34% of the gate count, **with a caveat** (see
  Known limitations).

## The gadget: the re-share is inside the gate

Per gate, the caller supplies the operand blocks, the target's two share blocks (S1, S2),
**5 fresh random wires R**, **4 fresh random wires chaff**, and clean ancillas (out×3,
scratch, scratch2). The gadget performs ONE permutation-form flip of S2's majority, keyed

```
W = u ⊕ V,      u = E(S1) ⊕ E(R)      (the folded re-share)
                V = the gate's cleartext output over the operand encodings
```

landing the permuted majority on the out-wires. The output interface is a **relabeling
that costs zero gates**:

```
share1' = R                    share2' = (S2[0], S2[1], out0, out1, out2)
```

Correctness: `E(share2') = E(S2) ⊕ u ⊕ V`, `E(share1') = E(R)`, so the value gains
exactly `V`. Composition means calling the gadget again on the new blocks — outputs into
inputs, repeated writes to one wire, repeated reads of one encoding all verified bare.

Why it is safe, in two sentences: `u` is fresh-uniform, so the flip key `W` is uniform
and **independent of V** — the gate output is never a flip amount anywhere, which kills
every `V·B`-class weight-2 witness (`cov(W·B, V) = −cov(B, V) = 0`). Each gadget
carrier-touches only S2's majority while share-1 generations (S1, then R) are only ever
read as masked monomials, so every per-gadget relation the exact attacker can form
contains one fresh never-exposed majority unknown and the system never closes.

## Gate types and cost

`V` is a compile-time ANF; four types cover the SG construction (dd = both operands
2-shared, ss = both single-encoded; clean restores scratch2, dirty leaves masked garbage,
−~11%):

| vtype  | V           | realizes            | dd clean/dirty | ss clean/dirty |
|--------|-------------|---------------------|----------------|----------------|
| `r57`  | 1 ⊕ b ⊕ ab  | c ^= a ∨ ¬b         | **193** / 169  | 128 / 114      |
| `nab`  | b ⊕ ab      | c ^= ¬a ∧ b         | 190 / 166      | 125 / 111      |
| `and`  | ab          | c ^= a ∧ b          | 193 / 169      | 128 / 114      |
| `copy` | op          | c ^= op (1-input)   | —              | 70             |

A full SG (one logical gate over SG-2-shared wires, `r_t ^= f⊕g`, `s_t ^= g`, six
sub-gates; `big_gate_gadget.py`) costs **649 gates with fresh (read-only) input shares**; in a deep circuit
where input shares were written by earlier SGs, the product gates run dd and an SG costs
~909.

## How the ~190 gates are spent (and why each discipline rule exists)

The `a·b` term uses a **masked Toffoli cascade** per output bit instead of the flat
10×10 monomial expansion: `scratch2 ← (b ⊕ m0 ⊕ m1)·B_i(s)`, then a's monomials multiply
against scratch2, with two correction runs canceling the mask spill. Every rule below
closes a leak class that was *measured*, then fixed, then re-measured:

* **Two masks, not one** — a single mask is stripped by the weight-2 mirror pair
  `(m∧scratch2) ⊕ (m∧mask∧B) = m·b·B` (measured 0.375; kept reproducible as the
  `n_masks=1` positive control in the development files).
* **Masks = R0,R1 for `r57`/`nab`** — u contains them linearly, so the corrections cancel
  two u-emissions outright. **`and` gets dedicated masks (chaff[2:4])**: it emits u's
  R0,R1-linear gates (no s2-emit to cancel them), and reusing R0,R1 as masks lets those
  bare `mask·B` coordinates echo against the build's `mask·B₂` across groups (measured
  0.118).
* **Build/unbuild order** — pads of b at the extremes, masks strictly interior,
  inter-mask segment balanced (for a single-encoded b: the full maj-triple, whose XOR is
  maj — balanced), unbuild in *exact reverse*. Violations measured at 0.375–0.76 (the
  worst: an unbuild intermediate `(b⊕R1)·B` paired with the bare public wire R1).
* **Chunked runs with provably balanced boundaries** — single-role chunks (cascade /
  corr-m0 / corr-m1) of ≤5 gates; every boundary carries a *balanced* gate: linear-u
  monomials or chaff pairs (each chaff wire emitted twice = net zero, sequenced so every
  boundary interval keeps one uncanceled). Maj-pair u-monomials are **not** balanced
  (E[(−1)^{sᵢsⱼ}] = ½) and don't count — using them as boundaries left a 0.079 residual.
* **Shared-B groups** — the permutation has B₀ = B₁ as functions, so rounds 0,1 share one
  scratch/scratch2 build; both runs are emitted in the **identical order**, collapsing
  every cross-run pair into the single-run window family.

**Permutation constraints** (exhaustive search over all 576×576 valid
majority-preserving/flipping pairs): every `B_i` non-constant (a constant B emits W bare
→ V leaks exactly); **no XOR-subset of the distinct B_i constant** (else `b·B ⊕ b·B₂ = b`
lands in the exact span — the first cost-minimal candidate had B₂ = ¬B and leaked b and V
exactly); **E[B_i] balanced** (kills a second-order residual ~(2E[B]−1)·0.1); minimal ANF
overhead. The shipped pair: `SB_U0 = id`, `SB_U1 = (7,5,6,0,3,1,2,4)`, overhead 13.

## Verification (test_gate_gadget.py + dedicated harnesses)

Per configuration: end-to-end correctness; **independent replay** (a from-scratch
simulator re-executes the recorded gate list from initial values and demands bit-exact
agreement with every flip/new-value/final state — no out-of-band writes can hide);
structure (written wires = declared ancillas exactly, comp=0 everywhere); **exact test on
the raw non-deduplicated trace** with the eliminator fuzzed by planted/perturbed
relations every run; exhaustive weight-1/weight-2 scan (all 16 pair functions via a
matmul scanner cross-validated against brute force to 1e-9) against a NULL random target.

Composition evidence (development harnesses): 2-gate output→input chain and 3-gate
strict chains, 0 exact leaks, weight-1 ≤ 0.0084 at n=200k; the **six-gadget SG**: 0 exact
leaks on all 28 checked values × 3 seeds, SG-level weight-1 ≤ 0.0087 at n=200k, full
weight-2 at the NULL/noise level. Single gadget weight-1 bounded ≤ 0.005 at n=400k.

## Known limitations

* **All "zero"s are sampling bounds**, not proofs: ≤~0.005 (w1, n=400k), ≤~0.02 (w2,
  n=48k), ≤~0.05 (composed w2, n=10k).
* **Single-encoded (ss) operands carry a residual weight-2 class on the operand
  itself** (~0.03–0.07; measured 0.068 for `and`-ss): a cross-group pair can cover *both*
  pads of a 5-wire encoding (2 pads), voiding the padding argument; the dd representation
  is structurally immune (4 spread pads cannot be covered by two coordinates without
  crossing balanced boundaries), and every dd configuration measures clean. This is
  share-level exposure only — 2-shared logical values built from such operands (the SG
  case) measured clean at n=200k — but for unconditional per-gadget guarantees use dd
  (the default).
* Weight-3+ statistics are out of scope and some are genuinely nonzero.
* The emission-order rules are load-bearing. Do not reorder gates.
