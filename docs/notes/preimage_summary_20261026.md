# Preimage cryptanalysis of the baseline mixed output — efforts and conclusions

*2026-10-26*

## 1. Objective and setup

The target is the declared mixed output of the baseline / as-is run of the TDP
pipeline: a reversible circuit **F** on 256 wires with **666,745 gates** that computes a
64-bit permutation **C** on the honest input slice. The honest input places
`x` on wires 0–63 and zeros elsewhere; the payload `C(x)` appears on wires 64–127 at
the output.

**Goal.** Find an input `x` with `C(x) = 0^64` (a preimage of zero), using only the
declared output **F** and the attacker's own rewrites of it. The identity of the
internal `C` values exposed by the earlier census is *not* used — it was derived with
knowledge of `C` and is therefore off-limits. `C` itself is used only in clearly
labelled diagnostics that measure what an attack could have seen; it never drives an
attack. Every candidate `x` is verified against the real circuit.

Three equivalent forms of the same `C` were available and used as targets:

| circuit | gates | origin |
|---|---|---|
| F | 666,745 | declared baseline/asis mixed output |
| plateau | 355,990 | C-blind ANF resynthesis of F (30 rounds) |
| short-279k | 279,263 | fcompress shortening sweep (`fc4_c5_n64`), verified equivalent |

A preimage found on any of them is the same `x`, verifiable on all.

## 2. Attacks attempted

### 2.1 Direct SAT inversion

Each circuit was encoded into CNF (one variable per initial wire and per gate birth;
Toffoli gates as an AND auxiliary plus a native XOR clause for CryptoMiniSat), with the
non-input wires fixed to 0 at entry and the payload fixed to `0^64` at exit. Instances
on F and on the plateau were run with 8 threads each.

**Outcome:** no solution. Both single-circuit baselines ran for more than 30 hours
without returning.

### 2.2 Structural reconstruction (vocabulary walks)

Build the vocabulary of every function (values on 8,192 sampled inputs, up to
complement) born on any wire of the derived circuits, then walk `g57` gates forward
from the inputs and backward from the payload, accepting a gate whenever the implied
value is present in the vocabulary. A complete chain would reconstruct `C` and hence
`x`.

**Outcome:** no chain. A diagnostic (using `C`) showed the vocabulary covers 684 of
`C`'s 1,000 gate births with seven circuits and 735 with all 65 fair-game circuits, but
mere membership does not identify a genuine `C` gate: the C-blind resyntheses and
stage-3 remixes contain many non-`C` values of exactly the tested form. Forward walks
were spurious from the first gate (the vocabulary near the inputs is saturated with
low-degree polynomials of `x`); backward walks were genuine only with the smallest
vocabulary and stalled after 20–40 gates. A lineage rule and a degree-2 span filter
reduced false matches but did not make the walk reliable.

### 2.3 The replication landmark

The raw sandwich computes `C(x)` in the middle and copies it, so an unmixed circuit has
an interior cut where two disjoint 64-wire bundles carry the same functions; constraining
that bundle to zero would isolate `C` in the circuit's first half. We tested, C-blind,
whether this landmark survives, at three levels of generality.

- **Verbatim / up to NOT — absent.** Scanning every cut of F and its three rewrites,
  the number of functions carried by two or more wires peaks at only 15–21 and only at
  the very start (depth 0.005–0.009, the input-copy step); it is zero through the entire
  interior. F's 64 output functions are never co-present at an interior cut (0–2 across
  the middle, all 64 only at the final gate). Stage-6 packing and ANF resynthesis
  rebuild the circuit toward the output, destroying the clean mid-circuit copy.

- **Affine (degree-1) — weak.** At each cut, how many output functions lie in the
  GF(2)-affine span of the live wires. The midpoint carries only a weak partial replica:
  F peaks at 10 of 64 at depth 0.485, the two remixes at 5. Insufficient to invert.

- **Degree-2 — the finding (see Section 3).**

### 2.4 Exploiting the degree-2 structure

Three sound ways to turn the degree-2 midpoint relations into an inversion were built
and run. All are sound (the added constraints are logically implied by `payload = 0`)
and self-verifying.

- **Two-circuit linked SAT.** Encode the plateau and F over a shared 64-variable input,
  both payloads zero, plus clauses equating internal variables whose functions coincide
  on samples. The two equivalent circuits share **42,771** identical internal functions,
  so a value forced cheaply in the small circuit propagates into the large one. Variants
  with 6,000 and 30,000 links (~1.7M variables).

- **Hint-augmented SAT.** The full plateau with `payload = 0`, plus the sparse degree-2
  relations imposed at one cut (50 hints) and at two cuts (94 hints, 59 of 64 output
  bits) as redundant implied constraints.

- **Prefix-only SAT.** Encode only the prefix up to the degree-2 cut, impose the
  relations there, and enumerate the prefix's satisfying inputs, checking each against
  the full circuit. On the plateau, 59 constraints across two cuts pin the input to
  about 2^5 candidates over 178k of the 356k gates. On short-279k, the earlier midpoint
  gives a shorter prefix (112k gates) but only 40 constraints.

**Outcome:** no candidate or preimage from any variant. The prefix solve — the
best-shaped, smallest instance — produced zero satisfying inputs after 16+ hours on the
279k circuit and 24+ hours on the plateau; the linked and hint variants likewise
returned nothing.

## 3. Key finding: the degree-2 midpoint replica

The replication landmark the raw sandwich provides survives the pipeline **not
verbatim and not affinely, but up to degree 2**. At each cut we measured how many of the
64 output functions are GF(2) polynomials of degree at most 2 in the live wires, over
enough samples (36,864) that the ~33,000 degree-2 monomials stay below the sample count
(membership false-positive about 2^-4000, so the counts are not coincidental).

The count rises sharply toward the middle and peaks per circuit:

| circuit | peak degree-2 output bits | depth | profile at 0.40 / 0.45 / 0.48 / 0.50 |
|---|---|---|---|
| plateau (356k) | 50 of 64 | 0.45 | 13 / 50 / 30 / 44 |
| F (666k) | 36 of 64 | 0.48 | 3 / 20 / 36 / 19 |
| short-279k | 40 of 64 | 0.40 | 40 / 37 / 26 / 29 |
| remix (1.38M) | 20 of 64 | 0.48 | 2 / 12 / 20 / 12 |
| remix2 (1.11M) | 16 of 64 | 0.50 | 3 / 8 / 8 / 16 |

Three properties make this a genuine, usable structural weakness rather than an
artifact:

- **Sparse.** Despite the near-full-rank degree-2 span, the representations are compact:
  on the plateau at depth 0.45 the 50 relations use a median of 10 monomials (max 50),
  and four output bits (17, 31, 41, 57) equal a single interior wire verbatim. On the
  279k circuit the median is 6.

- **Nested and growing.** Classifying each output bit by its minimal replica on the
  plateau at depth 0.45: 4 are a bare wire (direct), 5 are affine, 50 are degree-2, and
  14 have none. The classes nest, so higher degree exposes genuinely new bits — the
  affine-to-quadratic jump is almost entirely new (45 of the 50 need a product).

- **Complementary across circuits.** The plateau and F expose largely different output
  bits (overlap 29 of 50 and 36). Two plateau cuts cover 59 of 64 bits; adding F's cut
  reaches **62 of 64**. So almost every output bit is a compact degree-2 function of
  interior wires at some midpoint cut.

An incidental observation: each step of compression *increased* the degree-2 exposure
(666k → 36, 356k → 50), consistent with ANF resynthesis rebuilding the computation into
low-degree windows. The fcompress-shortened 279k circuit peaks lower (40) and earlier
(depth 0.40), indicating the low-degree leak comes specifically from ANF resynthesis
rather than from compression in general.

## 4. Conclusions

1. **No preimage was found.** Across three equivalent circuit sizes (666k, 356k, 279k)
   and five sound SAT formulations (full, two-circuit linked, degree-2 hinted at one and
   two cuts, and half-circuit with the bundle constrained), no solver produced `x` with
   `C(x) = 0^64`. The best-shaped instance — the 279k circuit's 112k-gate prefix with its
   degree-2 bundle pinned — returned no satisfying assignment at all after 16 hours.

2. **The structural leak is real and nearly complete.** The sandwich's mid-circuit
   replication of `C` survives the full pipeline at degree 2: around the midpoint, almost
   every output bit (62 of 64 taken across two circuits) is a sparse degree-2 function of
   the interior wires. This is a concrete, quantified representational weakness that the
   verbatim and affine scans missed.

3. **The leak does not translate into a feasible inversion by these means.** The
   obstacle is not the structure but the solver: inverting a 100k-to-180k-gate reversible
   circuit is intractable for CryptoMiniSat here even when the degree-2 relations pin the
   input to a handful of candidates and remove the uncompute half. Shrinking the circuit
   from 666k to 279k and halving the prefix did not cross the tractability threshold.

4. **Interpretation.** The degree-2 midpoint replica is best read as a property of the
   *representation* — the resynthesis expresses `C`'s output as low-degree functions of
   its own intermediates — not as a shortcut to `x`. Exploiting it still requires solving
   for an input consistent with the prefix, which is the hard part. A different engine
   (algebraic / Gröbner over the sparse quadratic system, or a dedicated reversible-circuit
   inverter) would be the next thing to try; standard SAT is not enough.

## 5. Tools and artifacts

All tools are on the server under `~/threemask_cmp/py6/`; results under
`~/threemask_cmp/run/preimage/`.

- `sat_invert.py` — direct SAT inversion (baselines).
- `vocab.py`, `vocab_index.py`, `blind_walk2.py`, `blind_walk3.py`, `vocab_diag*.py` —
  vocabulary construction and structural walks with C-using diagnostics.
- `payload_cut.py`, `replication_scan.py` — verbatim replication scans.
- `affine_replica_scan.py` — affine-span scan.
- `degree2_replica_scan.py` — degree-2 span scan (the key measurement).
- `combo_extract.py` — explicit sparse degree-2 combinations with provenance.
- `classify_replicas.py` — per-bit direct/affine/quadratic classification.
- `combined_link_sat.py` — two-circuit linked SAT.
- `plateau_hint_sat.py`, `plateau_hint_sat2.py` — hint-augmented SAT (one and two cuts).
- `prefix_quad_sat.py` — prefix-only solve with degree-2 constraints and input
  enumeration.

The full chronological log is in `docs/notes/stage2_stage6_campaign_20261021.md`
(section "Preimage campaign on the baseline mixed output").
