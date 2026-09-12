# Structural complexity of the SAT attack on GSS-MIX

*2026-08-16. An analysis of the implemented SAT attack as an **algorithm** — what its
loop bounds are, how they depend on the wire count `n`, which of its reuse channels the
safeguards close, and whether the resulting cost is polynomial or (sub)exponential.
Companion to `experiments/gss_sat_scaling_20260810/FINAL_SAT_SCALING_ESTIMATE.md`, which
reports the empirical campaign. Where the two disagree, this document says so explicitly.*

Supporting measurements and scripts: `experiments/gss_sat_structure_20260816/`.

---

## 0. The three answers, up front

**(1) Why the safeguards raise the attack's cost, structurally.** The attack is a preimage
search on a reversible circuit. Without safeguards that problem is not a search at all: a
reversible circuit given in the clear, with its full output known, inverts in `O(|C|)` by
running the gate list backwards. Every safeguard exists to delete one of the conditions that
makes that sentence true. The load-bearing one is the **zero slice**: of the `4n` final wires
only `n` are pinned to the target, so backward simulation has `3n` unknown bits to guess and
stops being a deduction. What remains is a search over the `n` free input bits, and the rest
of the pipeline exists to stop a solver from finding structure that beats that search.

**(2) How the loop bounds change.** They change by **constant factors only**. The encoder's
loops run `Θ(G)` times where `G` is the final gate count, and `G = Θ(n log² n)` — the same
asymptotic class as the source computation `|C| = round(n·(log₂n)²)`. Every pipeline stage
multiplies `G` by an `n`-independent constant (gadgetisation ≈ 88×, phase A ≈ 1.55×, split
≈ 2.01×, crossing ≈ 1.83×, fcompress ≈ 0.93×). **No safeguard changes any exponent.** What
the safeguards actually destroy is *reuse*, and that is measurable: see §5.

**(3) Polynomial or exponential.** Four statements with four different scopes:

- **Proved:** the encoding is polynomial — `Θ(n log² n)` variables and clauses.
- **Proved:** the attack is bounded above by `2ⁿ · Θ(n log² n)`, because the `n` input
  variables form a unit-propagation backdoor.
- **Measured, not proved:** every structural shortcut that would beat `2ⁿ` yields only a
  constant-factor discount, and CDCL's reuse coefficient *decays* as roughly `n^-0.7`.
- **Formally unidentified:** whether CDCL's conflict count is polynomial or exponential in
  `n`. It cannot be settled from the current record, and no lower bound is available — see
  §6.4 for why the classical proof-complexity toolkit does not apply here.

**And one correction that outranks all of the above.** Below `n ≈ 48` the scaling question is
moot, because **plain exhaustive enumeration of the `n` input bits beats the SAT attack by
three to five orders of magnitude**, and this is now verified against the real artifacts (§7).
The archived TIMEOUTs at `n ≤ 40` are not evidence about the construction; they are evidence
that CDCL is a poor algorithm for this problem.

---

## 0.5 The exact problem

Write the `4n` wires as four `n`-wire blocks `(a, b, c, d)`. The deliverable `G` computes

```
G(a, b, c, d)  =  ( junk ,  b XOR C(a) ,  junk ,  junk )      ON THE ZERO SLICE b = c = d = 0
```

so `G(x, 0, 0, 0)` puts `C(x)` on wires `n..2n-1`. **The attacker's task:** given `y = C(x)` and
**only `G`**, find `x'` with `C(x') = y`. `C` is a random g57 circuit on `n` wires, hence a
bijection, so `x'` is unique.

Both attacks compared in this document operate under exactly that restriction. The CNF encoder
opens `FINAL.mpmct1` and nothing else. The enumerator evaluates `G` forward. Neither reads `C`,
the sandwich, or any intermediate artifact. The `O(|C|)` backward inversion in §3 is the
*pre-safeguard baseline* — what the zero slice exists to prevent — not an available attack.

**Two properties of that statement were verified rather than assumed**
(`experiments/gss_sat_structure_20260816/offslice_probe.txt`, 200 trials per case, `n=11`):

| perturbation | `block2 == b XOR C(a)` | `block2 == C(a)` | mean Hamming from `b XOR C(a)` |
|---|---:|---:|---:|
| `b ≠ 0`, `c = d = 0` | 0/200 | 0/200 | 5.67 of 11 |
| `c ≠ 0`, `b = d = 0` | 0/200 | 0/200 | 5.38 of 11 |
| `d ≠ 0`, `b = c = 0` | 0/200 | 0/200 | 5.58 of 11 |

- **The `b XOR C(a)` form holds only on the zero slice.** Off-slice it never holds, and the
  deviation is a full `n/2` avalanche — S1 fires and a disturbed `C'` runs, exactly as
  `docs/design/SLICED_SANDWICH.md` specifies. There is no clean XOR channel in `b` to run differentials
  through.
- **All `3n` pins are live, not just `b`.** Setting `c` or `d` alone already destroys the answer.
  The attacker cannot relax any block.

A per-wire influence census over all `3n` junk wires
(`junk_influence_census.txt`; 2,048 logical inputs per wire at `n=11`, 256 at `n=128`) confirms
this quantitatively: a single junk-wire flip changes on average **63.2 of 128** answer bits at
`n=128` — full avalanche — and leaves the answer intact for only 0.479% of inputs. The
exceptions are a handful of wires: at `n=128`, one junk wire (258) showed no influence across
256 samples and three more (329, 449, 480) were unchanged for 9–51% of inputs. That is a real
anomaly worth noting, but it is small (4 of 384 wires), it is a bound of the form "no influence
observed in 256 samples" rather than a proof of inertness, and it only shrinks the *backward*
guess space — which is already dominated by the forward attack.

---

## 0.6 The loop ledger — the literal before/after

The question is how loop-iteration counts that depend on the wire count change when the
safeguards are introduced. The answer is unusually clean:

| loop | iterations | before safeguards | after safeguards |
|---|---|---|---|
| encoder validation pass | `G` | `Θ(n log²n)` over `\|C\|` | same class, ×1093 constant |
| encoder emit pass | `G`, emitting `2k+2` clauses each | — | same class, ×1093 constant |
| one full unit-propagation sweep | `O(G + K)` | `Θ(n log²n)` | same class, ×1093 constant |
| conflict analysis per conflict | ~propagations/conflict | — | `Θ(1)`: measured 661–1,207, flat in `n` |
| inprocessing passes | budget-driven | — | constant fraction recovered, decaying |
| **the search loop** | **number of conflicts** | **does not exist** | **≤ 2ⁿ; empirically 0 → >2.9M at n=11** |

> **The safeguards do not lengthen any loop of the attack beyond a constant factor. They create a
> loop that did not previously exist.**

Inverting `C` in the clear is a single backward pass, `for g = |C| down to 1` — zero search. The
zero slice deletes that pass, and what replaces it is a search whose only complete formulation is
over the `n` free input bits. Every other loop in the attack stayed in `Θ(n log²n)`.

The ladder in §8 shows all three points of this story directly, as solver behaviour:

| what the attacker holds | search-loop iterations observed |
|---|---|
| `C` in the clear | **0 conflicts at every width from n=8 to n=256**, 5/5 replicates — measured, §0.7 |
| the sandwich alone | **0 conflicts on 6/6 circuits** — a search space exists but the solver never needed a conflict |
| the full deliverable | 4/6 circuits exceeded **2.35M conflicts** without terminating |

### 0.6.1 The source baseline, measured

Rather than argue that inverting `C` in the clear is not a search, it was run. Fresh random g57
circuits with the library's own `|C| = round(n·(log₂n)²)`, all `n` outputs pinned to the target,
same per-gate Tseitin shape as the campaign encoder, same pinned Kissat 4.0.4, 5 replicates per
width (`experiments/gss_sat_structure_20260816/source_baseline.txt`):

| n | `\|C\|` | vars | clauses | conflicts | decisions | propagations | wall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 72 | 80 | 440 | **0** | **0** | 80 | 0.004 s |
| 32 | 800 | 832 | 4,832 | **0** | **0** | 832 | 0.007 s |
| 64 | 2,304 | 2,368 | 13,888 | **0** | **0** | 2,368 | 0.008 s |
| 128 | 6,272 | 6,400 | 37,760 | **0** | **0** | 6,400 | 0.013–0.021 s |
| 256 | 16,384 | 16,640 | 98,560 | **0** | **0** | 16,640 | 0.031–0.042 s |

**Zero conflicts and zero decisions at every width, 55/55 runs, with propagations exactly equal to
the variable count** — one propagation per variable, no search whatsoever. This is the same
signature as `gss_n010_r01`, and it is the mechanism: with all `n` outputs pinned, the last gate's
controls are unwritten wires whose final values are pinned, so the AND is determined, the pre-gate
value follows, and backward induction fixes the entire circuit in one sweep.

### 0.6.2 What the zero slice is *not* doing

A natural guess is that the safeguard works by hiding output bits. It does not. Pinning only `k`
of the `n` outputs of `C` and hiding the rest still gives **0 conflicts** at every `k` tested, down
to hiding half the outputs (`n=32`, hidden ∈ {0…16}; `n=64`, hidden ∈ {0…16}; 3 replicates each):

> Removing output constraints *enlarges* the solution set to `2^(n-k)` preimages, so the instance
> becomes easier to satisfy, not harder.

The zero slice achieves something that hiding bits cannot: it keeps the witness **unique** while
still denying propagation a path. It does that by adding `3n` junk wires whose *inputs* are pinned
to zero and whose *outputs* are free. The pin count stays at `4n` and the witness stays unique
(`C` is a bijection), but the pins are now positioned so that **neither direction chains**: forward
propagation from the zero-pinned inputs dies after 0.42–0.54% of the circuit (§3), and backward
propagation from the target stalls at the first gate whose controls include an unpinned junk wire.

That — not "less information" — is the structural reason the search loop exists.

---

## 0.7 Worked instantiation: n = 128, m = 6000

The library default at `n=128` is `|C| = round(128·(log₂128)²) = 6272`, so **m = 6000 is the
shipped configuration to within 4.3%**, and every measurement on the three public `n=128`
deliverables transfers directly. Sizes below use the measured `G/|A| = 516.6` from those three
deliverables and `|A| = 2m + 2s + n` with `s = max(n, round(n log₂n)) = 896`.

| quantity | source `C` (m=6000) | after GSS-MIX | ratio |
|---|---:|---:|---:|
| gates | 6,000 | ≈ 7,191,000 | ×1,199 |
| wires | 128 | 512 | ×4 |
| CNF variables | **6,128** | ≈ 7,191,600 | ×1,173 |
| CNF clauses | **36,128** | ≈ 56,160,000 | ×1,554 |
| one unit-propagation sweep | 6,128 propagations | ≈ 7.19M propagations | ×1,173 |
| **search-loop iterations** | **0 conflicts** | **≥ 216,311,044, still unsolved** | **0 → ∞** |

**The source row is measured, 5/5 replicates** (`n128_source_msweep.txt`): 6,128 variables,
36,128 clauses, **0 conflicts, 0 decisions, exactly 6,128 propagations, 0.013–0.014 s**. The
baseline is also **m-independent** — 40/40 runs at `m ∈ {1000, 2000, 4000, 6000, 6272, 8000,
12000, 20000}` all give 0 conflicts with propagations equal to the variable count. No quantity of
source gates creates a search when the whole output vector is known.

**The mixed row is bounded, not measured.** No `n=128` instance has ever been solved. The three
real ones:

| circuit | wall | conflicts | peak RSS |
|---|---:|---:|---:|
| legacy public C1 | ≥85,542.37 s (23.76 h) | ≥216,311,044 | 7,775,140 KiB |
| legacy public C2 | ≥85,499.96 s (23.75 h) | ≥220,980,111 | 7,913,236 KiB |
| current-policy n=128 | ≥90,724 s (25.20 h) | — (host deleted) | — |

That 7.4–7.5 GB peak is why this configuration cannot be re-run on a 7 GB workstation; the
constraint is memory, not patience.

**Attack ranking at n=128 / m=6000:**

| attack | cost | status |
|---|---:|---|
| Kissat / CDCL | **unknown** — ≥23.75 h and ≥216M conflicts on three circuits | no solve ever observed |
| exhaustive enumeration | ≈ 2^144.8 operations | measured throughput, projected cost |
| **seed search** | **≈ 2^79 operations** (18,760 candidates/s/core) | **the binding attack** |
| semantic structural probe | ~10 s, removes 7.14% of the instance | does not break it; see §9 |

---

## 1. The attack as an algorithm

### 1.1 What is implemented

`experiments/gss_sat_scaling_20260810/mpmct1_zero_slice_to_cnf.cpp` Tseitin-encodes the final
circuit; `run_job.py` runs pinned Kissat 4.0.4 under an external wall cap.

```
INPUT: FINAL.mpmct1  (W = 4n wires, G gates, K = total control count), n, TARGET (n bits)

ENCODE
  state[w] <- w+1                for w in 0..4n-1              # :180-181,  4n iterations
  emit unit  -(w+1)             for w in n..4n-1               # :182,      3n zero pins
  for each gate g (target t, complement c, controls (w_i,p_i), width k):   # G iterations
      old <- state[t] ;  res <- fresh variable                 # :187-188
      l_i <- (p_i ? state[w_i] : -state[w_i])                  # :192-195
      for i in 1..k:  emit ( l_i, -old,  res)                  # :201    2k ternary clauses
                      emit ( l_i,  old, -res)                  # :202
      emit (-l_1..-l_k,  old,  res)                            # :204    2 clauses of width k+2
      emit (-l_1..-l_k, -old, -res)                            # :206
      state[t] <- res                                          # :209
  emit unit  ±state[n+b]        for b in 0..n-1                # :216-219, n target pins

SOLVE   kissat --seed=S --time=W --sat --verbose=1 --statistics=1   (sequential, one core)
DECODE  read the initial wire variables, re-simulate, compare wires n..2n-1 to TARGET
```

The per-gate group encodes `res = old XOR AND(l_1..l_k) XOR c` **bi-implicationally**. Unit
propagation therefore crosses a gate with equal ease in either direction. Any directional
asymmetry the attacker faces is a property of the **boundary conditions**, not of the encoding.

### 1.2 Exact instance size

| quantity | closed form | source |
|---|---|---|
| variables `V` | `4n + G` | `:161` |
| clauses `C` | `2K + 2G + 4n` | `:163` |
| literal occurrences | `8K + 4G + 4n` | `:189-207` |
| unit clauses | `4n` (`3n` zero pins + `n` target pins) | `:182`, `:216-219` |

Verified on all 29 archived rows of `aggregate/current/results.tsv`: `cnf_variables − 4n − final_gates = 0`
for 29/29, and `K = (C − 2G − 4n)/2` is integral for 29/29. Independently re-derived here by
directly counting controls in the public `n=128` deliverables — e.g. `public_c3`: `G = 7,475,700`,
`K = 21,718,654`, giving `V = 7,476,212` and `C = 58,389,220`.

A locally rebuilt encoder emits **byte-identical CNF** (same SHA-256) to the campaign's pinned
prebuilt binary on a real artifact, so every measurement below is against the real attack
(`experiments/gss_sat_structure_20260816/encoder_equivalence.txt`).

### 1.3 The size class

`|C| = |D| = round(n·(log₂n)²)` is the library convention (`scripts/gss_mix.sh:108-116`), and each
pipeline stage multiplies gate count by an `n`-independent constant. Measured `G / (n log₂²n)`
across the archive:

| n | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| `G` | 90,108 | 305,081 | 925,771 | 2,591,259 | 6,958,181 | 17,912,700 |
| `G / (n log₂²n)` | 1251.5 | 1191.7 | 1157.2 | 1124.5 | 1109.4 | 1093.3 |

A slowly declining ratio: `G = Θ(n log^k n)` with `k` slightly below 2. Fitting a pure power law
over this range gives the frequently quoted `n^1.53` — that exponent is a polylog seen through a
32× window in `n`, not a genuine power. Either way:

> **`V` and `C` are `Θ(n log² n)`. The safeguards do not work by inflating the representation.**

---

## 2. Where the `n`-dependence becomes a search

### 2.1 The `n` input bits are a unit-propagation backdoor (proved)

**Claim.** The set `S = {variables 1..n}` (the initial values of wires `0..n-1`) has the property
that for *every* assignment to `S`, unit propagation decides the whole formula.

**Proof.** `state[]` is initialised to the identity map (`:180-181`), so wire `w`'s initial value is
variable `w+1`. Line `:182` emits `3n` negative units fixing variables `n+1..4n`, leaving exactly the
`n` variables of `S` free. Each gate reads `state[target]` and `state[controls[i]]` — both strictly
earlier variables — and writes a fresh variable (`:188`, `:209`). Induct over gates in file order:
at each gate `old` and every `l_i` are already assigned, and the group is a complete bi-implicational
definition, so `res` is forced. One topological sweep fixes all `G` auxiliaries. Total fixed:
`4n + G` — everything. ∎

**Demonstrated, not merely argued.** Run `gss_n010_r01` reports:

```
c conflicts:      0        c decisions:    0        c searches: 0
c propagations:   137065   c variables_original: 137065
```

Zero decisions, zero conflicts, and **exactly one propagation per variable**. The solver's default
phase happened to coincide with the challenge input; propagation then determined all 137,065
variables in a single sweep with no search. That is the backdoor lemma observed directly.

*Not proved: minimality.* No minimal-backdoor extraction has ever been run on these instances. The
correct statement is `|backdoor| ≤ n`, which is all the upper bound needs. A backdoor of size
`b < n` would give a `2^b · poly` attack and has not been ruled out.

### 2.2 The boundary asymmetry

| | constrained | free |
|---|---|---|
| initial wires | **3n** (pinned to 0, `:182`) | **n** (the logical input) |
| final wires | **n** (pinned to target, `:216-219`) | **3n** (junk, explicitly not asserted) |

Forward the search space is `2ⁿ`; naive backward simulation must guess `3n` bits. There is no
meet-in-the-middle, because the meeting point would need the `3n` junk bits. This asymmetry is
created by the `N` copy step and the `2n` frozen band, and it is the one exactly-exponential
effect any safeguard has.

*Honest refinement:* a constraint **solver** attacking backwards still bottoms out at the same
`n`-variable backdoor. The correct claim is that backward simulation loses its `O(|C|)` determinism
and is never better than forward — not that a backward attack costs `2^{3n}`.

### 2.3 The ceiling

Combining §2.1 and §1.3: **the attack costs at most `2ⁿ · Θ(n log² n)`**. This exponent is
identical before and after every safeguard. Everything the safeguards do lives in the constant,
and in whether a solver can get *below* the ceiling.

---

## 3. The pre-safeguard baseline

**The plain fact.** A reversible circuit given in the clear, with its full output vector known,
inverts in `O(|C|)`: every MPMCT gate is its own inverse given its controls. For the source
computation that is `O(n log₂²n)` gate evaluations — quasi-linear, no search.

The structural difference the safeguards make is directly measurable at both boundaries of the
instance. Two quantities, both computed here on real artifacts:

- **Constant frontier** — how far forward 3-valued propagation from the `3n` pinned zeros survives
  before every wire is input-dependent, and how many gate variables it determines for free. (This
  is exactly what unit propagation gets before any search.)
- **Backward cone** — how many gates are in the cone of influence of the `n` asserted output bits.
  Gates outside it are irrelevant and deletable at zero cost.

Six independent seeds at `n=11`, one artifact per pipeline stage (means):

| stage | gates | controls/gate | affine % | gate vars determined | constant frontier | in backward cone |
|---|---:|---:|---:|---:|---:|---:|
| sandwich only (2n wires) | 351 | 1.90 | 10.5% | 0 | **57.6% of circuit** | **51.6%** |
| + gadgetisation (4n) | 30,808 | 1.56 | 46.3% | 127 | 1.56% | 99.05% |
| + phase A | 78,565 | 2.80 | 21.9% | 44 | 1.06% | 99.98% |
| + split | 87,753 | 2.61 | 29.9% | 74 | 1.47% | 99.97% |
| + crossing + fcompress | 145,514 | 3.74 | 11.6% | 100 | 1.45% | 99.96% |

And the three public `n=128` deliverables, independently generated:

| | c1 | c2 | c3 |
|---|---:|---:|---:|
| gates | 7,468,981 | 7,469,889 | 7,475,700 |
| controls/gate | 2.9049 | 2.9047 | 2.9052 |
| affine (NOT/CNOT) gates | 16.344% | 16.328% | 16.349% |
| **nonlinear gates** | **83.656%** | **83.672%** | **83.651%** |
| gate vars determined by propagation | 673 | 620 | 680 |
| …as a fraction of the instance | 0.0090% | 0.0083% | 0.0091% |
| constant frontier dies at | 0.42% | 0.50% | **0.54% of circuit** |
| gates in the backward cone | — | — | **99.98%** |

Read together: **the sandwich alone hands the attacker half the circuit for free** — 57.6% of it
lies inside the constant frontier and 48.4% of it is outside the target's cone and can be deleted.
Gadgetisation is the stage that collapses both to ~1%, and at `n=128` the instance has an
irreducible core of 99.98% of `Θ(n log² n)` gate variables with essentially zero free deduction
from either boundary: only 1,402 of 7,475,700 gates are deletable, and propagation gifts 680 of
7,475,700 gate variables.

---

## 4. Safeguard-by-safeguard: which loop, which reuse channel

| safeguard | attack loop / reuse channel attacked | before | after | mechanism |
|---|---|---|---|---|
| random `D` block, `\|D\|=\|C\|` | backward simulation | `O(\|A\|)` returns `C⁻¹` | returns `D⁻¹(p)` — same size, same distribution | the wrong answer, undetectable by inspection |
| `N` copy step + `2n` band | output boundary width | `4n` output bits known | **n** pinned, **3n** free | destroys backward determinism; forces `2ⁿ` forward |
| slice blocks S1,S2 (`s = n log₂n` each) | aux-assignment freedom | any of `2ⁿ` aux settings computes `C` | exactly one slice does | encoded as the `3n` unit clauses |
| random interleave of S1/S2 | block identification | S1/S2 contiguous, deletable | gate-by-gate classification, vocabulary does not separate | no separable prefix/suffix |
| N-column float | **seam cut** (MITM / 2-way split) | seam at one known offset, `O(1)` cut | span 15 → ~76 positions at n=16 | dissolves the `C\|N\|D` seam |
| product-share masks `[2,2,2,3]` | algebraic degree | degree-1 fit over `4n+1` regressors recovers the state | needs degree ≥3: `1+4n+C(4n,2)+C(4n,3)` = 22.4M regressors at n=128 | a **polynomial** step (cubic), not a wall |
| **nonlinear band fill** | global GF(2) invariants → cross-circuit propagation | `2n` free affine input→output relations; injecting them turned n=32 from a >2 h timeout into **10 min** | no such invariants exist | the single largest measured per-safeguard SAT effect in the corpus |
| band roll + retire-refill epochs | census/fingerprint identification of the band | one write-count threshold separates the band perfectly; lifetime census recovers 254/256 band wires | separable → not; 0 fingerprints at ≥0.50 coverage | cost +12.4% and +7.9% gates |
| **phase A** DB re-encoding | **template reuse** (BVA, subsumption) | `O(G)` pattern-match against a 7-realisation public library | BVA yield `factored/variables` **20.99% → 1.30%** across n=16→256 | each window independently re-spelled from a 320 GB store |
| **split stage** | local explicability / probing | every gate is a g57; local windows are locally explicable | a polarity obligation opened at position `p` is discharged **`Θ(\|G\|)` gates away** (mean 0.57 of the whole circuit) | zero synthetic gates; pure non-locality |
| **crossing walk** | provenance locality / cached partials | 1 image per source gate in 1 place | median **3** descendants at median span **413** gates | width damper `3^-(w-1)` pushes spread to the median, not the tail |
| **fcompress** | the attacker's free preprocessing | 17% strippable from raw split output | residual **92.8–93.5%** over 28 circuits — at most ~7% strippable | deterministic and attacker-computable, so running it cannot weaken hiding |

### 4.1 What this table does and does not establish

It is a **size ledger plus a mechanism narrative**, not a cost ledger. Two limits must be stated:

1. **The campaign never built the "before" arm.** `run_job.py` encodes only `paths["final"]`. No CNF
   was built and no solver invoked for any pre-final stage at any `n` in the campaign. §8 reports a
   matched-`n` ladder built here to start closing that gap.
2. **Safeguard effect and size effect are confounded.** Every stage changes `G`, and the CNF is
   exactly affine in `G`. No experiment anywhere varies a safeguard while holding gate count fixed.

The one matched-instance dose series in the corpus (the 2026-07-13 point-function campaign, a
different pipeline) points *against* a per-gate hardening effect: `log(conflicts) ~ log(gates)` has
slope **0.92**, i.e. sublinear in instance size, and **188 of 188** completed sidecars found the
planted point at every mixing dose probed.

---

## 5. Reuse: the channel the safeguards actually close

CDCL's only advantage over enumeration is that a clause learned in one region prunes others. All
figures below are read directly from the archived verbose Kissat logs.

### 5.1 Per-conflict cost is flat; the conflict count is what grows

| n | 8 | 16 | 32 | 64 | 96 | 256 |
|---|---:|---:|---:|---:|---:|---:|
| propagations / conflict | 5,438 | 1,299 | 661 | 799 | 792 | 1,207 |
| decisions / conflict | 4.31 | 4.87 | 9.00 | 14.61 | 25.24 | **109.97** |

Propagations per conflict are statistically flat in `n`. One conflict at `n=256` touches ~1,200
assignments in a formula with 17.9M variables — **0.007% of it**. The clause it produces is a
statement about a vanishing fraction of the instance. All cost growth lives in the conflict count.

*Caveat:* the decisions/conflict trend rests on one circuit per width above `n=12`, and `n=256`
supplies most of the x-variance. "Roughly linear in `n`" is defensible; a precise exponent is not.

### 5.2 Learned-clause reuse decays

| n | 16 | 32 | 64 | 256 |
|---|---:|---:|---:|---:|
| clauses learned | 6,938,339 | 92,588,637 | 327,603,261 | 208,509,452 |
| clauses used | 220,899,757 | 651,534,568 | 1,615,401,402 | 738,253,015 |
| **uses per learned clause** | **31.8** | **7.04** | **4.93** | **3.54** |

Monotone decay of roughly `n^-0.7`. And this is not a run-length artifact: the `n=256` run had
**24× more wall time** than the `n=12` runs and achieved ~30× *less* reuse per clause.

### 5.3 The mechanism is glue, and Kissat's own tier boundary shows it

Kissat recomputes tier-1 as the glue level at which cumulative clause usage crosses ~50%. Glue
counts how many distinct decision levels a learned clause ties together; low glue means a local,
broadly-valid lemma.

| n | tier-1 glue | share of uses at tier 1 | bulk bucket |
|---|---:|---:|---|
| 11 | 1 | 97.64% | — |
| 16 | 1 | 91.18% | — |
| 32 | 1 | 53.73% | glue 3–5 |
| 64 | **4** | 53.08% | glue 6–9 |
| 96 | **4** | 54.82% | glue 6–9 |
| 256 | **4** | 53.84% | **glue 6–14** |

At `n ≤ 16`, a tiny core of **glue-1** clauses — genuinely local, globally valid circuit facts —
carries 91–98% of all clause usage. By `n ≥ 64` that core is gone and you need glue up to 11–16 to
cover 90% of usage. Individual clauses are not longer; **there is simply no longer a small set of
transferable local lemmas.** High-glue clauses are both rarer to become relevant and first in line
for database reduction — which is how "less reusable" becomes "deleted before reuse".

This is the precise, mechanical answer to *"how do the safeguards reduce the amount of computation
that can be reused across different parts of the attack?"* The split and crossing stages spread each
source gate's influence over a median span of 413 gates to ≥3 descendants; a conflict in one window
therefore cannot be explained by a small set of variables in that window, so the 1UIP clause spans
many decision levels.

### 5.4 Structure-exploiting inprocessing recovers a flat-to-decaying fraction

Fractions of `variables_original`, all techniques at their defaults (all ON):

| n | units | substituted | sweep equivalences | factored (BVA) | congruent |
|---|---:|---:|---:|---:|---:|
| 16 | 0.90% | 5.14% | 2.71% | **20.99%** | 1.50% |
| 32 | 0.70% | 4.71% | 2.19% | 8.66% | 1.50% |
| 64 | 0.57% | 4.55% | 2.40% | 3.27% | 1.31% |
| 96 | 0.50% | 4.48% | 2.24% | 2.48% | 1.33% |
| 256 | 0.42% | 3.74% | 2.11% | **1.30%** | 1.26% |

**Nothing grows.** BVA — the technique that finds repeated structure — decays 16× across a 32× span
in `n`, meaning the *absolute* number of factorable repeated patterns is roughly constant while the
formula grows 59×. Post-preprocessing variable survival is **flat at 32–34% for every `n ≥ 16`**
(`variables_remaining_percent` in `results.tsv`): preprocessing removes a constant fraction and
never moves the instance toward the `n`-variable logical core.

### 5.5 Two honesty caveats on this evidence

**(a) The detector has near-zero coverage.** At `n=256`, `--sweepmaxvars = 8,192` is **0.046%** of a
17.9M-variable formula, and 2 of 384 sweep attempts completed. Through that keyhole Kissat still
returned ~1.58M structural facts (8.8% of the formula). The supportable inference is *"a
budget-starved detector found 1.58M facts"*, **not** *"no more structure exists"*.

**(b) The formula is 59% XOR and no XOR-aware solver was ever run.** Kissat's own gate extractor
reports, at `n=256`: `congruent_gates_xors = 62,004,824` (59%), `ites = 43,187,671` (41%),
`ands = 33,559` (0.03%), with 98% of matches being XOR matches. Kissat 4.0.4 has no
Gaussian-elimination-over-GF(2) engine. Every archived cost is an upper bound on the best-known
attack by an unmeasured margin.

---

## 6. Asymptotic verdict

### 6.1 Proved: the encoding is polynomial

`V = 4n + G`, `C = 2K + 2G + 4n`, exact from source and verified with zero error on all 29 rows;
`G = Θ(n log² n)`. Every safeguard contributes an `n`-independent constant. **No safeguard changes
the asymptotic order of the representation.**

### 6.2 Proved: the attack is bounded above by `2ⁿ · Θ(n log² n)`

By §2.1. The exponent is identical before and after every safeguard.

### 6.3 Structurally argued, not proved

- Per-conflict solver cost is `Θ(1)` in `n`, so all CDCL cost growth is in the conflict count.
- Reuse decays `≈ n^-0.7`; the glue-1 core that carries 91–98% of usage at `n ≤ 16` is gone by
  `n ≥ 64`. CDCL is measurably degenerating toward plain DPLL over the `n`-support.
- Every structural preprocessing route measured yields a constant-factor discount that does not
  decay with `n`. The honest form is **"measured, under one solver at default flags, one circuit per
  width, with no lower bound on what any preprocessor could achieve."**
- Two open proof gaps: no treewidth lower bound, and no backdoor-minimality result.

### 6.4 Formally unidentified — and why no lower bound is available

The six prespecified censored-AFT fits differ by `ΔAIC ≤ 0.244` within family: the data cannot
distinguish a power law from exponential growth. More fundamentally:

> **The formula is satisfiable with a unique witness.** This removes the entire classical toolkit.

- Resolution size/width lower bounds, Ben-Sasson–Wigderson, and the exponential lower bounds for
  **Tseitin formulas over expanders** are all statements about *refuting unsatisfiable* formulas. A
  satisfiable formula has no refutation.
- The Tseitin resemblance is a trap. This CNF *is* a Tseitin-style encoding of an XOR system over a
  circuit graph whose safeguards deliberately spread dependence — but the Tseitin lower bound
  requires **odd total charge**, i.e. an inconsistent system. Here the charge is consistent by
  construction, because the system is the evaluation of an actual reversible circuit on an actual
  input. Expansion is what makes the *odd*-charge instance hard and says nothing about the
  even-charge one, which in the pure-XOR case falls to Gaussian elimination in polynomial time.
  **The structural property the safeguards buy lands on the wrong side of the theorem.**
- The standard bridge is real but one-directional: `F ∧ (blocking clause on the witness)` is
  unsatisfiable and its refutation complexity is meaningful, so any run that *certifies exhaustion*
  must pay it. But a run that merely *finds* the witness owes no refutation. This dataset contains
  the counterexample: `gss_n010_r01`, 0.265 s, 0 conflicts. **No per-instance lower bound holds for
  this family.**
- `P` vs `NP` bars an unconditional lower bound on the *problem* (inversion self-reduces to `n` SAT
  queries), but not algorithm-specific lower bounds. That barrier applies to every concrete one-way
  function candidate and carries no information about this construction's quality.
- **Zero UNSAT results exist anywhere in either campaign.** Every non-solve is a wall-clock timeout,
  an administrative censor, or a solver resource failure. Nothing in the repository establishes that
  any instance is hard.

### 6.5 The verdict sentence

> Polynomial-size encoding (**proved**, `Θ(n log² n)`); `Θ(1)` per-conflict solver cost
> (**measured**, flat across a 32× span in `n`); over a `2ⁿ` search space that preprocessing is
> **measured — not proved —** unable to shrink, with the reuse coefficient decaying as `≈ n^-0.7`.
> The `2ⁿ` exponent is identical before and after every safeguard; the safeguards buy roughly
> **10 bits of constant**. Whether CDCL's conflict count is polynomial or exponential in `n` is
> formally unidentified — but below `n ≈ 48` the question is moot, for the reason in §7.

---

## 7. What the solver costs on `G`, and why it is not the binding attack

### 7.0 The measured SAT times

Per mixed circuit, from `aggregate/current/results.tsv`. Hosts are normalised within 6%
(speed indices 0.9987–1.0624).

| n | `G` (gates) | solved | actual solve time | conflicts | enumeration on the same `G` |
|---:|---:|---:|---|---|---|
| 8 | 90,108–95,211 | 3/3 | 59.8 / 96.6 / 314.7 s | 11.6k / 88.2k / 542k | 0.0044 s |
| 9 | 111,680–113,684 | 2/3 | 626.6 / 627.5 s | 948k / 951k | 0.011 s |
| 10 | 135,662–141,416 | 1/3 | 0.265 s *(0 conflicts, degenerate)* | 0 | 0.026 s |
| 11 | 157,472–167,382 | 3/6 | 4,311.7 / 4,568.5 / 5,255.8 s | 4.02M / 4.17M / 5.26M | 0.061 s |
| 12 | 184,771–192,888 | 1/4 | 4,517.1 s | 4.70M | 0.14 s |
| 16 | 305,081 | 0/1 | ≥3,600 s | ≥7.97M | 3.6 s |
| 20 | 446,349 | 0/1 | ≥7,200 s | ≥24.7M | 85 s |
| 24 | 596,778 | 0/1 | ≥10,800 s | ≥47.8M | 30 min |
| 32 | 925,771 | 0/1 | ≥21,600 s | ≥109.4M | 8.4 core-days · 19 min on 640 threads |
| 40 | 1,322,067 | 0/1 | ≥28,800 s | ≥162.0M | 8.4 core-years · 4.8 fleet-days |
| 48 | 1,697,222 | 0/1 | ≥43,200 s | ≥220.6M | 2,750 core-years · 4.3 fleet-years |
| 64 | 2,591,259 | 0/1 | ≥86,400 s | ≥380.2M | out of reach |
| 96 | 4,667,022 | 0/1 | ≥86,400 s | ≥329.0M | out of reach |
| 128 | 6,958,181 | 0/1 | ≥90,724 s *(admin censor)* | — | out of reach |
| 256 | 17,912,700 | 0/1 | ≥172,801 s | ≥257.5M | out of reach |

**A SAT solve time exists only for `G` up to about 193k gates (`n ≤ 12`).** Above that all 29
circuits are right-censors: floors at the cap, not measurements.

Two readings matter:

- **`G` sets the clock rate; `n` sets the number of ticks.** Conflict throughput is nearly flat —
  2,213/s at `n=16`, 5,064/s at `n=32`, 4,401/s at `n=64`, 1,490/s at `n=256`: a factor of 3.4
  across a 59× span in instance size. Wall time ≈ conflicts ÷ ~2,000/s, and circuit size barely
  moves it. The sharpest illustration: `n=10` at `G = 137,025` solved in **0.265 s**, while `n=11`
  at `G = 157,472` censored past 1,200 s — 15% more gates, four-plus orders of difference.
- **"The time given `G`" is a distribution, not a number.** At `n=11` three circuits solved at
  4,312–5,256 s while three siblings censored at 1,200–1,800 s. At `n=12` one solved at 4,517 s
  while another burned 9.24M conflicts in 7,200 s without solving. The three `n=8` solve times
  span 5.3×.

### 7.1 Why it is not the binding attack

The zero-slice preimage problem has exactly `2ⁿ` candidate inputs. The structure-blind attack is to
evaluate all of them — 64 at a time in machine words, one pass over the gate list per 64 candidates.

**Verified here against the real artifacts** (`experiments/gss_sat_structure_20260816/scripts/enumerate.cpp`,
single laptop core, `-O3 -march=native`, no AVX-512, no tuning):

- The **real `n=11` deliverable** — 149,477 gates — was solved by exhaustive enumeration in
  **0.0475 seconds**. The archived `n=11` circuits of comparable size took Kissat
  **4,311.7 / 4,568.5 / 5,255.8 seconds** and **4.02M / 4.17M / 5.26M conflicts** on server hardware.
  Correctness confirmed: the search halted at exactly the batch containing the known challenge input.
- Sustained throughput on the **real `n=128` public deliverable** (7,475,700 gates, 21.7M controls):
  **8.97 × 10⁷ word-gate-operations/second** = 5.74 × 10⁹ candidate-gate-evaluations/second.

Projecting with the conservative measured rate of `8.6 × 10⁷` word-gate-ops/s and the archived gate
counts (`2^(n−6) · G` word-gate-ops):

| n | G (archived) | one core | 640-thread fleet | archived Kissat outcome |
|---:|---:|---:|---:|---|
| 16 | 305,081 | **3.6 s** | — | TIMEOUT, ≥3,600 s, ≥7,966,864 conflicts |
| 20 | 446,349 | **85 s** | — | TIMEOUT, ≥7,200 s, ≥24,725,750 conflicts |
| 24 | 596,778 | **30 min** | 2.8 s | TIMEOUT, ≥10,800 s, ≥47,840,496 conflicts |
| 32 | 925,771 | 8.4 core-days | **19 min** | TIMEOUT, ≥21,600 s, ≥109,360,030 conflicts |
| 40 | 1,322,067 | 8.4 core-years | **4.8 days** | TIMEOUT, ≥28,800 s, ≥161,985,852 conflicts |
| 48 | 1,697,222 | 2,750 core-years | 4.3 years | TIMEOUT, ≥43,200 s, ≥220,577,239 conflicts |
| 64 | 2,591,259 | 2.8 × 10⁸ core-years | out of reach | TIMEOUT, ≥86,400 s, ≥380,182,485 conflicts |

*(Fleet = the 640-thread configuration recorded in the 2026-07-13 campaign inventory. Two unused
tuning factors remain: AVX-512 lanes and tighter gate packing.)*

**Consequences.**

1. Three archived TIMEOUTs (`n=16, 20, 24`) are instances one core finishes in seconds to half an
   hour — inside their own wall caps by factors of ~1,000×, ~85× and ~6×. Twelve of the eighteen
   right-censored observations feeding the censored-AFT fits are at `n ≤ 24`.
2. **Below `n ≈ 48` the scaling question is not open.** The construction at those widths is broken
   by an algorithm this repository already contains the pieces for.
3. The censored fits are therefore describing the wrong quantity at the small end. They measure
   Kissat's difficulty, which is bounded above by an algorithm that is 3–5 orders of magnitude faster.
4. The genuine security claim begins around `n ≥ 64` and rests on `2ⁿ` forward evaluation, not on
   anything Kissat did.

This does not contradict the campaign's own careful framing — it never claimed the censors were
population medians or bounds. It does mean the censors at `n ≤ 40` should not be cited as evidence
of hardness at all.

---

## 8. The matched-`n` before/after ladder

The campaign's largest structural gap is that no CNF was ever built for a pre-final stage. A ladder
was built here to start closing it: six independent `n=11` circuits, one CNF per pipeline stage, the
**same** target on every rung (every stage is semantics-preserving), the same pinned Kissat 4.0.4,
a common 2,700 s cap.

*Scope limit: these local runs used an empty replacement store, so phase A performed growth but **no
DB re-encoding**. The ladder therefore isolates the structural safeguards, not phase A's main
mechanism.*

| rung | stage | mean gates | solved | censored | median conflicts | conflicts per circuit (`≥` = censored) |
|---|---|---:|---:|---:|---:|---|
| R0 | sandwich only (2n wires) | 351 | **6/6** | 0 | **0** | 0, 0, 0, 0, 0, 0 |
| R1 | + gadgetisation (4n) | 30,807 | 6/6 | 0 | 13,239 | 5,874 · 8,654 · 10,609 · 15,869 · 18,655 · 23,326 |
| R2 | + phase A (growth only) | 78,565 | 6/6 | 0 | 192,527 | 10,712 · 11,564 · 11,659 · 373,396 · 416,928 · 443,670 |
| R3 | + split | 87,753 | 6/6 | 0 | 9,720 | 6,812 · 7,229 · 7,370 · 12,070 · 12,497 · 372,362 |
| R4 | + crossing walk | 155,862 | **2/6** | 4 | 2,618,817 | 9,117 · 13,279 · ≥2,549,882 · ≥2,687,753 · ≥2,719,363 · ≥2,728,819 |
| R5 | + fcompress (deliverable) | 145,514 | **2/6** | 4 | 2,371,070 | 13,931 · 14,591 · ≥2,351,095 · ≥2,391,046 · ≥2,700,069 · ≥2,887,321 |

Two transitions are robust at this sample size:

1. **The unmixed sandwich is not a search at all.** All six solved with **zero conflicts and zero
   decisions** — pure unit propagation. This is §3's structural result (57.6% of the circuit inside
   the constant frontier, 48.4% outside the target's cone) showing up directly as solver behaviour.
2. **The crossing walk is where the solve rate collapses**, 6/6 → 2/6 at a common cap. That is the
   stage whose stated job is anti-inversion spread, and it is the only stage that changes the
   *outcome* rather than the conflict count.

Everything between is dominated by **instance variance**: the R2 and R3 distributions are strongly
bimodal (three circuits at ~10⁴ conflicts, three at ~4×10⁵), which is why R3's median sits *below*
R2's. With six circuits per rung those medians are indicative, not resolved — the same large
instance variance the campaign reports at `n=11`–`12`.

**And every one of these 36 instances is solved by exhaustive enumeration in 0.0475 seconds** (§7).
The ladder measures how hard each stage makes life for *Kissat*, not how hard the circuit is.

---

## 9. Residual risks

**9.1 Linear algebra over the CNOT/XOR structure — quantitatively dead as posed.** The obvious
attack is "Gaussian-eliminate the affine part, brute-force the ANDs". Measured on the delivered
artifacts, **83.65% of gates are nonlinear** (control width ≥ 2) and mean control width is 2.905,
leaving ~6.25M AND terms at `n=128`. Done properly with consistency pruning the method collapses to
exactly the `2ⁿ` branch-on-the-support search, because the linear system is fully determined once the
`n` free bits are fixed. **Not a shortcut.**

**9.2 XOR-aware solving — untried.** The formula is 59% XOR by Kissat's own extractor and Kissat has
no GF(2) engine. One CryptoMiniSat run per existing CNF costs nothing and could move every number in
the archive.

**9.3 The semantic structural channel — now measured, and it is the largest free reduction.**
What §3 measures is the *syntactic* frontier: unit propagation is exactly 3-valued simulation, so it
marks a gate variable constant only when it is forced gate-locally. The **semantic** question is
strictly larger — is a gate variable constant as a Boolean function of the `n` free input bits on
the zero slice, and are two gate variables always equal or always complementary? Those are exactly
the facts a SAT sweeper hunts, and Kissat sees 0.046% of the formula per sweep environment while
completing 2 of 384 attempts.

This has now been run at 100% coverage on all three real `n=128` deliverables
(`semantic_probe.txt`, 1,024 bit-sliced zero-slice samples, **~10 s and 1.6 GB per artifact**):

| | c1 | c2 | c3 |
|---|---:|---:|---:|
| semantically constant gate variables | 36,830 | 36,761 | **36,779** (0.492%) |
| …versus what unit propagation finds | 673 | 620 | **680** (0.009%) |
| **gap** | **54.7×** | **59.3×** | **54.1×** |
| variables in equal/complementary classes | 689,316 | 695,773 | 689,110 |
| number of classes (largest) | 189,972 (410) | 193,383 (458) | 190,514 (416) |
| **eliminable by these facts alone** | **7.179%** | **7.218%** | **7.161%** |

**The counts are converged, not sampling artefacts.** Re-running c3 at 256 / 512 / 1,024 / 2,048
samples gives the constant count as **36,779 every time** — bit-identical, including the
const-0 / const-1 split (18,814 / 17,965). The equal-class count converges downward from 735,126 to
686,343 as false pairs are eliminated, settling near 7.14% (`semantic_convergence.txt`).

**What this means.** A 10-second probe removes ~7.14% of the instance — comparable to Kissat's
*entire* inprocessing union (7.564% at `n=256`) after 48 hours at 0.046% coverage. It does not break
anything: 7% is a constant-factor discount on a `2ⁿ` search, and the channel *shrinks* with `n`
(constants are 1.72% of the instance at `n=11` versus 0.492% at `n=128`). But it is the largest free
reduction available and it had never been run. The pipeline does suppress it — the eliminable
fraction falls from 17.79% after gadgetisation to 9.77% at the deliverable in the `n=11` ladder — but
it never reaches zero. The precedent for this channel mattering is the linear-band-fill result: a
structural fact recovered *outside* the solver by cheap forward simulation turned `n=32` from a
>2 h timeout into 10 minutes.

**9.4 Target width is a security parameter and is treated nowhere as one.** `2ⁿ` prices one exact,
full-width, externally chosen target. A partial target with `j` of `n` bits pinned costs ~`2^j`; `T`
targets cost `2ⁿ` total, not `T·2ⁿ`, because one sweep computes `G(x)` for all `x`.

**9.5 The generation seed caps security at a constant, independent of `n`.** `scripts/gss_mix.sh:104`
draws the master seed as `secrets.randbelow(2**63 - 16)` (reduced from 64 bits to avoid bash signed
overflow), and stage seeds are `SEED, SEED+1, SEED`. `docs/GSS_MIX.md` states that stages 1+2
regenerate **bit-identically** from `(n, seed)` — and `n` is public. A candidate seed is testable by
regenerating the gadget and comparing its zero-slice evaluation on one random input.

Critically, a candidate seed is testable by regenerating **`C` alone** — not the gadget.
`src/bin/gen_sandwich_gadget.rs:180-181` seeds a dedicated stream for the source computation
(`fastrand::seed(seed)`; the file header states "`seed` fixes C only"), and
`|C| = round(n·(log₂n)²)` is a public function of the published `n`. So the test is: regenerate
`C_s` (6,272 gates at `n=128`), evaluate `C_s(x₀)`, compare against
`y* = G(x₀,0,0,0)|block2` — which the attacker computes **once** from the public circuit. A
128-bit match identifies the seed, and it yields `C` outright, not merely a preimage.

**Measured** (`experiments/gss_sat_structure_20260816/scripts/seedrate.cpp`, one core, `-O3
-march=native`):

| n | `\|C\|` | candidate seeds/s/core | expected search (2^62) | on 640 threads | total work |
|---:|---:|---:|---:|---:|---:|
| 128 | 6,272 | 18,760 | 7.79 × 10⁶ core-years | 12,200 years | **~2^79 ops** |
| 256 | 16,384 | 7,334 | 1.99 × 10⁷ core-years | 31,100 years | **~2^81 ops** |

A factor of 2.6 between `n=128` and `n=256` — not a factor of `2^128`.

- Crossover with `2ⁿ` enumeration is near **`n ≈ 60–64`**: at `n=64` enumeration costs
  `2^64 × 2,591,259/64 ≈ 2^79.3` word-gate-ops while seed search costs ≈ `2^77.5`.
- **Therefore the structure-blind cost of this construction peaks near 2^79 operations at
  `n ≈ 64` and stays flat thereafter.** `n=128` and `n=256` have essentially the same ceiling. No
  document in the corpus states this.
- *Caveats:* the benchmark does not reproduce fastrand's exact stream, so the constant could move
  by a small factor, not orders. The attack assumes the artifact was generated by `gss_mix.sh`
  with its 63-bit draw. The inner loop is a tiny PRNG plus `|C|` XOR-AND operations with no memory
  traffic and no inter-candidate dependence — close to ideal for GPU or ASIC, so a large-scale
  attacker's wall-clock is far below the CPU figures above.
- Related: `provenance.json` publishes the full per-stage gate ledger, which is a seed-dependent
  fingerprint usable to reject candidate seeds before any function comparison.

This is the most actionable finding in this document. Widening the seed to `≥ n` bits (or drawing
independent per-stage seeds with total entropy `≥ n`) is a prerequisite for any asymptotic hardness
claim above `n = 63` to be well-posed.

---

## 10. What would settle the question

Ordered by information gain per CPU-hour.

1. **Semantic structural probe — ~2 s per artifact.** 1024-sample bit-sliced zero-slice simulation.
   Returns the real constant frontier, all equal/complementary/constant-XOR pairs and every wire's
   empirical support at **100% coverage**, against Kissat's 0.046%. Highest gain by orders of magnitude.
2. **Native enumeration at `n = 16, 20, 24` against the real artifacts — 3.6 s, 85 s, 30 min.**
   Partially done here at `n=11` and `n=128` throughput; finishing it closes the scaling question
   below `n ≈ 48` outright.
3. **The same at `n = 32, 40` on the fleet — 19 min, 4.8 days.**
4. **XOR-aware solving on the 29 existing CNFs.** No new artifacts needed.
5. **Finish the before/after ladder at `n = 12, 16, 20` with a real replacement store**, so phase A's
   DB re-encoding is included and the one genuinely missing arm exists at more than one width.
6. **Size-controlled ablation** — vary one safeguard while holding final gate count fixed. The only
   way to break the confound between "harder per gate" and "more gates".
7. **Widen the generation seed** (§9.5). Not a measurement but a prerequisite.
8. **Run the never-run tools**: `security_tests/attacks/kissat_hardcore_probe.c` (component decomposition),
   `security_tests/demixing/fcone.rs`, `security_tests/attacks/circuit_structure_scan.py`, bliss/nauty for the automorphism group. Each
   converts a "closed by proxy" verdict into a measurement.
9. **A treewidth lower bound and a minimal-backdoor extraction.** Cheap in CPU, expensive in human
   time; these convert the two open proof gaps into either closed doors or a break.

---

## 11. Lower bounds — what is actually establishable

Full study and data: `experiments/gss_sat_structure_20260816/REFUTATION_STUDY.md`.

**Headline.** No lower bound on the runtime of solving is established, and none is reachable by
these methods. The only proved bounds live in models that hide the gate list, and **both of those
models' hypotheses are false as deployed**. What survives is a tight bound in the structure-blind
model, capped by a constant.

### 11.1 Ceilings first — no lower bound can exceed these

| statement | model | status |
|---|---|---|
| enumeration: `2^(n−6)·G` word-ops ≈ `2^144.8` at n=128 | any classical machine, real artifacts | **measured** |
| seed search: **≈2^76.6–2^79** operations, **constant in n** | attacker holds `G` only | **measured** |

The seed ceiling binds and does not grow with `n`, so **no lower bound above ~2^79 can hold at any
width.** Everything below is capped by that.

### 11.2 Proved, but on hypotheses that fail as deployed

| statement | model | status |
|---|---|---|
| `Pr[invert] ≤ (q+1)/2ⁿ`; `q = Ω(2ⁿ)` for constant success | forward-oracle inversion of a **uniform random permutation** | **proved and tight** — enumeration matches with equality |
| `R` regenerations identify the seed w.p. `R/(2^63−16)` | seed→circuit as an **opaque random injection** | **proved in the model; the model is refuted** (see 11.3) |
| `Θ(2^(n/2))` quantum queries | quantum query model | **proved in the black-box model**; seed-space Grover bypasses it |

The first hypothesis fails because the family has support `≤ 2^63`: it is not a random permutation
and is not even 1-wise independent for `n ≥ 64`.

### 11.3 The seed→circuit map is not an injection — verified

`fastrand 2.3.0` is WyRand, whose **state is a bare additive counter** `s + i·K` with
`K = 0x2d358dccaa6c78a5`; only the output is mixed. Verified byte-exact against the shipped
`gen_sandwich_gadget` (`wyrand_stream_structure.txt`), at n=16, |C|=256:

| seed | result |
|---|---|
| `s + 1·K` | the wire-draw stream shifted by **one draw** (`c8d,b6e,9b0` → `8db,6e9,b0c`) |
| `s + 3·K` | **C with its first gate deleted** — matches base shifted by one gate in **255/255** positions |
| `s+1`, `s+2`, `s+12345` (controls) | **0/256** matches |

So the 2^63 seeds are windows into a single cyclic gate stream, not independent draws. This
**refutes the argument** that 2^62 regenerations is a lower bound: candidate tests are no longer
independent.

*What it does not do is make the attack 179× faster.* Walking the stream makes generation amortized
`O(1)`, but each window is a different circuit whose first gate changes everything downstream, so
evaluation stays `O(|C|)` per candidate and no incremental trick carries across windows. Measured
split at n=128: generation 86.7% of a candidate test, evaluation 13.3%. **Making generation free
caps the speedup at 7.52×**, i.e. ~2^76.6, not 2^71.5.

### 11.4 What the UNSAT arm licenses — and what it does not

Blocking the unique witness yields genuinely unsatisfiable instances, so refutation size becomes a
real measurable object. Three limits apply immediately:

1. **The backdoor caps the whole programme.** The `n`-input UP backdoor bounds tree-like resolution
   (and everything p-simulating it) at `2ⁿ·poly`. So the *best possible* proof-complexity result
   here is "resolution is no better than enumerating the `n` free bits" — which is already known,
   and sits far above the seed ceiling.
2. **Refutation hardness does not transfer to solving hardness.** `T_solve ≤ T_refute`; a solver may
   find the witness early and owe no refutation.
3. **R1 is not evidence about mixing.** Its 0-conflict result at every width is a theorem about any
   one-wire-modifying circuit with its *full* output pinned. Its security content is a warning: leak
   the other `3n` output coordinates and both solving and refuting collapse to `O(|G|)`.

### 11.5 What the measurements do support

- **CDCL's cost is not a floor.** Depth-3 cube decomposition beats direct CDCL by **≥70×** on the
  same instance (40,328 vs ≥2,826,110 conflicts), and enumeration beats both by ~10⁵×. Citing the
  campaign's conflict counts as evidence of hardness overstates by at least that much.
- **A sharp phase transition** at cube depth k=3→2: 142× cost for 2× the residual space.
- **The estimand is a mixture, not a unimodal law.** On a fixed circuit with **160** redrawn targets:
  65 solved at 3,781–15,317 conflicts (median 12,318), 95 censored at ≥734,018–≥1,808,893 — a **≥48×
  gap with zero observations inside it**. The lognormal/Weibull AFT family used throughout the
  campaign is the wrong model, and no median from a one-target-per-circuit design is meaningful.
- **A median result does hold, narrowly.** 95/160 censored gives an exact one-sided 95% lower bound
  of **0.5258 > 0.5**, so the median target on that circuit exceeds 900 s and ~10⁶ conflicts. It took
  160 targets: 24/40 gave 0.4578 and 69/120 gave 0.4958, both short despite point estimates near 0.6.
  Scope is one circuit, one width, one solver, one cap — it does not extrapolate.
- **`gss_n010_r01` was a coincidence, not a collapse** — reproduced exactly by setting the target to
  `G(0)`. It is a `2^-n` event, not a property of the construction, and no security bound follows
  from it.
- **Refutation scaling has exactly one completed point.** n=8 refutes at 8,920 conflicts; n=11, 12
  and 14 all censored (≥2.83M, ≥3.46M, ≥6.23M), as did all five n=11 circuits in R2 (2.46M–2.88M).
  Those censored counts measure *conflict throughput × cap*, not difficulty, so they are not
  comparable to one another and no exponent can be fitted. One point is the honest state of this arm.

### 11.6 The defensible statement

> **In the structure-blind model — an attacker who evaluates `G` but does not exploit its gate list
> — inverting a random target requires `Ω(2ⁿ)` evaluations, and this is tight: enumeration matches
> it. That bound is capped by seed search at ≈2^76.6–2^79 operations, constant in `n`, so the
> effective structure-blind lower bound is `min(2ⁿ·poly, ~2^76.6)` and saturates near `n ≈ 60`.
> Beyond that model nothing is established: an unconditional bound would separate P from NP, the
> proof-complexity route is capped at `2ⁿ·poly` by the backdoor and does not transfer from
> refutation to solving, and the measured CDCL costs are upper bounds on a search strategy that a
> depth-3 cube split already beats by 70×.**

The practical reading is unchanged from §9.5: widen the seed. Until it is at least `n` bits, no
property of the mixing pipeline can raise the structure-blind bound at all.

---

## Provenance

First-hand measurements in this document (scripts and outputs in
`experiments/gss_sat_structure_20260816/`): CNF-size identity and encoder byte-equivalence; gate-width
and affine-fraction censuses; constant-propagation frontiers; backward cone of influence; the six-seed
stage ladder; Kissat inprocessing, glue-tier and reuse statistics extracted from the archived verbose
logs; bit-sliced enumeration against the real `n=11` and `n=128` artifacts.

Figures taken from the existing campaign record are cited to
`experiments/gss_sat_scaling_20260810/` and `docs/`. Design-doc figures reproduced here (band-fill
invariant injection, split/crossing spread, mask-plan degrees, BVA decay) are quoted from those
sources and were not independently re-derived.

No seed, challenge input, solver model, witness or private manifest value was read or is reproduced.
